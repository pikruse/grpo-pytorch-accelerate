import json, os, shutil, re, random, requests, io, sys
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
from datasets import load_dataset
from math_verify import parse, verify, ExprExtractionConfig
import deepspeed
from tqdm import tqdm

from data_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list


# 设置环境变量，启用分词器的并行处理
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


model_path = "/model/Qwen/Qwen2.5-7B"      # 模型存储路径
beta = 0.03                                # 优化器的beta参数（此处未使用，可能为占位符）
num_pre_Q = 8                              # 每个GPU的预训练问题数量（用于计算微批量大小）
Q_batch_size = 1                           # 每个问题的小批量大小
all_steps = 1000                           # 总训练步数
max_prompt_length = 400                    # 最大提示长度
save_steps = 200                           # 每隔多少步保存一次模型


# DeepSpeed配置
ds_config = {
    "train_micro_batch_size_per_gpu": Q_batch_size*num_pre_Q,  # 每个GPU的微批量大小
    "gradient_accumulation_steps": 2,                          # 梯度累积步数
    "optimizer": {
        "type": "AdamW",                                       # 优化器类型为AdamW
        "params": { "lr": 1e-6 }                               # 优化器的学习率参数
    },
    "bf16": {"enabled": True},                                 # 启用bfloat16精度
    "zero_optimization": {
        "stage": 1,                                            # ZeRO优化阶段，1表示基本阶段
        "allgather_partitions": True,                          # 启用allgather分区
        "allgather_bucket_size": 2e8,                          # allgather桶大小
        "overlap_comm": True,                                  # 启用通信重叠
        "reduce_scatter": True,                                # 启用reduce scatter
        "reduce_bucket_size": 2e8,                             # reduce桶大小
        "contiguous_gradients": True,                          # 启用连续梯度
        "stage3_gather_16bit_weights_on_model_save": True,     # 在模型保存时启用阶段3的16位权重gather
        "offload_optimizer": {"device": "cpu"}                 # 将优化器卸载到CPU
    }
}


# 数据服务器地址，假设数据通过该地址提供
data_server = "http://localhost:5678"


def get_batch():
    """
    从数据服务器获取一批数据。

    该函数尝试从数据服务器获取数据，如果成功，则解析并返回数据；否则，返回None。

    Returns:
        dict or None: 解析后的数据字典或None。
    """

    try:
        # 发送GET请求到数据服务器
        r = requests.get(f"{data_server}/get").content
        if r == b'empty': return None
    except: return None

    # 将字节串转换为字节列表
    dd = bytes_list_to_list(r)
    # 解析第一个字节串为JSON，并存储在字典中
    data = json.loads(dd[0]) 
    # 将第二个字节串转换为张量，并存储在 'inputs' 键中
    data['inputs'] = bytes_to_tensor(dd[1])
    # 将第三个字节串转换为张量，并存储在 'rewards' 键中
    data['rewards'] = bytes_to_tensor(dd[2])
    # 将第四个字节串转换为张量，并存储在 'refs' 键中
    data['refs'] = bytes_to_tensor(dd[3])
    # 返回数据字典
    return data


# 从指定路径加载预训练的分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 从指定路径加载预训练的语言模型
model = AutoModelForCausalLM.from_pretrained(model_path, 
        torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
# 将加载的模型赋值给生成模型变量
gen_model = model


# 加载数据集（这里使用的是 OpenAI 的 GSM8K 数据集）
#dataset = load_dataset("meta-math/GSM8K_zh", "default", split="train")
dataset = load_dataset("openai/gsm8k", "main", split="train")

# 处理数据集，将问题和答案分开，并去除答案中的特定标记
QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]


# 生成配置（GenerationConfig）
generation_config = GenerationConfig(
            max_new_tokens=512,   # 生成的最大新token数量，设置为512
            do_sample=True,       # 是否进行采样，设置为True以启用随机采样
            temperature=0.9,      # 采样温度，控制生成文本的多样性，值越高多样性越高
            num_return_sequences=num_pre_Q,   # 每个输入返回的序列数量，设置为num_pre_Q
            pad_token_id=tokenizer.pad_token_id,   # 填充token的ID，使用分词器的填充token ID
        )


# 系统提示（system_prompt）
system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""


# 生成答案函数（gen_answers）
def gen_answers(prompts):
    """
    根据提示生成答案。

    该函数接收一组提示，使用预训练的分词器和模型生成相应的答案。

    Args:
        prompts (list of str): 输入的提示列表。

    Returns:
        list of str: 生成的答案列表。
    """
    # 初始化提示文本列表
    tip_text = []
    for x in prompts:
        # 对每个提示，应用聊天模板，添加系统提示和用户提示
        tip_text.append(tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
    
    # 对提示文本进行分词，并进行填充
    tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    # 获取提示的长度
    prompt_length = tip_inputs["input_ids"].shape[-1]
    # 如果提示长度超过最大限制，则返回空列表
    if prompt_length > max_prompt_length: return []
    # 将所有张量移动到模型所在的设备（GPU或CPU）
    tip_inputs = {k: v.to(model.device) for k, v in tip_inputs.items()}

    # 设置推理模式，禁用梯度计算以节省显存
    with torch.inference_mode():
        # 使用生成配置生成补全的ID
        tip_completion_ids = gen_model.generate(**tip_inputs, generation_config=generation_config)
    # 排除提示部分，获取补全的ID
    completion_ids = tip_completion_ids[:, prompt_length:]
    # 将补全的ID解码为文本，并去除结束标记
    answers = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in completion_ids]

    # 返回生成的答案列表
    return answers


# 奖励校正函数（reward_correct）
def reward_correct(item, answer):
    """
    计算答案的正确性奖励。

    该函数使用正则表达式提取答案中的数字，并与标准答案进行比较。

    Args:
        item (dict): 包含标准答案的字典。
        answer (str): 生成的答案。

    Returns:
        float: 奖励值，1表示正确，-1表示错误，-1.0表示无法提取数字。
    """
    # 定义正则表达式模式，匹配整数、小数或分数
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    # 在答案中查找所有匹配的数字
    nums = re.findall(pattern, answer) # 使用正则表达式在answer中查找所有数字
    # 如果没有找到数字，则返回 -1.0
    if len(nums) == 0: return -1.0
    # 获取最后一个数字
    lastnum = nums[-1] # 用answer中最后一个数字和ground_truth做比较
    # 解析最后一个数字
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    # 解析标准答案
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])

    # 如果解析后的答案与标准答案匹配，则返回1；否则返回-1
    return 1 if verify(ans, ground_truth) else -1


# 奖励格式化函数（reward_format）
def reward_format(item, answer):
    """
    计算答案的格式奖励。

    该函数检查答案是否符合特定的格式要求。

    Args:
        item (dict): 包含标准答案的字典。
        answer (str): 生成的答案。

    Returns:
        float: 奖励值，1.25表示符合格式，-1表示不符合。
    """
    # 定义正则表达式模式，要求答案符合 <think></think><answer></answer> 的格式
    # pattern = r"^<think>(?:(?!</?think>)[\s\S]*?)</think>\s*<answer>(?:(?!</?answer>)[\s\S]*?)</answer><\|im_end\|>$"
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"

    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1


# 生成样本函数（gen_samples）
def gen_samples(inputs):
    """
    生成样本数据。

    该函数接收输入数据，生成相应的答案，并计算奖励。

    Args:
        inputs (list of dict): 输入数据列表，每个字典包含问题和相关信息。

    Returns:
        tuple: 包含输入ID、输出ID、奖励和答案的元组。
    """
    # 从输入数据中提取问题
    prompts = [x["Q"] for x in inputs]
    # 生成答案
    answers = gen_answers(prompts)
    # 如果没有生成答案，则返回None
    if len(answers) == 0: return None, None, None, None
    # 初始化奖励列表
    rewards = []
    for i, inp in enumerate(inputs):
        for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
            # 计算每个答案的总奖励，包括正确性和格式
            rewards.append(reward_correct(inp, a) + reward_format(inp, a))
    # 应用聊天模板生成提示文本
    prompts_text = [tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
    # 对提示文本进行分词和填充
    prompt_inputs = tokenizer(prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    # 对答案进行分词和填充
    output_ids = tokenizer(answers, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
    
    # 返回输入ID、输出ID、奖励和答案
    return prompt_inputs["input_ids"], output_ids["input_ids"], torch.tensor(rewards, dtype=torch.float32), answers


# 生成模式函数（generate_mode）
def generate_mode(num=10, rank=0):
    """
    进入生成模式。

    该函数生成指定数量的样本，并将其上传到数据服务器。

    Args:
        num (int, optional): 要生成的样本数量，默认为10。
        rank (int, optional): 进程排名，默认为0。
    """
    # 如果是主进程，则打印进入生成模式的信息
    if rank == 0: print('enter generate mode')
    for ii in range(num):
        # 从QAs列表中随机抽取Q_batch_size个样本
        inputs = random.sample(QAs, Q_batch_size)
        # 生成样本数据
        prompt_inputs, output_ids, rewards, answers = gen_samples(inputs)
        # 如果没有生成样本，则继续
        if prompt_inputs is None: continue
        if rank == 0: 
            # 如果是主进程，则打印奖励信息
            print('rewards:', rewards)
            if ii == 5:
                # 如果是第6个样本，则打印第一个答案
                print('answers:', answers[0])
        # 如果所有奖励的差异小于0.01，则继续
        if (rewards.max() - rewards.min()).item() < 0.01: continue
        # 计算重复次数
        rep = output_ids.shape[0] // prompt_inputs.shape[0]
        # 获取提示长度
        prompt_length = prompt_inputs.shape[1]
        # 重复提示以匹配输出ID的形状
        Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
        # 将提示和输出ID连接起来
        merged_ids = torch.cat([Qrep, output_ids], dim=1)
        # 将数据打包为字节列表
        xdata = make_bytes_list([json.dumps({"plen": prompt_length}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(rewards)])
        # 将数据上传到数据服务器
        requests.post(f"{data_server}/upload", data=xdata)
    # 如果是主进程，则打印退出生成模式的信息
    if rank == 0: print('exit generate mode')


# 检查命令行参数
if 'genonly' in sys.argv:
    # 将模型移动到GPU（CUDA）
    model.to('cuda')
    # 调用生成模式函数，生成大量样本（999999次）
    generate_mode(999999)
    # 退出程序
    sys.exit()


# DeepSpeed初始化
engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                               model_parameters=model.parameters())
# 将引擎赋值给生成模型变量
gen_model = engine


# GRPO_step 函数
def GRPO_step(batch):
    """
    执行GRPO步骤，计算损失并返回。

    Args:
        batch (dict): 包含以下键的字典：
            - 'plen' (int): 提示长度。
            - 'inputs' (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length)。
            - 'rewards' (torch.Tensor): 奖励张量，形状为 (batch_size, num_pre_Q)。
            - 'refs' (torch.Tensor): 参考对数概率张量，形状为 (batch_size, sequence_length)。

    Returns:
        torch.Tensor: 计算得到的损失。
    """
    # 获取提示长度
    prompt_length = batch['plen']
    # 将输入张量移动到引擎所在的设备（通常是GPU）
    inputs = batch['inputs'].to(engine.device)
    # 将奖励张量移动到引擎所在的设备
    rewards = batch['rewards'].to(engine.device)

    def get_per_token_logps(logits, input_ids):
        """
        计算每个token的对数概率。

        Args:
            logits (torch.Tensor): 模型输出的logits，形状为 (batch_size, sequence_length, vocab_size)。
            input_ids (torch.Tensor): 输入token的ID，形状为 (batch_size, sequence_length)。

        Returns:
            torch.Tensor: 每个token的对数概率，形状为 (batch_size, sequence_length - 1)。
        """
        # 排除最后一个logit，因为它对应于下一个token的预测，形状变为 (batch_size, sequence_length - 1, vocab_size)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        # 排除第一个输入ID，因为我们没有对应的logits，形状变为 (batch_size, sequence_length - 1)
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

        # 初始化存储每个token对数概率的列表
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            # 计算log softmax，形状为 (sequence_length - 1, vocab_size)
            log_probs = logits_row.log_softmax(dim=-1)
            # 获取每个token对应的对数概率，形状为 (sequence_length - 1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            # 将对数概率添加到列表中
            per_token_logps.append(token_log_prob)
        # 将列表转换为张量，形状为 (batch_size, sequence_length - 1)
        return torch.stack(per_token_logps)
    
    # 计算每个token的对数概率
    per_token_logps = get_per_token_logps(engine(inputs).logits, inputs)
    # 根据提示长度截取对数概率
    per_token_logps = per_token_logps[:,prompt_length-1:]
    # 将参考对数概率张量移动到与per_token_logps相同的设备
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)

    # 计算每个token的KL散度
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    # 创建完成掩码，标记非填充token的位置
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    # 计算分组奖励的均值和标准差
    mean_grouped_rewards = rewards.view(-1, num_pre_Q).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, num_pre_Q).std(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    # 计算优势
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    # 计算每个token的损失
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    # 计算总损失
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    return loss


# 进入生成模式
generate_mode(rank=torch.distributed.get_rank())


# 训练循环
# 定义训练步数范围
progress = range(1, all_steps+1)

# 如果是主进程，则使用tqdm显示进度条
if torch.distributed.get_rank() == 0: progress = tqdm(progress)

for step in progress:
    # 从数据服务器获取一批数据
    batch = get_batch()
    while batch is None:
        # 如果没有数据，则进入生成模式
        generate_mode(rank=torch.distributed.get_rank())
        # 重新获取数据
        batch = get_batch()

    # 计算损失
    loss = GRPO_step(batch)

    # 反向传播计算梯度
    engine.backward(loss)
    # 更新模型参数
    engine.step()

    if torch.distributed.get_rank() == 0:
        # 更新进度条描述，显示当前损失
        progress.set_description(f"Loss: {loss.item():.6f}")

    if step % save_steps == 0:
        # 同步所有进程
        dist.barrier()
        if torch.distributed.get_rank() == 0:
            # 打印保存模型的信息
            print('saving model')
            # 定义保存路径
            save_name = f"./step_{step}"
            # 获取模型的状态字典
            state_dict = engine.module.state_dict()
            # 将状态字典中的张量移动到CPU
            state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
            # 保存模型
            engine.module.save_pretrained(save_name, state_dict=state_dict)
            # 保存分词器
            tokenizer.save_pretrained(save_name)
        # 同步所有进程
        dist.barrier()
