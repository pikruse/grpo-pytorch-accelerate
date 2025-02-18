import json, os, shutil, re, random, io
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from bottle import request
import bottle, threading, queue


def tensor_to_bytes(t):
    """
    将PyTorch张量转换为字节。

    该函数使用 `torch.save` 将张量序列化，并将其存储在内存缓冲区中，然后返回缓冲区的字节内容。

    Args:
        t (torch.Tensor): 需要转换的PyTorch张量。

    Returns:
        bytes: 序列化后的张量字节数据。
    """
    # 创建一个内存缓冲区，用于存储序列化后的数据
    buffer = io.BytesIO()
    # 使用 torch.save 将张量序列化并写入缓冲区
    torch.save(t, buffer)
    # 返回缓冲区的字节内容
    return buffer.getvalue()


def bytes_to_tensor(b):
    """
    将字节数据转换为PyTorch张量。

    该函数使用 `torch.load` 从内存缓冲区中加载序列化后的张量，并返回该张量。

    Args:
        b (bytes): 需要转换的字节数据。

    Returns:
        torch.Tensor: 反序列化后的PyTorch张量。
    """
    return torch.load(io.BytesIO(b))


def make_bytes_list(blist):
    """
    将字节列表转换为单个字节串。

    该函数首先将列表的长度写入缓冲区，然后依次将每个字节串的长度及其内容写入缓冲区，最后返回整个缓冲区的字节内容。

    Args:
        blist (list of bytes): 需要转换的字节列表。

    Returns:
        bytes: 包含列表长度和每个字节串长度及其内容的单个字节串。
    """
    # 创建一个内存缓冲区，用于存储序列化后的数据
    buffer = io.BytesIO()
    # 将列表的长度写入缓冲区，使用4字节大端编码
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        # 将每个字节串的长度写入缓冲区，使用4字节大端编码
        buffer.write(len(b).to_bytes(4, 'big'))
        # 将字节串的内容写入缓冲区
        buffer.write(b)
    # 返回缓冲区的字节内容
    return buffer.getvalue()


def bytes_list_to_list(b):
    """
    将单个字节串转换为字节列表。

    该函数首先从字节串中读取列表的长度，然后依次读取每个字节串的长度及其内容，并将其添加到列表中，最后返回字节列表。

    Args:
        b (bytes): 需要转换的单个字节串。

    Returns:
        list of bytes: 反序列化后的字节列表。
    """
    # 创建一个内存缓冲区，用于读取字节数据
    buffer = io.BytesIO(b)
    # 从缓冲区中读取列表的长度，使用4字节大端编码
    num = int.from_bytes(buffer.read(4), 'big')
    # 初始化字节列表
    blist = []
    for _ in range(num):
        # 读取每个字节串的长度，使用4字节大端编码
        l = int.from_bytes(buffer.read(4), 'big')
        # 读取字节串的内容并添加到列表中
        blist.append(buffer.read(l))
    # 返回字节列表
    return blist


if __name__ == '__main__':   
    
    # 设置环境变量，启用分词器的并行处理
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    
    # 模型存储路径
    model_path = "/model/Qwen/Qwen2.5-7B"

    # 从指定路径加载预训练的因果语言模型（AutoModelForCausalLM），并设置模型参数
    ref_model = AutoModelForCausalLM.from_pretrained(model_path,
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
    # 将模型设置为评估模式，禁用批归一化层和Dropout层的训练行为
    ref_model.eval()
    # 禁用模型参数的梯度计算，以节省显存和加速推理
    ref_model.requires_grad_(False)

    def get_per_token_logps(input_ids):
        """
        计算每个输入token的对数概率。

        该函数接收输入token的ID，计算模型输出的对数概率，并返回每个token的对数概率。

        Args:
            input_ids (torch.Tensor): 输入token的ID张量，形状为 (batch_size, sequence_length)。

        Returns:
            torch.Tensor: 每个token的对数概率张量，形状为 (batch_size, sequence_length)。
        """
        # 获取模型的输出logits，形状为 (batch_size, sequence_length, vocab_size)
        logits = ref_model(input_ids).logits  # (B, L, V)

        # 排除最后一个logit，因为它对应于下一个token的预测，形状变为 (batch_size, sequence_length - 1, vocab_size)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        # 排除第一个输入ID，因为我们没有对应的logits，形状变为 (batch_size, sequence_length - 1)
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

        # 计算输入token的对数概率。使用循环以减少内存峰值。
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

    # 原始数据队列，用于存储接收到的数据
    raw_queue = queue.Queue()
    # 结果队列，用于存储处理后的数据
    result_queue = queue.Queue()

    # 创建一个 Bottle 应用实例
    app = bottle.Bottle()

    # 定义一个POST接口，路径为 /upload
    @app.route('/upload', method='POST')
    
    def do_upload():
        # 读取请求的主体内容（二进制数据）
        dd = request.body.read()
        # 将字节串转换为字节列表
        dd = bytes_list_to_list(dd)
        # 解析第一个字节串为JSON，并存储在字典的 'base' 键中
        data = {'base': json.loads(dd[0])} 
        # 将第二个字节串转换为张量，并存储在 'inputs' 键中
        data['inputs'] = bytes_to_tensor(dd[1])
        # 将第三个字节串转换为张量，并存储在 'rewards' 键中
        data['rewards'] = bytes_to_tensor(dd[2])
        # 将数据放入原始数据队列中
        raw_queue.put(data)
        # 打印接收到的数据形状和奖励信息
        print('receive', data['inputs'].shape, data['rewards'])

    # 定义一个GET接口，路径为 /get
    @app.route('/get', method='GET')
    def do_get():
        # 检查结果队列是否为空
        # 如果为空，则返回 'empty' 的字节串
        if result_queue.empty(): return b'empty'
        # 如果不为空，则从队列中获取结果并返回
        return result_queue.get()

    # 使用 Tornado 服务器在指定的主机和端口上运行 Bottle 应用
    def run_server(): bottle.run(app, host='0.0.0.0', port=59875, server='tornado')
    # 启动一个非守护线程来运行服务器
    threading.Thread(target=run_server, daemon=False).start()

    while True:
        # 从原始数据队列中获取数据
        d = raw_queue.get()
        # 获取提示长度
        prompt_length = d['base']['plen']
        # 设置推理模式，禁用梯度计算以节省显存
        with torch.inference_mode():
            # 计算每个token的对数概率
            per_token_logps = get_per_token_logps(d['inputs'].to(ref_model.device))
        # 根据提示长度截取对数概率
        per_token_logps = per_token_logps[:,prompt_length-1:]
        # 将数据打包为字节列表
        xdata = make_bytes_list([json.dumps(d['base']).encode(),     # 将 'base' 数据编码为JSON并转换为字节
                                 tensor_to_bytes(d['inputs']),       # 将 'inputs' 张量转换为字节
                                 tensor_to_bytes(d['rewards']),      # 将 'rewards' 张量转换为字节
                                 tensor_to_bytes(per_token_logps)])  # 将对数概率张量转换为字节
        # 将打包后的数据放入结果队列中
        result_queue.put(xdata)

    
