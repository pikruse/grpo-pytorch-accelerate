# imports 
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

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# define hyperparams
model_path = "/model/Qwen/Qwen2.5-7B"      
beta = 0.03                                
num_pre_Q = 8                              
Q_batch_size = 1                           
all_steps = 1000                           
max_prompt_length = 400                    
save_steps = 200


# accelerate config
accelerate_config = {

}

# create dataloaders
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, 
        torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
gen_model = model


# use gsm8k as a test dataset
dataset = load_dataset("openai/gsm8k", "main", split="train")

# get questions, answers from dataset
QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]


# describe how model will generate
generation_config = GenerationConfig(
            max_new_tokens=512,   
            do_sample=True,       
            temperature=0.9,      
            num_return_sequences=num_pre_Q,   
            pad_token_id=tokenizer.pad_token_id,   
        )

# define system prompt to encourage reasoning
system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""


def gen_answers(prompts):

    
    tip_text = []
    for x in prompts:
        
        tip_text.append(tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
    
    
    tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    
    prompt_length = tip_inputs["input_ids"].shape[-1]
    
    if prompt_length > max_prompt_length: return []
    
    tip_inputs = {k: v.to(model.device) for k, v in tip_inputs.items()}

    
    with torch.inference_mode():
        
        tip_completion_ids = gen_model.generate(**tip_inputs, generation_config=generation_config)
    
    completion_ids = tip_completion_ids[:, prompt_length:]
    
    answers = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in completion_ids]

    
    return answers

# function to reward correctness
def math_reward_correct(item, answer):

    """
    evaluate mathematical answer correctness, 
    """

    # pattern: 1 or more digits, followed by a period, followed by 1 or more digits OR 1 or more digits, followed by a forward slash, followed by 1 or more digits OR 1 or more digits
    # for our text questions, we can change this to a "yes/no" pattern or something similar 
    pattern = r'\d+\.\d+|\d+/\d+|\d+'

    # find all numbers in the answer
    nums = re.findall(pattern, answer)

    # if no numbers are found, return -1
    if len(nums) == 0: 
        return -1.0

    # get the last number (the answer)
    lastnum = nums[-1]

    # parse the answer with the math_verify library
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])

    # parse the ground truth with the math_verify library
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])

    # return 1 if the answer is correct, else return -1
    return 1 if verify(ans, ground_truth) else -1

# define reward function for node connectivity
def yn_reward_correct(gt, pred) -> int: 
    """
    given a yes/no answer and ground truth, return 1 if correct, -1 if incorrect
    """

    # extract answer from prediction
    ans = ''.join(re.findall(r"<answer>(.*?)</answer>", pred)) 
    ans = ans.lower() 

    # if the model didn't produce an answer, return -1
    if ans == "":
        return -1
    
    # if the model produced an answer, compare it to the ground truth - return 1 if correct, -1 if incorrect
    if ans == gt:
        return 1
    else:
        return -1

# function to reward formatting
def reward_format(item, answer):
    """
    if the answer is in the correct format, reward 1.25, else reward -1
    """
    
    # answer format
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"

    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1

# 
def gen_samples(inputs):

    prompts = [x["Q"] for x in inputs]

    answers = gen_answers(prompts)
    
    if len(answers) == 0: return None, None, None, None
    
    rewards = []
    for i, inp in enumerate(inputs):
        for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
            
            rewards.append(reward_correct(inp, a) + reward_format(inp, a))
    
    prompts_text = [tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
    
    prompt_inputs = tokenizer(prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    
    output_ids = tokenizer(answers, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
    
    
    return prompt_inputs["input_ids"], output_ids["input_ids"], torch.tensor(rewards, dtype=torch.float32), answers



def generate_mode(num=10, rank=0):

    if rank == 0: print('enter generate mode')
    for ii in range(num):
        
        inputs = random.sample(QAs, Q_batch_size)
        
        prompt_inputs, output_ids, rewards, answers = gen_samples(inputs)
        
        if prompt_inputs is None: continue
        if rank == 0: 
            
            print('rewards:', rewards)
            if ii == 5:
                
                print('answers:', answers[0])
        
        if (rewards.max() - rewards.min()).item() < 0.01: continue

        rep = output_ids.shape[0] // prompt_inputs.shape[0]
        
        prompt_length = prompt_inputs.shape[1]
        
        Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
        
        merged_ids = torch.cat([Qrep, output_ids], dim=1)
        
        xdata = make_bytes_list([json.dumps({"plen": prompt_length}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(rewards)])
        
        requests.post(f"{data_server}/upload", data=xdata)
    
    if rank == 0: print('exit generate mode')



if 'genonly' in sys.argv:
    
    model.to('cuda')
    
    generate_mode(999999)
    
    sys.exit()



engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                               model_parameters=model.parameters())

gen_model = engine



def GRPO_step(batch):

    
    prompt_length = batch['plen']
    
    inputs = batch['inputs'].to(engine.device)
    
    rewards = batch['rewards'].to(engine.device)

    def get_per_token_logps(logits, input_ids):

        
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

        
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            
            log_probs = logits_row.log_softmax(dim=-1)
            
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            
            per_token_logps.append(token_log_prob)
        
        return torch.stack(per_token_logps)
    
    
    per_token_logps = get_per_token_logps(engine(inputs).logits, inputs)
    
    per_token_logps = per_token_logps[:,prompt_length-1:]
    
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)

    
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    
    mean_grouped_rewards = rewards.view(-1, num_pre_Q).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, num_pre_Q).std(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    return loss



generate_mode(rank=torch.distributed.get_rank())




progress = range(1, all_steps+1)


if torch.distributed.get_rank() == 0: progress = tqdm(progress)

for step in progress:
    
    batch = get_batch()
    while batch is None:
        
        generate_mode(rank=torch.distributed.get_rank())
        
        batch = get_batch()

    
    loss = GRPO_step(batch)

    
    engine.backward(loss)
    
    engine.step()

    if torch.distributed.get_rank() == 0:
        
        progress.set_description(f"Loss: {loss.item():.6f}")

    if step % save_steps == 0:
        
        dist.barrier()
        if torch.distributed.get_rank() == 0:
            
            print('saving model')
            
            save_name = f"./step_{step}"
            
            state_dict = engine.module.state_dict()
            
            state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
            
            engine.module.save_pretrained(save_name, state_dict=state_dict)
            
            tokenizer.save_pretrained(save_name)
        
        dist.barrier()
