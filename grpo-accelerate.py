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
model_path = "meta-llama/Llama-3.1-8B-Instruct"      
beta = 0.03                                
num_pre_Q = 8  #                          
Q_batch_size = 1 #                            
all_steps = 1000 # number of iterations to train                        
max_prompt_length = 400 # how much text to produce
save_steps = 200 # when to checkpoint

# deepspeed config
ds_config = {
    "train_micro_batch_size_per_gpu": Q_batch_size*num_pre_Q,  
    "gradient_accumulation_steps": 2,                          
    "optimizer": {
        "type": "AdamW",                                       
        "params": { "lr": 1e-6 }                               
    },
    "bf16": {"enabled": True},                                 
    "zero_optimization": {
        "stage": 1,                                            
        "allgather_partitions": True,                          
        "allgather_bucket_size": 2e8,                          
        "overlap_comm": True,                                  
        "reduce_scatter": True,                                
        "reduce_bucket_size": 2e8,                             
        "contiguous_gradients": True,                          
        "stage3_gather_16bit_weights_on_model_save": True,     
        "offload_optimizer": {"device": "cpu"}                 
    }
}

# create dataloaders
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, 
        torch_dtype=torch.bfloat16, _attn_implementation="sdpa")

# create gen model and ref model
ref_model = model
gen_model = model

# use gsm8k as a test dataset
dataset = load_dataset("openai/gsm8k", "main", split="train")

# get questions, answers from dataset
mathQAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]

# set generation params
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

# probably want to replace this with dataloader/dataset
# def get_batch():
#     """
#     get a dict of inputs, rewards, and refs
#     """

#     # inputs from gsmk
#     data['inputs'] = 

#     #
#     data['rewards'] = 

#     data['refs'] = 

#     return data

# function to generate answers from prompts
def gen_answers(prompts):
    """
    given a list of prompts, generate answers text answers using the model.
    additionally, generate reference answers with the ref model
    """

    # make blank list for tip text
    tip_text = []

    # loop through prompts
    for x in prompts:
        
        # apply chat template to system prompt and user prompt
        tip_text.append(tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True)) # our prompt is the user prompt
    
    # convert the text to tokens
    tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    
    # grab the prompt length
    prompt_length = tip_inputs["input_ids"].shape[-1]
    
    # if the prompt length is greater than the max prompt length, return an empty list
    if prompt_length > max_prompt_length: 
        return []
    
    # map everything to device
    tip_inputs = {k: v.to(model.device) for k, v in tip_inputs.items()}

    # generate the completion ids with model
    with torch.inference_mode():
        tip_completion_ids = gen_model.generate(**tip_inputs, generation_config=generation_config)
    
    # remove the prompt from the completion ids, just isolate the completion
    completion_ids = tip_completion_ids[:, prompt_length:]
    
    # get answers in text form
    answers = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in completion_ids]

    # return answers
    return answers

# function to reward correctness
def reward_correct_math(item, answer):
    """
    evaluate mathematical answer correctness, return 1 if correct, -1 if incorrect
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
def reward_correct_yn(gt, pred) -> int: 
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

# function to generate samples from some inputs
def gen_samples(inputs):
    """
    generate answer tokens from inputs and calculate rewards.
    inputs are a list of Q/A dictionaries, each containing a question and answer.
    """

    # extract questions from input data
    prompts = [x["Q"] for x in inputs]

    # generate answers from prompts
    answers = gen_answers(prompts)
    
    # if no answers are generated, return None for all values
    if len(answers) == 0: return None, None, None, None
    
    # for each input, calculate the reward
    rewards = []
    for i, inp in enumerate(inputs):

        # loop through answers in the batch
        for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]: # answers are grouped by input, so we need to group them by input here. 8 answers per input
            
            # reward is equal to the sum of the correctness reward and the formatting reward
            rewards.append(reward_correct_math(inp, a) + reward_format(inp, a))
    
    # format the prompt template with the system prompt and user prompt
    prompts_text = [tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
    
    # convert the prompts/answers to tokens
    prompt_inputs = tokenizer(prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    output_ids = tokenizer(answers, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
    
    # return the prompt inputs, output ids, rewards, and answers
    return prompt_inputs["input_ids"], output_ids["input_ids"], torch.tensor(rewards, dtype=torch.float32), answers


# make a function to generate samples, instead of eval
def generate_mode(num=10, rank=0):

    # if we're on main process in distributed 
    if rank == 0: print('enter generate mode')

    # loop through number of samples
    for ii in range(num):
        
        # sample a batch of QAs
        inputs = random.sample(mathQAs, Q_batch_size)
        
        # generate samples from inputs
        prompt_inputs, output_ids, rewards, answers = gen_samples(inputs)
        
        # if no samples are generated, continue
        if prompt_inputs is None: continue

        # if we're on main process in distributed, print rewards
        if rank == 0: 
            print('rewards:', rewards)
            if ii == 5:
                print('answers:', answers[0]) # on the 5th iteration, print the answers
        
        # if the difference between the max and min rewards is less than 0.01, continue
        if (rewards.max() - rewards.min()).item() < 0.01: 
            continue
        
    #     # get the number of output ids to prompt ids
    #     rep = output_ids.shape[0] // prompt_inputs.shape[0]
        
    #     # get the max length of the prompt inputs
    #     prompt_length = prompt_inputs.shape[1]
        
    #     # repeat the prompt inputs to match the output ids
    #     Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
        
    #     # merge the prompt ids and output ids
    #     merged_ids = torch.cat([Qrep, output_ids], dim=1)
        
    #     # encode the prompt length, bytes of the merged ids, and bytes of the rewards
    #     xdata = make_bytes_list([json.dumps({"plen": prompt_length}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(rewards)])
        
    #     # upload the data to the data server
    #     requests.post(f"{data_server}/upload", data=xdata)
    
    # if rank == 0: print('exit generate mode')


# if we're only generating, generate and exit
if 'genonly' in sys.argv:
    model.to('cuda')
    generate_mode(999999)
    sys.exit()


# initialize the deepspeed engine
engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                               model_parameters=model.parameters())
gen_model = engine


# define a function for one step of GRPO
def GRPO_step(batch):
    
    # get the prompt length, inputs, rewards
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    rewards = batch['rewards'].to(engine.device)
    refs = batch['refs'].to(engine.device)

    # write a function to get logps 
    def get_per_token_logps(logits, input_ids):

        # "line up" logits and input ids
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        
        # make a list of per_token_logps for batch
        per_token_logps = []

        # loop through each row in the batch
        for logits_row, input_ids_row in zip(logits, input_ids):
            
            # apply softmax then logarithm in one to normalize then log the logits
            log_probs = logits_row.log_softmax(dim=-1)
            
            # get the log probs corresponding to the token indices in the input 
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            
            # append to the batch list
            per_token_logps.append(token_log_prob)
        
        # stack the list as a torch tensor
        return torch.stack(per_token_logps)
    
    # get logps_per_token for input
    per_token_logps = get_per_token_logps(engine(inputs).logits, inputs)
    
    # isolate for the answer
    per_token_logps = per_token_logps[:,prompt_length-1:]
    
    # move the refs to the device as well
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)

    # calculate kl between refs and new preds
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    
    # get a mask for the actual answer tokens
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    # get prereqs for calculating advantages
    mean_grouped_rewards = rewards.view(-1, num_pre_Q).mean(dim=1) # num_pre_q answers per input
    std_grouped_rewards = rewards.view(-1, num_pre_Q).std(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    
    # advantage = normalized reward
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    # loss = logps * advantages
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    
    # normalize the loss over all tokens
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    return loss

# make pbar
progress = range(1, all_steps+1)
if torch.distributed.get_rank() == 0: 
    progress = tqdm(progress)

# training loop
for step in progress:
    # get a batch for each epoch

    batch = get_batch()

    # generate if batch is none
    while batch is None:
        # generate_mode(rank=torch.distributed.get_rank())
        batch = get_batch()

    # get the grpo loss
    loss = GRPO_step(batch)

    # backprop / step
    # don't need zero_grad() since loss is incremented by rl function
    engine.backward(loss)
    engine.step()

    # tracking
    if torch.distributed.get_rank() == 0:    
        progress.set_description(f"Loss: {loss.item():.6f}")

    # saving
    if step % save_steps == 0:

        # wait for all gpus to reach barrier
        dist.barrier()
        if torch.distributed.get_rank() == 0:
            print('saving model')
            
            # move to cpu and save model/tokenizer
            save_name = f"./step_{step}"
            state_dict = engine.module.state_dict()
            state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
            engine.module.save_pretrained(save_name, state_dict=state_dict)
            tokenizer.save_pretrained(save_name)
        
        dist.barrier()