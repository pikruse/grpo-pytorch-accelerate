import os
import json
import random
import re
import torch
import deepspeed
from tqdm import tqdm
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset

# Set environment variable to enable parallel processing for the tokenizer
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

model_path = "/model/Qwen/Qwen2.5-7B"      # Model storage path
beta = 0.03                                # Optimizer beta parameter (unused here, possibly a placeholder)
num_pre_Q = 8                              # Number of pre-training questions per GPU (used to calculate micro-batch size)
Q_batch_size = 1                           # Mini-batch size per question
all_steps = 1000                           # Total training steps
max_prompt_length = 400                    # Maximum prompt length
save_steps = 200                           # Save the model every this many steps

# DeepSpeed configuration
ds_config = {
    "train_micro_batch_size_per_gpu": Q_batch_size * num_pre_Q,  # Micro-batch size per GPU
    "gradient_accumulation_steps": 2,                            # Gradient accumulation steps
    "optimizer": {
        "type": "AdamW",                                       # Optimizer type: AdamW
        "params": {"lr": 1e-6}                                   # Optimizer learning rate parameter
    },
    "bf16": {"enabled": True},                                 # Enable bfloat16 precision
    "zero_optimization": {
        "stage": 1,                                            # ZeRO optimization stage; 1 indicates the basic stage
        "allgather_partitions": True,                          # Enable allgather partitions
        "allgather_bucket_size": 2e8,                          # Allgather bucket size
        "overlap_comm": True,                                  # Enable overlapping of communication
        "reduce_scatter": True,                                # Enable reduce scatter
        "reduce_bucket_size": 2e8,                             # Reduce bucket size
        "contiguous_gradients": True,                          # Enable contiguous gradients
        "stage3_gather_16bit_weights_on_model_save": True,     # In stage 3, gather 16-bit weights when saving the model
        "offload_optimizer": {"device": "cpu"}                 # Offload optimizer to CPU
    }
}

# Load the pre-trained tokenizer from the specified path
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Load the pre-trained language model from the specified path
model = AutoModelForCausalLM.from_pretrained(model_path, 
        torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
# Assign the loaded model to the generation model variable
gen_model = model

# Load dataset (here we use OpenAI's GSM8K dataset)
#dataset = load_dataset("meta-math/GSM8K_zh", "default", split="train")
dataset = load_dataset("openai/gsm8k", "main", split="train")

# Process the dataset by separating questions and answers, and removing specific markers from the answers
QAs = [{'Q': x, 'A': y.split('####')[-1].strip()} 
       for x, y in zip(dataset['question'], dataset['answer'])]

# Generation configuration (GenerationConfig)
generation_config = GenerationConfig(
    max_new_tokens=512,                # Maximum number of new tokens to generate, set to 512
    do_sample=True,                    # Whether to perform sampling; set to True to enable random sampling
    temperature=0.9,                   # Sampling temperature controlling text diversity; higher means more diversity
    num_return_sequences=num_pre_Q,    # Number of sequences returned per input, set to num_pre_Q
    pad_token_id=tokenizer.pad_token_id,  # Padding token ID, using the tokenizer's padding token ID
)

# System prompt (system_prompt)
system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

# Answer generation function (gen_answers)
def gen_answers(prompts):
    """
    Generate answers based on the prompts.
    
    This function takes a list of prompts and uses the pre-trained tokenizer and model to generate corresponding answers.
    
    Args:
        prompts (list of str): List of input prompts.
    
    Returns:
        list of str: List of generated answers.
    """
    # Initialize list for prompt texts
    tip_text = []
    for x in prompts:
        # For each prompt, apply the chat template by adding system and user prompts
        tip_text.append(tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x}
        ], tokenize=False, add_generation_prompt=True))
    
    # Tokenize the prompt texts and apply padding
    tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    # Get the prompt length
    prompt_length = tip_inputs["input_ids"].shape[-1]
    # If the prompt length exceeds the maximum limit, return an empty list
    if prompt_length > max_prompt_length:
        return []
    # Move all tensors to the device where the model is located (GPU or CPU)
    tip_inputs = {k: v.to(model.device) for k, v in tip_inputs.items()}

    # Set inference mode (disable gradient computation to save memory)
    with torch.inference_mode():
        # Generate completion IDs using the generation configuration
        tip_completion_ids = gen_model.generate(**tip_inputs, generation_config=generation_config)
    # Exclude the prompt portion and obtain the completion IDs
    completion_ids = tip_completion_ids[:, prompt_length:]
    # Decode the completion IDs into text and remove the end-of-text marker
    answers = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in completion_ids]

    # Return the list of generated answers
    return answers

# Reward correction function (reward_correct)
def reward_correct(item, answer):
    """
    Compute the correctness reward for an answer.
    
    This function uses a regular expression to extract numbers from the answer and compares them with the ground truth.
    
    Args:
        item (dict): Dictionary containing the ground truth answer.
        answer (str): Generated answer.
    
    Returns:
        float: Reward value; 1 indicates correct, -1 indicates incorrect, -1.0 indicates failure to extract a number.
    """
    # Define a regex pattern to match decimals, fractions, or integers
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    # Find all numbers in the answer using the regex
    nums = re.findall(pattern, answer)  # Use regex to find all numbers in answer
    # If no numbers are found, return -1.0
    if len(nums) == 0:
        return -1.0
    # Get the last number (to compare the last number in the answer with the ground truth)
    lastnum = nums[-1]
    # Parse the last number
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    # Parse the ground truth answer
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    # Return 1 if the parsed answer matches the ground truth; otherwise, return -1
    return 1 if verify(ans, ground_truth) else -1

# Reward formatting function (reward_format)
def reward_format(item, answer):
    """
    Compute the format reward for an answer.
    
    This function checks whether the answer conforms to specific format requirements.
    
    Args:
        item (dict): Dictionary containing the ground truth answer.
        answer (str): Generated answer.
    
    Returns:
        float: Reward value; 1.25 indicates correct format, -1 indicates non-conformity.
    """
    # Define a regex pattern that requires the answer to follow the <think></think><answer></answer> format
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1

# Sample generation function (gen_samples)
def gen_samples(inputs):
    """
    Generate sample data.
    
    This function takes input data, generates corresponding answers, and computes rewards.
    
    Args:
        inputs (list of dict): List of input data, each dictionary containing a question and related information.
    
    Returns:
        tuple: A tuple containing input IDs, output IDs, rewards, and answers.
    """
    # Extract questions from the input data
    prompts = [x["Q"] for x in inputs]
    # Generate answers
    answers = gen_answers(prompts)
    # If no answers are generated, return None
    if len(answers) == 0:
        return None, None, None, None
    # Initialize the rewards list
    rewards = []
    for i, inp in enumerate(inputs):
        for a in answers[i * num_pre_Q:(i + 1) * num_pre_Q]:
            # Compute the total reward for each answer (including correctness and format)
            rewards.append(reward_correct(inp, a) + reward_format(inp, a))
    # Apply chat template to generate prompt texts
    prompts_text = [tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x}
    ], tokenize=False, add_generation_prompt=True) for x in prompts]
    # Tokenize and pad the prompt texts
    prompt_inputs = tokenizer(prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    # Tokenize and pad the answers
    output_ids = tokenizer(answers, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
    # Return the input IDs, output IDs, rewards, and answers
    return prompt_inputs["input_ids"], output_ids["input_ids"], torch.tensor(rewards, dtype=torch.float32), answers

# New function to generate a training batch locally (without any external server)
def local_get_batch():
    """
    Generate a training batch locally by sampling from the QAs dataset.
    
    Returns:
        dict: A dictionary with the following keys:
            - 'plen': prompt length (int)
            - 'inputs': merged tensor of prompt and generated output (torch.Tensor)
            - 'rewards': computed rewards (torch.Tensor)
            - 'refs': reference per-token log probabilities (torch.Tensor)
    """
    # Randomly sample Q_batch_size examples from QAs
    qa_samples = random.sample(QAs, Q_batch_size)
    prompt_inputs, output_ids, rewards, answers = gen_samples(qa_samples)
    if prompt_inputs is None:
        return None
    # Repeat prompt inputs to match the shape of output_ids
    rep = output_ids.shape[0] // prompt_inputs.shape[0]
    prompt_length = prompt_inputs.shape[1]
    Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
    # Concatenate the prompt and generated output to form the full sequence
    merged_ids = torch.cat([Qrep, output_ids], dim=1)
    
    # Compute reference log probabilities for the merged sequence using the current model
    with torch.inference_mode():
        ref_logits = gen_model(merged_ids).logits

    # Compute per-token log probabilities (same as in GRPO_step)
    def get_per_token_logps(logits, input_ids):
        logits = logits[:, :-1, :]  # Exclude the last logit (next-token prediction)
        input_ids = input_ids[:, 1:]  # Exclude the first token (no corresponding logit)
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    ref_logps = get_per_token_logps(ref_logits, merged_ids)
    
    # Return the batch as a dictionary
    return {
        'plen': prompt_length,
        'inputs': merged_ids,
        'rewards': rewards,
        'refs': ref_logps
    }

# GRPO_step function
def GRPO_step(batch):
    """
    Execute a GRPO step: compute and return the loss.
    
    Args:
        batch (dict): A dictionary containing:
            - 'plen' (int): Prompt length.
            - 'inputs' (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            - 'rewards' (torch.Tensor): Reward tensor of shape (batch_size, num_pre_Q).
            - 'refs' (torch.Tensor): Reference log probability tensor of shape (batch_size, sequence_length).
    
    Returns:
        torch.Tensor: The computed loss.
    """
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    rewards = batch['rewards'].to(engine.device)

    def get_per_token_logps(logits, input_ids):
        logits = logits[:, :-1, :]  # Exclude the last logit
        input_ids = input_ids[:, 1:]  # Exclude the first token
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    per_token_logps = get_per_token_logps(engine(inputs).logits, inputs)
    per_token_logps = per_token_logps[:, prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    
    # Compute KL divergence per token
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    # Create a mask for non-padding tokens
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    
    # Calculate mean and standard deviation of grouped rewards
    mean_grouped_rewards = rewards.view(-1, num_pre_Q).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, num_pre_Q).std(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    
    # Compute per-token loss
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    return loss

# DeepSpeed initialization
engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                               model_parameters=model.parameters())
# Assign the engine to the generation model variable
gen_model = engine

# Training loop (local mode without external server interactions)
progress = range(1, all_steps + 1)
if torch.distributed.get_rank() == 0:
    progress = tqdm(progress)

for step in progress:
    batch = local_get_batch()
    while batch is None:
        batch = local_get_batch()
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