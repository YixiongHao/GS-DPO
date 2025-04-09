import os
import json
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import wandb
from tqdm import tqdm

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Initialize distributed training
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()

# Dataset class for DPO training
class DPODataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = item["question"]
        chosen = item["response_j"]  # Winning answer
        rejected = item["response_k"]  # Losing answer
        
        # Format for instruction-tuned model
        prompt_format = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

        {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        # Tokenize inputs
        prompt_tokens = self.tokenizer(prompt_format, return_tensors="pt", add_special_tokens=False)
        
        # Get prompt length to identify where response begins
        prompt_len = prompt_tokens.input_ids.size(1)
        
        # Combine prompt with responses
        chosen_input = self.tokenizer(
            prompt_format + chosen + self.tokenizer.eos_token,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        rejected_input = self.tokenizer(
            prompt_format + rejected + self.tokenizer.eos_token,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Return tensors without batch dimension
        return {
            "chosen_input_ids": chosen_input.input_ids[0],
            "chosen_attention_mask": chosen_input.attention_mask[0],
            "rejected_input_ids": rejected_input.input_ids[0],
            "rejected_attention_mask": rejected_input.attention_mask[0],
            "prompt_len": prompt_len
        }

# Collate function for batching
def collate_fn(batch):
    max_chosen_len = max(item["chosen_input_ids"].size(0) for item in batch)
    max_rejected_len = max(item["rejected_input_ids"].size(0) for item in batch)
    
    chosen_input_ids = []
    chosen_attention_mask = []
    rejected_input_ids = []
    rejected_attention_mask = []
    prompt_lens = []
    
    for item in batch:
        chosen_input_ids.append(pad_sequence(item["chosen_input_ids"], max_chosen_len))
        chosen_attention_mask.append(pad_sequence(item["chosen_attention_mask"], max_chosen_len))
        rejected_input_ids.append(pad_sequence(item["rejected_input_ids"], max_rejected_len))
        rejected_attention_mask.append(pad_sequence(item["rejected_attention_mask"], max_rejected_len))
        prompt_lens.append(item["prompt_len"])
    
    return {
        "chosen_input_ids": torch.stack(chosen_input_ids),
        "chosen_attention_mask": torch.stack(chosen_attention_mask),
        "rejected_input_ids": torch.stack(rejected_input_ids),
        "rejected_attention_mask": torch.stack(rejected_attention_mask),
        "prompt_lens": torch.tensor(prompt_lens)
    }

def pad_sequence(seq, target_len):
    if len(seq) >= target_len:
        return seq[:target_len]
    else:
        return torch.cat([seq, torch.zeros(target_len - len(seq), dtype=torch.long)])

# Compute DPO loss
def compute_dpo_loss(model, batch, beta=0.1, reference_model=None):
    chosen_input_ids = batch["chosen_input_ids"].to(model.device)
    chosen_attention_mask = batch["chosen_attention_mask"].to(model.device)
    rejected_input_ids = batch["rejected_input_ids"].to(model.device)
    rejected_attention_mask = batch["rejected_attention_mask"].to(model.device)
    prompt_lens = batch["prompt_lens"].to(model.device)
    
    # Forward pass for chosen responses
    chosen_outputs = model(
        input_ids=chosen_input_ids,
        attention_mask=chosen_attention_mask,
        return_dict=True
    )
    
    # Forward pass for rejected responses
    rejected_outputs = model(
        input_ids=rejected_input_ids,
        attention_mask=rejected_attention_mask,
        return_dict=True
    )
    
    # Calculate logprobs for policy model
    chosen_logprobs = get_response_logprobs(
        chosen_outputs.logits, 
        chosen_input_ids, 
        prompt_lens
    )
    
    rejected_logprobs = get_response_logprobs(
        rejected_outputs.logits, 
        rejected_input_ids, 
        prompt_lens
    )
    
    # If reference model is provided, calculate reference logprobs for KL penalty
    if reference_model is not None:
        with torch.no_grad():
            ref_chosen_outputs = reference_model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                return_dict=True
            )
            
            ref_rejected_outputs = reference_model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                return_dict=True
            )
            
            ref_chosen_logprobs = get_response_logprobs(
                ref_chosen_outputs.logits, 
                chosen_input_ids, 
                prompt_lens
            )
            
            ref_rejected_logprobs = get_response_logprobs(
                ref_rejected_outputs.logits, 
                rejected_input_ids, 
                prompt_lens
            )

            chosen_kl = chosen_logprobs - ref_chosen_logprobs
            rejected_kl = rejected_logprobs - ref_rejected_logprobs
    else:
        chosen_kl = 0
        rejected_kl = 0
    
    # DPO loss calculation
    logits = beta * (chosen_logprobs - rejected_logprobs - (chosen_kl - rejected_kl))
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    
    return loss

# Helper function to get response logprobs
def get_response_logprobs(logits, input_ids, prompt_lens):
    batch_size = logits.shape[0]
    logprobs_list = []
    
    for i in range(batch_size):
        prompt_len = prompt_lens[i]
        
        # Shift logits and input_ids by 1
        shifted_logits = logits[i, :-1, :]
        shifted_labels = input_ids[i, 1:]
        
        # Select response part (after prompt)
        response_logits = shifted_logits[prompt_len-1:]
        response_labels = shifted_labels[prompt_len-1:]
        
        # Get log probabilities for each token
        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
        
        # Get the log prob of the actual token
        token_logprobs = log_probs.gather(-1, response_labels.unsqueeze(-1)).squeeze(-1)
        
        # Filter out padding tokens
        mask = (response_labels != 0).float()
        
        # Sum up the log probs
        response_logprob = (token_logprobs * mask).sum() / (mask.sum() + 1e-8)
        logprobs_list.append(response_logprob)
    
    return torch.stack(logprobs_list)

# Load and prepare the dataset
def prepare_dataset(tokenizer, local_rank, world_size, batch_size=4):
    # Load dataset
    selected_dataset = load_dataset("parquet", data_files="../scratch/huggingface_cache/hub/datasets--vincentmin--eli5_rlhf_explainlikeim5/blobs/29854e787f65d51246c3b8fb570ff997c10bbf9c146b1e65a78b774b7954c6dc", split="train")

    # # Take only 20% of the training data
    # total_samples = len(full_dataset)
    
    # # Use deterministic seed for sampling
    # selected_indices = random.sample(range(total_samples), int(0.2 * total_samples))
    # selected_dataset = full_dataset.select(selected_indices)
    
    # if local_rank == 0:
    #     print(f"Selected {len(selected_dataset)} samples from {total_samples} total samples")
    
    # Create dataset and dataloader
    dpo_dataset = DPODataset(selected_dataset, tokenizer)
    
    # Create distributed sampler
    sampler = torch.utils.data.distributed.DistributedSampler(
        dpo_dataset, 
        num_replicas=world_size,
        rank=local_rank
    )
    
    dataloader = DataLoader(
        dpo_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader, sampler

# Load evaluation prompts
def load_evaluation_prompts(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_prompts = data["questions"]["benign"]
            prompts = random.sample(all_prompts, 200)
        return prompts
    except FileNotFoundError:
        # Create default evaluation prompts if file doesn't exist
        default_prompts = [
            "Explain quantum computing to me.",
            "How do black holes work?",
            "Why is the sky blue?",
            "How does the internet actually work?",
            "What causes climate change and what can we do about it?"
        ]

        print(f"File not found. Using default prompts: {default_prompts}")
        
        return default_prompts

# Evaluate model
def evaluate_model(model, tokenizer, eval_prompts, local_rank):
    if local_rank != 0:
        return {}  # Only run on main process
    
    model.eval()
    results = []
    
    for i, prompt in enumerate(eval_prompts):
        # Format prompt for the model
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

        {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        # Generate response
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        response = tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)
        
        # Create evaluation prompt for self-evaluation
        judge_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful research assistant that objectively evaluates conversations.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Please rate the following response on a scale from 0 to 10 based on how detailed and helpful it is.
        0 means not detailed at all, 10 means extremely detailed and helpful.

        Question: {prompt}

        Response: {response}

        Please provide only a numerical rating between 0 and 10.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        
        # Generate evaluation
        judge_input = tokenizer(judge_prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            judge_outputs = model.generate(
                input_ids=judge_input,
                max_new_tokens=10,
                temperature=0.1,
                top_p=0.95,
                do_sample=False
            )

        rating_text = tokenizer.decode(judge_outputs[0][judge_input.size(1):], skip_special_tokens=True)
        
        # Extract the numerical rating
        try:
            rating = int(''.join(filter(str.isdigit, rating_text.split()[0])))
            # Ensure rating is within bounds
            rating = max(0, min(10, rating))
        except (ValueError, IndexError):
            print('Failed to parse rating:', rating_text)
            continue
        
        results.append(rating)
    
    model.train()
    return torch.tensor(results)

# Main training function
def train(args, model, tokenizer, train_dataloader, train_sampler, eval_prompts, local_rank, reference_model):
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Set up scheduler
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Setup wandb for the main process
    if local_rank == 0:
        wandb.init(
            project="llama3-dpo",
            name=f"llama3-8b-dpo-lora-{args.beta}-lr{args.learning_rate}",
            config=args.__dict__
        )
        
        # Create directory for checkpoints
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Wrap model in DDP
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Training loop
    global_step = 0
    best_avg_rating = 0
    
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        
        progress_bar = tqdm(
            train_dataloader, 
            disable=(local_rank != 0),
            desc=f"Epoch {epoch+1}/{args.num_epochs}"
        )
        
        for step, batch in enumerate(progress_bar):
            # Forward and backward pass
            loss = compute_dpo_loss(ddp_model, batch, beta=args.beta, reference_model=reference_model)
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            # Update weights after accumulation steps
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            
            # Update progress bar
            if local_rank == 0:
                progress_bar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})
                
                # Log to wandb
                if global_step % args.logging_steps == 0:
                    wandb.log({
                        "train/loss": loss.item() * args.gradient_accumulation_steps,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "global_step": global_step
                    })

            #del loss
            #torch.cuda.empty_cache()
            global_step += 1
            
            # Evaluate periodically
            if global_step % args.eval_steps == 0:
                # Evaluate model
                eval_results = evaluate_model(ddp_model.module, tokenizer, eval_prompts, local_rank)
                
                if local_rank == 0 and eval_results:
                    # Calculate average rating
                    avg_rating = torch.mean(eval_results.float()).item()
                    
                    del eval_results
                    torch.cuda.empty_cache()

                    # Log evaluation results
                    wandb.log({
                        "eval/avg_detail_rating": avg_rating,
                        "global_step": global_step
                    })
                    
                    # Save if this is the best model
                    if avg_rating > best_avg_rating:
                        best_avg_rating = avg_rating
                        # Save best model
                        ddp_model.module.save_pretrained(os.path.join(args.output_dir, "best"))
                        tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))
                        
                        wandb.log({
                            "eval/best_avg_rating": best_avg_rating,
                            "global_step": global_step
                        })
                
                # Save checkpoint at regular intervals
                if local_rank == 0 and global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    ddp_model.module.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
        
        # Log epoch metrics
        if local_rank == 0:
            wandb.log({
                "train/epoch_loss": epoch_loss / len(train_dataloader),
                "train/epoch": epoch + 1,
                "global_step": global_step
            })
    
    # Save the final model
    if local_rank == 0:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        ddp_model.module.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        wandb.finish()

def main():
    # Setup argument parser
    import argparse
    parser = argparse.ArgumentParser(description="DPO training for Llama 3 8B")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", 
                        help="Model ID on Hugging Face")
    parser.add_argument("--output_dir", type=str, default="~/scratch/GS-DPO", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter for DPO")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Gradient accumulation steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=1000, help="Steps between saving checkpoints")
    parser.add_argument("--logging_steps", type=int, default=50, help="Steps between logging")
    parser.add_argument("--eval_file", type=str, default="/home/hice1/yhao96/GS-DPO/eval_data_flattended.json", 
                        help="Evaluation prompts file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    os.environ["OMP_NUM_THREADS"] = "60"
    
    # Setup distributed training
    local_rank, world_size = setup_distributed()
    
    if local_rank == 0:
        print(f"Training with {world_size} GPUs")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model in BF16 for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank}
    )

    # Create reference model (frozen, used for KL regularization)
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )

    # Keep reference model in eval mode and don't compute gradients for it
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Prepare dataset and dataloaders
    train_dataloader, train_sampler = prepare_dataset(
        tokenizer,
        local_rank,
        world_size,
        batch_size=args.batch_size
    )
    
    # Load evaluation prompts
    eval_prompts = load_evaluation_prompts(args.eval_file)
    
    if local_rank == 0:
        print(f"Loaded {len(eval_prompts)} evaluation prompts")
    
    # Train the model
    train(args, model, tokenizer, train_dataloader, train_sampler, eval_prompts, local_rank, reference_model)
    
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()