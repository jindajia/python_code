import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
from datasets import load_dataset
import deepspeed

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning GPT-2 with DeepSpeed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed from distributed launcher")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.00015, help="Learning rate")
    return parser.parse_args()

args = parse_args()

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load and preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
tokenized_dataset = dataset.map(preprocess_function, batched=True)
dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size)

# DeepSpeed configuration
deepspeed_config = {
    "fp16": {
      "enabled": True,
      "loss_scale": 0,
      "loss_scale_window": 1000,
      "hysteresis": 2,
      "min_loss_scale": 1
    },
    "zero_optimization": {
      "stage": 3,
      "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True
      },
      "offload_param": {
        "device": "cpu",
        "pin_memory": True
      },
      "overlap_comm": True,
      "contiguous_gradients": True,
      "sub_group_size": 1e9,
      "reduce_bucket_size": 5e8,
      "stage3_max_live_parameters": 1e9,
      "stage3_max_reuse_distance": 1e9,
      "stage3_prefetch_bucket_size": 5e7,
      "stage3_param_persistence_threshold": 1e5,
      "reduce_scatter": True,
      "reduce_bucket_size": 2e8,
      "allgather_bucket_size": 2e8,
      "zero_allow_untested_optimizer": True,
      "zero_quantized_weights": True,
      "zero_quantized_gradients": True
    },
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": 3e-5,
        "betas": [0.8, 0.999],
        "eps": 1e-8,
        "weight_decay": 3e-7
      }
    },
    "scheduler": {
      "type": "WarmupLR",
      "params": {
        "warmup_min_lr": 0,
        "warmup_max_lr": 3e-5,
        "warmup_num_steps": 1000
      }
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 2000,
    "wall_clock_breakdown": False
  }


# DeepSpeed initialization
model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     config_params=deepspeed_config)

# Training loop
for epoch in range(args.num_epochs):
    for batch in dataloader:
        # Forward pass
        inputs = {k: v.to(model_engine.local_rank) for k, v in batch.items()}
        outputs = model_engine(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        # Backward pass
        model_engine.backward(loss)
        model_engine.step()

        if model_engine.local_rank == 0 and model_engine.global_steps % deepspeed_config["steps_per_print"] == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Save the model
if model_engine.local_rank == 0:
    model_engine.save_checkpoint("/ocean/projects/asc200010p/jjia1/scripts/gpt_result/codeparrot/debug/transformer-deepspeed")

print("Training completed.")
