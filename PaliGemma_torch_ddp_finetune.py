# To run this script
# torchrun --standalone --nproc_per_node=4 PaliGemma_torch_ddp_finetune.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,8,9'

import random
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from datasets import load_dataset
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from peft import get_peft_model, LoraConfig

# ------------------------------------------------------------------------------
# 1. Training Configuration
# ------------------------------------------------------------------------------

@dataclass
class TrainingConfig:

    # Training settings
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    # total effective batch size = num_process (num_gpus) * per_device_batch_size * gradient_accumulation_steps = 4 * 8 * 1 = 32
    warmup_steps: int = 100 # ~5-10% of max_iters
    max_iters: int = 2010 # (21,435 samples / total effective batch size) * num_epochs=3
    log_interval: int = 100 # print loss and current learning rate every 100 steps
    # To Do: update logs to wandb
    save_interval: int = 1000 # save adapter weights every 1000 steps
    num_dataloader_workers = 16 # num_gpus x 4

    out_dir: str = "paligemma_vqav2_custom/"
    learning_rate: float = 5e-5
    weight_decay: float = 1e-6
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0

    # Model settings
    model_id: str = "google/paligemma2-3b-pt-224"
    lora_rank: int = 16
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"
    ])
    task_type: str = "CAUSAL_LM"
    dtype: torch.dtype = torch.bfloat16

    # Data settings
    dataset_name: str = "merve/vqav2-small"
    split: str = "validation"
    cache_dir: str = "./hf_datasets/"

    # DDP settings
    ddp: bool = False # reverts to True during DDP initialization
    backend: str = "nccl"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337

config = TrainingConfig()

# ------------------------------------------------------------------------------
# 2. DDP Initialization
# ------------------------------------------------------------------------------

if int(os.environ.get("LOCAL_RANK", -1)) != -1:
    config.ddp = True
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    config.device = f"cuda:{rank}"
    dist.init_process_group(backend=config.backend, init_method="env://")
    world_size = dist.get_world_size()
    print(f"Initialized DDP on rank {rank} with world size {world_size}")
else:
    config.ddp = False
    rank = 0
    world_size = 1

# Set seed for reproducibility
torch.manual_seed(config.seed + rank)
random.seed(config.seed + rank)

# ------------------------------------------------------------------------------
# 3. Data Loader
# ------------------------------------------------------------------------------

# PyTorch Dataset class
class VQADataset(Dataset):
    def __init__(self, dataset, processor, dtype):
        self.dataset = dataset
        self.processor = processor
        self.image_token = "<image>"
        self.dtype = dtype

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        text = f"{self.image_token} answer en {example['question']}"
        label = example['multiple_choice_answer']
        image = example["image"].convert("RGB")
        return {"text": text, "image": image, "label": label}

# Load huggingface dataset and processor
raw_dataset = load_dataset(config.dataset_name, split=config.split, cache_dir=config.cache_dir)
processor = PaliGemmaProcessor.from_pretrained(config.model_id, cache_dir=config.cache_dir)

# Create PyTorch dataset
train_dataset = VQADataset(raw_dataset, processor, config.dtype)

# Updated collate function that avoids moving data to GPU
def my_collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]
    images = [item["image"] for item in batch]
    tokens = processor(text=texts, images=images, suffix=labels,
                       return_tensors="pt", padding="longest")
    
    # Only convert floating-point tensors to the desired dtype (bfloat16), donot disturb LongInt tensors
    new_tokens = {}
    for k, v in tokens.items():
        if v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            new_tokens[k] = v.to(config.dtype)
        else:
            new_tokens[k] = v
    return new_tokens

# DataLoader with distributed sampler
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if config.ddp else None
train_dataloader = DataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=config.per_device_batch_size,
    shuffle=(train_sampler is None),
    num_workers=config.num_dataloader_workers,
    pin_memory=False,
    collate_fn=my_collate_fn,
)

# ------------------------------------------------------------------------------
# 4. Model Initialization and LoRA Setup
# ------------------------------------------------------------------------------

model = PaliGemmaForConditionalGeneration.from_pretrained(
    config.model_id,
    device_map="auto" if not config.ddp else {"": config.device},
    torch_dtype=config.dtype,
    attn_implementation="eager", # strongly recommended by hf when training gemma2 models
    cache_dir="./hf_models/",
)

lora_config = LoraConfig(
    r=config.lora_rank,
    target_modules=config.lora_target_modules,
    task_type=config.task_type,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.train()


# ------------------------------------------------------------------------------
# 5. Optimizer and Learning Rate Scheduler
# ------------------------------------------------------------------------------

# configure weight decay for all params except for bias and norm parameters
decay_params = [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in ["bias", "norm"])]
no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in ["bias", "norm"])]
optim_groups = [
    {'params': decay_params, 'weight_decay': config.weight_decay},
    {'params': no_decay_params, 'weight_decay': 0.0}
]

# Adam optimizer with weight decay
optimizer = AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)

# increase lr linearly from 0 to 5e-5 until warmup steps and then remains constant
def get_lr(it):
    if it < config.warmup_steps:
        return config.learning_rate * it / config.warmup_steps
    return config.learning_rate

# ------------------------------------------------------------------------------
# 6. Training Loop
# ------------------------------------------------------------------------------

# save adapter weights only
def save_checkpoint(iteration):
    if rank == 0:
        ckpt_path = os.path.join(config.out_dir, f"iter_{iteration}")
        os.makedirs(ckpt_path, exist_ok=True)
        model.save_pretrained(ckpt_path, save_adapter=True)
        print(f"Saved checkpoint to {ckpt_path}")

data_iter = iter(train_dataloader)
iter_num = 0

while iter_num < config.max_iters:

    # iterate over dataloader
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_dataloader)
        batch = next(data_iter)

    # Move batch to GPU device in the main process
    batch = {k: v.to(config.device, non_blocking=True) for k, v in batch.items()}

    # get current learning rate
    current_lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    # Forward Pass and BackPropogation
    for _ in range(config.gradient_accumulation_steps):
        outputs = model(**batch)
        loss = outputs.loss / config.gradient_accumulation_steps
        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # display key metrics
    iter_num += 1
    if iter_num % config.log_interval == 0 and rank == 0:
        current_loss = outputs.loss.item() * config.gradient_accumulation_steps
        print(f"\n - Iter {iter_num}: loss {current_loss:.4f}, lr {current_lr:.8f}")

    # save adapter weights at save intervals
    if iter_num % config.save_interval == 0 and rank == 0:
        save_checkpoint(iter_num)

# save adapter weights at final step
save_checkpoint(config.max_iters)

if config.ddp:
    dist.destroy_process_group()