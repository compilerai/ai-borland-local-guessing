import logging
import argparse
import random
import os

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _safe_load

from hyperparameters.config import LLMConfig
from .pre_process import load_and_format_dataset
from utils.loggerClass import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for stack-offset extraction")

    # Data
    parser.add_argument("--dataset_path",     required=True,  help="Path to train JSON")
    parser.add_argument("--val_dataset_path", required=True,  help="Path to val JSON")
    parser.add_argument("--output_dir",       required=True,  help="Where to save checkpoints")

    # Model
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-1.5B",
                        help="HuggingFace model id or local path")

    # LoRA
    parser.add_argument("--lora_r",     type=int,   default=16)
    parser.add_argument("--lora_alpha", type=int,   default=32)

    # Training
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--early_stopping",  type=int,   default=3,
                        help="Early stopping patience (epochs). Set 0 to disable.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Set to 'true' to auto-resume from the latest checkpoint in output_dir, "
                             "or provide a specific path string to a checkpoint folder.")

    # Logging
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log_tag", default=None,
                        help="Tag for the log filename: logs/output_<tag>.txt. "
                             "Defaults to the output_dir basename.")
    parser.add_argument("--report_to", default="none",
                        choices=["none", "wandb", "tensorboard"],
                        help="Experiment tracker")

    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def vram_used_gb():
    return torch.cuda.memory_allocated() / 1e9

def detect_dtype() -> torch.dtype:
    """
    Use bfloat16 only on Ampere+ GPUs (A100, RTX 30xx …).
    Fall back to float16 for V100 / T4 / older hardware.
    """
    if not torch.cuda.is_available():
        return torch.float32
    cc_major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if cc_major >= 8 else torch.float16

# Main
def main() -> None:
    args = parse_args()

    # Derive log tag from output_dir name if not explicitly provided
    log_tag = args.log_tag or os.path.basename(args.output_dir.rstrip("/\\")) or "run"
    console_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(tag=log_tag, console_level=console_level)
    set_seed(args.seed)

    config = LLMConfig()

    dtype = detect_dtype()
    LOGGER.info(f"Using dtype: {dtype}")

    # 1. Datasets
    LOGGER.info("Loading datasets ...")
    train_dataset = load_and_format_dataset(args.dataset_path)
    val_dataset   = load_and_format_dataset(args.val_dataset_path)
    LOGGER.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # 2. Tokenizer
    LOGGER.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    lengths = [len(tokenizer.apply_chat_template(r["messages"], tokenize=True)) 
            for r in train_dataset.select(range(500))]
    LOGGER.info(f"Avg tokens/sample: {sum(lengths)/len(lengths):.0f}, max: {max(lengths)}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        LOGGER.info("Set pad_token = eos_token")

    # Qwen models use right-padding by default; keep it consistent
    tokenizer.padding_side = "right"

    # 3. Model
    LOGGER.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    # Disable caching during training (incompatible with gradient checkpointing)
    model.config.use_cache = False
    LOGGER.info(f"  VRAM after model load : {vram_used_gb():.1f} GB")

    # 4. LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # 5. Training arguments
    use_bf16 = (dtype == torch.bfloat16)

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # Batching
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        # Optimisation
        learning_rate=config.lr,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=0.05,
        num_train_epochs=config.num_train_epochs,
        weight_decay=0.01,

        # Precision
        bf16=use_bf16,
        fp16=not use_bf16,

        # Evaluation & checkpointing
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,          # keep only the 2 best checkpoints

        # Gradient checkpointing saves VRAM at a small throughput cost
        gradient_checkpointing=True,
        # Fixes the use_reentrant warning
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging
        logging_steps=config.logging_steps,
        report_to=args.report_to,

        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,

        # The Padding Trap Killers
        group_by_length=True,       # Groups similar-length sequences together
        dataloader_num_workers=4,   # Spins up CPU workers to handle sorting in parallel
    )

    # 6. Callbacks
    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping)
        )
        LOGGER.info(f"Early stopping enabled (patience={args.early_stopping})")

    # 7. Trainer
    LOGGER.info("Initialising SFTTrainer ...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        max_seq_length=config.max_seq_length,
        packing=False,
        callbacks=callbacks,
    )

    # 8. Train
    LOGGER.info("Starting fine-tuning ...")
    
    # Process the resume parameter
    resume_flag = args.resume_from_checkpoint
    
    if resume_flag:
        if resume_flag.lower() == "true":
            resume_flag = True
            LOGGER.info(f"Auto-resuming training from the latest checkpoint found in {args.output_dir}")
        elif resume_flag.lower() == "false" or resume_flag.lower() == "none":
            resume_flag = None
            LOGGER.info("Resuming training from explicit checkpoint path: false")
        else:
            LOGGER.info(f"Resuming training from explicit checkpoint path: {resume_flag}")

    trainer.train(resume_from_checkpoint=resume_flag)

    # 9. Save
    LOGGER.info(f"Saving best model to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()