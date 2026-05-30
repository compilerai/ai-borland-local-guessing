"""
benchmark_throughput.py

Measures real training throughput (tokens/sec) on your actual hardware.
Mirrors SFTTrainer exactly: gradient checkpointing enabled, variable-length
padded batches, same LoRA config.

Usage:
    python benchmark_throughput.py \
        --model_name Qwen/Qwen2.5-Coder-7B \
        --seq_len 1828 \
        --batch_size 4 \
        --grad_accum 4 \
        --train_examples 13211
"""

import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",     default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--seq_len",        type=int, default=1828)
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--grad_accum",     type=int, default=4)
    p.add_argument("--train_examples", type=int, default=13211)
    p.add_argument("--warmup_steps",   type=int, default=5)
    p.add_argument("--measure_steps",  type=int, default=20)
    return p.parse_args()


def detect_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def hr(label=""):
    print(f"\n{'='*60}  {label}")


def vram_used_gb():
    return torch.cuda.memory_allocated() / 1e9


def make_batch(batch_size, avg_seq_len, vocab_size, pad_id, device):
    """Variable-length padded batch — matches SFTTrainer collator behavior."""
    low    = max(32,   int(avg_seq_len * 0.70))
    high   = min(4096, int(avg_seq_len * 1.30))
    lengths = torch.randint(low, high + 1, (batch_size,)).tolist()
    max_len = max(lengths)

    ids  = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len,           dtype=torch.long)
    lbls = torch.full((batch_size, max_len), -100,    dtype=torch.long)

    for i, L in enumerate(lengths):
        row = torch.randint(0, vocab_size, (L,))
        ids[i, :L]  = row
        mask[i, :L] = 1
        lbls[i, :L] = row

    return ids.to(device), mask.to(device), lbls.to(device), sum(lengths)


def main():
    args   = parse_args()
    dtype  = detect_dtype()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr("hardware")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU        : {props.name}")
        print(f"  VRAM total : {props.total_memory / 1e9:.1f} GB")
        print(f"  dtype      : {dtype}")

    hr("loading model + LoRA")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id or 0

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    # --- gradient checkpointing: critical to match SFTTrainer ---
    # Without this the benchmark uses ~30% more VRAM than real training
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    print(f"  VRAM after model load : {vram_used_gb():.1f} GB")

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    print(f"  Trainable params      : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  VRAM after LoRA       : {vram_used_gb():.1f} GB")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )

    vocab_size  = tokenizer.vocab_size or model.config.vocab_size
    total_steps = args.warmup_steps + args.measure_steps

    hr("benchmark config")
    print(f"  batch_size            : {args.batch_size}")
    print(f"  avg seq_len           : {args.seq_len}  (batches vary ±30%)")
    print(f"  grad_accum            : {args.grad_accum}")
    print(f"  effective batch       : {args.batch_size * args.grad_accum}")
    print(f"  gradient checkpointing: enabled  (matches SFTTrainer)")
    print(f"  warmup / measure      : {args.warmup_steps} / {args.measure_steps} steps")

    hr("running benchmark")
    step_times   = []
    token_counts = []
    peak_vram    = 0.0

    for step in range(total_steps):
        phase = "warmup" if step < args.warmup_steps else "measuring"
        print(f"  step {step+1:3d}/{total_steps}  [{phase}]", end="\r", flush=True)

        input_ids, attention_mask, labels, real_tokens = make_batch(
            args.batch_size, args.seq_len, vocab_size, pad_id, device
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1e9)

        if step >= args.warmup_steps:
            step_times.append(elapsed)
            token_counts.append(real_tokens)

    print()

    hr("results")
    avg_step   = sum(step_times) / len(step_times)
    avg_tps    = sum(token_counts) / sum(step_times)
    max_tps    = max(tc / st for tc, st in zip(token_counts, step_times))
    min_tps    = min(tc / st for tc, st in zip(token_counts, step_times))

    print(f"  avg step time         : {avg_step:.2f}s")
    print(f"  min / max step        : {min(step_times):.2f}s / {max(step_times):.2f}s")
    print(f"  peak VRAM used        : {peak_vram:.1f} GB")
    print()
    print(f"  avg tokens/sec        : {avg_tps:,.0f}")
    print(f"  min / max tok/sec     : {min_tps:,.0f} / {max_tps:,.0f}")

    hr("epoch time estimate")
    eff_batch       = args.batch_size * args.grad_accum
    steps_per_epoch = -(-args.train_examples // eff_batch)
    epoch_tokens    = args.train_examples * args.seq_len
    epoch_sec       = epoch_tokens / avg_tps
    epoch_min       = epoch_sec / 60

    print(f"  train examples        : {args.train_examples:,}")
    print(f"  effective batch       : {eff_batch}")
    print(f"  steps / epoch         : {steps_per_epoch:,}")
    print(f"  tokens / epoch        : {epoch_tokens:,}")
    print()
    print(f"  1 epoch               : {epoch_min:.0f} min  ({epoch_sec:.0f}s)")
    print(f"  2 epochs (config)     : {epoch_min*2:.0f} min  (~{epoch_min*2/60:.1f}h)")
    print(f"  + eval overhead       : ~10-15 min per epoch")
    print("=" * 60)


if __name__ == "__main__":
    main()