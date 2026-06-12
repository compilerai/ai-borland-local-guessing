import os
import json
import torch
import argparse
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from .pre_process import load_and_format_dataset
from utils.loggerClass import setup_logging

LOGGER = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline Inference (No LoRA) — Base Model Only")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-7B", help="Base model HuggingFace ID")
    # --adapter_dir is intentionally removed. The base model is loaded directly.
    parser.add_argument("--test_dataset_path", required=True, help="Path to your raw evaluation/test JSON file")
    parser.add_argument("--output_results_json", default="baseline_eval_results.json", help="Where to save prediction artifacts")
    parser.add_argument("--sample_limit", type=int, default=0, help="Limit evaluation to N samples for speed. 0 for all.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of sequences to generate concurrently.")
    return parser.parse_args()

def detect_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    cc_major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if cc_major >= 8 else torch.float16

def extract_json_block(text: str) -> str:
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[-1].split("```")[0].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        return cleaned[start:end+1]
    return cleaned

def compare_json_objects(pred_str: str, target_str: str) -> bool:
    try:
        pred_obj = json.loads(extract_json_block(pred_str))
        target_obj = json.loads(extract_json_block(target_str))
        return pred_obj == target_obj
    except Exception:
        return False

def main() -> None:
    args = parse_args()
    setup_logging(tag="inference_baseline_no_lora", console_level=logging.INFO)
    dtype = detect_dtype()

    # 1. Load Tokenizer directly from the base model (no adapter dir)
    # CHANGE 1: tokenizer source is args.base_model, not args.adapter_dir.
    LOGGER.info(f"Loading tokenizer from base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 2. Load base model ONLY — no PeftModel wrapping
    # CHANGE 2: we load the model and use it directly. No adapter is merged.
    LOGGER.info(f"Loading base model (no LoRA): {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()

    # 3. Load & Format Data (identical to fine-tuned version)
    LOGGER.info("Formatting raw data using pre-processor...")
    hf_dataset = load_and_format_dataset(args.test_dataset_path)
    if args.sample_limit > 0:
        hf_dataset = hf_dataset.select(range(min(args.sample_limit, len(hf_dataset))))

    # 4. Prepare and Sort Dataset
    LOGGER.info("Preparing prompts and sorting by length for optimal batching...")
    eval_data = []

    for idx, sample in enumerate(hf_dataset):
        messages = sample["messages"]
        user_messages = [m for m in messages if m["role"] != "assistant"]
        target_messages = [m for m in messages if m["role"] == "assistant"]

        if not user_messages or not target_messages:
            continue

        ground_truth = target_messages[0]["content"]
        prompt_input = tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        eval_data.append({
            "idx": idx,
            "prompt": prompt_input,
            "ground_truth": ground_truth,
            "length": len(prompt_input)
        })

    eval_data.sort(key=lambda x: x["length"])
    LOGGER.info(f"Loaded and sorted {len(eval_data)} target evaluations.")

    correct_predictions = 0
    valid_json_parses = 0
    evaluation_records = []

    # 5. Batched Generation Loop (identical to fine-tuned version)
    LOGGER.info(f"Starting batched generation loop (Batch Size: {args.batch_size})...")
    with torch.no_grad():
        for i in tqdm(range(0, len(eval_data), args.batch_size), desc="Evaluating Batches"):
            batch = eval_data[i : i + args.batch_size]
            prompts = [x["prompt"] for x in batch]
            ground_truths = [x["ground_truth"] for x in batch]
            original_indices = [x["idx"] for x in batch]

            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[1]

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            generated_tokens = outputs[:, input_length:]
            prediction_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for pred_text, truth, o_idx in zip(prediction_texts, ground_truths, original_indices):
                pred_text = pred_text.strip()
                is_valid_json = False

                try:
                    json.loads(extract_json_block(pred_text))
                    is_valid_json = True
                    valid_json_parses += 1
                except Exception:
                    pass

                is_exact_match = compare_json_objects(pred_text, truth)
                if is_exact_match:
                    correct_predictions += 1

                evaluation_records.append({
                    "sample_idx": o_idx,
                    "prediction": pred_text,
                    "ground_truth": truth,
                    "json_parse_success": is_valid_json,
                    "exact_match_success": is_exact_match
                })

    # 6. Compute & Log Metrics
    total_samples = len(evaluation_records)
    json_parse_rate = (valid_json_parses / total_samples) * 100 if total_samples > 0 else 0
    exact_match_rate = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0

    LOGGER.info("========================================")
    LOGGER.info("     BASELINE EVALUATION SUMMARY        ")
    LOGGER.info("   (Base model — no LoRA adapter)       ")
    LOGGER.info("========================================")
    LOGGER.info(f"Total Evaluated Samples: {total_samples}")
    LOGGER.info(f"Valid JSON Output Rate:  {json_parse_rate:.2f}% ({valid_json_parses}/{total_samples})")
    LOGGER.info(f"Exact Match Accuracy:    {exact_match_rate:.2f}% ({correct_predictions}/{total_samples})")
    LOGGER.info("========================================")

    evaluation_records.sort(key=lambda x: x["sample_idx"])
    metrics_summary = {
        # CHANGE 3: label clearly distinguishes this from the fine-tuned run
        "model": args.base_model,
        "adapter": None,
        "metrics": {
            "total_samples": total_samples,
            "json_parse_rate_pct": json_parse_rate,
            "exact_match_accuracy_pct": exact_match_rate
        },
        "results": evaluation_records
    }

    with open(args.output_results_json, "w", encoding="utf-8") as out_f:
        json.dump(metrics_summary, out_f, indent=4)

    LOGGER.info(f"Results saved to: {args.output_results_json}")

if __name__ == "__main__":
    main()