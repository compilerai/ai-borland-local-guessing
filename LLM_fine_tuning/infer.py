import os
import json
import torch
import argparse
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .pre_process import load_and_format_dataset
from utils.loggerClass import setup_logging

LOGGER = logging.getLogger(__name__)

# FIX 4: Compute a safe max_new_tokens budget.
# Largest observed valid JSON is ~900 chars ≈ 360 tokens. 420 gives headroom
# without the long hallucination window that 512 allowed.
MAX_NEW_TOKENS = 420


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched Inference and Evaluation for LoRA Adapter")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-7B", help="Base model HuggingFace ID")
    parser.add_argument("--adapter_dir", default="./output_LLM_part_Qwen7B", help="Path where the final adapter is saved")
    parser.add_argument("--test_dataset_path", required=True, help="Path to your raw evaluation/test JSON file")
    parser.add_argument("--output_results_json", default="eval_results.json", help="Where to save prediction artifacts")
    parser.add_argument("--sample_limit", type=int, default=0, help="Limit evaluation to N samples for speed. 0 for all.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of sequences to generate concurrently.")
    return parser.parse_args()


def detect_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    cc_major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if cc_major >= 8 else torch.float16


def extract_json_block(text: str) -> str:
    """
    FIX 3: Upgraded extractor — validates that the extracted slice is actually
    parseable JSON before returning it. Previously the function returned the
    slice blindly, so trailing hallucinated text after a valid JSON object
    (system-prompt bleed, Q&A loops, ETwitter spam) would land inside the
    returned string and cause json.loads() to fail even though the JSON itself
    was correct.

    Strategy:
    1. Strip markdown fences.
    2. Find the first '{'.
    3. Walk forward from there trying progressively smaller rfind('}') windows
       until json.loads() succeeds. This handles the common case where the
       model generated valid JSON then appended garbage after the closing '}'.
    """
    cleaned = text.strip()

    # Strip markdown fences
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[-1].split("```")[0].strip()

    start = cleaned.find("{")
    if start == -1:
        return cleaned

    # Walk candidate end positions from right to left, return the first
    # substring that parses as valid JSON. This gracefully handles any
    # post-JSON hallucination regardless of its length or content.
    search_region = cleaned[start:]
    end = len(search_region)
    while end > 0:
        candidate_end = search_region.rfind("}", 0, end)
        if candidate_end == -1:
            break
        candidate = search_region[: candidate_end + 1]
        try:
            json.loads(candidate)
            return candidate          # First (longest) valid JSON wins
        except json.JSONDecodeError:
            end = candidate_end       # Shrink window and try again

    # Fallback: return everything from the first '{' to the last '}'
    end = cleaned.rfind("}")
    if end != -1:
        return cleaned[start : end + 1]
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
    setup_logging(tag="inference_eval_batched", console_level=logging.INFO)
    dtype = detect_dtype()

    # 1. Load Tokenizer & Force Left-Padding for Generation
    LOGGER.info(f"Loading tokenizer from adapter path: {args.adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # CRITICAL: Batched generation requires left padding!
    tokenizer.padding_side = "left"

    # 2. Load Model & Adapter
    LOGGER.info(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    LOGGER.info(f"Merging LoRA adapter from: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    # 3. Load & Format Data
    LOGGER.info("Formatting raw data using pre-processor...")
    hf_dataset = load_and_format_dataset(args.test_dataset_path)
    if args.sample_limit > 0:
        hf_dataset = hf_dataset.select(range(min(args.sample_limit, len(hf_dataset))))

    # 4. Prepare and Sort Dataset (The Padding Trap Killer)
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

    # Sort by prompt length to minimize padding waste inside batches
    eval_data.sort(key=lambda x: x["length"])
    LOGGER.info(f"Loaded and sorted {len(eval_data)} target evaluations.")

    correct_predictions = 0
    valid_json_parses = 0
    evaluation_records = []

    # 5. Batched Generation Loop
    LOGGER.info(f"Starting batched generation loop (Batch Size: {args.batch_size}, Max New Tokens: {MAX_NEW_TOKENS})...")
    with torch.no_grad():
        for i in tqdm(range(0, len(eval_data), args.batch_size), desc="Evaluating Batches"):
            batch = eval_data[i : i + args.batch_size]
            prompts = [x["prompt"] for x in batch]
            ground_truths = [x["ground_truth"] for x in batch]
            original_indices = [x["idx"] for x in batch]

            # Tokenize the entire batch
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[1]

            # stop_strings=["}"] was removed — it triggers on the first nested '}'
            # inside variable_mappings (each object closes with '}'), not the outer
            # closing brace of the top-level object. This would cut the output
            # mid-array after the first variable, producing invalid JSON every time.
            # extract_json_block() already handles all post-JSON hallucination by
            # walking backwards until it finds a slice that parses — no stop_strings needed.
            # max_new_tokens tightened from 512 → 420 to shrink the hallucination window.
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Slice out only the newly generated tokens and decode
            generated_tokens = outputs[:, input_length:]
            prediction_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # Evaluate the batch
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

    # 6. Compute Metrics
    total_samples = len(evaluation_records)
    json_parse_rate = (valid_json_parses / total_samples) * 100 if total_samples > 0 else 0
    exact_match_rate = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0

    LOGGER.info("========================================")
    LOGGER.info("          EVALUATION SUMMARY            ")
    LOGGER.info("========================================")
    LOGGER.info(f"Total Evaluated Samples: {total_samples}")
    LOGGER.info(f"Valid JSON Output Rate:  {json_parse_rate:.2f}% ({valid_json_parses}/{total_samples})")
    LOGGER.info(f"Exact Match Accuracy:    {exact_match_rate:.2f}% ({correct_predictions}/{total_samples})")
    LOGGER.info("========================================")

    # Save results (resort back to original index order)
    evaluation_records.sort(key=lambda x: x["sample_idx"])
    metrics_summary = {
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