# Stage 1 : Dataset Generation

export TOKENIZERS_PARALLELISM=false

# python -m generateData.generate_c_programs --num_samples 10000 \
#     --max_random_func 10 \
#     --output_dir ./data

# python -m generateData.map_assembly \
#     --json_path ./data/dataset_labels.json \
#     --asm_dir ./data/assembly_codes

# python -m generateData.parse_labels \
#     --json_path ./data/dataset_labels.json

# this saves the data to ./data/dataset_labels_sanitized.json
# python -m generateData.stripping_debug_asm_lines \
#     --json_path ./data/dataset_labels.json

# this splits the data to ./data/dataset_train.json and ./data/dataset_val.json 
# python -m generateData.train_val_split \
#     --json_path ./data/dataset_labels_sanitized.json

# python -m generateData.slice_data

# Stage 2 : LLM Finetuning

# this will store model and LORA adapters inside output_dir/models and output_dir/LORA_adapter 
python -m LLM_fine_tuning.train \
    --model_name Qwen/Qwen2.5-Coder-7B \
    --dataset_path ./data/dataset_train_small.json \
    --val_dataset_path ./data/dataset_val_small.json \
    --output_dir ./output_LLM_part_Qwen7B

# this will pull model and LORA adapters inside output_dir/models and output_dir/LORA_adapter -> store the predictions inside output_dir/preds
# python -m LLM_fine_tuning.infer \
#     --model_dir ./output_LLM_part_Qwen7B \
#     --dataset_path ./data/dataset_val.json \

# # this will for eval
# python -m LLM_fine_tuning.eval \
#     --ground_truth ./data/dataset_val.json \
#     --model_pred ./output_LLM_part_Qwen7B/preds

# python dummy_checks/benchmark_throughput.py \
#     --model_name Qwen/Qwen2.5-Coder-7B \
#     --seq_len 1828 \
#     --batch_size 8 \
#     --grad_accum 2 \
#     --train_examples 13211 \
#     --warmup_steps 5 \
#     --measure_steps 20

# python dummy_checks/plot_metric.py 