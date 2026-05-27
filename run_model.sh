# python -m generateData.main --num_samples 10000 \
#     --max_random_func 10 \
#     --output_dir ./data

python -m generateData.map_assembly \
    --json_path ./data/dataset_labels.json \
    --asm_dir ./data/assembly_codes