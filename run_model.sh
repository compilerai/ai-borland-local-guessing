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
python -m generateData.train_val_split \
    --json_path ./data/dataset_labels_sanitized.json