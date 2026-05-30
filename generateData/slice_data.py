import json
import random

def main():
    input_path = "./data/dataset_val.json"
    output_path = "./data/dataset_val_small.json"
    keep_percentage = 0.30

    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Shuffle to ensure a diverse mix of compiler behaviors
    random.seed(42)
    random.shuffle(data)

    # Calculate the exact cut index
    cut_index = int(len(data) * keep_percentage)
    sliced_data = data[:cut_index]

    print(f"Original dataset size: {len(data)} files")
    print(f"Sliced dataset size:   {len(sliced_data)} files")

    print(f"Saving smaller dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sliced_data, f, indent=2)
        
    print("Done! You are ready to train.")

if __name__ == "__main__":
    main()