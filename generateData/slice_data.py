import json
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description="Script for slicing data")
    parser.add_argument("--input_path", required=True, help="Path to dataset json")
    parser.add_argument("--output_path", required=True, help="Path to output dataset json")
    parser.add_argument("--keep_percentage", type=float, required=True, help="Percentage to keep (e.g., 0.5)")

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    keep_percentage = args.keep_percentage

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