import argparse
import json
import random
from sklearn.model_selection import train_test_split

TRAIN_SPLIT_RATIO = 0.80
VAL_SPLIT_RATIO = 0.20
SEED = 42

def main():
    parser = argparse.ArgumentParser(description="Script for train test split")
    parser.add_argument("--json_path", required=True, help="Path to the dataset_labels_sanitized.json")
    args = parser.parse_args()

    # 1. Load your sanitized JSON
    with open(args.json_path, "r") as f:
        data = json.load(f)

    # 2. Shuffle the files
    random.shuffle(data)

    # 3. Perform a 80% Train, 20% Validation
    train_data, val_data = train_test_split(data, test_size=VAL_SPLIT_RATIO, random_state=SEED)

    # 4. Save them as separate files
    for name, subset in [("train", train_data), ("val", val_data)]:

        output_path = args.json_path.replace(".json", f"_{name}.json")
        
        with open(output_path, "w") as f:
            json.dump(subset, f, indent=2)
        print(f"Saved {len(subset)} records to {name} split.")

if __name__ == "__main__":
    main()