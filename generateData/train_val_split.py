import argparse
import json
import logging
from sklearn.model_selection import train_test_split
from pathlib import Path
from utils.json_utils import save_compact_json

TRAIN_SPLIT_RATIO = 0.80
VAL_SPLIT_RATIO = 0.20
SEED = 42

LOGGER = logging.getLogger(__name__)

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

def main():
    parser = argparse.ArgumentParser(description="Script for train validation split")
    parser.add_argument("--json_path", required=True, help="Path to dataset json")
    args = parser.parse_args()
    setup_logger()

    # Load JSON
    with open(args.json_path, "r") as f:
        data = json.load(f)

    # Train / validation split
    train_data, val_data = train_test_split( 
        data,
        test_size=VAL_SPLIT_RATIO,
        random_state=SEED,
        shuffle=True
    )

    # Save splits
    for name, subset in [
        ("train", train_data),
        ("val", val_data)
    ]:
        output_path = Path(args.json_path).parent / f"dataset_{name}.json"

        with open(output_path, "w") as f:
            json.dump(subset, f, indent=2)

        save_compact_json(subset, output_path, LOGGER)
        LOGGER.info(
            f"Saved {len(subset)} records to {output_path}"
        )

if __name__ == "__main__":
    main()