import argparse
import os
import json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="buidling real dataset")
    parser.add_argument("--input_path", required=True)
    return parser.parse_args()

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def main():
    args = parse_args()
    output_file = os.path.join(args.input_path, "dataset_labels.json")
    # stem -> {"c": ..., "asm": ...}
    file_pairs = {}

    for root, dirs, files in os.walk(args.input_path):
        for file in files:
            if not (file.endswith(".c") or file.endswith(".asm")):
                continue

            stem, ext = os.path.splitext(file)
            if stem not in file_pairs:
                file_pairs[stem] = {}

            file_pairs[stem][ext[1:]] = os.path.join(root, file)

    dataset = []

    for stem, paths in file_pairs.items():
        # skip incomplete pairs
        if "c" not in paths or "asm" not in paths:
            continue

        c_content = read_file(paths["c"])
        asm_content = read_file(paths["asm"])


        dataset.append({
            "file_id": stem,
            "source_code_c": c_content,
            "assembly_code": asm_content
        })      

    output_file = os.path.join(args.input_path, "dataset_labels.json")  

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} pairs to {output_file}")


if __name__ == "__main__":

    main()