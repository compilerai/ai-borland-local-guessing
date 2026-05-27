import argparse
import json
import logging
import os
from tqdm import tqdm

from utils.json_utils import save_compact_json

LOGGER = logging.getLogger(__name__)

OUTPUT_PATH_EXTENTION = "_sanitized.json"

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def strip_debug_artifacts(raw_asm: str) -> str:
    """Removes C comments, liveness markers, and debug symbols to mimic a stripped binary."""
    cleaned_lines = []
    for line in raw_asm.split('\n'):
        stripped_line = line.strip()
        
        # 1. Skip pure comment lines (the embedded C code)
        if stripped_line.startswith(';'):
            continue
            
        # 2. Skip Borland proprietary debug markers
        if stripped_line.startswith('?live') or stripped_line.startswith('?debug'):
            continue
            
        # 3. Strip inline comments from actual assembly lines (e.g., "mov eax, 1 ; set counter")
        if ';' in line:
            line = line.split(';')[0]
            
        cleaned_lines.append(line.rstrip())
        
    # Remove excessive blank lines left behind by the stripping process
    return '\n'.join([line for line in cleaned_lines if line.strip() != ''])

def main():
    parser = argparse.ArgumentParser(description="Sanitize Assembly for ML Training")
    parser.add_argument("--json_path", required=True, help="Path to the dataset_labels.json")
    args = parser.parse_args()
    setup_logger()

    if not os.path.exists(args.json_path):
        LOGGER.error("File not found: %s", args.json_path)
        return

    LOGGER.info("Loading dataset to strip cheat codes...")
    with open(args.json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    for record in tqdm(dataset, desc="Stripping Assembly"):
        raw_asm = record.get("assembly_code", "")
        if raw_asm:
            record["assembly_code"] = strip_debug_artifacts(raw_asm)

    output_path = args.json_path.replace(".json", OUTPUT_PATH_EXTENTION)
    LOGGER.info("Saving sanitized dataset to %s...", output_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    
    save_compact_json(dataset, output_path, LOGGER)
    LOGGER.info("Dataset sanitized and ready for the neural network!")

if __name__ == "__main__":
    main()