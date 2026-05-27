import argparse
import logging
import os
import json
import uuid
import re
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map Borland Assembly to Ground Truth JSON")
    parser.add_argument("--json_path", required=True, help="Path to the dataset_labels.json file")
    parser.add_argument("--asm_dir", required=True, help="Directory containing the generated .asm files")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()

def save_compact_json(data: list, output_path: str):
    """Saves the JSON while keeping variable_mappings on a single line (High-Speed Version)."""
    placeholders = {}
    
    # 1. Convert variable_mapping to compact strings and swap with UUIDs
    for record in tqdm(data, desc="Formatting variable mapping strings"):
        for func in record.get("label", {}).get("functions", []):
            new_mappings = []
            for mapping in func.get("variable_mapping", []):
                compact_str = json.dumps(mapping, separators=(', ', ': '))
                ph = f"__VAR_MAPPING_{uuid.uuid4().hex}__"
                placeholders[ph] = compact_str
                new_mappings.append(ph)
            func["variable_mapping"] = new_mappings
    
    LOGGER.info("Serializing main JSON structure...")
    # 2. Dump main JSON
    raw_json = json.dumps(data, indent=2)
    
    # 3. Swap placeholders back using highly-optimized SINGLE-PASS Regex
    pattern = re.compile(r'"__VAR_MAPPING_[0-9a-f]{32}__"')
    
    pbar = tqdm(total=len(placeholders), desc="Swapping compact JSON blocks")
    
    def replacer(match):
        pbar.update(1)
        ph_key = match.group(0).strip('"')
        return placeholders[ph_key]
        
    raw_json = pattern.sub(replacer, raw_json)
    pbar.close()
        
    # 4. Write to disk
    LOGGER.info("Writing updated JSON to disk...")
    with open(output_path, "w", encoding="utf-8") as jf:
        jf.write(raw_json)

def main():
    args = parse_args()
    setup_logger(args.log_level)

    if not os.path.exists(args.json_path):
        LOGGER.error("JSON file not found: %s", args.json_path)
        return

    if not os.path.exists(args.asm_dir):
        LOGGER.error("Assembly directory not found: %s", args.asm_dir)
        return

    LOGGER.info("Loading JSON dataset...")
    with open(args.json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    LOGGER.info("Loaded %d records. Mapping assembly files...", len(dataset))
    
    success_count = 0
    missing_count = 0

    # Iterate through the JSON and pull in the corresponding assembly code
    for record in tqdm(dataset, desc="Stitching Assembly"):
        file_id = record.get("file_id")
        
        if file_id is None:
            LOGGER.warning("Found a record with no file_id. Skipping.")
            continue
            
        asm_filename = f"sample_{file_id:05d}.asm"
        asm_filepath = os.path.join(args.asm_dir, asm_filename)
        
        if os.path.exists(asm_filepath):
            with open(asm_filepath, "r", encoding="utf-8", errors="replace") as asm_file:
                asm_content = asm_file.read()
                
            record["assembly_code"] = asm_content
            success_count += 1
        else:
            missing_count += 1
            LOGGER.debug("Missing assembly file: %s", asm_filename)

    LOGGER.info("Mapping complete. Success: %d | Missing: %d", success_count, missing_count)
    
    save_compact_json(dataset, args.json_path)
    LOGGER.info("Done!")

if __name__ == "__main__":
    main()