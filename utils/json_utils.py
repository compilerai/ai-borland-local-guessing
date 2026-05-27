import re
import json
from tqdm import tqdm
import uuid

def save_compact_json(data: list, output_path: str, LOGGER):
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
