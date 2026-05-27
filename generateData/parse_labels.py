import argparse
import logging
import os
import json
import uuid
import re
from collections import defaultdict
from tqdm import tqdm
from utils.json_utils import save_compact_json

LOGGER = logging.getLogger(__name__)

def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Stack Labels from Borland Assembly (Optimized)")
    parser.add_argument("--json_path", required=True, help="Path to the dataset_labels.json file")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()

def extract_function_asm(full_asm: str, func_name: str) -> str:
    """Extracts the specific assembly block for a given function."""
    pattern = re.compile(rf'_{func_name}\s+proc\s+near(.*?)_{func_name}\s+endp', re.IGNORECASE | re.DOTALL)
    match = pattern.search(full_asm)
    return match.group(1) if match else ""

def get_stack_allocation(func_asm: str) -> tuple:
    """Accumulates all stack allocations to handle Borland multi-page chunking."""
    total_size = 0
    instructions = []
    
    # Scan the prologue window (first 40 lines is plenty)
    for line in func_asm.split('\n')[:40]:
        line = line.split(';')[0].strip()
        match = re.search(r'(add\s+esp\s*,\s*-([0-9a-f]+h?)|sub\s+esp\s*,\s*([0-9a-f]+h?))', line, re.IGNORECASE)
        if match:
            instructions.append(match.group(1).strip())
            
            # Parse size properly accounting for hex 'h' suffix
            val_str = match.group(2) if match.group(2) else match.group(3)
            val_str = val_str.lower()
            size = int(val_str[:-1], 16) if val_str.endswith('h') else int(val_str)
            total_size += size
            
    inst_str = " | ".join(instructions) if instructions else None
    return inst_str, (total_size if total_size > 0 else None)

def map_variable_offsets(func_asm: str, var_names: list) -> dict:
    """Maps variables to their [ebp-X] offsets using Liveness Tracing + Frequency Analysis."""
    
    # --- STEP 1: Parse LEA instructions for Register Caching ---
    reg_to_offset = {}
    lea_pattern = re.compile(r'lea\s+([a-z]{3})\s*,\s*(?:(?:dword|word|byte)\s+ptr\s*)?\[ebp\s*([+-]\s*[0-9a-f]+h?)\]', re.IGNORECASE)
    for match in lea_pattern.finditer(func_asm):
        reg = match.group(1).lower()
        m_clean = match.group(2).replace(' ', '').lower()
        sign = -1 if '-' in m_clean else 1
        m_clean = m_clean.replace('-', '').replace('+', '')
        val = int(m_clean[:-1], 16) if m_clean.endswith('h') else int(m_clean)
        
        # Only keep the earliest LEA per register to capture prologue setup
        if reg not in reg_to_offset:
            reg_to_offset[reg] = sign * val

    # --- STEP 2: Parse Liveness Comments (e.g. ?live1@16: ; EBX = &var_yabfe) ---
    var_to_reg = {}
    for line in func_asm.split('\n'):
        if '?live' in line or ';' in line:
            # Matches 'EBX = &var_name' or 'EBX = var_name'
            pairs = re.findall(r'([a-z]{3})\s*=\s*&?([a-z0-9_]+)', line, re.IGNORECASE)
            for reg, var in pairs:
                if var not in var_to_reg:
                    var_to_reg[var] = reg.lower()

    # --- STEP 3: Organize blocks for standard Frequency Analysis ---
    blocks = []
    current_comments = []
    current_code = []

    for line in func_asm.split('\n'):
        stripped = line.strip()
        if not stripped: continue
        if stripped.startswith(';'):
            if current_code:
                blocks.append((current_comments, current_code))
                current_comments = []
                current_code = []
            current_comments.append(stripped)
        else:
            current_code.append(stripped)
            
    if current_code or current_comments:
        blocks.append((current_comments, current_code))

    # --- STEP 4: Resolve the Offsets ---
    var_offsets = {}
    
    for var_name in var_names:
        best_offset = None
        
        # Priority A: Check the Liveness + LEA Cache (Catches optimized arrays/structs)
        if var_name in var_to_reg:
            reg = var_to_reg[var_name]
            if reg in reg_to_offset:
                best_offset = reg_to_offset[reg]

        # Priority B: Fallback to Frequency Analysis (Catches localized scalars)
        if best_offset is None:
            offset_candidates = defaultdict(int)
            for comments, code in blocks:
                comment_text = "\n".join(comments)
                if var_name in comment_text:
                    for code_line in code:
                        matches = re.findall(r'\[ebp\s*([+-]\s*[0-9a-f]+h?)\]', code_line, re.IGNORECASE)
                        for m in matches:
                            m_clean = m.replace(' ', '').lower()
                            sign = -1 if '-' in m_clean else 1
                            m_clean = m_clean.replace('-', '').replace('+', '')
                            val = int(m_clean[:-1], 16) if m_clean.endswith('h') else int(m_clean)
                            offset_candidates[sign * val] += 1
                            
            if offset_candidates:
                # Find the most frequently used offset below the variable's C comment
                best_offset = max(offset_candidates.items(), key=lambda x: x[1])[0]

        # Final Formatting
        if best_offset is not None:
            offset_str = f"ebp{best_offset}" if best_offset < 0 else f"ebp+{best_offset}"
            var_offsets[var_name] = {
                "assembly_reference": offset_str,
                "allocation_space_offset": best_offset
            }
            
    return var_offsets

def main():
    args = parse_args()
    setup_logger(args.log_level)

    if not os.path.exists(args.json_path):
        LOGGER.error("JSON file not found: %s", args.json_path)
        return

    LOGGER.info("Loading dataset...")
    with open(args.json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    LOGGER.info("Parsing assembly and extracting ground truth labels...")
    
    total_vars_found = 0
    total_vars_missing = 0

    for record in tqdm(dataset, desc="Extracting Labels"):
        full_asm = record.get("assembly_code", "")
        if not full_asm:
            continue
            
        for func in record.get("label", {}).get("functions", []):
            func_name = func.get("function_name")
            func_asm = extract_function_asm(full_asm, func_name)
            
            if not func_asm:
                continue
                
            # 1. Map the Stack Allocation
            alloc_inst, alloc_size = get_stack_allocation(func_asm)
            func["stack_allocation_instruction"] = alloc_inst
            func["stack_allocation_size_bytes"] = alloc_size
            
            # 2. Map the Variables
            var_mappings = func.get("variable_mapping", [])
            var_names = [v["variable_name"] for v in var_mappings]
            
            extracted_offsets = map_variable_offsets(func_asm, var_names)
            
            for v_map in var_mappings:
                v_name = v_map["variable_name"]
                if v_name in extracted_offsets:
                    v_map["assembly_reference"] = extracted_offsets[v_name]["assembly_reference"]
                    v_map["allocation_space_offset"] = extracted_offsets[v_name]["allocation_space_offset"]
                    total_vars_found += 1
                else:
                    total_vars_missing += 1

    LOGGER.info("Extraction complete!")
    LOGGER.info("Successfully labeled %d variables. Missing/Optimized out: %d", total_vars_found, total_vars_missing)
    
    LOGGER.info("Saving finalized JSON dataset...")
    save_compact_json(dataset, args.json_path, LOGGER)
    LOGGER.info("Dataset fully finalized and ready for ML Training!")

if __name__ == "__main__":
    main()