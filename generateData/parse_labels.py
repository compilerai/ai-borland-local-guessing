import argparse
import logging
import os
import json
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
    """Calculates frame size, safely handling multi-chunk allocations and push-optimizations."""
    total_size = 0
    instructions = []

    prologue_lines = func_asm.split('\n')[:40]

    # 1. Standard add/sub esp parsing
    for line in prologue_lines:
        line = line.split(';')[0].strip()
        match = re.search(r'(add\s+esp\s*,\s*-([0-9a-f]+h?)|sub\s+esp\s*,\s*([0-9a-f]+h?))', line, re.IGNORECASE)
        if match:
            instructions.append(match.group(1).strip())
            val_str = match.group(2) if match.group(2) else match.group(3)
            val_str = val_str.lower()
            size = int(val_str[:-1], 16) if val_str.endswith('h') else int(val_str)
            total_size += size

    # 2. Push-Optimization Fallback (Crucial for small frames/pointers)
    if not instructions:
        push_count = 0
        for line in prologue_lines[:15]:
            # Stop looking once we hit real function body code (calls, jumps)
            if re.match(r'\s*(call|jmp)\b', line, re.IGNORECASE):
                break
            # Count register pushes used for stack alignment/allocation
            if re.match(r'\s*push\s+(eax|ebx|ecx|edx|esi|edi)\b', line, re.IGNORECASE):
                push_count += 1

        if push_count > 0:
            total_size = push_count * 4
            instructions = [f"push reg (x{push_count})"]

    inst_str = " | ".join(instructions) if instructions else None
    return inst_str, (total_size if total_size > 0 else None)

def _parse_ebp_offset(raw: str) -> int | None:
    """
    Parses a raw [ebp OP hex_or_dec] capture group into a signed integer offset.
    Handles formats like '-4', '- 4', '+8', '8', '0ffch', '-0ffch'.
    Returns None if parsing fails.
    """
    clean = raw.replace(' ', '').lower()
    sign = -1 if '-' in clean else 1
    clean = clean.replace('-', '').replace('+', '')
    try:
        value = int(clean[:-1], 16) if clean.endswith('h') else int(clean)
        return sign * value
    except ValueError:
        LOGGER.debug("Failed to parse ebp offset from raw string: %r", raw)
        return None

# Anchored safe register list to prevent false positive liveness traces
_REGS = {'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'esp', 'ebp'}

# PRIMARY write: mov/fstp/fst/fistp into [ebp±X]. The (size + 'ptr') prefix is
# OPTIONAL as a whole -- Borland often omits it for register-to-memory stores
# (e.g. "mov [ebp-48],eax"), and 'qword' is included for double-precision
# fstp stores (e.g. "fstp qword ptr [ebp-8]").
_WRITE_PRIMARY = re.compile(
    r'(?:mov|fstp|fst|fistp|fisttp)\s+'
    r'(?:(?:dword|word|byte|qword)\s+ptr\s*)?\[ebp\s*([+\-]\s*[0-9a-f]+h?)\]',
    re.IGNORECASE
)

# LEA target: used for the "= &var_name" address-of pattern, where the LEA's
# offset IS var_name's own slot (the subsequent mov stores the *pointer*,
# not var_name itself).
_LEA = re.compile(
    r'lea\s+[a-z]{2,3}\s*,\s*(?:(?:dword|word|byte|qword)\s+ptr\s*)?\[ebp\s*([+\-]\s*[0-9a-f]+h?)\]',
    re.IGNORECASE
)

# Last-resort fallback: any [ebp±X] reference (read, write, lea, push, etc.)
_ANY_EBP = re.compile(r'\[ebp\s*([+\-]\s*[0-9a-f]+h?)\]', re.IGNORECASE)

def map_variable_offsets(func_asm: str, var_names: list) -> dict:
    """Maps variables using the First-Seen Heuristic and Anchored Liveness."""
    
    # Anchored safe register list to prevent false positive liveness traces
    REGS = {'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'esp', 'ebp'}
    
    # Organize into Comment/Code Blocks
    blocks = []
    current_comments = []
    current_code = []

    for line in func_asm.split('\n'):
        stripped = line.strip()
        if not stripped: continue
        
        # Treat both ';' comments and Borland '?live' markers as comment lines
        is_comment = stripped.startswith(';') or re.match(r'^\?live\d+@\d+:', stripped)
        
        if is_comment:
            if current_code:
                blocks.append((current_comments, current_code))
                current_comments = []
                current_code = []
            current_comments.append(stripped)
        else:
            current_code.append(stripped)
            
    if current_code or current_comments:
        blocks.append((current_comments, current_code))

    var_offsets = {}
    
    # Regex to detect hoisted C-declarations so we can ignore them
    decl_pattern = re.compile(r';\s*(?:volatile\s+)?(?:int|char|short|double|float|long|struct)\b')

    for var_name in var_names:
        best_offset = None
        assigned_reg = None

        # Skip pointer variables — their offset is meaningless for the ML task
        if var_name.startswith("ptr_"):
            var_offsets[var_name] = {
                "assembly_reference": None,
                "allocation_space_offset": None
            }
            continue
        
        for comments, code in blocks:
            comment_text = "\n".join(comments)
            
            # Check for explicitly assigned Liveness registers
            pairs = re.findall(r'\b([a-z]{2,3})\s*=\s*&?(\w+)', comment_text, re.IGNORECASE)
            for reg, v in pairs:
                if v == var_name and reg.lower() in REGS:
                    assigned_reg = reg.lower()

            # Filter out hoisted declarations to prevent the "Fused Block" bug
            active_comments = [c for c in comments if not decl_pattern.search(c)]

            # First-Seen Heuristic
            if any(var_name in c for c in active_comments):
                
                # 1. Prefer write instructions using your robust _WRITE_PRIMARY regex
                for code_line in code:
                    write_match = _WRITE_PRIMARY.search(code_line)
                    if write_match:
                        # Wire up your clean offset parser!
                        best_offset = _parse_ebp_offset(write_match.group(1))
                        if best_offset is not None:
                            break # Found the write target!
                
                # 2. Fallback to standard first-seen using _ANY_EBP
                if best_offset is None:
                    for code_line in code:
                        matches = _ANY_EBP.findall(code_line)
                        if matches:
                            best_offset = _parse_ebp_offset(matches[0])
                            if best_offset is not None:
                                break
                
                if best_offset is not None:
                    break # Found it! Stop searching other blocks entirely.

        # Final Formatting & Fallbacks
        if best_offset is not None:
            offset_str = f"ebp{best_offset}" if best_offset < 0 else f"ebp+{best_offset}"
            var_offsets[var_name] = {
                "assembly_reference": offset_str,
                "allocation_space_offset": best_offset
            }
        elif assigned_reg is not None:
            # Register-only fallback
            var_offsets[var_name] = {
                "assembly_reference": f"register {assigned_reg}",
                "allocation_space_offset": None
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

    LOGGER.info("Parsing assembly and extracting pristine ground truth labels...")

    total_vars_found = 0
    total_vars_missing = 0

    for record in tqdm(dataset, desc="Extracting Labels"):
        full_asm = record.get("assembly_code", "")
        if not full_asm:
            continue

        for func in record.get("label", {}).get("functions", []):
            func_name = func.get("function_name")
            if not func_name:
                LOGGER.warning("Encountered a function entry with no function_name -- skipping.")
                continue

            func_asm = extract_function_asm(full_asm, func_name)

            if not func_asm:
                LOGGER.debug("Could not locate assembly block for function: %s", func_name)
                continue

            alloc_inst, alloc_size = get_stack_allocation(func_asm)
            func["stack_allocation_instruction"] = alloc_inst
            func["stack_allocation_size_bytes"] = alloc_size

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
    LOGGER.info(
        "Successfully labeled %d variables. Missing/Optimized out: %d",
        total_vars_found, total_vars_missing
    )

    LOGGER.info("Saving finalized JSON dataset...")
    save_compact_json(dataset, args.json_path, LOGGER)
    LOGGER.info("Dataset fully finalized and ready for ML Evaluation!")

if __name__ == "__main__":
    main()