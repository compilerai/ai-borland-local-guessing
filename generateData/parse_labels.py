import argparse
import logging
import os
import json
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
    parser = argparse.ArgumentParser(description="Extract Stack Labels from Borland Assembly")
    parser.add_argument("--json_path", required=True, help="Path to the dataset_labels.json file")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def extract_function_asm(full_asm: str, func_name: str) -> str:
    """Extracts the assembly block for a given function name."""
    pattern = re.compile(
        rf'_{func_name}\s+proc\s+near(.*?)_{func_name}\s+endp',
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(full_asm)
    return match.group(1) if match else ""


def get_stack_allocation(func_asm: str) -> tuple:
    """
    Calculates frame size, handling:
      - Single add/sub esp,-N
      - Multi-chunk allocations (two add esp,-N lines for large frames)
      - Push-register optimisation fallback for small frames (push ecx / push ebx)
    """
    total_size = 0
    instructions = []
    prologue_lines = func_asm.split('\n')[:40]

    # Pass 1: standard add esp,-N / sub esp,N patterns
    for line in prologue_lines:
        line = line.split(';')[0].strip()
        match = re.search(
            r'(add\s+esp\s*,\s*-([0-9a-f]+h?)|sub\s+esp\s*,\s*([0-9a-f]+h?))',
            line, re.IGNORECASE
        )
        if match:
            instructions.append(match.group(1).strip())
            val_str = match.group(2) if match.group(2) else match.group(3)
            val_str = val_str.lower()
            size = int(val_str[:-1], 16) if val_str.endswith('h') else int(val_str)
            total_size += size

    # Pass 2: push-register fallback for micro / small-frame functions.
    # Real-world functions with 1-2 locals often use "push ecx" instead of
    # "add esp,-4". We count these pushes and convert to bytes.
    if not instructions:
        push_count = 0
        for line in prologue_lines[:15]:
            # Stop at the first real function-body instruction
            if re.match(r'\s*(call|jmp)\b', line, re.IGNORECASE):
                break
            if re.match(r'\s*push\s+(eax|ebx|ecx|edx|esi|edi)\b', line, re.IGNORECASE):
                push_count += 1
        if push_count > 0:
            total_size = push_count * 4
            instructions = [f"push reg (x{push_count})"]

    inst_str = " | ".join(instructions) if instructions else None
    return inst_str, (total_size if total_size > 0 else None)


def _parse_ebp_offset(raw: str) -> int | None:
    """
    Parses a raw [ebp OP value] capture group into a signed integer offset.
    Handles: '-4', '- 4', '+8', '8', '0ffch', '-0ffch'.
    Returns None on parse failure rather than silently returning 0.
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


# Pre-compiled regexes for performance across large datasets
_WRITE_PRIMARY = re.compile(
    r'(?:mov|fstp|fst|fistp|fisttp)\s+'
    r'(?:(?:dword|word|byte|qword)\s+ptr\s*)?\[ebp\s*([+\-]\s*[0-9a-f]+h?)\]',
    re.IGNORECASE
)
_ANY_EBP = re.compile(r'\[ebp\s*([+\-]\s*[0-9a-f]+h?)\]', re.IGNORECASE)

# Pattern to detect declaration-only comment lines (e.g. "; int j, parity, sign")
# These should NOT trigger the first-seen offset heuristic — they describe type info,
# not liveness. Without this filter, the parser grabs the wrong offset for the
# first variable mentioned after a declaration block.
_DECL_COMMENT = re.compile(
    r';\s*(?:volatile\s+)?(?:int|char|short|double|float|long|struct|unsigned)\b'
)


def map_variable_offsets(func_asm: str, var_names: list) -> dict:
    """
    Maps each variable name to its stack offset using a 3-tier strategy:

    Tier 1 — Anchored liveness tracing:
        Borland emits ?live comments like "; EBX = var_name" or "; ESI = &n".
        We capture the register and use it as a register-allocation marker.

    Tier 2 — First-Seen write heuristic (priority):
        When a comment block mentions the variable, scan the associated code
        for a write instruction (mov/fstp/fst) to [ebp±N]. This is the most
        reliable offset source.

    Tier 3 — First-Seen read fallback:
        If no write is found, fall back to any [ebp±N] reference (lea, cmp,
        push) in the same code block.

    Tier 4 — Register-only fallback:
        If no stack slot is found but a liveness register was noted, record
        "register <reg>" so the model learns register allocation for any
        variable type — not just pointers.

    OOD note: Tier 4 is critical for real-world OOD performance. Training
    data previously only had null assembly_reference for ptr_ variables.
    Real code register-allocates any variable (parity, sign, t, ii, etc.).
    """
    REGS = {'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'esp', 'ebp'}

    # ------------------------------------------------------------------ #
    # Organise assembly into (comment_lines, code_lines) blocks.
    # Block boundaries are Borland ';' comments and '?live' markers.
    # ------------------------------------------------------------------ #
    blocks = []
    current_comments: list[str] = []
    current_code: list[str] = []

    for line in func_asm.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
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

    # ------------------------------------------------------------------ #
    # Per-variable offset search
    # ------------------------------------------------------------------ #
    var_offsets: dict = {}

    for var_name in var_names:
        best_offset = None
        assigned_reg = None

        # OOD FIX: Pointer variables never have a meaningful stack slot of their
        # own — they hold another variable's address. Skip offset search entirely.
        if var_name.startswith("ptr_"):
            var_offsets[var_name] = {"assembly_reference": None, "allocation_space_offset": None}
            continue

        for comments, code in blocks:
            comment_text = "\n".join(comments)

            # Tier 1: Anchored liveness register — "; EBX = var_name" or "; ESI = &n"
            pairs = re.findall(r'\b([a-z]{2,3})\s*=\s*&?(\w+)', comment_text, re.IGNORECASE)
            for reg, v in pairs:
                if v == var_name and reg.lower() in REGS:
                    assigned_reg = reg.lower()

            # Filter out pure declaration comments — they mention variable names
            # but the associated code describes a different variable's initialisation.
            active_comments = [c for c in comments if not _DECL_COMMENT.search(c)]

            if not any(var_name in c for c in active_comments):
                continue

            # Tier 2: Explicit write instruction → most reliable offset source
            for code_line in code:
                write_match = _WRITE_PRIMARY.search(code_line)
                if write_match:
                    offset = _parse_ebp_offset(write_match.group(1))
                    if offset is not None:
                        best_offset = offset
                        break

            # Tier 3: Any [ebp±N] reference fallback (lea, cmp, push, read)
            if best_offset is None:
                for code_line in code:
                    matches = _ANY_EBP.findall(code_line)
                    if matches:
                        offset = _parse_ebp_offset(matches[0])
                        if offset is not None:
                            best_offset = offset
                            break

            if best_offset is not None:
                break   # Found a definitive offset — stop scanning further blocks

        # ------------------------------------------------------------------ #
        # Format result
        # ------------------------------------------------------------------ #
        if best_offset is not None:
            offset_str = f"ebp{best_offset}" if best_offset < 0 else f"ebp+{best_offset}"
            var_offsets[var_name] = {
                "assembly_reference": offset_str,
                "allocation_space_offset": best_offset
            }
        elif assigned_reg is not None:
            # Tier 4: Register-only — variable was never spilled to the stack.
            # This is the critical OOD fix: the model must learn that ANY variable
            # (not just ptr_ ones) can have null allocation_space_offset when the
            # compiler keeps it in a register throughout the function's lifetime.
            var_offsets[var_name] = {
                "assembly_reference": f"register {assigned_reg}",
                "allocation_space_offset": None
            }
        # If neither found: variable is ghost / optimized out entirely.
        # It is intentionally absent from var_offsets; the caller counts it as missing.

    return var_offsets


def save_compact_json(dataset: list, out_path: str, logger: logging.Logger) -> None:
    """Saves the dataset JSON to disk."""
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
        logger.info("Saved dataset to %s", out_path)
    except IOError as e:
        logger.error("Failed to save JSON: %s", e)


def main():
    args = parse_args()
    setup_logger(args.log_level)

    if not os.path.exists(args.json_path):
        LOGGER.error("JSON file not found: %s", args.json_path)
        return

    LOGGER.info("Loading dataset from %s ...", args.json_path)
    with open(args.json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    LOGGER.info("Parsing assembly and extracting ground truth labels...")
    total_vars_found = 0
    total_vars_missing = 0
    total_register_only = 0

    for record in tqdm(dataset, desc="Extracting Labels"):
        full_asm = record.get("assembly_code", "")
        if not full_asm:
            continue

        for func in record.get("label", {}).get("functions", []):
            func_name = func.get("function_name")
            if not func_name:
                LOGGER.warning("Function entry with no function_name — skipping.")
                continue

            func_asm = extract_function_asm(full_asm, func_name)
            if not func_asm:
                LOGGER.debug("Could not locate assembly block for: %s", func_name)
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
                    if extracted_offsets[v_name]["allocation_space_offset"] is None:
                        total_register_only += 1
                    else:
                        total_vars_found += 1
                else:
                    total_vars_missing += 1

    LOGGER.info("Extraction complete!")
    LOGGER.info("Stack-allocated:   %d", total_vars_found)
    LOGGER.info("Register-only:     %d  (null offset, has assembly_reference)", total_register_only)
    LOGGER.info("Missing/ghost:     %d  (not in extracted_offsets)", total_vars_missing)

    LOGGER.info("Saving finalized JSON dataset...")
    save_compact_json(dataset, args.json_path, LOGGER)
    LOGGER.info("Dataset fully finalized and ready for ML training!")


if __name__ == "__main__":
    main()