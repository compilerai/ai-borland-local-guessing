import argparse
import logging
import os
import random
import string
from collections import defaultdict
from tqdm import tqdm
import json
import uuid
import re

LOGGER = logging.getLogger(__name__)

# Directory names
SOURCE_C_CODE_DIR = "source_codes"

# --- CONFIGURATION & TEMPLATES ---
C_TYPES = ["int", "char", "short", "double", "float", "long"]
OPAQUE_SINKS = ["baz", "foober", "MYmyread", "MYmywrite", "bazz"]

# Borland 32-bit type sizes
TYPE_SIZES = {
    "char": 1,
    "short": 2,
    "int": 4,
    "long": 4,
    "float": 4,
    "double": 8
}

# Struct Templates (Hardcoded to guarantee perfect size calculations despite C padding rules)
STRUCT_DEFS = {
    "struct Point": {
        "decl": "struct Point {\n    int x;\n    int y;\n};",
        "size": 8,
        "first_field": "x",
        "first_field_type": "int"
    },
    "struct DataNode": {
        "decl": "struct DataNode {\n    char flag;\n    double value;\n};",
        "size": 16, # 1 byte char + 7 bytes padding + 8 bytes double = 16 bytes
        "first_field": "flag",
        "first_field_type": "char"
    },
    "struct Buffer": {
        "decl": "struct Buffer {\n    int id;\n    char buf[12];\n};",
        "size": 16, # 4 byte int + 12 byte array = 16 bytes
        "first_field": "id",
        "first_field_type": "int"
    },
    # COMPLEXITY UPGRADE 1: Nested Structs
    "struct NetworkPacket": {
        "decl": "struct NetworkPacket {\n    short header;\n    struct Buffer payload;\n    int checksum;\n};",
        "size": 24, # 2 byte short + 2 pad + 16 struct + 4 int
        "first_field": "header",
        "first_field_type": "short"
    }
}

# Explicit compilation order to prevent C compiler failures
STRUCT_DECL_ORDER = [
    "struct Point",
    "struct DataNode",
    "struct Buffer",
    "struct NetworkPacket" # Relies on Buffer, MUST come last
]

ALL_TYPES = C_TYPES + list(STRUCT_DEFS.keys())

# Hyper-parameters
# PER FUNCTION
MIN_NUM_HOIST_DECL = 6
MAX_NUM_HOIST_DECL = 15
IS_ARRAY_PROB_THRESHOLD = 0.35
MIN_ARRAY_SIZE = 8
MAX_ARRAY_SIZE = 256
VOLATILE_THRESHOLD = 0.8
OPAQUE_SINK_THRESHOLD = 0.8
NOISE_GHOST_VARS_PROB = 0.8

# Noise tuning
MAX_NOISE_BLOCKS = 4
NOISE_ARTITHMETIC_OPS_PROB = 0.6
NOISE_CONDITIONAL_PROB = 0.5
NOISE_FOR_LOOP_PROB = 0.4
BINARY_OPS_PROB = 0.3
THIRD_VAR_BEING_CONST_PROB = 0.5

# COMPLEXITY UPGRADES
SWITCH_STATEMENT_PROB = 0.35
POINTER_ALIAS_PROB = 0.40


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI BORLAND LOCAL GUESSING - COMPLEX GENERATOR")
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--max_random_func", default=10, type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()

def get_rand_name(prefix="var_") -> str:
    """Generates a random variable name."""
    return prefix + "".join(random.choices(string.ascii_lowercase, k=6))

def generate_random_val(v_type: str) -> str:
    """Generates a type-appropriate literal value for assignment."""
    if v_type in ["float", "double"]:
        return f"{random.uniform(0.5, 999.5):.2f}"
    elif v_type == "char":
        return f"'{random.choice(string.ascii_uppercase)}'"
    else:
        return str(random.randint(0, 1000))

def add_escape(v_name, is_array, actual_field_name):
    sink = random.choice(OPAQUE_SINKS)
    curr_lines = []

    if random.random() > OPAQUE_SINK_THRESHOLD:
        condition_target = f"{v_name}[0]" if is_array else v_name
        condition_target = f"{condition_target}.{actual_field_name}" if actual_field_name else condition_target
        curr_lines.append(f"    if ({condition_target} > -1) {{")
        curr_lines.append(f"        {sink}((void*)&{v_name});")
        curr_lines.append("    }")
    else:
        curr_lines.append(f"    {sink}((void*)&{v_name});")
    return curr_lines

def generate_function_body(func_id: int) -> tuple[str, dict]:
    lines = []
    func_name = f"synth_func_{func_id}"
    lines.append(f"void {func_name}() {{")

    # Initialize function_meta_data at the start of every function body.
    # Previously this was never created, causing a NameError at runtime.
    function_meta_data = {
        "function_name": func_name,
        "stack_allocation_instruction": None,
        "stack_allocation_size_bytes": None,
        "variable_mapping": []
    }

    # Step 1: Hoisted Declarations
    num_vars = random.randint(MIN_NUM_HOIST_DECL, MAX_NUM_HOIST_DECL)
    var_info_map = defaultdict(list)
    all_vars = []

    for var_id in range(num_vars):
        v_type = random.choice(ALL_TYPES)
        v_name = get_rand_name()
        is_array = random.random() < IS_ARRAY_PROB_THRESHOLD
        is_ghost_variable = random.random() > NOISE_GHOST_VARS_PROB
        array_size = 0

        if is_array:
            array_size = random.randint(MIN_ARRAY_SIZE, MAX_ARRAY_SIZE)
            lines.append(f"    {v_type} {v_name}[{array_size}];")
        else:
            var_info_map[v_type].append(v_name)

        all_vars.append([v_type, v_name, is_array, is_ghost_variable, array_size])

    for var_type, var_name in var_info_map.items():
        vol = "volatile " if random.random() > VOLATILE_THRESHOLD else ""
        var_dec_line = f"    {vol}{var_type} " + ", ".join(var_name) + ";"
        lines.append(var_dec_line)

    # COMPLEXITY UPGRADE 2: Pointer Aliasing Declarations
    new_pointers = []
    # Use list(all_vars) to iterate over a copy, preventing infinite loops!
    for v_type, v_name, is_array, is_ghost, _ in list(all_vars):
        if not is_array and not is_ghost and random.random() < POINTER_ALIAS_PROB:
            ptr_name = get_rand_name("ptr_")
            lines.append(f"    {v_type}* {ptr_name} = &{v_name};")
            # Add to a temporary list first
            new_pointers.append(["pointer", ptr_name, False, False, 0])

    # Safely extend all_vars after the loop is done
    all_vars.extend(new_pointers)

    lines.append("")

    # Step 2: Dummy assignments
    for var_type, var_name, is_array, _, _ in all_vars:
        if var_type == "pointer":
            continue # Pointers are assigned at declaration, skip them here!

        if var_type in STRUCT_DEFS:
            field = STRUCT_DEFS[var_type]["first_field"]
            f_type = STRUCT_DEFS[var_type]["first_field_type"]
            target = f"{var_name}[0].{field}" if is_array else f"{var_name}.{field}"
            lines.append(f"    {target} = {generate_random_val(f_type)};")
        else:
            target = f"{var_name}[0]" if is_array else var_name
            lines.append(f"    {target} = {generate_random_val(var_type)};")

    lines.append("")

    # Step 3: Adding noise
    # FIX 2: This block was entirely missing before. The noise injection described
    # in the PDF (P(Loop)=0.4, P(Conditional)=0.5, etc.) never fired, meaning the
    # generated dataset lacked the register-pressure complexity needed for training.
    OPS = ["+", "-", "*"]
    REL_OPS = ["<", ">", "<=", ">=", "!=", "=="]
    INT_TYPES = ["int", "short", "long", "char"]

    noise_candidates = []
    for v in all_vars:
        v_type, v_name, is_array, _, _ = v

        # Prevent pointers from being corrupted by arithmetic noise
        if v_type == "pointer":
            continue

        actual_type = STRUCT_DEFS[v_type]["first_field_type"] if v_type in STRUCT_DEFS else v_type
        access_str = f"{v_name}[0]" if is_array else v_name
        if v_type in STRUCT_DEFS:
            access_str += f".{STRUCT_DEFS[v_type]['first_field']}"
        noise_candidates.append([actual_type, access_str])

    grouped_lines = []

    for _ in range(random.randint(1, MAX_NOISE_BLOCKS)):
        if not noise_candidates:
            break

        # Arithmetic noise: var = var OP var (integers only to avoid UB)
        if random.random() < NOISE_ARTITHMETIC_OPS_PROB and len(noise_candidates) >= 2:
            c1 = random.choice(noise_candidates)
            c2 = random.choice(noise_candidates)
            op = random.choice(OPS)
            # Restrict to int-compatible types to avoid float/struct arithmetic UB
            if c1[0] in INT_TYPES and c2[0] in INT_TYPES:
                noise_block = [f"    {c1[1]} = {c1[1]} {op} {c2[1]};"]
                grouped_lines.append(noise_block)

        # Conditional noise: if (var REL_OP var) { var = const; }
        if random.random() < NOISE_CONDITIONAL_PROB and len(noise_candidates) >= 2:
            c1 = random.choice(noise_candidates)
            c2 = random.choice(noise_candidates)
            rel_op = random.choice(REL_OPS)
            noise_block = [
                f"    if ({c1[1]} {rel_op} {c2[1]}) {{",
                f"        {c1[1]} = {generate_random_val(c1[0])};",
                f"    }}"
            ]
            grouped_lines.append(noise_block)

        # For-loop noise: loop variable over a small range, modifying a candidate
        if random.random() < NOISE_FOR_LOOP_PROB and noise_candidates:
            c1 = random.choice(noise_candidates)
            loop_var = get_rand_name("i_")
            limit = random.randint(2, 10)
            # Only do arithmetic inside the loop for integer-compatible types
            if c1[0] in INT_TYPES:
                noise_block = [
                    f"    {{",
                    f"        int {loop_var};",
                    f"        for ({loop_var} = 0; {loop_var} < {limit}; {loop_var}++) {{",
                    f"            {c1[1]} = {c1[1]} + {loop_var};",
                    f"        }}",
                    f"    }}"
                ]
            else:
                # For floats/structs, just re-assign inside the loop to create pressure
                noise_block = [
                    f"    {{",
                    f"        int {loop_var};",
                    f"        for ({loop_var} = 0; {loop_var} < {limit}; {loop_var}++) {{",
                    f"            {c1[1]} = {generate_random_val(c1[0])};",
                    f"        }}",
                    f"    }}"
                ]
            grouped_lines.append(noise_block)

        # Switch-statement noise (COMPLEXITY UPGRADE): switch on an int candidate
        if random.random() < SWITCH_STATEMENT_PROB and noise_candidates:
            int_candidates = [c for c in noise_candidates if c[0] in INT_TYPES]
            if int_candidates:
                c1 = random.choice(int_candidates)
                c2 = random.choice(int_candidates)
                case_vals = random.sample(range(0, 20), k=3)
                noise_block = [
                    f"    switch ({c1[1]} % 3) {{"
                ]
                for cv in case_vals:
                    noise_block.append(f"        case {cv}:")
                    noise_block.append(f"            {c2[1]} = {generate_random_val(c2[0])};")
                    noise_block.append(f"            break;")
                noise_block.append(f"        default:")
                noise_block.append(f"            {c2[1]} = {generate_random_val(c2[0])};")
                noise_block.append(f"            break;")
                noise_block.append(f"    }}")
                grouped_lines.append(noise_block)

    # Step 4: Binary-op noise between two randomly chosen candidates (extra register pressure)
    if random.random() < BINARY_OPS_PROB and len(noise_candidates) >= 3:
        c1, c2, c3 = random.sample(noise_candidates, 3)
        if c1[0] in INT_TYPES and c2[0] in INT_TYPES:
            op = random.choice(OPS)
            if random.random() < THIRD_VAR_BEING_CONST_PROB:
                rhs = generate_random_val(c1[0])
            else:
                rhs = f"{c2[1]} {op} {c3[1]}" if c3[0] in INT_TYPES else generate_random_val(c1[0])
            grouped_lines.append([f"    {c1[1]} = {rhs};"])

    # Step 5: Escapes and Metadata Tracking
    for v_type, v_name, is_array, is_ghost, array_size in all_vars:
        if is_ghost: continue

        # Pointers don't need opaque sinks, they are already on the stack.
        if v_type != "pointer":
            # Allow 20% of non-ghost variables to bypass escapes entirely, 
            # freeing the Borland compiler to optimize them into registers.
            if random.random() > 0.20:
                actual_field_name = STRUCT_DEFS[v_type]["first_field"] if v_type in STRUCT_DEFS else None
                grouped_lines.append(add_escape(v_name, is_array, actual_field_name))

        # Size logic mapping (Pointers are ALWAYS 4 bytes in 32-bit x86)
        if v_type == "pointer":
            final_size = 4
        else:
            base_size = STRUCT_DEFS[v_type]["size"] if v_type in STRUCT_DEFS else TYPE_SIZES[v_type]
            final_size = base_size * array_size if is_array else base_size

        function_meta_data["variable_mapping"].append({
            "variable_name": v_name,
            "assembly_reference": None,
            "allocation_space_offset": None,
            "size_bytes": final_size
        })

    # Step 6: Shuffle and Assemble
    random.shuffle(grouped_lines)
    for group in grouped_lines:
        lines.extend(group)

    lines.append("}\n")
    return "\n".join(lines), function_meta_data

def generate_c_code(file_id: int, max_random_func: int) -> tuple[str, dict]:
    lines = ["#include <stdio.h>", "#include <stdlib.h>\n"]

    # Explicitly use the safe declaration order
    for s_key in STRUCT_DECL_ORDER:
        lines.append(STRUCT_DEFS[s_key]["decl"])
    lines.append("")

    for sink in OPAQUE_SINKS:
        lines.append(f"void {sink}(void*);")
    lines.append("\n")

    file_label = {
        "file_id": file_id,
        "source_code_c": "",
        "assembly_code": "",
        "label": {"functions": []}
    }

    num_functions = random.randint(1, max_random_func)
    for i in range(num_functions):
        function_code, function_meta_data = generate_function_body(func_id=i)
        lines.append(function_code)
        file_label["label"]["functions"].append(function_meta_data)

    final_c_code = "\n".join(lines)
    file_label["source_code_c"] = final_c_code

    return final_c_code, file_label

if __name__ == "__main__":
    args = parse_args()
    setup_logger(args.log_level)

    os.makedirs(args.output_dir, exist_ok=True)
    source_code_dir = os.path.join(args.output_dir, SOURCE_C_CODE_DIR)
    os.makedirs(source_code_dir, exist_ok=True)
    LOGGER.info("Generating samples for %d", args.num_samples)

    success_count = 0
    dataset_records = []
    for i in tqdm(range(args.num_samples), desc="Generating samples"):
        file_path = os.path.join(source_code_dir, f"sample_{i:05d}.c")
        try:
            c_code, file_label_data = generate_c_code(i, args.max_random_func)
            with open(file_path, "w") as f:
                f.write(c_code)
            dataset_records.append(file_label_data)
            success_count += 1
        except IOError as e:
            LOGGER.error("Failed to generate c code to %s : %s", file_path, e)

    json_path = os.path.join(args.output_dir, "dataset_labels.json")
    try:
        placeholders = {}
        for record in tqdm(dataset_records, desc="Formatting variable mapping strings"):
            for func in record["label"]["functions"]:
                new_mappings = []
                for mapping in func["variable_mapping"]:
                    compact_str = json.dumps(mapping, separators=(', ', ': '))
                    ph = f"__VAR_MAPPING_{uuid.uuid4().hex}__"
                    placeholders[ph] = compact_str
                    new_mappings.append(ph)
                func["variable_mapping"] = new_mappings

        LOGGER.info("Serializing main JSON structure...")
        raw_json = json.dumps(dataset_records, indent=2)

        pattern = re.compile(r'"__VAR_MAPPING_[0-9a-f]{32}__"')
        pbar = tqdm(total=len(placeholders), desc="Swapping compact JSON blocks")

        def replacer(match):
            pbar.update(1)
            ph_key = match.group(0).strip('"')
            return placeholders[ph_key]

        raw_json = pattern.sub(replacer, raw_json)
        pbar.close()

        LOGGER.info("Writing JSON to disk...")
        with open(json_path, "w") as jf:
            jf.write(raw_json)

        LOGGER.info("Successfully saved ground truth JSON to %s", json_path)
    except IOError as e:
        LOGGER.error("Failed to save JSON file: %s", e)

    LOGGER.info("Successfully generated %d C programs and saved to %s", success_count, source_code_dir)