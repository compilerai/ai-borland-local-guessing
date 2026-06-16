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

SOURCE_C_CODE_DIR = "source_codes"

C_TYPES = ["int", "char", "short", "double", "float", "long"]
OPAQUE_SINKS = ["baz", "foober", "MYmyread", "MYmywrite", "bazz"]

STRUCT_DEFS = {
    "struct Point": {
        "decl": "struct Point {\n    int x;\n    int y;\n};",
        "first_field": "x",
        "first_field_type": "int"
    },
    "struct DataNode": {
        "decl": "struct DataNode {\n    char flag;\n    double value;\n};",
        "first_field": "flag",
        "first_field_type": "char"
    },
    "struct Buffer": {
        "decl": "struct Buffer {\n    int id;\n    char buf[12];\n};",
        "first_field": "id",
        "first_field_type": "int"
    },
    "struct NetworkPacket": {
        "decl": "struct NetworkPacket {\n    short header;\n    struct Buffer payload;\n    int checksum;\n};",
        "first_field": "header",
        "first_field_type": "short"
    }
}

STRUCT_DECL_ORDER = ["struct Point", "struct DataNode", "struct Buffer", "struct NetworkPacket"]
ALL_TYPES = C_TYPES + list(STRUCT_DEFS.keys())

REALISTIC_VAR_NAMES = [
    "i", "j", "k", "n", "m", "x", "y", "z", "idx", "val", "tmp", "len", "count", "flag", "result",
    "offset", "size", "err", "ret", "sign", "parity", "total", "sum", "mean", "buf", "ptr", "p", "q",
    "fd", "src", "dst", "out", "in_", "cur", "prev", "next", "lo", "hi", "mid", "lim", "num", "data",
    "key", "hash", "seed", "mask", "mode", "state", "L", "N", "K", "T", "M", "R", "S"
]

REALISTIC_PARAM_NAMES = [
    "n", "m", "count", "fd", "size", "val", "flag", "len", "offset", "p", "q", "src", "dst", "buf",
    "key", "idx", "a", "b", "c", "x", "y", "lo", "hi", "mean", "sign", "mode", "seed", "M", "N", "K"
]

# Hyper-parameters
MIN_NUM_HOIST_DECL = 6
MAX_NUM_HOIST_DECL = 15
IS_ARRAY_PROB_THRESHOLD = 0.35
MIN_ARRAY_SIZE = 8
MAX_ARRAY_SIZE = 256
VOLATILE_THRESHOLD = 0.8
OPAQUE_SINK_THRESHOLD = 0.8
NOISE_GHOST_VARS_PROB = 0.8

MAX_NOISE_BLOCKS = 4
NOISE_ARTITHMETIC_OPS_PROB = 0.6
NOISE_CONDITIONAL_PROB = 0.5
NOISE_FOR_LOOP_PROB = 0.4
BINARY_OPS_PROB = 0.3
THIRD_VAR_BEING_CONST_PROB = 0.5
SWITCH_STATEMENT_PROB = 0.35
POINTER_ALIAS_PROB = 0.40

MULTI_HOP_ALIAS_PROB = 0.25
ARRAY_ELEMENT_ESCAPE_PROB = 0.25
STRUCT_FIELD_ESCAPE_PROB = 0.25
POINTER_ARITH_PROB = 0.20
SCOPED_ARRAY_PROB = 0.30
PARAM_ADDR_TAKEN_PROB = 0.50
RETURN_NON_VOID_PROB = 0.60

MICRO_FUNC_PROB = 0.15


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(asctime)s | %(levelname)s | %(message)s", force=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI BORLAND LOCAL GUESSING - V3 (Collision-Proof)")
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--max_random_func", default=10, type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


# ====================================================================================
# MEMORY CHECK: All naming functions now require the "used_names" set to prevent dupes
# ====================================================================================
def get_rand_name(used_names: set, prefix="var_") -> str:
    """Generates a unique variable name checking against the used_names memory bank."""
    while True:
        if random.random() < 0.60:
            name = f"{random.choice(REALISTIC_VAR_NAMES)}{random.randint(0, 99)}"
        else:
            name = prefix + "".join(random.choices(string.ascii_lowercase, k=4))
        
        if name not in used_names:
            used_names.add(name)
            return name


def generate_function_params(func_id: int, is_micro_func: bool, used_names: set) -> list:
    """Generates unique parameters and logs them in the memory bank."""
    num_params = random.randint(0, 1) if is_micro_func else random.randint(0, 4)
    params = []
    
    for _ in range(num_params):
        p_type = random.choice(["int", "char", "short", "long", "double", "float"])
        while True:
            p_name = random.choice(REALISTIC_PARAM_NAMES) + f"_{func_id}_{random.randint(0, 9)}"
            if p_name not in used_names:
                used_names.add(p_name)
                break
        params.append((p_type, p_name))
    return params


def generate_random_val(v_type: str) -> str:
    if v_type in ["float", "double"]:
        return f"{random.uniform(0.5, 999.5):.2f}"
    elif v_type == "char":
        return f"'{random.choice(string.ascii_uppercase)}'"
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


def generate_pointer_indirection_block(v_name: str, v_type: str, is_array: bool, used_names: set) -> list:
    sink = random.choice(OPAQUE_SINKS)
    ptr_name = get_rand_name(used_names, "p")
    target = f"&{v_name}[0]" if is_array else f"&{v_name}"
    return [f"    {{", f"        {v_type}* {ptr_name} = {target};", f"        {sink}((void*){ptr_name});", f"    }}"]


def generate_multi_hop_alias_block(v_name: str, v_type: str, is_array: bool, used_names: set) -> list:
    sink = random.choice(OPAQUE_SINKS)
    ptr_a, ptr_b = get_rand_name(used_names, "a"), get_rand_name(used_names, "b")
    target = f"&{v_name}[0]" if is_array else f"&{v_name}"
    return [
        f"    {{", f"        {v_type}* {ptr_a} = {target};",
        f"        {v_type}* {ptr_b} = {ptr_a};", f"        {sink}((void*){ptr_b});", f"    }}"
    ]


def generate_array_element_escape_block(v_name: str, arr_size: int) -> list:
    sink = random.choice(OPAQUE_SINKS)
    idx = random.randint(1, max(1, arr_size - 1))
    return [f"    {sink}((void*)&{v_name}[{idx}]);"]


def generate_struct_field_escape_block(v_name: str, v_type: str) -> list:
    sink = random.choice(OPAQUE_SINKS)
    field = STRUCT_DEFS[v_type]["first_field"]
    return [f"    {sink}((void*)&{v_name}.{field});"]


def generate_pointer_arithmetic_block(v_name: str, arr_size: int, used_names: set) -> list:
    sink = random.choice(OPAQUE_SINKS)
    step = random.randint(1, max(1, arr_size // 4))
    ptr_name = get_rand_name(used_names, "p")
    return [
        f"    {{", f"        char* {ptr_name} = (char*)&{v_name} + {step};",
        f"        {sink}((void*){ptr_name});", f"    }}"
    ]


def generate_scoped_array_block(used_names: set) -> list:
    arr_size = random.choice([8, 16, 64, 128, 256])
    buf_name, loop_var = get_rand_name(used_names, "buf"), get_rand_name(used_names, "i")
    limit, sink = random.randint(2, 10), random.choice(OPAQUE_SINKS)
    return [
        f"    {{", f"        int {loop_var};",
        f"        for ({loop_var} = 0; {loop_var} < {limit}; {loop_var}++) {{",
        f"            char {buf_name}[{arr_size}];", f"            {sink}((void*)&{buf_name});",
        f"        }}", f"    }}"
    ]


def generate_function_body(func_id: int) -> tuple[str, dict]:
    lines = []
    
    # Initialize the Memory Bank for this specific function scope
    used_names = set()

    is_micro_func = random.random() < MICRO_FUNC_PROB
    params = generate_function_params(func_id, is_micro_func, used_names)
    func_name = f"synth_func_{func_id}"

    ret_type = random.choice(["int", "int", "int", "double", "char"]) if random.random() < RETURN_NON_VOID_PROB else "void"

    if params:
        lines.append(f"{ret_type} {func_name}({', '.join(f'{pt} {pn}' for pt, pn in params)}) {{")
    else:
        lines.append(f"{ret_type} {func_name}(void) {{")

    function_meta_data = {
        "function_name": func_name,
        "stack_allocation_instruction": None,
        "stack_allocation_size_bytes": None,
        "variable_mapping": []
    }

    for pt, pn in params:
        function_meta_data["variable_mapping"].append({
            "variable_name": pn,
            "assembly_reference": None,
            "allocation_space_offset": None,
            "size_bytes": None
        })

    num_vars = random.randint(1, 2) if is_micro_func else random.randint(MIN_NUM_HOIST_DECL, MAX_NUM_HOIST_DECL)
    array_prob_thresh = 0.0 if is_micro_func else IS_ARRAY_PROB_THRESHOLD

    var_info_map = defaultdict(list)
    all_vars = []

    for var_id in range(num_vars):
        v_type = random.choice(C_TYPES) if is_micro_func else random.choice(ALL_TYPES)
        v_name = get_rand_name(used_names)
        is_array = random.random() < array_prob_thresh
        is_ghost_variable = random.random() > NOISE_GHOST_VARS_PROB
        array_size = random.randint(MIN_ARRAY_SIZE, MAX_ARRAY_SIZE) if is_array else 0

        if is_array:
            lines.append(f"    {v_type} {v_name}[{array_size}];")
        else:
            var_info_map[v_type].append(v_name)

        all_vars.append([v_type, v_name, is_array, is_ghost_variable, array_size])

    for var_type, var_name in var_info_map.items():
        vol = "volatile " if random.random() > VOLATILE_THRESHOLD else ""
        lines.append(f"    {vol}{var_type} " + ", ".join(var_name) + ";")

    new_pointers = []
    for v_type, v_name, is_array, is_ghost, _ in list(all_vars):
        if not is_array and not is_ghost and random.random() < POINTER_ALIAS_PROB:
            ptr_name = get_rand_name(used_names, "p")
            lines.append(f"    {v_type}* {ptr_name} = &{v_name};")
            new_pointers.append(["pointer", ptr_name, False, False, 0])
    all_vars.extend(new_pointers)
    lines.append("")

    for var_type, var_name, is_array, _, _ in all_vars:
        if var_type == "pointer": continue
        if var_type in STRUCT_DEFS:
            field = STRUCT_DEFS[var_type]["first_field"]
            f_type = STRUCT_DEFS[var_type]["first_field_type"]
            target = f"{var_name}[0].{field}" if is_array else f"{var_name}.{field}"
            lines.append(f"    {target} = {generate_random_val(f_type)};")
        else:
            target = f"{var_name}[0]" if is_array else var_name
            lines.append(f"    {target} = {generate_random_val(var_type)};")

    if params and not is_micro_func:
        for var_type, var_name, is_array, is_ghost, _ in all_vars:
            if is_ghost or var_type == "pointer": continue
            if random.random() < 0.30:
                compat_params = [
                    pn for pt, pn in params
                    if pt == var_type or (pt in ["int", "long", "short", "char"] and var_type in ["int", "long", "short", "char"])
                ]
                if compat_params:
                    p_name = random.choice(compat_params)
                    target = f"{var_name}[0]" if is_array else var_name
                    if var_type not in STRUCT_DEFS:
                        lines.append(f"    {target} = ({var_type}){p_name};")
    lines.append("")

    OPS = ["+", "-", "*"]
    REL_OPS = ["<", ">", "<=", ">=", "!=", "=="]
    INT_TYPES = ["int", "short", "long", "char"]

    noise_candidates = []
    for v in all_vars:
        v_type, v_name, is_array, _, _ = v
        if v_type == "pointer": continue
        actual_type = STRUCT_DEFS[v_type]["first_field_type"] if v_type in STRUCT_DEFS else v_type
        access_str = f"{v_name}[0]" if is_array else v_name
        if v_type in STRUCT_DEFS: access_str += f".{STRUCT_DEFS[v_type]['first_field']}"
        noise_candidates.append([actual_type, access_str])
    for pt, pn in params:
        if pt not in STRUCT_DEFS:
            noise_candidates.append([pt, pn])

    grouped_lines = []
    max_noise_iters = 1 if is_micro_func else MAX_NOISE_BLOCKS

    for _ in range(random.randint(1, max_noise_iters)):
        if not noise_candidates: break

        if random.random() < NOISE_ARTITHMETIC_OPS_PROB and len(noise_candidates) >= 2:
            c1, c2, op = random.choice(noise_candidates), random.choice(noise_candidates), random.choice(OPS)
            if c1[0] in INT_TYPES and c2[0] in INT_TYPES:
                grouped_lines.append([f"    {c1[1]} = {c1[1]} {op} {c2[1]};"])

        if random.random() < NOISE_CONDITIONAL_PROB and len(noise_candidates) >= 2:
            c1, c2, rel_op = random.choice(noise_candidates), random.choice(noise_candidates), random.choice(REL_OPS)
            grouped_lines.append([
                f"    if ({c1[1]} {rel_op} {c2[1]}) {{",
                f"        {c1[1]} = {generate_random_val(c1[0])};",
                f"    }}"
            ])

        if random.random() < NOISE_FOR_LOOP_PROB and noise_candidates and not is_micro_func:
            c1, loop_var, limit = random.choice(noise_candidates), get_rand_name(used_names, "i"), random.randint(2, 10)
            if c1[0] in INT_TYPES:
                grouped_lines.append([
                    f"    {{", f"        int {loop_var};",
                    f"        for ({loop_var} = 0; {loop_var} < {limit}; {loop_var}++) {{",
                    f"            {c1[1]} = {c1[1]} + {loop_var};", f"        }}", f"    }}"
                ])
            else:
                grouped_lines.append([
                    f"    {{", f"        int {loop_var};",
                    f"        for ({loop_var} = 0; {loop_var} < {limit}; {loop_var}++) {{",
                    f"            {c1[1]} = {generate_random_val(c1[0])};", f"        }}", f"    }}"
                ])

        if random.random() < SWITCH_STATEMENT_PROB and noise_candidates and not is_micro_func:
            int_candidates = [c for c in noise_candidates if c[0] in INT_TYPES]
            if int_candidates:
                c1, c2 = random.choice(int_candidates), random.choice(int_candidates)
                case_vals = random.sample(range(0, 20), k=3)
                block = [f"    switch ({c1[1]} % 3) {{"]
                for cv in case_vals:
                    block.extend([f"        case {cv}:", f"            {c2[1]} = {generate_random_val(c2[0])};", f"            break;"])
                block.extend([f"        default:", f"            {c2[1]} = {generate_random_val(c2[0])};", f"            break;", f"    }}"])
                grouped_lines.append(block)

    if random.random() < BINARY_OPS_PROB and len(noise_candidates) >= 3:
        c1, c2, c3 = random.sample(noise_candidates, 3)
        if c1[0] in INT_TYPES and c2[0] in INT_TYPES:
            if random.random() < THIRD_VAR_BEING_CONST_PROB:
                rhs = generate_random_val(c1[0])
            else:
                rhs = f"{c2[1]} {random.choice(OPS)} {c3[1]}" if c3[0] in INT_TYPES else generate_random_val(c1[0])
            grouped_lines.append([f"    {c1[1]} = {rhs};"])

    if random.random() < SCOPED_ARRAY_PROB and not is_micro_func:
        grouped_lines.append(generate_scoped_array_block(used_names))

    for v_type, v_name, is_array, is_ghost, array_size in all_vars:
        if is_ghost: continue
        if v_type != "pointer" and random.random() > 0.20:
            roll = random.random()
            if is_array and roll < ARRAY_ELEMENT_ESCAPE_PROB and array_size > 2:
                grouped_lines.append(generate_array_element_escape_block(v_name, array_size))
            elif is_array and roll < ARRAY_ELEMENT_ESCAPE_PROB + POINTER_ARITH_PROB:
                grouped_lines.append(generate_pointer_arithmetic_block(v_name, array_size, used_names))
            elif not is_array and v_type in STRUCT_DEFS and roll < STRUCT_FIELD_ESCAPE_PROB:
                grouped_lines.append(generate_struct_field_escape_block(v_name, v_type))
            elif not is_array and roll < MULTI_HOP_ALIAS_PROB:
                grouped_lines.append(generate_multi_hop_alias_block(v_name, v_type, is_array, used_names))
            elif not is_array and roll < MULTI_HOP_ALIAS_PROB + POINTER_ALIAS_PROB:
                grouped_lines.append(generate_pointer_indirection_block(v_name, v_type, is_array, used_names))
            else:
                actual_field_name = STRUCT_DEFS[v_type]["first_field"] if v_type in STRUCT_DEFS else None
                grouped_lines.append(add_escape(v_name, is_array, actual_field_name))

        function_meta_data["variable_mapping"].append({
            "variable_name": v_name,
            "assembly_reference": None,
            "allocation_space_offset": None,
            "size_bytes": None
        })

    for pt, pn in params:
        if random.random() < PARAM_ADDR_TAKEN_PROB:
            sink = random.choice(OPAQUE_SINKS)
            if random.random() < 0.40:
                ptr_name = get_rand_name(used_names, "p")
                grouped_lines.append([
                    f"    {{", f"        {pt}* {ptr_name} = &{pn};", f"        {sink}((void*){ptr_name});", f"    }}"
                ])
            else:
                grouped_lines.append([f"    {sink}((void*)&{pn});"])

    random.shuffle(grouped_lines)
    for group in grouped_lines:
        lines.extend(group)

    if ret_type != "void":
        if ret_type == "double": lines.append("    return 0.0;")
        elif ret_type == "char": lines.append("    return '\\0';")
        else: lines.append("    return 0;")

    lines.append("}\n")
    return "\n".join(lines), function_meta_data


def generate_c_code(file_id: int, max_random_func: int) -> tuple[str, dict]:
    lines = ["#include <stdio.h>", "#include <stdlib.h>\n"]
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

    LOGGER.info("Successfully generated %d / %d C programs and saved to %s",
                success_count, args.num_samples, source_code_dir)