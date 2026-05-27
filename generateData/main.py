import argparse
import logging
import os
import random
import string
from collections import defaultdict
from tqdm import tqdm
import json

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
    }
}

ALL_TYPES = C_TYPES + list(STRUCT_DEFS.keys())

# Hyper-parameters
# PER FUNCTION
MIN_NUM_HOIST_DECL = 4
MAX_NUM_HOIST_DECL = 12
IS_ARRAY_PROB_THRESHOLD = 0.40
MIN_ARRAY_SIZE = 8
MAX_ARRAY_SIZE = 512
VOLATILE_THRESHOLD = 0.8
OPAQUE_SINK_THRESHOLD = 0.8
NOISE_GHOST_VARS_PROB = 0.8
# Noise tuning
MAX_NOISE_BLOCKS = 3
NOISE_ARTITHMETIC_OPS_PROB = 0.6
NOISE_CONDITIONAL_PROB = 0.5
NOISE_FOR_LOOP_PROB = 0.4
BINARY_OPS_PROB = 0.3
THIRD_VAR_BEING_CONST_PROB = 0.5

def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "AI BORLAND LOCAL GUESSING"
    )
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--max_random_func", default=10, type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
    return parser.parse_args()

def get_rand_name(prefix="var_") -> str:
    """Generates a random variable name."""
    return prefix + "".join(random.choices(string.ascii_lowercase, k=5))

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
        condition_target = f"{condition_target}.{actual_field_name}" if actual_field_name else condition_target # For struct variables
        curr_lines.append(f"    if ({condition_target} > -1) {{")
        curr_lines.append(f"        {sink}((void*)&{v_name});")
        curr_lines.append("    }")
    else:
        curr_lines.append(f"    {sink}((void*)&{v_name});")
    return curr_lines

def generate_function_body(func_id:int) -> tuple[str, dict]:
    lines = []
    
    func_name = f"synth_func_{func_id}"
    lines.append(f"void {func_name}() {{")

    # Step 1: Hoisted Declarations
    num_vars = random.randint(MIN_NUM_HOIST_DECL, MAX_NUM_HOIST_DECL)
    # mapping variable type to variable names
    var_info_map = defaultdict(list)
    # tracking all variables :: each entry would be a subarray with 4 entities - var_type, var_name, is_array, is_ghost_variable 
    all_vars = []

    for var_id in range(num_vars):
        v_type = random.choice(ALL_TYPES) # Now includes structs
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
        var_dec_line = f"    {vol}{var_type}"

        for name in var_name:
            var_dec_line += f" {name},"
        
        var_dec_line = var_dec_line[:-1] + ";"   # replace last ',' with ';'
        
        lines.append(var_dec_line)
    
    lines.append("")

    # Step 2: Dummy assignments
    for var_type, var_name, is_array, _, _ in all_vars:
        # If it's a struct, we assign to its first field to avoid C syntax errors
        if var_type in STRUCT_DEFS:
            field = STRUCT_DEFS[var_type]["first_field"]
            f_type = STRUCT_DEFS[var_type]["first_field_type"]
            if is_array:
                lines.append(f"    {var_name}[0].{field} = {generate_random_val(f_type)};")
            else:
                lines.append(f"    {var_name}.{field} = {generate_random_val(f_type)};")
        else:
            if is_array:
                lines.append(f"    {var_name}[0] = {generate_random_val(var_type)};")
            else:
                lines.append(f"    {var_name} = {generate_random_val(var_type)};")
    
    lines.append("")
    # This will be useful in step 3 and step 4 for more generality
    grouped_lines = []

    # Step 3 : Adding noise : Arithmetic Dependencies ---
    # Interleaving some mathematical assignments to add register pressure
    # v : [var_type, var_name, is_array, is_ghost_variable]
    OPS = ["+", "-", "*"]
    REL_OPS = ["<", ">", "<=", ">=", "!=", "=="] # Added relational operators
    INT_TYPES = ["int", "short", "long", "char"]

    noise_candidates = []
    for v in all_vars:
        v_type, v_name, is_array, _, _ = v
        
        # 1. Get the actual underlying data type (primitive vs struct field)
        actual_type = STRUCT_DEFS[v_type]["first_field_type"] if v_type in STRUCT_DEFS else v_type
        
        # 2. Build the proper C access string
        access_str = f"{v_name}[0]" if is_array else v_name
        if v_type in STRUCT_DEFS:
            access_str += f".{STRUCT_DEFS[v_type]['first_field']}"
            
        # Store as [actual_type, access_string] so your downstream v[1] calls work perfectly!
        noise_candidates.append([actual_type, access_str])

    # Now ALL variables (arrays, structs, primitives) can participate in noise!
    all_scalers = noise_candidates
    int_scalars = [v for v in noise_candidates if v[0] in INT_TYPES]

    if len(all_scalers)  >= 2:
        noisy_blocks = random.randint(1, MAX_NOISE_BLOCKS)
        for ops in range(noisy_blocks):
            
            target_var = random.choice(all_scalers)
            available_sources = [v for v in all_scalers if v[1] != target_var[1]]

            if available_sources:
                source_var = random.choice(available_sources)
                ops_selected = random.choice(OPS)
                available_sources_3rd_var = [v for v in all_scalers if v[1] not in {target_var[1], source_var[1]}]

                if random.random() > THIRD_VAR_BEING_CONST_PROB or not available_sources_3rd_var:
                    third_var = random.randint(1, 200)
                else:
                    third_var = random.choice(available_sources_3rd_var)[1]

                # Step 3.1 :: simple arithmetic ops
                curr_lines = []
                if random.random() > BINARY_OPS_PROB:
                    # trinary ops
                    curr_lines.append(f"    {target_var[1]} = {source_var[1]} {ops_selected} {third_var};")
                else:
                    # binary ops
                    curr_lines.append(f"    {target_var[1]} {ops_selected}= {third_var};")
                grouped_lines.append(curr_lines)

            # Step 3.2 ::  Simple conditional ops
            if random.random() < NOISE_CONDITIONAL_PROB:
                cond_var = random.choice(all_scalers)
                available_targets = [v for v in all_scalers if v[1] != cond_var[1]]

                if available_targets:
                    target_var = random.choice(available_targets)
                    rel_op = random.choice(REL_OPS)
                    compare_val = random.randint(0, 100)
                    modify_val = random.randint(1, 15)
                    
                    curr_lines = []
                    curr_lines.append(f"    if ({cond_var[1]} {rel_op} {compare_val}) {{")
                    curr_lines.append(f"        {target_var[1]} += {modify_val};")
                    curr_lines.append(f"    }} else {{")
                    curr_lines.append(f"        {target_var[1]} -= {modify_val};")
                    curr_lines.append(f"    }}")
                    grouped_lines.append(curr_lines)

            # Step 3.3 ::  Simple loops
            if random.random() < NOISE_FOR_LOOP_PROB and len(int_scalars) >= 2:
                
                loop_cond_var = random.choice(int_scalars)
                available_breaks = [v for v in int_scalars if v[1] != loop_cond_var[1]]

                if available_breaks:
                    max_break_condn = random.choice(available_breaks)
                    rel_op = random.choice(REL_OPS)
                    target_var = random.choice(all_scalers)
                    compare_val = random.randint(0, 100)
                    modify_val = random.randint(1, 15)

                    curr_lines = []
                    curr_lines.append(f"    for ({loop_cond_var[1]} = 0; {loop_cond_var[1]} < {max_break_condn[1]}; ++{loop_cond_var[1]}) {{")
                    curr_lines.append(f"        if ({target_var[1]} {rel_op} {compare_val}) {{")
                    curr_lines.append(f"            {target_var[1]} += {modify_val};")
                    curr_lines.append(f"        }} else {{")
                    curr_lines.append(f"            {target_var[1]} -= {modify_val};")
                    curr_lines.append(f"        }}")
                    curr_lines.append(f"    }}")
                    grouped_lines.append(curr_lines)
    
    # Step 4: Store the function meta data for labelling
    function_meta_data = {
        "function_name": func_name,
        "stack_allocation_instruction": None, # Will leave it later for predicting
        "stack_allocation_size_bytes": None, # ||
        "variable_mapping": []
    }

    # Step 5: the escapes - taking address
    for v_type, v_name, is_array, is_ghost_variable, array_size in all_vars:
        if is_ghost_variable:
            continue
        actual_field_name = STRUCT_DEFS[v_type]["first_field"] if v_type in STRUCT_DEFS else None
        grouped_lines.append(add_escape(v_name, is_array, actual_field_name))

        # Size logic mapping
        base_size = STRUCT_DEFS[v_type]["size"] if v_type in STRUCT_DEFS else TYPE_SIZES[v_type]
        final_size = base_size * array_size if is_array else base_size

        function_meta_data["variable_mapping"].append({
            "variable_name": v_name,
            "assembly_reference": None,
            "allocation_space_offset": None,
            "size_bytes": final_size
        })
    
    # Step 6: Suffle all groups and re-create the function
    random.shuffle(grouped_lines)

    for group in grouped_lines:
        for line in group:
            lines.append(line)

    lines.append("}\n")
    return "\n".join(lines), function_meta_data

def generate_c_code(file_id:int, max_random_func:int) -> tuple[str, dict]:
    lines = []

    lines.append("#include <stdio.h>")
    lines.append("#include <stdlib.h>\n")

    # Inject Struct Definitions globally
    for s_def in STRUCT_DEFS.values():
        lines.append(s_def["decl"])
    lines.append("")

    # Provide opaque function prototypes
    for sink in OPAQUE_SINKS:
        lines.append(f"void {sink}(void*);")
    lines.append("\n")

    file_label = {
        "file_id": file_id,
        "source_code_c": "", 
        "assembly_code": "", # Empty, ready for parsing script
        "label": {
            "functions": []
        }
    }

    # Step 2. Randomize how many functions are in this specific C file (e.g., 1 to 10)
    num_functions = random.randint(1, max_random_func)
    for i in range(num_functions):
        # Pass a unique ID for the function name
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
            
            # Save metadata to dataset array
            dataset_records.append(file_label_data)
            success_count += 1
        except IOError as e:
            LOGGER.error("Failed to generate c code to %s : %s", file_path, e)

    # Final step : Output the JSON Ground Truth File
    json_path = os.path.join(args.output_dir, "dataset_labels.json")
    try:
        with open(json_path, "w") as jf:
            json.dump(dataset_records, jf, indent=2)
        LOGGER.info("Successfully saved ground truth JSON to %s", json_path)
    except IOError as e:
        LOGGER.error("Failed to save JSON file: %s", e)
    
    LOGGER.info("Successfully generated %d C programs and saved to %s", success_count, source_code_dir)