import argparse
import logging
import os
import random
import string
from collections import defaultdict

LOGGER = logging.getLogger(__name__)

# --- CONFIGURATION & TEMPLATES ---
C_TYPES = ["int", "char", "short", "double", "float", "long"]
OPAQUE_SINKS = ["baz", "foober", "MYmyread", "MYmywrite", "bazz"]

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
NOISE_ARTITHMETIC_OPS_PROB = 0.8
MAX_NOISE_ARITHMETIC_OPS = 10
BINARY_OPS_PROB = 0.3
MAX_CONSTANT_VAR = 200
THIRD_VARIABLE_PROB = 0.5
NOISE_CONDITIONAL_PROB = 0.7
NOISE_FOR_LOOP_PROB = 0.7

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

def add_escape(v_name, is_array):
    sink = random.choice(OPAQUE_SINKS)

    curr_lines = []

    if random.random() > OPAQUE_SINK_THRESHOLD:
        condition_target = f"{v_name}[0]" if is_array else v_name
        curr_lines.append(f"    if ({condition_target} > -1) {{")
        curr_lines.append(f"        {sink}((void*)&{v_name});")
        curr_lines.append("    }")
    else:
        curr_lines.append(f"    {sink}((void*)&{v_name});")
    return curr_lines

def generate_function_body(func_id:int) -> str:
    """Generates the code for a single C function."""
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
        # randomly choose one datatype for var_id from C_TYPES
        v_type = random.choice(C_TYPES)
        v_name = get_rand_name()

        is_array = random.random() < IS_ARRAY_PROB_THRESHOLD
        is_ghost_variable = random.random() > NOISE_GHOST_VARS_PROB
        if is_array:

            array_size = random.randint(MIN_ARRAY_SIZE, MAX_ARRAY_SIZE)
            lines.append(f"    {v_type} {v_name}[{array_size}];")
        else:
            var_info_map[v_type].append(v_name)
        
        all_vars.append([v_type, v_name, is_array, is_ghost_variable])
    
    for var_type, var_name in var_info_map.items():
        vol = "volatile " if random.random() > VOLATILE_THRESHOLD else ""
        var_dec_line = f"    {vol}{var_type}"

        for name in var_name:
            var_dec_line += f" {name},"
        
        var_dec_line = var_dec_line[:-1] + ";"   # replace last ',' with ';'
        
        lines.append(var_dec_line)
    
    lines.append("")

    # Step 2: Dummy assignments
    for var_type, var_name, is_array, is_ghost_variable in all_vars:
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
    scalars = [v for v in all_vars if not v[2] and v[0] in C_TYPES]

    if len(scalars)  >= 1:
        noisy_arithmetic_ops = random.randint(1, MAX_NOISE_ARITHMETIC_OPS)
        for ops in range(noisy_arithmetic_ops):
            
            source_var = random.choice(scalars)
            target_var = random.choice(scalars)
            ops_selected = random.choice(OPS)

            third_var = random.choice(scalars)[1] if random.random() > THIRD_VARIABLE_PROB else random.randint(1, MAX_CONSTANT_VAR)
            alt_var = random.choice(scalars)[1] if random.random() > THIRD_VARIABLE_PROB else random.randint(1, MAX_CONSTANT_VAR)

            # Step 3.1 :: simple arithmetic ops
            curr_lines = []
            if source_var[1] != target_var[1] and random.random() > BINARY_OPS_PROB:
                # trinary ops
                curr_lines.append(f"    {target_var[1]} = {source_var[1]} {ops_selected} {third_var};")
            else:
                # binary ops
                curr_lines.append(f"    {target_var[1]} {ops_selected}= {third_var};")
            grouped_lines.append(curr_lines)

            # Step 3.2 ::  Simple conditional ops
            curr_lines = []
            cond_var = random.choice(scalars)
            curr_lines.append(f"    if ({cond_var[1]} != {alt_var}) {{")
            curr_lines.append(f"        {random.choice(scalars)[1]} += {third_var};")
            curr_lines.append(f"    }} else {{")
            curr_lines.append(f"        {random.choice(scalars)[1]} -= {third_var};")
            curr_lines.append(f"    }}")
            grouped_lines.append(curr_lines)

            # Step 3.3 ::  Simple loops
            curr_lines = []
            loop_cond_var = random.choice(scalars)
            initial_val = random.randint(0, 5)
            max_break_condn = random.choice(scalars)
            curr_lines.append(f"    for ({loop_cond_var[1]} = {initial_val}; {loop_cond_var[1]} < {max_break_condn[1]}; ++{loop_cond_var[1]}) {{")
            curr_lines.append(f"        if ({cond_var[1]} != {alt_var}) {{")
            curr_lines.append(f"            {random.choice(scalars)[1]} += {third_var};")
            curr_lines.append(f"        }} else {{")
            curr_lines.append(f"            {random.choice(scalars)[1]} -= {third_var};")
            curr_lines.append(f"        }}")
            curr_lines.append(f"    }}")
            grouped_lines.append(curr_lines)

    # Step 4: the escapes - taking address
    for _, v_name, is_array, is_ghost_variable in all_vars:
        if is_ghost_variable:
            continue
        grouped_lines.append(add_escape(v_name, is_array))
    
    # Step 5: Suffle all groups and re-create the function
    random.shuffle(grouped_lines)

    for group in grouped_lines:
        for line in group:
            lines.append(line)

    lines.append("}\n")
    return "\n".join(lines)

def generate_c_code(file_id:int, max_random_func:int) -> str:

    lines = []

    lines.append("#include <stdio.h>")
    lines.append("#include <stdlib.h>\n")

    # Step 1. Provide opaque function prototypes at the top of the file
    for sink in OPAQUE_SINKS:
        lines.append(f"void {sink}(void*);")
    lines.append("\n")

    # Step 2. Randomize how many functions are in this specific C file (e.g., 1 to 10)
    num_functions = random.randint(1, max_random_func)
    for i in range(num_functions):
        # Pass a unique ID for the function name
        lines.append(generate_function_body(func_id=i))

    return "\n".join(lines)

if __name__ == "__main__":
    args = parse_args()
    setup_logger(args.log_level)

    os.makedirs(args.output_dir, exist_ok=True)

    LOGGER.info("Generating samples for %d", args.num_samples)
    
    success_count = 0
    for i in range(args.num_samples):
        file_path = os.path.join(args.output_dir, f"sample_{i:05d}.c")

        try:
            c_code = generate_c_code(i, args.max_random_func)

            with open(file_path, "w") as f:
                f.write(c_code)
            
            success_count += 1
        except IOError as e:
            LOGGER.error("Failed to generate c code to %s : %s", file_path, e)
    
    LOGGER.info("Successfully generated %d C programs and saved to %s", success_count, args.output_dir)