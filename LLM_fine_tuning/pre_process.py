import re
import json
import logging
from datasets import Dataset

LOGGER = logging.getLogger(__name__)

# Assembly / C extraction helpers
def extract_c_function(full_c: str, func_name: str) -> str:
    """
    Extracts a C function body by scanning for the opening brace and
    matching it to the closing brace. Works for any return type / signature.
    Falls back to a simple void-based search if the brace scan fails.
    """
    # Find the function signature line (handles any return type)
    pattern = re.compile(
        rf'\b{re.escape(func_name)}\s*\([^)]*\)\s*\{{',
        re.DOTALL
    )
    m = pattern.search(full_c)
    if not m:
        return ""

    start = m.start()
    brace_pos = m.end() - 1  # position of the opening '{'

    depth = 0
    for i in range(brace_pos, len(full_c)):
        if full_c[i] == '{':
            depth += 1
        elif full_c[i] == '}':
            depth -= 1
            if depth == 0:
                return full_c[start:i + 1].strip()

    return ""  # unmatched braces


def extract_asm_function(full_asm: str, func_name: str) -> str:
    """Extracts a Borland assembly block using proc/endp boundaries."""
    pattern = re.compile(
        rf'_{re.escape(func_name)}\s+proc\s+near(.*?)_{re.escape(func_name)}\s+endp',
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(full_asm)
    if not m:
        LOGGER.debug(f"No ASM block found for function: {func_name}")
        return ""
    return m.group(1).strip()

# Prompt construction — NO ground-truth leakage
def make_prompt(c_source: str, asm_source: str, variable_names_and_sizes: list[dict]) -> str:
    """
    Builds the user-turn prompt.

    `variable_names_and_sizes` is a list of {"name": str, "size_bytes": int}
    derived from the label — we pass ONLY name and size, never the answer
    (assembly_reference / allocation_space_offset).
    """
    registry_str = json.dumps(variable_names_and_sizes, indent=2)

    prompt = f"""You are a binary analysis engine specialised in Borland C 32-bit (x86) compilation.

    ### Task
    For each variable in the Registry, identify its stack location in the assembly.

    ### Borland Stack Conventions
    - Stack frame set up with `push ebp / mov ebp,esp / add esp,-N` (or `push ecx` for small frames).
    - Locals are at `[ebp-N]` (negative offsets from EBP).
    - Parameters are at `[ebp+N]` (positive offsets).
    - When `add esp,-4092` appears, a second `add esp,-M` usually follows for large frames.
    - A variable may be cached in a register via `lea reg,[ebp-N]`; trace the register to find N.
    - Register-allocated variables (never address-taken) have `assembly_reference: "register <reg>"` and `allocation_space_offset: null`.

    ### Variable Registry (name + declared size in bytes — do NOT invent other variables)
    {registry_str}

    ### C Source
    ```c
    {c_source}
    ```

    ### Assembly
    ```asm
    {asm_source}
    ```

    ### Output Format
    Respond with ONLY a JSON object, no prose, no markdown fences:
    {{
    "stack_allocation_instruction": "<the add/sub/push instruction(s) that allocate the frame, pipe-separated if multiple>",
    "stack_allocation_size_bytes": <total bytes allocated>,
    "variable_mappings": [
        {{
        "variable_name": "<name from registry>",
        "assembly_reference": "ebp-<N>",
        "allocation_space_offset": <-N>
        }}
    ]
    }}
    """
    return prompt.strip()

# Label → training target
def build_target(func_label: dict) -> str:
    """
    Converts a function label dict into the JSON string the model should
    produce. Keeps only the fields the model is asked to predict.
    """
    target = {
        "stack_allocation_instruction": func_label.get("stack_allocation_instruction"),
        "stack_allocation_size_bytes": func_label.get("stack_allocation_size_bytes"),
        "variable_mappings": [
            {
                "variable_name": v["variable_name"],
                "assembly_reference": v["assembly_reference"],
                "allocation_space_offset": v["allocation_space_offset"],
            }
            for v in func_label.get("variable_mapping", [])
        ],
    }
    return json.dumps(target, separators=(",", ":"))

# Dataset loader
def load_and_format_dataset(filepath: str) -> Dataset:
    """
    Loads the JSON dataset and converts each function into a ChatML
    messages record suitable for SFTTrainer.

    Each record: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    """
    LOGGER.info(f"Loading dataset from {filepath} ...")

    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []
    skipped = 0

    for file_record in raw_data:
        full_c   = file_record.get("source_code_c", "")
        full_asm = file_record.get("assembly_code", "")
        functions = file_record.get("label", {}).get("functions", [])

        for func in functions:
            func_name  = func["function_name"]
            var_labels = func.get("variable_mapping", [])

            c_func  = extract_c_function(full_c, func_name)
            asm_func = extract_asm_function(full_asm, func_name)

            if not c_func or not asm_func:
                LOGGER.warning(f"Skipping {func_name}: extraction failed.")
                skipped += 1
                continue

            # Build registry: name + size only (no answer leakage)
            registry = [
                {"name": v["variable_name"], "size_bytes": v["size_bytes"]}
                for v in var_labels
            ]

            prompt = make_prompt(c_func, asm_func, registry)
            target = build_target(func)

            records.append({
                "messages": [
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": target},
                ]
            })

    LOGGER.info(
        f"Formatted {len(records)} examples ({skipped} skipped) from {filepath}."
    )
    return Dataset.from_list(records)