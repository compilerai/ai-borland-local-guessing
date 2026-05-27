# Borland Assembly Stack Offset Prediction
### AI Dataset Generation & ML Pipeline for Stack Layout Recovery

## Problem Statement

Consider the following C code:

```c
int y;

void foo()
{
    int l;
    struct ptr_pair pp;

    pp.p1 = &l;
    pp.p2 = &y;

    w_deref_deref_arg(&pp);
    w_deref_arg(&pp);

    fv_pp(&pp);
}
```

Generated Borland assembly:

```asm
1: push ebp
2: mov ebp,esp
3: add esp,-12

4: lea eax,[ebp-4]
5: mov [ebp-12],eax

6: mov [ebp-8],offset _y
...
```

The compiler allocates stack memory:

```asm
add esp,-12
```

giving:

| Variable | Location | Offset in allocation |
|----------|----------|----------------------|
| `l` | `[ebp-4]` | 8 |
| `pp` | `[ebp-12]` | 0 |

The objective is:

Given a **C + Assembly pair**, automatically identify for every local variable:

1. Stack allocation instruction index
2. Offset inside allocated stack region

Formally:

```text
(C source + Assembly)
          ↓

Recover:

Variable → Allocation Instruction + Offset
```

---

## Overview

This repository provides an end-to-end pipeline for generating, compiling, extracting and preparing large-scale **C → Assembly mappings**.

Goal:

Train ML models (**LLMs + GNNs**) to recover stack layouts:

```text
Raw Borland Assembly
          ↓
ML Model
          ↓
Recovered Layout

[ebp-4]  → local_1
[ebp-8]  → buffer
esi      → cached_ptr
```

Compiler optimizations such as:

- `lea` register caching
- stack chunking
- hidden pointer propagation
- register spills

make static recovery difficult on stripped binaries.

We exploit debug artifacts to generate mathematically exact Ground Truth labels and later sanitize them.

---

# Pipeline Status
## ✅ Data Generation Completed

Current pipeline supports:

- Nested structs
- Multidimensional arrays
- Arithmetic obfuscation
- Control-flow noise
- Register spills
- Complex stack layouts

---

## 1. Generate C Programs

Creates obfuscated programs forcing difficult stack allocations.

```bash
python -m generateData.generate_c_programs \
    --num_samples 10000 \
    --max_random_func 10 \
    --output_dir ./data
```

Output:

```text
data/source_codes/
```

---

## 2. Assembly Mapping

Maps Borland generated assembly back into dataset structure.

```bash
python -m generateData.map_assembly \
    --json_path ./data/dataset_labels.json \
    --asm_dir ./data/assembly_codes
```

---

## 3. Ground Truth Extraction

Extract exact stack mappings.

Techniques:

- Liveness Cache Tracker
- `lea` propagation tracking
- Stack accumulator

Example:

```text
variable
    ↓

[ebp-X]
```

Run:

```bash
python -m generateData.parse_labels \
    --json_path ./data/dataset_labels.json
```

---

## 4. Sanitization

Remove:

- debug symbols
- embedded C code
- compiler hints
- liveness comments

Prevents target leakage.

```bash
python -m generateData.stripping_debug_asm_lines \
    --json_path ./data/dataset_labels.json
```

Output:

```text
data/dataset_labels_sanitized.json
```

---

## 5. Dataset Split

File-level splitting:

```bash
python -m generateData.train_val_split \
    --json_path ./data/dataset_labels_sanitized.json
```

Outputs:

```text
data/dataset_train.json
data/dataset_val.json
data/dataset_test.json
```

---

# Machine Learning Roadmap

## Phase 1 — LLM Fine Tuning

Baseline sequence model.

Models:

- Qwen2.5-Coder-7B
- Mistral-Nemo

Method:

```text
LoRA fine tuning
```

Input:

```text
Function level stripped assembly
```

Output:

```json
{
    "variable":"l",
    "stack":"ebp-4"
}
```

Goal:

> Establish >80% baseline accuracy

---

## Phase 2 — Graph Neural Networks

Convert assembly into CFGs.

```text
Assembly
    ↓

Control Flow Graph

Nodes = Basic Blocks
Edges = Jumps
```

Architectures:

- GAT
- GraphSAGE

Framework:

```text
PyTorch Geometric
```

Goal:

- Lightweight inference
- High accuracy
- CFG understanding
- Avoid LLM context limits

---

# Requirements

Core:

```bash
pip install tqdm scikit-learn
```

LLM:

```bash
pip install transformers peft trl
```

GNN:

```bash
pip install torch torch-geometric
```

Python:

```text
Python >= 3.9
```

---

# Final Objective

```text
Stripped Borland ASM
          ↓

AI Model

          ↓

Stack Recovery

[ebp-4]
[ebp-8]
esi
```

Recover compiler allocation logic **without debug symbols**.