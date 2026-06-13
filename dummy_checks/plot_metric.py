import os
import json
import matplotlib.pyplot as plt

def get_latest_checkpoint_state(base_dir: str) -> str:
    """Scans the base directory for the latest checkpoint folder and returns the path to its trainer_state.json."""
    if not os.path.exists(base_dir):
        return None

    checkpoint_folders = []
    
    # Iterate through all items in the base directory
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path) and item.startswith("checkpoint-"):
            try:
                # Extract the integer step number (e.g., '2487' from 'checkpoint-2487')
                step_num = int(item.split("-")[1])
                checkpoint_folders.append((step_num, full_path))
            except ValueError:
                continue

    # If checkpoint folders exist, sort them by step number descending and pick the highest
    if checkpoint_folders:
        checkpoint_folders.sort(key=lambda x: x[0], reverse=True)
        latest_checkpoint_dir = checkpoint_folders[0][1]
        state_path = os.path.join(latest_checkpoint_dir, "trainer_state.json")
        if os.path.exists(state_path):
            return state_path

    # Fallback: check the root directory if no checkpoint folders exist or lack the state file
    root_state_path = os.path.join(base_dir, "trainer_state.json")
    if os.path.exists(root_state_path):
        return root_state_path

    return None

# --- Configuration ---
OUTPUT_DIR = "./output_LLM_part_Qwen7B"

# --- Dynamic Fetching ---
state_path = get_latest_checkpoint_state(OUTPUT_DIR)

if not state_path:
    print(f"Error: Could not find a 'trainer_state.json' in {OUTPUT_DIR} or its checkpoint folders!")
    exit(1)

print(f"Found latest training state at: {state_path}")

# --- Data Extraction ---
try:
    with open(state_path, "r") as f:
        state = json.load(f)
except Exception as e:
    print(f"Failed to read JSON: {e}")
    exit(1)

log_history = state.get("log_history", [])

train_steps, train_loss = [], []
eval_steps, eval_loss = [], []

for entry in log_history:
    # Extract Training Loss
    if "loss" in entry and "step" in entry:
        train_steps.append(entry["step"])
        train_loss.append(entry["loss"])
    
    # Extract Evaluation Loss
    if "eval_loss" in entry and "step" in entry:
        eval_steps.append(entry["step"])
        eval_loss.append(entry["eval_loss"])

# --- Plotting ---
plt.figure(figsize=(10, 6))

if train_loss:
    plt.plot(train_steps, train_loss, label="Training Loss", color="blue", alpha=0.7)
else:
    print("Warning: No training loss data found in the state file.")

if eval_loss:
    plt.plot(eval_steps, eval_loss, label="Validation Loss", color="red", marker="o", linewidth=2)
else:
    print("Warning: No evaluation loss data found in the state file.")

plt.title("Training and Evaluation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss (Cross-Entropy)")
plt.legend()
plt.grid(True)

# --- SAVE TO DIRECTORY ---
# 1. Define the directory name
output_dir = "plots"

# 2. Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 3. Join the path and save
output_image = os.path.join(output_dir, "loss_curve.png")
plt.savefig(output_image, dpi=300, bbox_inches="tight")
print(f"Success! Saved loss curve to {output_image}")