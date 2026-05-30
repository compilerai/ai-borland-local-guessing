import json
import matplotlib.pyplot as plt

# Point this to your latest checkpoint folder or the root output dir if saved there
state_path = "./output_LLM_part_Qwen7B/checkpoint-2475/trainer_state.json" 

try:
    with open(state_path, "r") as f:
        state = json.load(f)
except FileNotFoundError:
    print(f"Could not find {state_path}. Check inside your checkpoint folders!")
    exit()

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

# Plotting
plt.figure(figsize=(10, 6))
if train_loss:
    plt.plot(train_steps, train_loss, label="Training Loss", color="blue", alpha=0.7)
if eval_loss:
    plt.plot(eval_steps, eval_loss, label="Validation Loss", color="red", marker="o")

plt.title("Training and Evaluation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss (Cross-Entropy)")
plt.legend()
plt.grid(True)

# Save the plot to an image file so you can view it easily
plt.savefig("loss_curve.png", dpi=300)
print("Saved loss curve to loss_curve.png!")