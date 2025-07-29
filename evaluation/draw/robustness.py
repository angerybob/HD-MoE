import json
import numpy as np
import matplotlib.pyplot as plt

model = "ds"
dataset = "coding"
if model == "ds":
    mlp_first, num_layers = True, 25
elif model == "mixtral":
    mlp_first, num_layers = False, 32
elif model == "qwen":
    mlp_first, num_layers = False, 28

expert_data_path = f'expert_trace/{model}/predict/experts_{dataset}_{model}_pre.json'
# Path to the performance results (speedup per layer)
performance_data_path = "evaluation/results/result_dynamic_per_layer.json"

output_figure_path = "evaluation/figs/robustness.pdf"
output_figure_path1 = "evaluation/figs/robustness.png"

try:
    with open(expert_data_path, 'r', encoding='utf-8') as f:
        expert_trace = json.load(f)
        actual_activations = expert_trace["selected_experts"]
        predicted_activations = expert_trace["predict_experts"]
except FileNotFoundError:
    print(f"Error: Expert trace file not found at '{expert_data_path}'")
    exit()

# Calculate the average prediction accuracy for each layer
prediction_accuracy_per_layer = []
for layer_id in range(1,num_layers):
    layer_key = str(layer_id + mlp_first)
    
    if layer_key not in actual_activations or layer_key not in predicted_activations:
        print(f"Warning: Data for layer key '{layer_key}' not found in expert trace. Skipping.")
        prediction_accuracy_per_layer.append(np.nan) # Append NaN if data is missing
        continue

    layer_accuracies = []
    # Iterate through each sample (token) for the current layer
    for i in range(len(actual_activations[layer_key])):
        actual_set = set(actual_activations[layer_key][i])
        predicted_set = set(predicted_activations[layer_key][i])
        
        if len(actual_set) > 0:
            accuracy = len(actual_set.intersection(predicted_set)) / len(actual_set)
            layer_accuracies.append(accuracy)

    # Calculate the average accuracy for this layer
    if layer_accuracies:
        average_accuracy = np.mean(layer_accuracies)
        prediction_accuracy_per_layer.append(average_accuracy)
    else:
        prediction_accuracy_per_layer.append(np.nan)

try:
    with open(performance_data_path, 'r', encoding='utf-8') as f:
        performance_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Performance results file not found at '{performance_data_path}'")
    exit()

speedup_per_layer_map = {}
for item in performance_data:
    if 'config' in item and 'layer_id' in item['config']:
        layer_id = item['config']['layer_id']
        speedup = item.get('dynamic_deployment', {}).get('speedup', 0)
        speedup_per_layer_map[layer_id] = speedup

# Create a list of speedups ordered by layer_id
speedup_per_layer = [speedup_per_layer_map.get(i, np.nan) for i in range(1,num_layers)]


plt.rcParams.update({
    "font.size": 16,
    "axes.labelweight": "bold",
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.figsize": (10, 6),
    "lines.linewidth": 3,
    "lines.markersize": 8,
})

fig, ax1 = plt.subplots()

# --- Plot Speedup on the left Y-axis (ax1) ---
color_speedup = 'tab:blue'
ax1.set_xlabel('Layer Index')
ax1.set_ylabel('Speedup', color=color_speedup, fontweight='bold')
ax1.plot(range(1,num_layers), speedup_per_layer, color=color_speedup, marker='o', linestyle='-', label='Speedup')
ax1.tick_params(axis='y', labelcolor=color_speedup)
ax1.grid(True, linestyle='--', alpha=0.6)
plt.xticks([1, 10, 20])
# Create a second Y-axis that shares the same X-axis
ax2 = ax1.twinx()

color_accuracy = 'tab:orange'
ax2.set_ylabel('Prediction Accuracy', color=color_accuracy, fontweight='bold')
ax2.plot(range(1,num_layers), prediction_accuracy_per_layer, color=color_accuracy, marker='s', linestyle='--', label='Prediction Accuracy')
ax2.tick_params(axis='y', labelcolor=color_accuracy)

# Set Y-axis limits
ax1.set_ylim(0,3) # Speedup should ideally start from 1
ax2.set_ylim(0.6, 0.9)

fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.savefig(output_figure_path1, format='png', bbox_inches='tight')
plt.savefig(output_figure_path, format='pdf', bbox_inches='tight')
