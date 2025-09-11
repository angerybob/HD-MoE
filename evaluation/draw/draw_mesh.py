import matplotlib.pyplot as plt
import numpy as np
import json
import os
from matplotlib.ticker import FuncFormatter
import pdb

# 确保输出目录存在
os.makedirs("evaluation/figs/TBT", exist_ok=True)

# 数据加载
with open("evaluation/results/result3_e2e.json", "r") as file:
    data = json.load(file)

# 定义硬件配置和批次大小
batch_sizes = [16, 32, 64, 128]
models = ["mixtral", "deepseek", "qwen"]

methods = ["TP", "EP", "compute_balancing", "node_link_balancing"]
mesh_shapes = [(4,4),(4, 8),(8,8)]  # 只考虑部分网格形状示例
models = ["mixtral", "deepseek", "qwen"]  # 模型类型
hardware_configs = [
    {"comp_TFLOPS": 5.0, "BW_GBPS": 50.0}
]
# 提取吞吐率（batch / latency）和加速比
throughput_data = {}
speedup_data = {}

# 提取每个配置的数据
for entry in data:
    config = entry['config']
    model = config['model']
    if model == "ds":
        model = "deepseek"
    batch = config['batch']
    mesh_shape = tuple(config['mesh_shape'])
    hardware = {"comp_TFLOPS": config["comp_TFLOPS"], "BW_GBPS": config["BW_GBPS"]}
    
    # 只处理(4,8)网格形状和指定批次大小
    if hardware != {"comp_TFLOPS": 5.0, "BW_GBPS": 50.0} or batch not in batch_sizes:
        continue
    
    # 初始化数据结构
    if model not in throughput_data:
        throughput_data[model] = {}
    hardware_key = (mesh_shape[0],mesh_shape[1])
    if hardware_key not in throughput_data[model]:
        throughput_data[model][hardware_key] = {}
    if batch not in throughput_data[model][hardware_key]:
        throughput_data[model][hardware_key][batch] = {}
    
    # 计算吞吐率 (batch/s)
    for method in methods:
        latency_us = entry[method]["latency_us"]
        throughput = latency_us/entry["TP"]["latency_us"]
        throughput_data[model][hardware_key][batch][method] = throughput
    
    # 计算加速比 (相对于node_link_balancing)
    node_link_latency = entry["node_link_balancing"]["latency_us"]
    for method in methods:
        if method == "node_link_balancing":
            continue
        latency_us = entry[method]["latency_us"]
        speedup = latency_us / node_link_latency
        if model not in speedup_data:
            speedup_data[model] = {}
        if hardware_key not in speedup_data[model]:
            speedup_data[model][hardware_key] = {}
        if batch not in speedup_data[model][hardware_key]:
            speedup_data[model][hardware_key][batch] = {}
        speedup_data[model][hardware_key][batch][method] = speedup

        

# 创建单一图形
plt.rcParams.update({
    "font.size": 14,
    "axes.labelweight": "bold",
    "axes.labelsize": 16,
    "legend.frameon": True,
    "lines.linewidth": 2,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})
fig, ax = plt.subplots(figsize=(36, 6))
ax2 = ax.twinx()
# 设置颜色
colors = {
    "TP": "#4E79A7",
    "EP": "#F28E2B",
    "compute_balancing": "#E15759",
    "node_link_balancing": "#59A14F"
}

# 批次大小的不同纹理
batch_patterns = {
    16: "",
    32: "",
    64: "",
    128: ""
}

# 计算每个配置的位置偏移
position = 0.2
bar_width = 0.1
group_spacing = 0.1
section_spacing = 0

# 存储所有数据点用于设置y轴范围
all_throughputs = []
all_speedups = []

# 存储模型和硬件配置的位置信息
model_positions = {}
hardware_positions = {}

# 先计算全局最大吞吐量，用于统一标签位置
for model in models:
    for hardware in mesh_shapes:
        hardware_key = (hardware[0],hardware[1])
        if (model in throughput_data and 
            hardware_key in throughput_data[model]):
            for batch in batch_sizes:
                for method in methods:
                    throughput = throughput_data[model][hardware_key][batch][method]
                    all_throughputs.append(throughput)

global_max_throughput = max(all_throughputs) if all_throughputs else 1
# 固定标签位置 - 修复未定义变量问题
# model_label_y = -global_max_throughput * 0.15
# hardware_label_y = -global_max_throughput * 0.08
# batch_label_y = -global_max_throughput * 0.01
lower_y=0.5
model_label_y = lower_y-0.28
hardware_label_y = lower_y-0.17
batch_label_y = lower_y-0.05
# 绘制所有数据
for model_idx, model in enumerate(models):
    model_start_pos = position
    model_positions[model] = (model_start_pos, None)  # 存储起始位置
    
    for hw_idx, hardware in enumerate(mesh_shapes):
        hardware_key = (hardware[0],hardware[1])
        hardware_start_pos = position
        hardware_positions[hardware_key] = (hardware_start_pos, None)  # 存储起始位置
        
        # 检查数据是否存在
        if (model not in throughput_data or 
            hardware_key not in throughput_data[model] or 
            batch_sizes[0] not in throughput_data[model][hardware_key]):
            continue
            
        # 存储每个方法的加速比点和位置
        method_points = {
            "TP": {"x": [], "y": []},
            "EP": {"x": [], "y": []},
            "compute_balancing": {"x": [], "y": []}
        }

        
        # 绘制每个批次大小的数据
        for batch_idx, batch in enumerate(batch_sizes):

            # 绘制每个方法的数据
            for method_idx, method in enumerate(methods):
                # 获取当前配置的数据
                batch_throughput = throughput_data[model][hardware_key][batch][method]
                
                # 计算柱子的位置
                bar_pos = position
                
                # 绘制柱子 - 使用颜色和纹理区分批次大小
                ax.bar(bar_pos, batch_throughput, bar_width, 
                       color=colors[method], edgecolor="black", alpha=0.8,
                       hatch=batch_patterns[batch])
                
                # 收集数据点用于设置y轴范围
                all_throughputs.append(batch_throughput)
                
                # 如果是前三种方法，收集加速比点
                if method != "node_link_balancing":
                    # 获取加速比
                    if (model in speedup_data and 
                        hardware_key in speedup_data[model] and 
                        batch in speedup_data[model][hardware_key] and 
                        method in speedup_data[model][hardware_key][batch]):
                        speedup = speedup_data[model][hardware_key][batch][method]
                        
                        # 存储加速比点和位置
                        method_points[method]["x"].append(bar_pos + bar_width/2)
                        method_points[method]["y"].append(speedup)
                        all_speedups.append(speedup)
                
                # 移动到下一个策略位置
                position += bar_width
            
            # 在批次大小之间添加小空隙
            position += bar_width * 0.2
        
        # 绘制连接线 - 同一配置组内同一方法的不同批次用线连接
        for method in ["TP", "EP", "compute_balancing"]:
            if method_points[method]["x"] and method_points[method]["y"]:
                # 按批次大小排序点
                sorted_indices = np.argsort(method_points[method]["x"])
                sorted_x = np.array(method_points[method]["x"])[sorted_indices]
                sorted_y = np.array(method_points[method]["y"])[sorted_indices]
                
                # 绘制连接线
                ax2.plot(sorted_x, sorted_y, 
                         color=colors[method], 
                         linestyle="--", 
                         marker="o" if method == "TP" else ("s" if method == "EP" else "^"),
                         markersize=15,
                         label=f"Speedup {method}")
        
        # 记录硬件配置的结束位置
        hardware_end_pos = position - bar_width * 0.2
        hardware_positions[hardware_key] = (hardware_start_pos, hardware_end_pos)
        hw_start, hw_end = hardware_positions[hardware_key]
        # 计算批次宽度
        batch_width = (hw_end - hw_start) / len(batch_sizes)
        for batch_idx, batch in enumerate(batch_sizes):
            batch_center = hw_start + batch_width * (batch_idx + 0.5)
            ax.text(batch_center, batch_label_y, str(batch), 
                    ha='center', va='top', fontsize=18, color='black')
        # 添加硬件配置标签 - 使用固定高度
        hw_label = f"({hardware[0]}, {hardware[1]})"
        hw_center = (hardware_start_pos + hardware_end_pos) / 2
        ax.text(hw_center, hardware_label_y, hw_label, 
                ha='center', va='top', fontsize=20, weight='bold')
        
        # 在硬件配置之间添加中等空隙
        position += group_spacing
    
    # 记录模型的结束位置
    model_end_pos = position - group_spacing
    model_positions[model] = (model_start_pos, model_end_pos)
    
    # 添加模型标签 - 使用固定高度
    model_center = (model_start_pos + model_end_pos) / 2
    ax.text(model_center, model_label_y, model, 
            ha='center', va='top', fontsize=28, weight='bold')
    
    # 在模型之间添加较大空隙
    position += section_spacing

# 添加批次大小图例
from matplotlib.patches import Patch
batch_legend_elements = [
    Patch(facecolor='white', edgecolor='black', hatch="", label='Batch 16'),
    Patch(facecolor='white', edgecolor='black', hatch="//", label='Batch 32'),
    Patch(facecolor='white', edgecolor='black', hatch="\\\\", label='Batch 64'),
    Patch(facecolor='white', edgecolor='black', hatch="xx", label='Batch 128')
]
#batch_legend = ax.legend(handles=batch_legend_elements, loc='upper left', fontsize=24)
#ax.add_artist(batch_legend)  # 保留此图例

# 添加方法图例
method_legend_elements = [
    Patch(facecolor=colors["TP"], edgecolor='black', label='TP'),
    Patch(facecolor=colors["EP"], edgecolor='black', label='EP'),
    Patch(facecolor=colors["compute_balancing"], edgecolor='black', label='Compute Balancing'),
    Patch(facecolor=colors["node_link_balancing"], edgecolor='black', label='Node-Link Balancing'),
    plt.Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=8, label='Speedup')
]
#ax.legend(handles=method_legend_elements, loc='upper right', fontsize=12)

# 设置x轴范围
ax.set_xlim([0, position - section_spacing])

# 设置y轴范围
if all_throughputs:
    min_throughput = min(all_throughputs)
    max_throughput = max(all_throughputs)
    ax.set_ylim([lower_y, max_throughput * 1.02])
    ax.set_ylabel("Normalized TBT", fontsize=26)

# 添加右侧y轴用于加速比

if all_speedups:
    min_speedup = min(all_speedups)
    max_speedup = max(all_speedups)
    ax2.set_ylim([min_speedup * 0.2, max_speedup * 1.05])
ax2.set_ylabel("Speedup", fontsize=26)

# 添加标题
#plt.title("Throughput and Speedup Comparison", fontsize=28, pad=20)

# 隐藏x轴刻度
ax.set_xticks([])
ax.tick_params(axis='y', labelsize=20) 
ax2.tick_params(axis='y', labelsize=20) 
# 添加网格线
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存图形
save_dir = "evaluation/figs/TBT/throughput_comparison_mesh.pdf"
plt.savefig(save_dir, bbox_inches='tight')
print(f"Figure saved at {save_dir}")
save_dir = "evaluation/figs/TBT/throughput_comparison_mesh.png"
plt.savefig(save_dir, bbox_inches='tight')
print(f"Figure saved at {save_dir}")
