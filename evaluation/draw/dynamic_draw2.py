import matplotlib.pyplot as plt
import numpy as np
import json

# 数据加载
with open("evaluation/results/result2_dynamic.json", "r") as file:
    data = json.load(file)

# 提取数据
samples = []
static_latencies = []
dynamic_latencies = []
speedups = []

# 只考虑 "ds" 模型和硬件配置 (comp_TFLOPS=5.0, BW_GBPS=50.0)
for entry in data:
    config = entry['config']
    if config["model"] == "ds" and config["comp_TFLOPS"] == 2.5 and config["BW_GBPS"] == 75.0:
        samples.append(config["sample"])
        static_latency = entry["static_deployment"]["latency_us"]*1e-3
        dynamic_latency = entry["dynamic_deployment"]["latency_us"]*1e-3
        static_latencies.append(static_latency)
        dynamic_latencies.append(dynamic_latency)
        speedups.append(entry["dynamic_deployment"]["speedup"])

# 绘图设置
plt.rcParams.update({
    "font.size": 25,
    "axes.labelweight": "normal",
    "axes.labelsize": 20,
    "legend.frameon": True,
    "lines.linewidth": 3
})
size=25
# 创建图表
fig, ax1 = plt.subplots(figsize=(10, 6))

# 设置柱状图（static_deployment 和 dynamic_deployment 延迟）
x = np.arange(len(samples))
width = 0.35  # 设置柱状图宽度
ax1.bar(x - width/2, static_latencies, width, label="Static Deployment", color="#4E79A7", edgecolor="black")
ax1.bar(x + width/2, dynamic_latencies, width, label="Dynamic Deployment", color="#F28E2B", edgecolor="black")

ax1.set_xlabel("Sample", fontsize=size)
ax1.set_ylabel("Latency (ms)", fontsize=size)
ax1.set_xticks(x)
ax1.set_xticklabels(samples, rotation=45, ha="right")
ax1.set_ylim([6.8, 12.5]) 

# 设置折线图（加速比）
ax2 = ax1.twinx()
ax2.plot(x, speedups, color="#59A14F", linestyle="--", marker="o", label="Speedup", markersize=8)
ax2.set_ylabel("Speedup", fontsize=size)
ax2.set_ylim([0.95, 1.44])  # 加速比范围
ax1.tick_params(axis='both', which='major', labelsize=size)  # 'both'表示x和y轴

# 设置次坐标轴（ax2）的刻度字体大小
ax2.tick_params(axis='y', labelsize=size)  # 仅设置y轴

# 图表标题
plt.title("Dynamic Placement Strategy", fontsize=size)

# 显示图形
plt.tight_layout()
plt.savefig("evaluation/figs/ablation/dynamic_deployment_latency_speedup2.pdf")
