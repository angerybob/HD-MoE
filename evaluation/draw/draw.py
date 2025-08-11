import matplotlib.pyplot as plt
import numpy as np
import json
import pdb

# 数据加载
with open("evaluation/results/result3_e2e.json", "r") as file:
    data = json.load(file)

# 定义硬件配置和批次大小
batch_sizes = [16, 32, 64, 128]
mesh_shapes = [(4, 8)]  # 只考虑部分网格形状示例
models = ["mixtral", "deepseek", "qwen"]  # 模型类型
hardware_configs = [
    {"comp_TFLOPS": 2.5, "BW_GBPS": 75.0},
    {"comp_TFLOPS": 5.0, "BW_GBPS": 50.0},
    {"comp_TFLOPS": 10.0, "BW_GBPS": 25.0}
]

# 提取吞吐率（batch / latency）和加速比（compute_balancing的speedup）
throughput_data = {}
speedup_data = {}

# 提取每个配置的数据
for entry in data:
    config = entry['config']
    model = config['model']
    if model=="ds":
        model="deepseek"
    batch = config['batch']
    mesh_shape = tuple(config['mesh_shape'])
    hardware = {"comp_TFLOPS": config["comp_TFLOPS"], "BW_GBPS": config["BW_GBPS"]}
    
    if model not in throughput_data:
        throughput_data[model] = {}
        speedup_data[model] = {}
    hardware_tuple = tuple(hardware.values())
    if hardware_tuple not in throughput_data[model]:
        throughput_data[model][hardware_tuple] = {}
        speedup_data[model][hardware_tuple] = {}

    # 计算吞吐率
    tp_latency = entry['TP']['latency_us']
    ep_latency = entry['EP']['latency_us']
    compute_latency = entry['compute_balancing']['latency_us']
    node_link_latency = entry['node_link_balancing']['latency_us']
    
    tp_throughput = tp_latency*1e-3
    ep_throughput = ep_latency*1e-3
    compute_throughput = compute_latency*1e-3
    node_link_throughput = node_link_latency*1e-3

    # 存储吞吐率数据
    if mesh_shape==(4,8):
        throughput_data[model][tuple(hardware.values())][batch] = {
            'TP': tp_throughput,
            'EP': ep_throughput,
            'compute_balancing': compute_throughput,
            'node_link_balancing': node_link_throughput
        }

        # 计算加速比
        speedup_tp = node_link_latency / tp_latency
        speedup_ep = node_link_latency / ep_latency
        speedup_compute = node_link_latency / compute_latency

        speedup_data[model][tuple(hardware.values())][batch] = {
            'speedup_TP': 1/speedup_tp,
            'speedup_EP': 1/speedup_ep,
            'speedup_compute': 1/speedup_compute
        }

# 绘制图形
plt.rcParams.update({"font.size": 22, "axes.labelweight": "normal", "axes.labelsize": 30, "legend.frameon": True, "lines.linewidth": 3})
fig, axs = plt.subplots(len(hardware_configs),len(models), figsize=(18, 11))

# 设置颜色
colors = ["#4E79A7", "#F28E2B", "#E15759", "#59A14F"]

# 对每个硬件配置和模型绘制图表
for row, hardware in enumerate(hardware_configs):
    for col, model in enumerate(models):
        ax = axs[row,col]
        #pdb.set_trace()
        batch_throughput = throughput_data[model][tuple(hardware.values())]
        batch_speedup = speedup_data[model][tuple(hardware.values())]

        # 绘制吞吐率的柱状图
        x = np.arange(len(batch_sizes))
        width = 0.2
        ax.bar(x - width, [batch_throughput[b]['TP'] for b in batch_sizes], width, color=colors[0], edgecolor="black")
        ax.bar(x, [batch_throughput[b]['EP'] for b in batch_sizes], width, color=colors[1], edgecolor="black")
        ax.bar(x + width, [batch_throughput[b]['compute_balancing'] for b in batch_sizes], width, color=colors[2], edgecolor="black")
        ax.bar(x + 2 * width, [batch_throughput[b]['node_link_balancing'] for b in batch_sizes], width, color=colors[3], edgecolor="black")

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("TBT (ms)")
        ax.set_title(f"{model} - {hardware['comp_TFLOPS']} TFLOPS, {hardware['BW_GBPS']} GB/s",fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        #ax.legend(loc="upper left", fontsize=20)
        combined_list = (
            [batch_throughput[b]['EP'] for b in batch_sizes] + 
            [batch_throughput[b]['TP'] for b in batch_sizes]
        )
        max_value = max(combined_list)
        ax.set_ylim([0.5*min([batch_throughput[b]['node_link_balancing'] for b in batch_sizes]), 1.1*max_value]) 

        # 绘制加速比的折线图
        ax2 = ax.twinx()
        speedup_tp = [batch_speedup[b]['speedup_TP'] for b in batch_sizes]
        speedup_ep = [batch_speedup[b]['speedup_EP'] for b in batch_sizes]
        speedup_compute = [batch_speedup[b]['speedup_compute'] for b in batch_sizes]

        ax2.plot(x + 2 * width, speedup_tp, color=colors[0], linestyle="--", marker="o", label="Speedup TP", markersize=8)
        ax2.plot(x + 2 * width, speedup_ep, color=colors[1], linestyle="--", marker="s", label="Speedup EP", markersize=8)
        ax2.plot(x + 2 * width, speedup_compute, color=colors[2], linestyle="--", marker="^", label="Speedup Compute", markersize=8)
        #pdb.set_trace()
        ax2.set_ylabel("Speedup")
        combined_list = (
            speedup_tp + 
            speedup_ep
        )
        max_value = max(combined_list)
        ax2.set_ylim([0.8, 1.1*max_value]) 

# 显示图形
plt.tight_layout()

save_dir = "evaluation/figs/TBT/throughput_comparison.pdf"
plt.savefig(save_dir)
print(f"Figure saved at {save_dir}")
save_dir = "evaluation/figs/TBT/throughput_comparison.png"
plt.savefig(save_dir)
print(f"Figure saved at {save_dir}")