import matplotlib.pyplot as plt
import numpy as np
import json

# 数据加载
with open("evaluation/results/result_ablation.json", "r") as file:
    data = json.load(file)

# 定义硬件配置
hardware_configs = [
    {"comp_TFLOPS": 2.5, "BW_GBPS": 75.0},
    {"comp_TFLOPS": 5.0, "BW_GBPS": 50.0},
    {"comp_TFLOPS": 10.0, "BW_GBPS": 25.0}
]
mesh_shapes = [(4,4),(4, 8),(8,8)] 
# 模型列表
models = ["deepseek"]

# 提取数据：根据硬件配置和模型获取延迟和加速比
latency_data = {model: {} for model in models}
speedup_data = {model: {} for model in models}

latency_data1 = {model: {} for model in models}
speedup_data1 = {model: {} for model in models}

for entry in data:
    config = entry['config']
    model = config['model']
    if model=="ds":
        model="deepseek"
    hardware = {"comp_TFLOPS": config["comp_TFLOPS"], "BW_GBPS": config["BW_GBPS"]}
    mesh_shape = tuple(config['mesh_shape'])
    # 只处理指定的模型和硬件配置
    if model in models:
        hardware_tuple = (hardware["comp_TFLOPS"], hardware["BW_GBPS"])
        
        # 获取延迟数据

        tp_latency = entry['TP']['communication_us']
        comp_latency = entry['compute_balancing']['communication_us']
        node_latency = entry['node_balancing']['communication_us']
        link_balancing_latency = entry['link_balancing']['communication_us']
        if mesh_shape==(4,8):
            latency_data[model].setdefault(hardware_tuple, {})
            latency_data[model][hardware_tuple] = {
                'TP': tp_latency*1e-3,
                'comp_balance': comp_latency*1e-3,
                'node_balance': node_latency*1e-3,
                'link_balance': link_balancing_latency*1e-3
            }
        if hardware_tuple==(5.0,50.0):  
            latency_data1[model].setdefault(mesh_shape, {})
            latency_data1[model][mesh_shape] = {
                'TP': tp_latency*1e-3,
                'comp_balance': comp_latency*1e-3,
                'node_balance': node_latency*1e-3,
                'link_balance': link_balancing_latency*1e-3
            }
        # 获取加速比数据
        node_balancing = entry['node_balancing']
        if mesh_shape==(4,8):
            speedup_data[model].setdefault(hardware_tuple, {})
            
            speedup_data[model][hardware_tuple] = {
                'speedup_TP': tp_latency/link_balancing_latency,
                'speedup_comp': comp_latency/link_balancing_latency,
                'speedup_node': node_latency/link_balancing_latency
            }
        if hardware_tuple==(5.0,50.0):  
            speedup_data1[model].setdefault(mesh_shape, {})
            
            speedup_data1[model][mesh_shape] = {
                'speedup_TP': tp_latency/link_balancing_latency,
                'speedup_comp': comp_latency/link_balancing_latency,
                'speedup_node': node_latency/link_balancing_latency
            }

# 绘图设置
plt.rcParams.update({
    "font.size": 25,
    "axes.labelweight": "normal",
    "axes.labelsize": 40,
    "legend.frameon": True,
    "lines.linewidth": 3
})
size=25
# 创建图表：两个模型放在一行
fig, axs = plt.subplots(1, 2, figsize=(18, 4))

# 设置颜色
colors = ["#4E79A7", "#F28E2B", "#E15759", "#59A14F"]


ax = axs[0]

# 获取每个硬件配置的延迟和加速比
model="deepseek"
hardware_tuple = (hardware["comp_TFLOPS"], hardware["BW_GBPS"])
latencies = latency_data[model][hardware_tuple]
speedup = speedup_data[model]

# x轴对应的硬件配置名称


x = np.arange(3)
width = 0.2
# 绘制延迟的柱状图

ax.bar(x - width, [latency_data[model][(hardware["comp_TFLOPS"], hardware["BW_GBPS"])]['TP'] for hardware in hardware_configs], width, label='TP', color=colors[0], edgecolor="black")
ax.bar(x, [latency_data[model][(hardware["comp_TFLOPS"], hardware["BW_GBPS"])]['comp_balance'] for hardware in hardware_configs], width, label='compute_balance', color=colors[1], edgecolor="black")
ax.bar(x + width, [latency_data[model][(hardware["comp_TFLOPS"], hardware["BW_GBPS"])]['node_balance'] for hardware in hardware_configs], width, label='node_balance', color=colors[2], edgecolor="black")
ax.bar(x + 2*width, [latency_data[model][(hardware["comp_TFLOPS"], hardware["BW_GBPS"])]['link_balance'] for hardware in hardware_configs], width, label='node_link_balance', color=colors[3], edgecolor="black")

# 绘制加速比的折线图
ax2 = ax.twinx()
#ax2.plot(x[1:4], speedup_values, color=colors[row], linestyle="--", marker="o", label="Speedup", markersize=8)
ax2.plot(x + 2 * width, [speedup[(hardware["comp_TFLOPS"], hardware["BW_GBPS"])]['speedup_TP'] for hardware in hardware_configs], color=colors[0], linestyle="--", marker="o", label="Speedup TP", markersize=8)
ax2.plot(x + 2*width, [speedup[(hardware["comp_TFLOPS"], hardware["BW_GBPS"])]['speedup_comp'] for hardware in hardware_configs], color=colors[1], linestyle="--", marker="s", label="Speedup Compute", markersize=8)
ax2.plot(x + 2 * width, [speedup[(hardware["comp_TFLOPS"], hardware["BW_GBPS"])]['speedup_node'] for hardware in hardware_configs], color=colors[2], linestyle="--", marker="^", label="Speedup Node", markersize=8)
ax.set_xlabel("Hardware Config",fontsize=size)
ax.set_ylabel("Latency (ms)",fontsize=size)
ax.set_title("Speedup for Different Configs",fontsize=size)
ax.set_xticks([0.1, 1.1, 2.1])
ax.set_xticklabels([f"({hw['comp_TFLOPS']},{hw['BW_GBPS']:.0f})" for hw in hardware_configs])
#ax.legend(loc="upper left", fontsize=20)

# 设置加速比的y轴
ax2.set_ylabel("Speedup",fontsize=size)
ax2.set_ylim([0.8, 1.7])  # 调整加速比的刻度范围

ax = axs[1]

# 获取每个硬件配置的延迟和加速比

speedup = speedup_data1[model]

# x轴对应的硬件配置名称

#speedup_values = [speedup['speedup_EP'], speedup['speedup_TP'], speedup['speedup_comp']]
x = np.arange(3)
width = 0.2
# 绘制延迟的柱状图

ax.bar(x - width, [latency_data1[model][mesh_shape]['TP'] for mesh_shape in mesh_shapes], width, label='TP', color=colors[0], edgecolor="black")
ax.bar(x, [latency_data1[model][mesh_shape]['comp_balance'] for mesh_shape in mesh_shapes], width, label='comp_balance', color=colors[1], edgecolor="black")
ax.bar(x + width, [latency_data1[model][mesh_shape]['node_balance'] for mesh_shape in mesh_shapes], width, label='node_balance', color=colors[2], edgecolor="black")
ax.bar(x + 2*width, [latency_data1[model][mesh_shape]['link_balance'] for mesh_shape in mesh_shapes], width, label='node_link_balance', color=colors[3], edgecolor="black")

# 绘制加速比的折线图
ax2 = ax.twinx()
#ax2.plot(x[1:4], speedup_values, color=colors[row], linestyle="--", marker="o", label="Speedup", markersize=8)
ax2.plot(x + 2 * width, [speedup[mesh_shape]['speedup_TP'] for mesh_shape in mesh_shapes], color=colors[0], linestyle="--", marker="o", label="Speedup TP", markersize=8)
ax2.plot(x + 2*width, [speedup[mesh_shape]['speedup_comp'] for mesh_shape in mesh_shapes], color=colors[1], linestyle="--", marker="s", label="Speedup Compute", markersize=8)
ax2.plot(x + 2 * width, [speedup[mesh_shape]['speedup_node'] for mesh_shape in mesh_shapes], color=colors[2], linestyle="--", marker="^", label="Speedup Node", markersize=8)
ax.set_xlabel("Mesh Size",fontsize=size)
ax.set_ylabel("Latency (ms)",fontsize=size)
ax.set_title("Speedup for Different Mesh Shapes",fontsize=size)
ax.set_xticks([0.1, 1.1, 2.1])
ax.set_xticklabels([f"{mesh_shape}" for mesh_shape in mesh_shapes])
#ax.legend(loc="upper left", fontsize=20)
ax.set_ylim([0.1, 1.7]) 
# 设置加速比的y轴
ax2.set_ylabel("Speedup",fontsize=size)
ax2.set_ylim([0.8, 1.7])  # 调整加速比的刻度范围



# 显示图形
plt.tight_layout()
plt.savefig("evaluation/figs/ablation/ablation_study_link_balancing_speedup.pdf")
