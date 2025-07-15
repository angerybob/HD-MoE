import matplotlib.pyplot as plt
import numpy as np
import json

# 数据加载
with open("/data/home/haochenhuang/deployment/result_ablation.json", "r") as file:
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
        tp_latency = entry['TP']['latency_us']
        ep_latency = entry['EP']['latency_us']
        node_balancing_latency = entry['compute_balancing']['latency_us']
        link_balancing_latency = entry['node_balancing']['latency_us']
        if mesh_shape==(4,8):
            latency_data[model].setdefault(hardware_tuple, {})
            latency_data[model][hardware_tuple] = {
                'TP': tp_latency*1e-3,
                'EP': ep_latency*1e-3,
                'compute_balance': node_balancing_latency*1e-3,
                'node_balance': link_balancing_latency*1e-3
            }
        if hardware_tuple==(5.0,50.0):
            latency_data1[model].setdefault(mesh_shape, {})
            latency_data1[model][mesh_shape] = {
                'TP': tp_latency*1e-3,
                'EP': ep_latency*1e-3,
                'compute_balance': node_balancing_latency*1e-3,
                'node_balance': link_balancing_latency*1e-3
            }
        # 获取加速比数据
        node_balancing = entry['node_balancing']
        if mesh_shape==(4,8):
            
            speedup_data[model].setdefault(hardware_tuple, {})
            
            speedup_data[model][hardware_tuple] = {
                'speedup_EP': node_balancing['speedup_EP'],
                'speedup_TP': node_balancing['speedup_TP'],
                'speedup_comp': node_balancing['speedup_comp']
            }
        if hardware_tuple==(5.0,50.0):
            speedup_data1[model].setdefault(mesh_shape, {})
            
            speedup_data1[model][mesh_shape] = {
                'speedup_EP': node_balancing['speedup_EP'],
                'speedup_TP': node_balancing['speedup_TP'],
                'speedup_comp': node_balancing['speedup_comp']
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
fig, axs = plt.subplots(1, 2, figsize=(18, 5))

# 设置颜色
colors = ["#4E79A7", "#F28E2B", "#E15759", "#59A14F"]


ax = axs[0]

# 获取每个硬件配置的延迟和加速比
model="deepseek"
hardware_tuple = (hardware["comp_TFLOPS"], hardware["BW_GBPS"])
latencies = latency_data[model][hardware_tuple]
speedup = speedup_data[model]

# x轴对应的硬件配置名称

latency_values = [latencies['TP'], latencies['EP'], latencies['compute_balance'], latencies['node_balance']]
#speedup_values = [speedup['speedup_EP'], speedup['speedup_TP'], speedup['speedup_comp']]
x = np.arange(3)
width = 0.2
# 绘制延迟的柱状图

ax.bar(x - width, [latency_data[model][(hardware["comp_TFLOPS"], hardware["BW_GBPS"])]['TP'] for hardware in hardware_configs], width, label='TP', color=colors[0], edgecolor="black")
ax.bar(x, [latency_data[model][(hardware["comp_TFLOPS"], hardware["BW_GBPS"])]['EP'] for hardware in hardware_configs], width, label='EP', color=colors[1], edgecolor="black")
ax.bar(x + width, [latency_data[model][(hardware["comp_TFLOPS"], hardware["BW_GBPS"])]['compute_balance'] for hardware in hardware_configs], width, label='compute_balance', color=colors[2], edgecolor="black")
ax.bar(x + 2*width, [latency_data[model][(hardware["comp_TFLOPS"], hardware["BW_GBPS"])]['node_balance'] for hardware in hardware_configs], width, label='node_balance', color=colors[3], edgecolor="black")
ax.set_ylim([0.6, 4.6]) 
# 绘制加速比的折线图
ax2 = ax.twinx()
#ax2.plot(x[1:4], speedup_values, color=colors[row], linestyle="--", marker="o", label="Speedup", markersize=8)
ax2.plot(x + 2 * width, [speedup[(hardware["comp_TFLOPS"], hardware["BW_GBPS"])]['speedup_TP'] for hardware in hardware_configs], color=colors[0], linestyle="--", marker="o", label="Speedup TP", markersize=8)
ax2.plot(x + 2 * width, [speedup[(hardware["comp_TFLOPS"], hardware["BW_GBPS"])]['speedup_EP'] for hardware in hardware_configs], color=colors[1], linestyle="--", marker="s", label="Speedup EP", markersize=8)
ax2.plot(x + 2 * width, [speedup[(hardware["comp_TFLOPS"], hardware["BW_GBPS"])]['speedup_comp'] for hardware in hardware_configs], color=colors[2], linestyle="--", marker="^", label="Speedup Compute", markersize=8)
ax.set_xlabel("Hardware Config",fontsize=size)
ax.set_ylabel("Latency (ms)",fontsize=size)
ax.set_title("Speedup for Different Configs",fontsize=size)
ax.set_xticks([0.1, 1.1, 2.1])
ax.set_xticklabels([f"({hw['comp_TFLOPS']},{hw['BW_GBPS']:.0f})" for hw in hardware_configs])
#ax.legend(loc="upper left", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=size)  # 'both'表示x和y轴

# 设置次坐标轴（ax2）的刻度字体大小
ax2.tick_params(axis='y', labelsize=size) 
# 设置加速比的y轴
ax2.set_ylabel("Speedup",fontsize=size)
ax2.set_ylim([0.8, 3.2])  # 调整加速比的刻度范围

ax = axs[1]

# 获取每个硬件配置的延迟和加速比

speedup = speedup_data1[model]

# x轴对应的硬件配置名称

#speedup_values = [speedup['speedup_EP'], speedup['speedup_TP'], speedup['speedup_comp']]
x = np.arange(3)
width = 0.2
# 绘制延迟的柱状图

ax.bar(x - width, [latency_data1[model][mesh_shape]['TP'] for mesh_shape in mesh_shapes], width, label='TP', color=colors[0], edgecolor="black")
ax.bar(x, [latency_data1[model][mesh_shape]['EP'] for mesh_shape in mesh_shapes], width, label='EP', color=colors[1], edgecolor="black")
ax.bar(x + width, [latency_data1[model][mesh_shape]['compute_balance'] for mesh_shape in mesh_shapes], width, label='compute_balance', color=colors[2], edgecolor="black")
ax.bar(x + 2*width, [latency_data1[model][mesh_shape]['node_balance'] for mesh_shape in mesh_shapes], width, label='node_balance', color=colors[3], edgecolor="black")
ax.set_ylim([0.5, 3.6]) 
# 绘制加速比的折线图
ax2 = ax.twinx()
#ax2.plot(x[1:4], speedup_values, color=colors[row], linestyle="--", marker="o", label="Speedup", markersize=8)
ax2.plot(x + 2 * width, [speedup[mesh_shape]['speedup_TP'] for mesh_shape in mesh_shapes], color=colors[0], linestyle="--", marker="o", label="Speedup TP", markersize=8)
ax2.plot(x + 2 * width, [speedup[mesh_shape]['speedup_EP'] for mesh_shape in mesh_shapes], color=colors[1], linestyle="--", marker="s", label="Speedup EP", markersize=8)
ax2.plot(x + 2 * width, [speedup[mesh_shape]['speedup_comp'] for mesh_shape in mesh_shapes], color=colors[2], linestyle="--", marker="^", label="Speedup Compute", markersize=8)
ax.set_xlabel("Mesh Size",fontsize=size)
ax.set_ylabel("Latency (ms)",fontsize=size)
ax.set_title("Speedup for Different Mesh Shapes",fontsize=size)
ax.set_xticks([0.1, 1.1, 2.1])
ax.set_xticklabels([f"{mesh_shape}" for mesh_shape in mesh_shapes])
#ax.legend(loc="upper left", fontsize=size)
ax.tick_params(axis='both', which='major', labelsize=size)  # 'both'表示x和y轴

# 设置次坐标轴（ax2）的刻度字体大小
ax2.tick_params(axis='y', labelsize=size) 
# 设置加速比的y轴
ax2.set_ylabel("Speedup",fontsize=size)
ax2.set_ylim([0.8, 3.2])  # 调整加速比的刻度范围



# 显示图形
plt.tight_layout()
plt.savefig("/data/home/haochenhuang/deployment/ablation_study_node_balancing_speedup.pdf")
