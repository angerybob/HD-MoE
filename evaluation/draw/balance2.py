import sys
import os

# 获取当前脚本的绝对路径，并向上追溯到项目根目录（HD-MoE/）
current_dir = os.path.dirname(os.path.abspath(__file__))  # balance2.py 的目录
project_root = os.path.dirname(os.path.dirname(current_dir))  # HD-MoE/ 目录

# 将项目根目录添加到 Python 路径
sys.path.append(project_root)
from node_allocation import MoE3DPNMOptimizer
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import ast
import pdb


def EP_deployment(L, E, D):
    """
    生成专家部署策略矩阵 P，维度为 E×D。
    P[e][d] = a 表示第 e 个专家在第 d 个设备上部署了 a 的权重（0 ≤ a ≤ 1）。
    """
    P = np.zeros((L,E, D))
    
    if D >= E:
        # D >= E 时，每个专家分配到多个设备，设备权重均匀分布
        k, r = divmod(D, E)  # 每个专家至少分到 k 个设备，前 r 个专家多分 1 个设备
        devices = np.arange(D)  # 设备索引
        np.random.shuffle(devices)  # 随机打乱设备顺序
        start = 0
        for e in range(E):
            num_devices = k + 1 if e < r else k  # 当前专家分到的设备数
            end = start + num_devices
            assigned_devices = devices[start:end]  # 随机分配到设备
            P[:,e, assigned_devices] = 1.0 / num_devices  # 权重均匀分布
            start = end  # 更新下一个起始位置
    else:
        # D < E 时，专家尽可能均衡分配到设备，每个专家只在一个设备
        m, r = divmod(E, D)  # 每个设备至少分到 m 个专家，前 r 个设备多分 1 个专家
        experts = np.arange(E)  # 专家索引
        np.random.shuffle(experts)  # 随机打乱专家顺序
        expert_idx = 0
        for d in range(D):
            num_experts = m + 1 if d < r else m  # 当前设备分到的专家数
            assigned_experts = experts[expert_idx : expert_idx + num_experts]  # 随机分配到专家
            P[:,assigned_experts, d] = 1.0  # 权重为 1
            expert_idx += num_experts
    return P
def generate_random_placement(D, mesh_shape):
    """
    生成随机的设备布局
    :param D: 设备数量
    :param mesh_shape: (X, Y)网格尺寸
    :return: D x X x Y的放置矩阵
    """
    X, Y = mesh_shape
    all_positions = [(x, y) for x in range(X) for y in range(Y)]
    
    if len(all_positions) < D:
        raise ValueError(f"Mesh size {X}x{Y} cannot accommodate {D} devices")
    
    selected = random.sample(all_positions, D)
    placement = np.zeros((D, X, Y), dtype=int)
    for d, (x, y) in enumerate(selected):
        placement[d, x, y] = 1
    return placement

#file_path = '/data/home/haochenhuang/deployment/results/30TFLOPS_20GBPS/arrays_30_TFLOPS_20_GBPS_in_layer_0.npz'

batch=128

try:
    with open('expert_trace/ds/experts_reasoning_ds.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('expert_trace/ds/experts_reasoning_ds.json', 'r', encoding='utf-8') as f1:
        sample = json.load(f1)
except FileNotFoundError:
    print("文件未找到，请检查文件路径和文件名。")
optimizer = MoE3DPNMOptimizer(E=64, SE=0,h=2048,D=32, B=batch,BW=75e9, comp=2.5e12,routing_trace=data)
P_tp=np.ones((optimizer.layer,optimizer.E,optimizer.D))/optimizer.D
P_ep=EP_deployment(optimizer.layer,optimizer.E,optimizer.D)


layer_id=10
file_path = f"results/reasoning_ds_2.5_TFLOPS_75.0_GBPS_for_4*8_mesh_128_batches/arrays_2.5_TFLOPS_75.0_GBPS_in_layer_{layer_id}.npz"


loaded_arrays = np.load(file_path)

# 访问加载的数组
P = loaded_arrays['arr1']
M = loaded_arrays['arr2']

def load(P,optimizer,layer_id):
    compute_load = np.sum(P * optimizer.f[:,:, None] * optimizer.B * 2 * optimizer.h*optimizer.IS, axis=1)[layer_id]
    #pdb.set_trace()
    single_comm = np.zeros((optimizer.layer,optimizer.D))

    for list_key, freq in optimizer.fg[optimizer.e][layer_id].items():
        group = ast.literal_eval(list_key)
        devices = np.sum(P[layer_id][group]>0,axis=0)
        redundant = freq * optimizer.B * optimizer.h * (devices>0)
        single_comm[layer_id] += redundant
    comm_load=single_comm[layer_id] / (optimizer.BW)
    load=comm_load+compute_load/optimizer.comp
    return load.reshape(4, 8)

def link(optimizer,M,P,layer_id,mesh_size):
    comm_time, link_load = optimizer.comm_time_acc(M, P, layer_id)

    # 构建 2D Mesh 链路矩阵
    #mesh_size = int(np.sqrt(optimizer.D))
    # 水平链路矩阵
    horizontal_links = np.zeros((mesh_size[0], mesh_size[1] - 1))
    # 垂直链路矩阵
    vertical_links = np.zeros((mesh_size[0] - 1, mesh_size[1]))

    for (src, dst), load in link_load.items():
        x1, y1 = src
        x2, y2 = dst
        if x1 == x2:  # 水平链路
            min_y = min(y1, y2)
            horizontal_links[x1, min_y] += load
        elif y1 == y2:  # 垂直链路
            min_x = min(x1, x2)
            vertical_links[min_x, y1] += load
    return horizontal_links,vertical_links


#pdb.set_trace()
mesh_shape=(4,8)
M_init=generate_random_placement(optimizer.D, mesh_shape)
horizontal_links_init,vertical_links_init=link(optimizer,M_init,P,layer_id,mesh_shape)
horizontal_links,vertical_links=link(optimizer,M,P,layer_id,mesh_shape)

vmin = min(np.min(vertical_links_init*1e6), np.min(vertical_links*1e6))
vmax = max(np.max(vertical_links_init*1e6), np.max(vertical_links*1e6))


# 创建一个包含两个子图的画布
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
heatmap=sns.heatmap(vertical_links_init*1e6, ax=axes[0], cmap="YlGnBu", vmin=vmin, vmax=vmax, annot=False)

cbar = heatmap.collections[0].colorbar

# 设置颜色条主标签字体大小
cbar.set_label('Link Load (us)', fontsize=24)  # 主标签字体大小

# 设置颜色条刻度标签字体大小
cbar.ax.tick_params(labelsize=24)  # 刻度标签字体大小
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=25)  # x轴标签
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=25)  # y轴标签
size=30
axes[0].set_title("Node Balancing", fontsize=size)
axes[0].set_xlabel("X", fontsize=size)
axes[0].set_ylabel("Y", fontsize=size)

# 绘制垂直链路热力图
heatmap=sns.heatmap(vertical_links*1e6, ax=axes[1], cmap="YlGnBu", vmin=vmin, vmax=vmax, annot=False)
cbar = heatmap.collections[0].colorbar

# 设置颜色条主标签字体大小
cbar.set_label('Link Load (us)', fontsize=24)  # 主标签字体大小
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=25)  # x轴标签
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=25)  # y轴标签
# 设置颜色条刻度标签字体大小
cbar.ax.tick_params(labelsize=24)  # 刻度标签字体大小
axes[1].set_title("Node-Link Balancing", fontsize=size)
axes[1].set_xlabel("X", fontsize=size)
axes[1].set_ylabel("Y", fontsize=size)

# 显示图形
plt.tight_layout()
plt.savefig('evaluation/figs/ablation/link.png')
plt.close()
print("Figure saved at evaluation/figs/ablation/link.png")

# 绘制柱状图
'''
sns.heatmap(
    vertical_links, 
    annot=False,  # 在单元格中显示数值
    fmt=".2f",     # 数值格式为整数
    cmap="viridis",  # 颜色映射（可选：'plasma', 'inferno', 'magma'等）
    linewidths=0.5,  # 单元格边框线宽
    cbar_kws={'label': 'congestion'}  # 颜色条标签
)

plt.title("load imbalance on 2D mesh", fontsize=14)
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.xticks(rotation=45)  # 横坐标标签旋转45度
plt.yticks(rotation=0)
plt.savefig('/data/home/haochenhuang/deployment/evaluation/vertical_links.png')
plt.close()'''
'''

combined_matrix = np.vstack([
    np.hstack([horizontal_links_init, np.zeros((mesh_shape[0], 1))]),
    np.hstack([vertical_links_init, np.zeros((mesh_shape[1] - 1, 0))])
])

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(combined_matrix, annot=False, fmt=".2f", cmap="YlGnBu")
plt.title("2D Mesh Link Congestion (Total Occupation Time)")
plt.xlabel("X Coordinate")
plt.xticks(rotation=45)
plt.ylabel("Y Coordinate")
plt.savefig('/data/home/haochenhuang/deployment/evaluation/horizontal_links_init imbalance.png')
plt.close()
'''