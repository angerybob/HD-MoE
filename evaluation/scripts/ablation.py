import sys
import os

# 获取当前脚本的绝对路径，并向上追溯到项目根目录（HD-MoE/）
current_dir = os.path.dirname(os.path.abspath(__file__))  # balance2.py 的目录
project_root = os.path.dirname(os.path.dirname(current_dir))  # HD-MoE/ 目录

# 将项目根目录添加到 Python 路径
sys.path.append(project_root)
from node_allocation import MoE3DPNMOptimizer
import numpy as np
import json
import pdb
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os




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


batch=128
model="ds"
if model=="mixtral":
    E,e,SE,h,IS,mlp_first,num_layers=8,2,0,4096,14336,False,32 #Mixtral
elif model=="ds":
    E,e,SE,h,IS,mlp_first,num_layers=64,6,0,2048,1408,True,26  #DeepSeekMoE  
elif model=="qwen":
    E,e,SE,h,IS,mlp_first,num_layers=64,8,0,3584,18944,False,28 #Qwen2
mesh_shape=(8,8)
BW=50
comp=5
D=mesh_shape[0]*mesh_shape[1]

dataset="reasoning"

data_path=f'expert_trace/{model}/experts_{dataset}_{model}.json'
try:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(data_path, 'r', encoding='utf-8') as f1:
        sample = json.load(f1)
except FileNotFoundError:
    print("文件未找到，请检查文件路径和文件名。")
optimizer = MoE3DPNMOptimizer(E=E,e=e, SE=SE,h=h,IS=IS,B=batch, D=D,BW=BW*1e9, comp=comp*1e12, num_layers=num_layers,mlp_first=mlp_first,routing_trace=data)
P_tp=np.ones((optimizer.layer,optimizer.E,optimizer.D))/optimizer.D
P_ep=EP_deployment(optimizer.layer,optimizer.E,optimizer.D)



tp_comp=0
tp_comm=0


ep_comp=0
ep_comm=0

comp_comp=0
comp_comm=0

comp=0
comm_node=0


comm_link=0


for layer_id in tqdm(range(optimizer.layer)):

    file_path = f'results/{dataset}_{model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_128_batches/arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id:.0f}.npz'
    loaded_arrays = np.load(file_path)
    comp_map=np.zeros((optimizer.E))
    # 访问加载的数组
    comp_path = f'results/comp_balance_only/{dataset}_{model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_{optimizer.B:.0f}_batches/arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id}.npz'
    P = loaded_arrays['arr1']
    M = loaded_arrays['arr2']
    loaded_comp_arrays = np.load(comp_path)
    P_comp = loaded_comp_arrays['arr1']
    M_comp = loaded_comp_arrays['arr2']




    #print("Evaluating TP and EP...")
    Z_tp=P_tp[layer_id]>0
    Z_ep=P_ep[layer_id]>0
    M_rand=generate_random_placement(optimizer.D, mesh_shape)
    optimizer.M=M_rand

    tp_comp+=optimizer.compute_time(P_tp)[layer_id]
    tp_comm+=2*optimizer.comm_time(P_tp)[layer_id]



    ep_comp+=optimizer.compute_time(P_ep)[layer_id]
    comm_temp,link=optimizer.comm_time_acc(M_rand,P_ep,layer_id)
    ep_comm+=comm_temp*2



    comp_comp+=optimizer.compute_time(P_comp)[layer_id]
    comm_temp,link=optimizer.comm_time_acc(M_comp,P_comp,layer_id)
    comm_temp+=optimizer.comm_time(P)[layer_id]
    comp_comm+=comm_temp*2

    #print("Evaluating node balance...")
    comp+=optimizer.compute_time(P)[layer_id]
    comm_temp,link=optimizer.comm_time_acc(M_rand,P,layer_id)
    comm_node+=comm_temp*2




    comm_temp,link=optimizer.comm_time_acc(M,P,layer_id)
    comm_link+=comm_temp*2





print(f"TP_communication: {tp_comm*1e6:.2f} us")
print(f"TP_computation: {tp_comp*1e6:.2f} us")
print(f"TP_latency: {(tp_comp+tp_comm)*1e6:.2f} us")


print(f"EP_communication: {ep_comm*1e6:.2f} us")
print(f"EP_computation: {ep_comp*1e6:.2f} us")
print(f"EP_latency: {(ep_comp+ep_comm)*1e6:.2f} us")

print(f"comp_communication: {comp_comm*1e6:.2f} us")
print(f"comp_computation: {comp_comp*1e6:.2f} us")
print(f"comp_latency: {(comp_comp+comp_comm)*1e6:.2f} us")


print(f"node_balancing_communication: {comm_node*1e6:.2f} us")
print(f"node_balancing_computation: {comp*1e6:.2f} us")
print(f"node_balancing_latency: {(comp+comm_node)*1e6:.2f} us")
print(f"node_balancing_speedup_EP:{(ep_comp+ep_comm)/(comp+comm_node):.2f}")
print(f"node_balancing_speedup_TP:{(tp_comp+tp_comm)/(comp+comm_node):.2f}")
print(f"node_balancing_speedup_comp:{(comp_comp+comp_comm)/(comp+comm_node):.2f}")





print(f"link_balancing: {comm_link*1e6:.2f} us")
print(f"link_balancing_computation: {comp*1e6:.2f} us")
print(f"link_balancing_latency: {(comp+comm_link)*1e6:.2f} us")
print(f"link_balancing_speedup:{(comm_node)/(comm_link):.2f}")
print(f"node_link_balancing_speedup_EP:{(ep_comp+ep_comm)/(comp+comm_link):.2f}")
print(f"node_link_balancing_speedup_TP:{(tp_comp+tp_comm)/(comp+comm_link):.2f}")
print(f"node_link_balancing_speedup_comp:{(comp_comp+comp_comm)/(comp+comm_link):.2f}")




result = {
    "config": {
        "mesh_shape": mesh_shape,  # 直接从变量 mesh_shape 获取
        "model": model,         # DeepSeekMoE 模型
        "dataset": dataset,
        "comp_TFLOPS": optimizer.comp*1e-12,    # 计算能力
        "BW_GBPS": optimizer.BW*1e-9          # 带宽
    },
    "TP": {
            "communication_us": round(tp_comm * 1e6, 2),
            "computation_us": round(tp_comp * 1e6, 2),
            "latency_us": round((tp_comp + tp_comm) * 1e6, 2)
    },
    "EP": {
            "communication_us": round(ep_comm * 1e6, 2),
            "computation_us": round(ep_comp * 1e6, 2),
            "latency_us": round((ep_comp + ep_comm) * 1e6, 2)
    },
    "compute_balancing": {
        "communication_us": round(comp_comm * 1e6, 2),
        "computation_us": round(comp_comp * 1e6, 2),
        "latency_us": round((comp_comp + comp_comm) * 1e6, 2)
        
    },
    "node_balancing": {
            "communication_us": round(comm_node * 1e6, 2),
            "computation_us": round(comp * 1e6, 2),
            "latency_us": round((comp + comm_node) * 1e6, 2),
            "speedup_EP": round((ep_comp + ep_comm) / (comp + comm_node), 2),
            "speedup_TP": round((tp_comp + tp_comm) / (comp + comm_node), 2),
            "speedup_comp": round((comp_comp+comp_comm)/(comp+comm_node),2)
    },
    "link_balancing": {
            "communication_us": round(comm_link * 1e6, 2),
            "computation_us": round(comp * 1e6, 2),
            "latency_us": round((comp + comm_link) * 1e6, 2),
            "speedup": round(comm_node / comm_link, 2),
            "speedup_EP": round((ep_comp + ep_comm) / (comp + comm_link), 2),
            "speedup_TP": round((tp_comp + tp_comm) / (comp + comm_link), 2),
            "speedup_comp": round((comp_comp+comp_comm)/(comp+comm_link),2)
    }
}

# 保存为JSON文件
import json
file_path = "evaluation/results/result_ablation.json"
new_data = result  # 你的新数据

# 如果文件存在，读取旧数据
if os.path.exists(file_path):
    with open(file_path, "r") as f:
        old_data = json.load(f)  # 假设原文件是 JSON 数组
    combined_data = old_data + [new_data]  # 合并数据
else:
    combined_data = [new_data]

# 写回文件
with open(file_path, "w") as f:
    json.dump(combined_data, f, indent=4)