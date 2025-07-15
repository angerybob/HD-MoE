import sys
sys.path.append("/data/home/haochenhuang/deployment") 
from node_allocation import MoE3DPNMOptimizer
import numpy as np
import json
import pdb
import random
from tqdm import tqdm
import matplotlib.pyplot as plt




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
mesh_shape=(8,8)
D=mesh_shape[0]*mesh_shape[1]
E,e,SE,h,IS,mlp_first,num_layers=64,6,0,2048,1408,True,26  #DeepSeekMoE  

try:
    with open('/data/home/haochenhuang/deployment/experts_reasoning_ds.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('/data/home/haochenhuang/deployment/experts_reasoning_ds.json', 'r', encoding='utf-8') as f1:
        sample = json.load(f1)
except FileNotFoundError:
    print("文件未找到，请检查文件路径和文件名。")
optimizer = MoE3DPNMOptimizer(E=E, e=e,SE=SE,h=h,IS=IS, B=batch,D=D,BW=75e9, comp=2.5e12,num_layers=num_layers,mlp_first=mlp_first,routing_trace=data)
P_tp=np.ones((optimizer.layer,optimizer.E,optimizer.D))/optimizer.D
P_ep=EP_deployment(optimizer.layer,optimizer.E,optimizer.D)

optimizer.X,optimizer.Y=mesh_shape
layer_id=11
tp_comp=0
tp_comm=0
tp_comp_dynamic=0
tp_comm_dynamic=0
ep_comp=0
ep_comm=0
ep_comp_dynamic=0
ep_comm_dynamic=0
comp=0
comm_node=0
comp_dynamic=0
comm_node_dynamic=0
comm_link=0
comm_link_dynamic=0

file_path = f'/data/home/haochenhuang/deployment/results/10.0_TFLOPS_25.0_GBPS_for_128_batches/arrays_10.0_TFLOPS_25.0_GBPS_in_layer_{layer_id:.0f}.npz'
loaded_arrays = np.load(file_path)

# 访问加载的数组
P = loaded_arrays['arr1']
M = loaded_arrays['arr2']

comp_map=np.zeros((optimizer.E))
random_samples = random.sample(sample[str(layer_id+1)], batch)
for sublist in random_samples:
    comp_map[sublist]+=8*optimizer.h**2

#print("Evaluating TP and EP...")
Z_tp=P_tp[layer_id]>0
Z_ep=P_ep[layer_id]>0
M_rand=generate_random_placement(optimizer.D, mesh_shape)
optimizer.M=M_rand

tp_comp+=optimizer.compute_time(P_tp)[layer_id]
tp_comm+=2*optimizer.comm_time(P_tp)[layer_id]

tp_comp_dynamic+=optimizer.compute_time_dynamic(P_tp,comp_map)[layer_id]
tp_comm_dynamic+=2*optimizer.comm_time_dynamic(P_tp,random_samples)[layer_id]

ep_comp+=optimizer.compute_time(P_ep)[layer_id]
comm_temp,link=optimizer.comm_time_acc(M_rand,P_ep,layer_id)
ep_comm+=comm_temp*2


ep_comp_dynamic+=optimizer.compute_time_dynamic(P_ep,comp_map)[layer_id]
ep_comm_dynamic+=2*optimizer.comm_time_acc_dynamic(M_rand,P_ep,layer_id,random_samples)
ep_ideal=2*optimizer.comm_time(P_ep)[layer_id]

print(f"TP_communication: {tp_comm*1e6:.2f} us")
print(f"TP_computation: {tp_comp*1e6:.2f} us")
print(f"TP_latency: {(tp_comp+tp_comm)*1e6:.2f} us")
print(f"TP_communication_dynamic: {tp_comm_dynamic*1e6:.2f} us")
print(f"TP_computation_dynamic: {tp_comp_dynamic*1e6:.2f} us")
print(f"TP_latency_dynamic: {(tp_comp_dynamic+tp_comm_dynamic)*1e6:.2f} us")
print(f"EP_communication: {ep_comm*1e6:.2f} us")
print(f"EP_communication_ideal: {ep_ideal*1e6:.2f} us")
print(f"EP_computation: {ep_comp*1e6:.2f} us")
print(f"EP_latency: {(ep_comp+ep_comm)*1e6:.2f} us")
print(f"EP_communication_dynamic: {ep_comm_dynamic*1e6:.2f} us")
print(f"EP_computation_dynamic: {ep_comp_dynamic*1e6:.2f} us")
print(f"EP_latency_dynamic: {(ep_comp_dynamic+ep_comm_dynamic)*1e6:.2f} us")

comp=optimizer.compute_time(P)[layer_id]
comm_node,link=optimizer.comm_time_acc(M_rand,P,layer_id)
comm_node*=2
print(f"node_balancing_communication: {comm_node*1e6:.2f} us")
print(f"node_balancing_computation: {comp*1e6:.2f} us")
print(f"node_balancing_latency: {(comp+comm_node)*1e6:.2f} us")
print(f"node_balancing_speedup:{(ep_comp+ep_comm)/(comp+comm_node):.2f}")