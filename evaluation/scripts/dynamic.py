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
import copy
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
def comp_overhead(comp,D,batch,h,L,intermediate,e):
    generation=3*batch*h**2
    score_context=2*batch*L*h
    projection=batch*h**2
    FFN=2*e*batch*intermediate*h**2
    return (generation+score_context+projection+FFN)/(D*comp)

def comm_overhead(BW,D,batch,h,alpha,e): 
    all_reduce=(5*batch*h*(D-1))/(BW*D)+4*alpha*D**0.5
    all2all=(4*batch*h*(e-1))/(BW*D)+(batch*h*(D-1))/(BW*D)+4*alpha*D**0.5 # An estimate
    return all_reduce+all2all # Consider Attention All-Reduce and MoE All2all

def mem_overhead(BWmem,D,batch,h,L,intermediate,E):
    KV_cache=2*batch*h*L
    generation=3*h*(batch+h)
    projection=h*(batch+h)
    FFN=2*E*intermediate*h**2+batch*h*(intermediate+1)
    return (KV_cache+generation+projection+FFN)/(D*BWmem)
#file_path = '/data/home/haochenhuang/deployment/results/30TFLOPS_20GBPS/arrays_30_TFLOPS_20_GBPS_in_layer_0.npz'

batch=int(128*4)
model="ds"
if model=="ds":
    E,e,SE,h,IS,mlp_first,num_layers=64,6,0,2048,1408,True,26  #DeepSeekMoE 
elif model=="mixtral":
    E,e,SE,h,IS,mlp_first,num_layers=8,2,0,4096,14336,False,32 #Mixtral
elif model=="qwen":
    E,e,SE,h,IS,mlp_first,num_layers=64,8,0,3584,18944,False,28 #Qwen2
 
dataset=["reasoning","math","coding","writing","roleplay"][0]
sample1=["reasoning","math","coding","writing","roleplay"][4]
mesh_shape=(4,8)
D=mesh_shape[0]*mesh_shape[1]
data_path=f'expert_trace/{model}/predict/experts_{dataset}_{model}_pre.json'
#data_path="/data/home/haochenhuang/deployment/evaluation/experts_reasoning_mixtral.json"
sample_path=f'expert_trace/{model}/predict/experts_{sample1}_{model}_pre.json'
try:
    with open(data_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
        data,pre=data1["selected_experts"],data1["predict_experts"]

    with open(sample_path, 'r', encoding='utf-8') as f1:
        sample2 = json.load(f1)
        sample,pre_sample=sample2["selected_experts"],sample2["predict_experts"]
except FileNotFoundError:
    print("文件未找到，请检查文件路径和文件名。")
optimizer = MoE3DPNMOptimizer(E=E,e=e, SE=SE,h=h,IS=IS,B=batch, D=D,BW=75*1e9, comp=2.5*1e12, num_layers=num_layers,mlp_first=mlp_first,routing_trace=data)
P_tp=np.ones((optimizer.layer,optimizer.E,optimizer.D))/optimizer.D
P_ep=EP_deployment(optimizer.layer,optimizer.E,optimizer.D)


sample_id=random.sample(range(0,len(sample[str(1)])), batch)
BWmem=625e9
L=1024
t_inf=comp_overhead(optimizer.comp,optimizer.D,batch,optimizer.h,L,optimizer.IS/optimizer.h,optimizer.e)+mem_overhead(BWmem,optimizer.D,batch,optimizer.h,L,optimizer.IS/optimizer.h,optimizer.E)
k=1
#pdb.set_trace()
while optimizer.optimal_broadcast_chunk(k=k)<t_inf:
    #pdb.set_trace()
    k+=1
k-=1
print(f"Number of experts to pre-broadcast: {k:.0f}")
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
comp_pre=0
comm_pre=0

for layer_id in tqdm(range(optimizer.layer)):
    file_path = f'results/{dataset}_{model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_128_batches/arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id:.0f}.npz'
    #file_path="/data/home/haochenhuang/deployment/results/reasoning_mixtral_10.0_TFLOPS_25.0_GBPS_for_8*8_mesh_128_batches/arrays_10.0_TFLOPS_25.0_GBPS_in_layer_1.npz"
    loaded_arrays = np.load(file_path)

    # 访问加载的数组
    P = loaded_arrays['arr1']
    M = loaded_arrays['arr2']
    comp_map=np.zeros((optimizer.E))
    random_samples = []
    next_samples=[]
    for i in sample_id:
        random_samples.append(sample[str(layer_id+optimizer.mlp_first)][i])
        next_samples.append(pre_sample[str(layer_id+optimizer.mlp_first)][i])
    for sublist in random_samples:
        comp_map[sublist]+=2*optimizer.h*optimizer.IS
    Z_tp=P_tp[layer_id]>0
    Z_ep=P_ep[layer_id]>0
    M_rand=generate_random_placement(optimizer.D, mesh_shape)



    tp_comp_dynamic+=optimizer.compute_time_dynamic(P_tp,comp_map)[layer_id]
    tp_comm_dynamic+=2*optimizer.comm_time_dynamic(P_tp,random_samples)[layer_id]



    ep_comp_dynamic+=optimizer.compute_time_dynamic(P_ep,comp_map)[layer_id]
    ep_comm_dynamic+=2*optimizer.comm_time_acc_dynamic(M_rand,P_ep,layer_id,random_samples)
    comp_dynamic+=optimizer.compute_time_dynamic(P,comp_map)[layer_id]


    Z=P>0
    M_init=generate_random_placement(optimizer.D, mesh_shape)

    #print("Evaluating link balance...")

    comm_temp,link=optimizer.comm_time_acc(M,P,layer_id)
    comm_link+=comm_temp*2

    comm_link_dynamic+=2*optimizer.comm_time_acc_dynamic(M,P,layer_id,random_samples)



    if layer_id==0: 
        comp_pre+=optimizer.compute_time_dynamic(P,comp_map)[0]
        comm_pre+=2*optimizer.comm_time_acc_dynamic(M,P,0,random_samples)

    #pdb.set_trace()
    if layer_id<optimizer.layer-1:
        pre_path = f'results/{dataset}_{model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_128_batches/arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id+1:.0f}.npz'
        pre_arrays = np.load(pre_path)
        P_next = pre_arrays['arr1']
        M_next=pre_arrays['arr2']
        Z_next=P_next[layer_id+1]>0
        p_copy=copy.deepcopy(P_next)
        z_copy=copy.deepcopy(Z_next)
        comp_map_next=np.zeros((optimizer.E))
        random_samples_next = []
        next_samples=[]
        for i in sample_id:
            random_samples_next.append(sample[str(layer_id+1+1)][i])
            next_samples.append(pre_sample[str(layer_id+1+1)][i])
        for sublist in random_samples_next:
            comp_map_next[sublist]+=2*optimizer.h*optimizer.IS
        for _ in range(k):
            prio, e=optimizer.priority_detection(p_copy,layer_id+1,next_samples)
            p_copy[layer_id+1,e,:]=0
            z_copy[e,:]=0
            for sublist in random_samples_next:
                if e in sublist:
                    activate_node=[]
                    
                    j,d=np.nonzero(P_next[layer_id+1][sublist])
                    for c in d:
                        activate_node.append(c)
                    #compute_load = np.sum(p_copy * comp_map[None,:,None], axis=1)

                    compute_load_next = np.sum(p_copy * comp_map_next[None,:,None], axis=1)
                    scatter_node=np.argmin(compute_load_next[layer_id+1,activate_node])
                    #z_copy[e,scatter_node]=1
                    compute_load_next[layer_id+1,scatter_node]+=2*optimizer.h*optimizer.IS
                    #pdb.set_trace()
        #pdb.set_trace()
        comp_pre+=np.max(compute_load_next[layer_id+1])/optimizer.comp
        comm_pre+=2*optimizer.comm_time_acc_dynamic(M_next,p_copy,layer_id+1,random_samples_next)
        #pdb.set_trace()
#pdb.set_trace()
print(f"TP_communication_dynamic: {tp_comm_dynamic*1e6:.2f} us")
print(f"TP_computation_dynamic: {tp_comp_dynamic*1e6:.2f} us")
print(f"TP_latency_dynamic: {(tp_comp_dynamic+tp_comm_dynamic)*1e6:.2f} us")

print(f"EP_communication_dynamic: {ep_comm_dynamic*1e6:.2f} us")
print(f"EP_computation_dynamic: {ep_comp_dynamic*1e6:.2f} us")
print(f"EP_latency_dynamic: {(ep_comp_dynamic+ep_comm_dynamic)*1e6:.2f} us")


print(f"link_balancing_dynamic: {comm_link_dynamic*1e6:.2f} us")
print(f"link_balancing_computation_dynamic: {comp_dynamic*1e6:.2f} us")
print(f"link_balancing_latency_dynamic: {(comp_dynamic+comm_link_dynamic)*1e6:.2f} us")

print(f"node_link_balancing_speedup_EP_dynamic:{(ep_comp_dynamic+ep_comm_dynamic)/(comp_dynamic+comm_link_dynamic):.2f}")
print(f"node_link_balancing_speedup_TP_dynamic:{(tp_comp_dynamic+tp_comm_dynamic)/(comp_dynamic+comm_link_dynamic):.2f}")

print(f"preb_communication_dynamic: {comm_pre*1e6:.2f} us")
print(f"preb_computation_dynamic: {comp_pre*1e6:.2f} us")
print(f"preb_latency_dynamic: {(comp_pre+comm_pre)*1e6:.2f} us")


print(f"preb_speedup_EP_dynamic:{(ep_comp_dynamic+ep_comm_dynamic)/(comp_pre+comm_pre):.2f}")
print(f"preb_speedup_TP_dynamic:{(tp_comp_dynamic+tp_comm_dynamic)/(comp_pre+comm_pre):.2f}")

print(f"preb_speedup_dynamic:{(comp_dynamic+comm_link_dynamic)/(comp_pre+comm_pre):.2f}")
config = {
    "mesh_shape": mesh_shape,  # 从变量 mesh_shape 获取
    "model": model,         # DeepSeekMoE 模型
    "dataset": dataset,
    "sample": sample1,
    "comp_TFLOPS": optimizer.comp*1e-12,    # 计算能力 (TFLOPS)
    "BW_GBPS": optimizer.BW*1e-9,         # 带宽 (GBPS)
    "batch": optimizer.B        # 从变量 batch 获取
}

result = {
    "config": config,
    "static_deployment": {
        "communication_us": round(comm_link_dynamic * 1e6, 2),
        "computation_us": round(comp_dynamic * 1e6, 2),
        "latency_us": round((comp_dynamic+comm_link_dynamic) * 1e6, 2)
        
    },
    "dynamic_deployment": {
        "communication_us": round(comm_pre * 1e6, 2),
        "computation_us": round(comp_pre * 1e6, 2),
        "latency_us": round((comp_pre + comm_pre) * 1e6, 2),
        "speedup": round((comp_dynamic+comm_link_dynamic) / (comp_pre + comm_pre), 2)
    }
}

file_path = "evaluation/results/result2_dynamic.json"
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
