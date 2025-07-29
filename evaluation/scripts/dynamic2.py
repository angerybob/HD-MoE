import sys
import os
import json
import random
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def EP_deployment(L, E, D):
    P = np.zeros((L,E, D))
    if D >= E:
        k, r = divmod(D, E); devices = np.arange(D); np.random.shuffle(devices); start = 0
        for e in range(E):
            num_devices = k + 1 if e < r else k; end = start + num_devices
            P[:,e, devices[start:end]] = 1.0 / num_devices; start = end
    else:
        m, r = divmod(E, D); experts = np.arange(E); np.random.shuffle(experts); expert_idx = 0
        for d in range(D):
            num_experts = m + 1 if d < r else m
            P[:,experts[expert_idx : expert_idx + num_experts], d] = 1.0; expert_idx += num_experts
    return P

def generate_random_placement(D, mesh_shape):
    X, Y = mesh_shape; all_positions = [(x, y) for x in range(X) for y in range(Y)]
    if len(all_positions) < D: raise ValueError(f"Mesh size {X}x{Y} cannot accommodate {D} devices")
    selected = random.sample(all_positions, D); placement = np.zeros((D, X, Y), dtype=int)
    for d, (x, y) in enumerate(selected): placement[d, x, y] = 1
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from node_allocation import MoE3DPNMOptimizer


batch=int(128*4)
model="ds"
if model=="ds":
    E,e,SE,h,IS,mlp_first,num_layers=64,6,0,2048,1408,True,26
elif model=="mixtral":
    E,e,SE,h,IS,mlp_first,num_layers=8,2,0,4096,14336,False,32
elif model=="qwen":
    E,e,SE,h,IS,mlp_first,num_layers=64,8,0,3584,18944,False,28

dataset=["reasoning","math","coding","writing","roleplay"][0]
sample1=["reasoning","math","coding","writing","roleplay"][2]
mesh_shape=(4,8)
D=mesh_shape[0]*mesh_shape[1]
data_path=f'expert_trace/{model}/predict/experts_{dataset}_{model}_pre.json'
sample_path=f'expert_trace/{model}/predict/experts_{sample1}_{model}_pre.json'
try:
    with open(data_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f); data,pre=data1["selected_experts"],data1["predict_experts"]
    with open(sample_path, 'r', encoding='utf-8') as f1:
        sample2 = json.load(f1); sample,pre_sample=sample2["selected_experts"],sample2["predict_experts"]
except FileNotFoundError:
    print("文件未找到，请检查文件路径和文件名。")
    sys.exit(1) # Exit if data not found

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

all_results = []

print("Processing layers for dynamic deployment analysis...")
for layer_id in tqdm(range(optimizer.layer)):
    # --- Load data for the CURRENT layer ---
    file_path = f'results/{dataset}_{model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_128_batches/arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id:.0f}.npz'
    try:
        loaded_arrays = np.load(file_path)
        P = loaded_arrays['arr1']
        M = loaded_arrays['arr2']
    except FileNotFoundError:
        print(f"Warning: Data for layer {layer_id} not found. Skipping.")
        continue

    comp_map = np.zeros((optimizer.E))
    random_samples = []
    current_layer_key = str(layer_id + optimizer.mlp_first)
    for i in sample_id:
        random_samples.append(sample[current_layer_key][i])
    for sublist in random_samples:
        comp_map[sublist] += 2 * optimizer.h * optimizer.IS

    comp_dynamic_layer = optimizer.compute_time_dynamic(P, comp_map)[layer_id]
    comm_link_dynamic_layer = 2 * optimizer.comm_time_acc_dynamic(M, P, layer_id, random_samples)
    comp_pre_layer = 0
    comm_pre_layer = 0

    if layer_id < optimizer.layer - 1:
        next_layer_id = layer_id + 1
        pre_path = f'results/{dataset}_{model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_128_batches/arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{next_layer_id:.0f}.npz'
        try:
            pre_arrays = np.load(pre_path)
            P_next = pre_arrays['arr1']
            M_next = pre_arrays['arr2']

            p_copy = copy.deepcopy(P_next)
            
            comp_map_next = np.zeros((optimizer.E))
            random_samples_next = []
            next_samples = []
            next_layer_key = str(next_layer_id + optimizer.mlp_first)
            
            for i in sample_id:
                if next_layer_key in sample and next_layer_key in pre_sample:
                    random_samples_next.append(sample[next_layer_key][i])
                    next_samples.append(pre_sample[next_layer_key][i])
            
            if random_samples_next: # Proceed only if we have data for the next layer
                for sublist in random_samples_next:
                    comp_map_next[sublist] += 2 * optimizer.h * optimizer.IS
                
                compute_load_next = np.sum(p_copy * comp_map_next[None,:,None], axis=1)

                for _ in range(k):
                    prio, e = optimizer.priority_detection(p_copy, next_layer_id, next_samples)
                    p_copy[next_layer_id, e, :] = 0
                    
                    for sublist in random_samples_next:
                        if e in sublist:
                            activate_node = []
                            j, d = np.nonzero(P_next[next_layer_id, sublist])
                            for c in d:
                                activate_node.append(c)
                            if activate_node:
                                scatter_node = activate_node[np.argmin(compute_load_next[next_layer_id, activate_node])]
                                compute_load_next[next_layer_id, scatter_node] += 2 * optimizer.h * optimizer.IS
                
                comp_pre_layer = np.max(compute_load_next[next_layer_id]) / optimizer.comp
                comm_pre_layer = 2 * optimizer.comm_time_acc_dynamic(M_next, p_copy, next_layer_id, random_samples_next)
                
        except FileNotFoundError:
            print(f"Warning: Data for next layer {next_layer_id} not found. Dynamic results for layer {layer_id} will be 0.")

    #  Build the result dictionary for the CURRENT layer
    config = {
        "layer_id": layer_id,
        "mesh_shape": mesh_shape,
        "model": model,
        "dataset": dataset,
        "sample": sample1,
        "comp_TFLOPS": optimizer.comp * 1e-12,
        "BW_GBPS": optimizer.BW * 1e-9,
        "batch": optimizer.B
    }

    static_latency = comp_dynamic_layer + comm_link_dynamic_layer
    dynamic_latency = comp_pre_layer + comm_pre_layer

    result_for_layer = {
        "config": config,
        "static_deployment": {
            "communication_us": round(comm_link_dynamic_layer * 1e6, 2),
            "computation_us": round(comp_dynamic_layer * 1e6, 2),
            "latency_us": round(static_latency * 1e6, 2)
        },
        "dynamic_deployment": {
            "communication_us": round(comm_pre_layer * 1e6, 2),
            "computation_us": round(comp_pre_layer * 1e6, 2),
            "latency_us": round(dynamic_latency * 1e6, 2),
            "speedup": round(static_latency / dynamic_latency, 2) if dynamic_latency > 0 else 0
        }
    }
    all_results.append(result_for_layer)

file_path = "evaluation/results/result_dynamic_per_layer.json" # Using a new name
os.makedirs(os.path.dirname(file_path), exist_ok=True)

new_data = all_results

if os.path.exists(file_path):
    try:
        with open(file_path, "r") as f:
            old_data = json.load(f)
        combined_data = old_data + new_data
    except json.JSONDecodeError:
        print(f"Warning: Could not decode existing JSON from {file_path}. Overwriting.")
        combined_data = new_data
else:
    combined_data = new_data

with open(file_path, "w") as f:
    json.dump(combined_data, f, indent=4)

print(f"Successfully saved {len(all_results)} layer results to {file_path}")