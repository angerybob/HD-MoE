import sys
sys.path.append("/data/home/haochenhuang/deployment") 
from node_allocation import MoE3DPNMOptimizer
import numpy as np
import json
import pdb
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy import stats
import ast


# 定义要检查的文件夹路径



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

def get_subregion_coordinates(H, W, rows_split, cols_split, i):
    """
    将大网格均匀划分为 rows_split×cols_split 个子区域，获取每个子区域中第 i 个格子的全局坐标。
    :param H: 大网格总行数（需是 rows_split 的倍数）
    :param W: 大网格总列数（需是 cols_split 的倍数）
    :param rows_split: 划分的行方向子区域数量（如2或1）
    :param cols_split: 划分的列方向子区域数量（如4或2）
    :param i: 子区域内格子的索引（按行优先，0 ≤ i < (H//rows_split)*(W//cols_split)）
    :return: 所有子区域中第i个格子的坐标列表，格式为 [(行, 列), ...]
    """
    if H % rows_split != 0 or W % cols_split != 0:
        raise ValueError(f"H必须是{rows_split}的倍数，W必须是{cols_split}的倍数")
    
    sub_H = H // rows_split  # 每个子区域的行数
    sub_W = W // cols_split  # 每个子区域的列数
    total_elements = sub_H * sub_W
    
    if i < 0 or i >= total_elements:
        raise ValueError(f"i必须在0到{total_elements-1}之间")
    
    coordinates = []
    # 遍历所有子区域的起始位置
    for start_row in range(0, H, sub_H):
        for start_col in range(0, W, sub_W):
            # 计算子区域内的相对行列（行优先）
            rel_row = i // sub_W
            rel_col = i % sub_W
            # 转换为全局坐标
            global_row = start_row + rel_row
            global_col = start_col + rel_col
            coordinates.append((global_row, global_col))
    return coordinates


def main():
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser(description="节点分配工具")

    # 2. 添加参数
    parser.add_argument("--layer-id", type=int, required=False, default=31, help="layer_id")
    parser.add_argument("--comp", type=float, required=False, default=5, help="computation throughput (TFLOPS)")
    parser.add_argument("--comm", type=float, required=False, default=50, help="communication bandwidth (GB/s)")
    parser.add_argument("--batch", type=int, required=False, default=128, help="batch size")
    parser.add_argument("--mesh-shape", type=str, required=False, default="(4,8)", help="mesh size")
    parser.add_argument("--model", type=str, required=False, default="mixtral", help="model")
    
    args = parser.parse_args()
    layer_id=args.layer_id
    comp=args.comp
    BW=args.comm
    batch = args.batch
    print(f"computation throughput (TFLOPS): {comp:.2f}")
    print(f"communication bandwidth (GB/s): {BW:.2f}")
    if args.model=="mixtral":
        E,e,SE,h,IS,mlp_first,num_layers=8,2,0,4096,14336,False,32 #Mixtral
        data_path="/data/home/haochenhuang/deployment/evaluation/experts_reasoning_mixtral.json"
        rows_split, cols_split=(1,2)
    elif args.model=="ds":
        E,e,SE,h,IS,mlp_first,num_layers=64,6,0,2048,1408,True,26  #DeepSeekMoE 
        data_path='/data/home/haochenhuang/deployment/experts_reasoning_ds.json' 
        rows_split, cols_split=(2,4)
    #pdb.set_trace()
    mesh_shape = eval(args.mesh_shape)
    #mesh_shape=(8,8)
    
    D=mesh_shape[0]*mesh_shape[1]
    
    #data_path="/data/home/haochenhuang/deployment/evaluation/experts_reasoning_mixtral.json"
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("文件未找到，请检查文件路径和文件名。")
    optimizer = MoE3DPNMOptimizer(E=E,e=e, SE=SE,h=h,IS=IS,B=batch, D=D,BW=BW*1e9, comp=comp*1e12, num_layers=num_layers,mlp_first=mlp_first,routing_trace=data)
    folder_path = Path(f'/data/home/haochenhuang/deployment/results/comp_balance_only/reasoning_{args.model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_{optimizer.B:.0f}_batches')

    assert folder_path.exists() and folder_path.is_dir()
    P_ep=EP_deployment(optimizer.layer,optimizer.E,optimizer.D)
    #mesh_shape=(8,8)
    M_rand=generate_random_placement(optimizer.D, mesh_shape)
    #pdb.set_trace()

    
    P=np.ones((optimizer.layer,optimizer.E,optimizer.D))/optimizer.D
    
    
    file_path = f'/data/home/haochenhuang/deployment/results/comp_balance_only/reasoning_{args.model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_{optimizer.B:.0f}_batches/arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id}.npz'
    #np.savez_compressed(file_path, arr1=P, arr2=M_rand)
    print(f"results saved at "+file_path)
    
    P_tp=np.ones((optimizer.layer,optimizer.E,optimizer.D))/optimizer.D
    
    
    optimizer.X,optimizer.Y=mesh_shape
    #layer_id=1
    Z_tp=P_tp[layer_id]>0
    Z_ep=P_ep[layer_id]>0
    #M_rand=generate_random_placement(optimizer.D, mesh_shape)
    optimizer.M=M_rand
    ep_comp=optimizer.compute_time(P_ep)[layer_id]
    ep_comm,link=optimizer.comm_time_acc(M_rand,P_ep,layer_id)
    ep_comm*=2
    tp_comp=optimizer.compute_time(P_tp)[layer_id]
    tp_comm=2*optimizer.comm_time(P_tp)[layer_id]
    print(f"TP_communication: {tp_comm*1e6:.2f} us")
    print(f"TP_computation: {tp_comp*1e6:.2f} us")
    print(f"TP_latency: {(tp_comp+tp_comm)*1e6:.2f} us")
    print(f"EP_communication: {ep_comm*1e6:.2f} us")
    print(f"EP_computation: {ep_comp*1e6:.2f} us")
    print(f"EP_latency: {(ep_comp+ep_comm)*1e6:.2f} us")
    comp_map=np.zeros((optimizer.E))
    random_samples = random.sample(data[str(layer_id+optimizer.mlp_first)], batch)
    for sublist in random_samples:
        comp_map[sublist]+=2*optimizer.h*optimizer.IS
    tp_comp_dynamic=optimizer.compute_time_dynamic(P_tp,comp_map)[layer_id]
    tp_comm_dynamic=2*optimizer.comm_time_dynamic(P_tp,random_samples)[layer_id]


    ep_comp_dynamic=optimizer.compute_time_dynamic(P_ep,comp_map)[layer_id]
    ep_comm_dynamic=2*optimizer.comm_time_acc_dynamic(M_rand,P_ep,layer_id,random_samples)
    
    print(f"TP_communication_dynamic: {tp_comm_dynamic*1e6:.2f} us")
    print(f"TP_computation_dynamic: {tp_comp_dynamic*1e6:.2f} us")
    print(f"TP_latency_dynamic: {(tp_comp_dynamic+tp_comm_dynamic)*1e6:.2f} us")

    print(f"EP_communication_dynamic: {ep_comm_dynamic*1e6:.2f} us")
    print(f"EP_computation_dynamic: {ep_comp_dynamic*1e6:.2f} us")
    print(f"EP_latency_dynamic: {(ep_comp_dynamic+ep_comm_dynamic)*1e6:.2f} us")

    P_temp=optimizer.ilp_solver_gurobi_comp(l=layer_id,moe_model=args.model,time_limit=60)
    l1,e1,d1=P_temp.shape
    print("Comp Balance Finished.")
    repeats = D // d1
    P=np.repeat(P_temp, repeats=repeats, axis=2)/repeats
    M=np.zeros((optimizer.D,mesh_shape[0],mesh_shape[1]))

    d=random.randint(0, repeats - 1)
    places=get_subregion_coordinates(mesh_shape[0], mesh_shape[1], rows_split, cols_split, d)
    for place in range(len(places)):
        x,y=places[place]
        M[d+repeats*place,x,y]=1
    #pdb.set_trace()
    comp=optimizer.compute_time(P)[layer_id]
    comm_node,link=optimizer.comm_time_acc(M,P,layer_id)
    comm_node+=optimizer.comm_time(P)[layer_id]
    comm_node*=2
    print(f"comp_balancing_communication: {comm_node*1e6:.2f} us")
    print(f"comp_balancing_computation: {comp*1e6:.2f} us")
    print(f"comp_balancing_latency: {(comp+comm_node)*1e6:.2f} us")
    print(f"comp_balancing_speedup_EP:{(ep_comp+ep_comm)/(comp+comm_node):.2f}")
    print(f"comp_balancing_speedup_TP:{(tp_comp+tp_comm)/(comp+comm_node):.2f}")
    
    
    comp_dynamic=optimizer.compute_time_dynamic(P,comp_map)[layer_id]
    comm_node_dynamic=2*optimizer.comm_time_acc_dynamic(M,P,layer_id,random_samples)
    print(f"comp_balancing_communication_dynamic: {comm_node_dynamic*1e6:.2f} us")
    print(f"comp_balancing_computation_dynamic: {comp_dynamic*1e6:.2f} us")
    print(f"comp_balancing_latency_dynamic: {(comp_dynamic+comm_node_dynamic)*1e6:.2f} us")
    print(f"comp_balancing_speedup_EP_dynamic:{(ep_comp_dynamic+ep_comm_dynamic)/(comp_dynamic+comm_node_dynamic):.2f}")
    print(f"comp_balancing_speedup_TP_dynamic:{(tp_comp_dynamic+tp_comm_dynamic)/(comp_dynamic+comm_node_dynamic):.2f}")
 
    np.savez_compressed(file_path, arr1=P, arr2=M)


    
if __name__ == "__main__":
    main()