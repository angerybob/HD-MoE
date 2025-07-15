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


def main():
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser(description="节点分配工具")

    # 2. 添加参数
    parser.add_argument("--layer-id", type=int, required=False, default=1, help="layer_id")
    parser.add_argument("--comp", type=float, required=False, default=10, help="computation throughput (TFLOPS)")
    parser.add_argument("--comm", type=float, required=False, default=25, help="communication bandwidth (GB/s)")
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
    elif args.model=="ds":
        E,e,SE,h,IS,mlp_first,num_layers=64,6,0,2048,1408,True,26  #DeepSeekMoE 
        data_path='/data/home/haochenhuang/deployment/experts_reasoning_ds.json' 
    elif args.model=="qwen":
        E,e,SE,h,IS,mlp_first,num_layers=64,8,0,3584,2560,False,28 #Qwen2
        data_path="/data/home/haochenhuang/deployment/evaluation/experts_reasoning_qwen.json"
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
    folder_path = Path(f'/data/home/haochenhuang/deployment/results/reasoning_{args.model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_{optimizer.B:.0f}_batches')

    assert folder_path.exists() and folder_path.is_dir()
    P_ep=EP_deployment(optimizer.layer,optimizer.E,optimizer.D)
    #mesh_shape=(8,8)
    M_rand=generate_random_placement(optimizer.D, mesh_shape)
    #pdb.set_trace()
    x=[]
    y=[]
    print("Begin to sampling communication latency...")
    #ref = f'/data/home/haochenhuang/deployment/results/reasoning_{args.model}_10.0_TFLOPS_25.0_GBPS_for_8*8_mesh_128_batches/arrays_10.0_TFLOPS_25.0_GBPS_in_layer_{layer_id:.0f}.npz'
    #loaded_arrays = np.load(ref)

    # 访问加载的数组
    #P_init = loaded_arrays['arr1']
    P_init=P_ep
    #M = loaded_arrays['arr2']
    
    # regress
    
    for b in tqdm(range(256,512+128,2)):
        
        random_samples = random.sample(data[str(layer_id+optimizer.mlp_first)], b)
        for sublist in random_samples:
            sublist+=list(range(optimizer.E-optimizer.SE,optimizer.E))
        #pdb.set_trace()
        comm=optimizer.comm_time_dynamic(P_init,random_samples)[layer_id]
        comm_acc=optimizer.comm_time_acc_dynamic(M_rand,P_init,layer_id,random_samples)
        x.append(comm)
        y.append(comm_acc)
        #pdb.set_trace()
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f"The parameter gamma is {slope:.2f}")
    print(f"intercept is {intercept:.2f}, r_value is {r_value:.2f}, p_value is {p_value:.2f}, std_err is {std_err:.2f} ")
    
    
    
    #slope=5
    

    # 绘制图形
    plt.figure(figsize=(8, 3))

    # 1. 绘制原始散点
    plt.scatter(x, y, color='blue', label='Communication Samples', s=50, alpha=0.7)

    # 2. 绘制拟合直线
    x_fit = np.linspace(min(x), max(x), 100)  # 生成拟合直线的x值
    y_fit = slope * x_fit + intercept          # 计算对应的y值
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Regressed Line: y = {slope:.2f}x + {intercept:.2f}')

    # 3. 添加图例和标签
    plt.legend(fontsize=12)
    plt.xlabel('Node Communication', fontsize=14)
    plt.ylabel('Schedule Communication', fontsize=14)
    plt.title('Regress Results', fontsize=16)

    # 4. 添加网格和调整边距
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(f'/data/home/haochenhuang/deployment/communication2.png')
    plt.close()
    
    pdb.set_trace()
    

    
    P=np.ones((optimizer.layer,optimizer.E,optimizer.D))/optimizer.D
    
    
    file_path = f'/data/home/haochenhuang/deployment/results/reasoning_{args.model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_{optimizer.B:.0f}_batches/arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id}.npz'
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

    P=optimizer.ilp_solver_gurobi(l=layer_id,gamma=slope,time_limit=1800)
    print("Node Balance Finished.")
    comp=optimizer.compute_time(P)[layer_id]
    comm_node,link=optimizer.comm_time_acc(M_rand,P,layer_id)
    comm_node*=2
    print(f"node_balancing_communication: {comm_node*1e6:.2f} us")
    print(f"node_balancing_computation: {comp*1e6:.2f} us")
    print(f"node_balancing_latency: {(comp+comm_node)*1e6:.2f} us")
    print(f"node_balancing_speedup_EP:{(ep_comp+ep_comm)/(comp+comm_node):.2f}")
    print(f"node_balancing_speedup_TP:{(tp_comp+tp_comm)/(comp+comm_node):.2f}")
    
    
    comp_dynamic=optimizer.compute_time_dynamic(P,comp_map)[layer_id]
    comm_node_dynamic=2*optimizer.comm_time_acc_dynamic(M_rand,P,layer_id,random_samples)
    print(f"node_balancing_communication_dynamic: {comm_node_dynamic*1e6:.2f} us")
    print(f"node_balancing_computation_dynamic: {comp_dynamic*1e6:.2f} us")
    print(f"node_balancing_latency_dynamic: {(comp_dynamic+comm_node_dynamic)*1e6:.2f} us")
    print(f"node_balancing_speedup_EP_dynamic:{(ep_comp_dynamic+ep_comm_dynamic)/(comp_dynamic+comm_node_dynamic):.2f}")
    print(f"node_balancing_speedup_TP_dynamic:{(tp_comp_dynamic+tp_comm_dynamic)/(comp_dynamic+comm_node_dynamic):.2f}")
    #pdb.set_trace()
    Z=P>0
    #M_init=generate_random_placement(optimizer.D, mesh_shape)
    M_init=M_rand
    #M=optimizer.optimize_placement_with_gurobi(Z,mesh_shape,layer_id,time_limit=360)
    max_iter=70
    M, cost_history =optimizer.optimize_placement_bo(M_init,Z,layer_id,max_iter=max_iter)

    #file_path = f'/data/home/haochenhuang/deployment/results/{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{optimizer.B:.0f}_batches/arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id}.npz'
    
    np.savez_compressed(file_path, arr1=P, arr2=M)

    optimizer.M=M
    comm_link,link=optimizer.comm_time_acc(M,P,layer_id)
    comm_link*=2

    dis=optimizer.evaluate_placement(M,Z,layer_id)
    print(f"Total communication distance is {dis:.2f} nodes")
    #M_rand=generate_random_placement(optimizer.D, mesh_shape)
    dis_rand=optimizer.evaluate_placement(M_rand,Z,layer_id)
    print(f"Total communication distance of random mapping is {dis_rand:.2f} nodes")

    print(f"link_balancing: {comm_link*1e6:.2f} us")
    print(f"link_balancing_computation: {comp*1e6:.2f} us")
    print(f"link_balancing_latency: {(comp+comm_link)*1e6:.2f} us")
    print(f"link_balancing_speedup:{(comm_node)/(comm_link):.2f}")
    print(f"node_link_balancing_speedup:{(ep_comp+ep_comm)/(comp+comm_link):.2f}")
    #S_tp=P_tp[layer_id][:, np.newaxis, :]*M_rand
    #S_ep=P_ep[layer_id][:, np.newaxis, :]*M_rand
    #dis_rand=optimizer.evaluate_placement(M_rand,Z_tp,layer_id)
    comm_link_dynamic=2*optimizer.comm_time_acc_dynamic(M,P,layer_id,random_samples)
    print(f"link_balancing_dynamic: {comm_link_dynamic*1e6:.2f} us")
    print(f"link_balancing_computation_dynamic: {comp_dynamic*1e6:.2f} us")
    print(f"link_balancing_latency_dynamic: {(comp_dynamic+comm_link_dynamic)*1e6:.2f} us")
    print(f"link_balancing_speedup_dynamic:{(comm_node_dynamic)/(comm_link_dynamic):.2f}")
    print(f"node_link_balancing_speedup_EP_dynamic:{(ep_comp_dynamic+ep_comm_dynamic)/(comp_dynamic+comm_link_dynamic):.2f}")
    print(f"node_link_balancing_speedup_TP_dynamic:{(tp_comp_dynamic+tp_comm_dynamic)/(comp_dynamic+comm_link_dynamic):.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, color='blue', linewidth=1)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Schedule Time", fontsize=12)
    plt.title("Bayesian Optimization", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'/data/home/haochenhuang/deployment/results/reasoning_{args.model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_{optimizer.B:.0f}_batches/BO_{max_iter}_in_layer_{layer_id}.png')
    plt.close()
    
if __name__ == "__main__":
    main()