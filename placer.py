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

try:
    with open('/data/home/haochenhuang/deployment/experts_math_ds.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("文件未找到，请检查文件路径和文件名。")
optimizer = MoE3DPNMOptimizer(E=64, h=2048, routing_trace=data)

# 步骤1: 剪枝搜索空间
#P=np.zeros((optimizer.layer,optimizer.E,optimizer.D))

para_strategy = optimizer.prune_search_space()
if para_strategy == 'TP':
    P=np.ones((optimizer.layer,optimizer.E,optimizer.D))/optimizer.D
else:
    #print(f"Valid expert groups after pruning: {valid_groups[0][:3]}...")
    #pdb.set_trace()
    # 步骤2: 求解最优放置
    #P = optimizer.ilp_solver()
    #print(f"Expert placement matrix:\n{P}")
# 使用Gurobi求解
    P = optimizer.ilp_solver_gurobi(layer_id=1,time_limit=40)

if P is not None:
    layer_id=1
    print("Optimal placement matrix:")
    #np.set_printoptions(threshold=np.inf)
    #print(P[layer_id])
    # 评估性能
    
    t_comp = optimizer.compute_time(P)[layer_id]
    t_comm = optimizer.comm_time(P)[layer_id]
    print(f"Total latency: {(t_comp + t_comm)*1e6:.2f} μs")
# 步骤3: 性能评估

    print(f"Computation Time: {t_comp*1e6:.2f} μs")
    print(f"Communication Time: {t_comm*1e6:.2f} μs")
    P_tp=np.ones((optimizer.layer,optimizer.E,optimizer.D))/optimizer.D
    tp_comp=optimizer.compute_time(P_tp)[layer_id]
    tp_comm=optimizer.comm_time(P_tp)[layer_id]
    print(f"TP latency: {(tp_comp+tp_comm)*1e6:.2f} μs = {tp_comp*1e6:.2f}(comp) μs + {tp_comm*1e6:.2f}(comm) μs")
    P_ep=EP_deployment(optimizer.layer,optimizer.E,optimizer.D)
    ep_comp=optimizer.compute_time(P_ep)[layer_id]
    ep_comm=optimizer.comm_time(P_ep)[layer_id]
    print(f"EP latency: {(ep_comp+ep_comm)*1e6:.2f} μs = {ep_comp*1e6:.2f}(comp) μs + {ep_comm*1e6:.2f}(comm) μs")
    #np.set_printoptions(threshold=np.inf)
    #print(P_ep[layer_id])

    mesh_shape=(8,8)
    Z=P[layer_id]>0
    M_init=generate_random_placement(optimizer.D, mesh_shape)
    #M=optimizer.optimize_placement_with_gurobi(Z,mesh_shape,layer_id,time_limit=360)
    M, cost_history =optimizer.optimize_placement_sa(M_init,Z,layer_id)
    #pdb.set_trace()
    dis=optimizer.evaluate_placement(M,Z,layer_id)
    print(f"Total communication distance is {dis:.2f} nodes")
    M_rand=generate_random_placement(optimizer.D, mesh_shape)
    dis_rand=optimizer.evaluate_placement(M_rand,Z,layer_id)
    print(f"Total communication distance of random mapping is {dis_rand:.2f} nodes")
    print("Evaluating random baselines...")
    random_scores = []
    num_trials = 20
    for _ in tqdm(range(num_trials)):
        random_placement = generate_random_placement(optimizer.D, mesh_shape)
        score = optimizer.evaluate_placement(random_placement, Z,layer_id)
        random_scores.append(score)
    
    # 结果分析
    print(f"\n{' Metric ':~^40}")
    print(f"随机布局平均得分({num_trials}次): {np.mean(random_scores):.2f} ± {np.std(random_scores):.2f}")
    #print(f"优化提升幅度: {(np.mean(random_scores)-ilp_score)/np.mean(random_scores)*100:.1f}%")

    # 扩展分析
    sorted_random = sorted(random_scores)
    print("\n随机布局得分分布：")
    print(f"最佳随机: {sorted_random[0]:.2f} nodes")
    #pdb.set_trace()
    print(f"25百分位: {sorted_random[int(num_trials*0.25)]:.2f} nodes")
    print(f"中位数: {sorted_random[int(num_trials*0.5)]:.2f} nodes")
    print(f"75百分位: {sorted_random[int(num_trials*0.75)]:.2f} nodes")
    print(f"最差随机: {sorted_random[-1]:.2f} nodes")

    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, color='blue', linewidth=1)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Evaluation Metric (Weighted MST Distance)", fontsize=12)
    plt.title("Simulated Annealing Convergence", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('Simulated Annealing Convergence_5e3.png')
    plt.close()
'''
# 步骤4: 动态调度
priorities = optimizer.priority_detection(P)
print(f"Top 3 migration priorities: {priorities[:3]}")

c, t_broadcast = optimizer.optimal_broadcast_chunk()
print(f"Optimal broadcast chunk size: {c/1e6:.2f} MB, Time: {t_broadcast*1e6:.2f} μs")'''