import sys
sys.path.append("/data/home/haochenhuang/deployment/astra-sim/extern/graph_frontend")
sys.path.append("/data/home/haochenhuang/deployment")
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import (
    Node,
    BoolList,
    GlobalMetadata,
    AttributeProto,
    COMM_COLL_NODE,
    ALL_REDUCE,
    ALL_TO_ALL,
    COMP_NODE,
    COMM_SEND_NODE,
    COMM_RECV_NODE
)
from node_allocation import MoE3DPNMOptimizer
import numpy as np
import json
import pdb
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast
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
def _find_optimal_aggregator(layer_id,M):
    """为简化实现，默认选择中心节点作为聚合点"""
    # 实际实现应基于专家分布，这里简化为几何中心
    x_center = np.mean([d[0] for d in M])
    y_center = np.mean([d[1] for d in M])
    min_dist = float('inf')
    best_d = 0
    for d, (x, y) in enumerate(M):
        dist = abs(x - x_center) + abs(y - y_center)
        if dist < min_dist:
            min_dist = dist
            best_d = d
    return best_d

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

def main() -> None:
    # metadata
    npus_count = 64  # 8 NPUs
    num_experts = 64
    #num_devices = 64
    num_layers = 26
    #expert_flops=np.zeros((num_layers,num_experts))
    h=2048
    batch=50
    # 每个专家的计算量假设为1GFLOP
    device_tflops = 10  # 单设备算力假设为100 TFLOPS
    network_bandwidth = 25  # 网络带宽100GB/s
    
    try:
        with open('/data/home/haochenhuang/deployment/experts_math.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("文件未找到，请检查文件路径和文件名。")
    optimizer = MoE3DPNMOptimizer(E=64, h=2048, routing_trace=data)
    P=EP_deployment(optimizer.layer, optimizer.E, optimizer.D)
    expert_flops = 8*optimizer.f*batch*h**2 
    expert_output = optimizer.f*batch*h
    layer_id=1
    mesh_shape=(8,8)
    X,Y=mesh_shape
    M_rand=generate_random_placement(optimizer.D, mesh_shape)
    S=P[layer_id][:, np.newaxis, :].reshape(optimizer.E,X,Y)*M_rand
    M1=S.reshape(optimizer.E, -1)
    values = 32 * np.sum(expert_output[layer_id, :] * P[layer_id, :, :], axis=0)
    max_value = np.max(values)
    for npu_id in range(npus_count):
        x,y=np.nonzero(M_rand[npu_id])
        new_id=y*mesh_shape[1]+x
        #pdb.set_trace()
        output_filename = f"/data/home/haochenhuang/deployment/evaluation/workload/chakra_trace.{new_id[0]}.et"
        #pdb.set_trace()
        with open(output_filename, "wb") as et:
            # Chakra Metadata
            encode_message(et, GlobalMetadata(version="0.0.4"))

            # create Chakra Node
            duration = sum(expert_flops[layer_id,:] * P[layer_id,:,npu_id]) / (device_tflops * 1e12) * 1e6
            
            node = Node()
            node.id=1
            node.name = f"Device"
            node.type = COMP_NODE
            node.attr.extend([
                AttributeProto(name="assigned_device", int64_val=new_id[0]),
            ])
            node.duration_micros = int(duration)

            # store Chakra ET file
            encode_message(et, node)
            c_id=1
            expert_gropus=list(np.nonzero(P[layer_id, :, npu_id])[0])
            for sublist in optimizer.fg[optimizer.e][layer_id].keys():
                sublist=ast.literal_eval(sublist)
                if bool(set(sublist).intersection(set(expert_gropus))):
                    c_id+=1
                    non_zero_coords = []
                    for i in sublist:
                        # 找到当前二维矩阵中不为 0 的元素的坐标
                        #pdb.set_trace()
                        d_id,rows, cols = np.nonzero(M_rand[np.nonzero(P[1][i])])
                        
                        for x, y in zip(rows, cols):
                            if (x, y) not in non_zero_coords:
                                non_zero_coords.append((x, y))
                    x_center = int(np.mean([d[0] for d in non_zero_coords]))
                    y_center = int(np.mean([d[1] for d in non_zero_coords]))
                    dst=y_center*mesh_shape[1]+x_center
                    #pdb.set_trace()
                    if str(sorted(sublist)) in optimizer.fg[optimizer.e][layer_id]:
                        comm_v=32*optimizer.fg[optimizer.e][layer_id][str(sorted(sublist))]*batch*h
            
            
            # 添加AllReduce通信节点
                    coll_node = Node()
                    coll_node.id=c_id
                    coll_node.name=f"All2all_{c_id}"
                    coll_node.type=COMM_SEND_NODE
                    #coll_node.comm_type=ALL_REDUCE
                    #coll_node.data_deps.extend([node.id]) 
                    coll_node.attr.append(AttributeProto(name="is_cpu_op", bool_val=False))
                    #coll_node.attr.append(AttributeProto(name="comm_type", int64_val=ALL_TO_ALL))
                    #coll_node.attr.append(AttributeProto(name="comm_size", int64_val=int(40*sum(expert_flops[layer_id,:] * P[layer_id,:,npu_id])/(8*h))))
                    coll_node.attr.append(AttributeProto(name="comm_size", int64_val=int(comm_v)))
                    coll_node.attr.append(AttributeProto(name="dst", int64_val=int(dst)))
                    #pdb.set_trace()
                    '''coll_node.duration_micros = int(optimizer.calc_comm_time(
                        npu_id, 0, expert_output
                    ))'''
                    encode_message(et, coll_node)
if __name__ == "__main__":
    main()



