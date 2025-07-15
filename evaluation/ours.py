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
)
from node_allocation import MoE3DPNMOptimizer
import numpy as np
import json
import pdb
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast
import argparse



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


def main() -> None:
    parser = argparse.ArgumentParser(description="节点分配工具")

    # 2. 添加参数
    parser.add_argument("--comp", type=float, required=False, default=10, help="computation throughput (TFLOPS)")
    parser.add_argument("--comm", type=float, required=False, default=25, help="communication bandwidth (GB/s)")
    parser.add_argument("--batch", type=int, required=False, default=128, help="batch size")
    args = parser.parse_args()
    
    # metadata
    npus_count = 64  # 8 NPUs
    num_experts = 64
    #num_devices = 64
    num_layers = 26
    #expert_flops=np.zeros((num_layers,num_experts))
    h=2048
    batch=args.batch
    # 每个专家的计算量假设为1GFLOP
    device_tflops = args.comp  # 单设备算力假设为100 TFLOPS
    network_bandwidth = args.comm  # 网络带宽100GB/s
    #layer_id=1
    try:
        with open('/data/home/haochenhuang/deployment/experts_math.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("文件未找到，请检查文件路径和文件名。")
    optimizer = MoE3DPNMOptimizer(E=num_experts, h=h, BW=network_bandwidth*1e9, comp=device_tflops*1e12,routing_trace=data)
    #P=optimizer.ilp_solver_gurobi(l=layer_id,time_limit=40)
    #file_path = f'/data/home/haochenhuang/deployment/evaluation/arrays_{device_tflops}_TFLOPS_{network_bandwidth}_GBPS_in_layer_{layer_id}.npz'
    #np.savez_compressed(file_path, arr1=P)
    P=np.zeros((optimizer.layer,optimizer.E,optimizer.D))
    for layer_id in range(optimizer.layer):
        file_path = f'/data/home/haochenhuang/deployment/results/{device_tflops:.0f}TFLOPS_{network_bandwidth:.0f}GBPS/arrays_{device_tflops:.0f}_TFLOPS_{network_bandwidth:.0f}_GBPS_in_layer_{layer_id}.npz'

        loaded_arrays = np.load(file_path)

        # 访问加载的数组
        P += loaded_arrays['arr1']
        
    comp_map=np.zeros((optimizer.layer,optimizer.E))
    random.seed(123)
    sample_id=random.sample(list(range(len(data["1"]))),batch)
    comm_map=np.zeros((optimizer.layer,optimizer.D))
    for l in range(optimizer.layer):
        for i in sample_id:
            sublist=data[str(l+1)][i]
            comp_map[l][sublist]+=8*optimizer.h**2
            e,d=np.nonzero(P[l,sublist])
            comm_map[l][d]+=4*optimizer.h
            
         
    expert_flops = comp_map
 


    max_value = 2*np.sum(np.max(comm_map, axis=1))
    print(max_value/1e9)
    #pdb.set_trace()
    for npu_id in range(npus_count):
        output_filename = f"/data/home/haochenhuang/deployment/evaluation/workload/chakra_trace.{npu_id}.et"
        with open(output_filename, "wb") as et:
            # Chakra Metadata
            encode_message(et, GlobalMetadata(version="0.0.4"))

            duration=0
            # create Chakra Node
            for l in range(optimizer.layer):
                duration += sum(expert_flops[l,:] * P[l,:,npu_id]) / (device_tflops * 1e12) * 1e6
            
            node = Node()
            node.id=1
            node.name = f"Device"
            node.type = COMP_NODE
            node.attr.extend([
                AttributeProto(name="assigned_device", int64_val=npu_id),
            ])
            node.duration_micros = int(duration)

            # store Chakra ET file
            encode_message(et, node)
            # 添加AllReduce通信节点
            coll_node = Node()
            coll_node.id=2
            coll_node.name=f"All2all"
            coll_node.type=COMM_COLL_NODE
            #coll_node.comm_type=ALL_REDUCE
            coll_node.data_deps.extend([node.id]) 
            coll_node.attr.append(AttributeProto(name="is_cpu_op", bool_val=False))
            coll_node.attr.append(AttributeProto(name="comm_type", int64_val=ALL_TO_ALL))
            #coll_node.attr.append(AttributeProto(name="comm_size", int64_val=int(40*sum(expert_flops[layer_id,:] * P[layer_id,:,npu_id])/(8*h))))
            coll_node.attr.append(AttributeProto(name="comm_size", int64_val=int(max_value)))
            #pdb.set_trace()
            '''coll_node.duration_micros = int(optimizer.calc_comm_time(
                npu_id, 0, expert_output
            ))'''
            encode_message(et, coll_node)
if __name__ == "__main__":
    main()



