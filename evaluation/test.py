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


def main() -> None:
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser(description="节点分配工具")

    # 2. 添加参数
    parser.add_argument("--comp", type=float, required=False, default=10, help="computation throughput (TFLOPS)")
    parser.add_argument("--comm", type=float, required=False, default=25, help="communication bandwidth (GB/s)")
    parser.add_argument("--batch", type=int, required=False, default=128, help="batch size")
    args = parser.parse_args()
    #layer_id=args.layer_id
    
    # metadata
    npus_count = 64  # 8 NPUs
    num_experts = 64
    #num_devices = 64
    num_layers = 26
    #expert_flops=np.zeros((num_layers,num_experts))
    h=2048
    batch=50
    # 每个专家的计算量假设为1GFLOP
    device_tflops = args.comp # 单设备算力假设为100 TFLOPS
    network_bandwidth = args.comm  # 网络带宽100GB/s
    
    try:
        with open('/data/home/haochenhuang/deployment/experts_math.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("文件未找到，请检查文件路径和文件名。")
    optimizer = MoE3DPNMOptimizer(E=64, h=2048, B=batch, BW=network_bandwidth*1e9, comp=device_tflops*1e12, routing_trace=data)
    P=np.ones((optimizer.layer,optimizer.E,optimizer.D))/optimizer.D
    
    
    layer_id=1
    
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
    expert_output = optimizer.f*batch*h
    max_value = 2*np.sum(np.max(comm_map, axis=1))
    print(max_value/1e9)
    for npu_id in range(npus_count):
        output_filename = f"/data/home/haochenhuang/deployment/evaluation/workload/chakra_trace.{npu_id}.et"
        with open(output_filename, "wb") as et:
            # Chakra Metadata
            encode_message(et, GlobalMetadata(version="0.0.4"))
            duration = 0
            comm_v=0
            # create Chakra Node
            for l in range(optimizer.layer):
                duration += sum(expert_flops[l,:] * P[l,:,npu_id]) / (device_tflops * 1e12) * 1e6
                comm_v += 32*sum(expert_output[l,:] * P[l,:,npu_id])
            
            node = Node()
            node.name = f"Device"
            node.id=1
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
            coll_node.name=f"AllReduce"
            coll_node.type=COMM_COLL_NODE
            #coll_node.comm_type=ALL_REDUCE
            coll_node.data_deps.extend([node.id]) 
            
            coll_node.attr.append(AttributeProto(name="comm_type", int64_val=ALL_REDUCE))
            coll_node.attr.append(AttributeProto(name="comm_size", int64_val=int(max_value)))
            #pdb.set_trace()
            
            '''coll_node.duration_micros = int(optimizer.calc_comm_time(
                npu_id, 0, expert_output
            ))'''
            encode_message(et, coll_node)
if __name__ == "__main__":
    main()



