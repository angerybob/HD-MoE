import sys
sys.path.extend([
    "/data/home/haochenhuang/deployment/astra-sim/extern/graph_frontend",
    "/data/home/haochenhuang/deployment"
])
from chakra.schema.protobuf.et_def_pb2 import (
    Node, GlobalMetadata, AttributeProto,
    COMM_COLL_NODE, ALL_REDUCE, COMP_NODE,Sint32List,Uint64List
)
from chakra.src.third_party.utils.protolib import encodeMessage
import numpy as np

def generate_mesh_coordinates(mesh_dim):
    """生成8x8网格坐标"""
    return [(x, y) for x in range(mesh_dim[0]) for y in range(mesh_dim[1])]

def create_computation_node(npu_id, flops, device_tflops):
    """创建计算节点"""
    node = Node()
    node.id = npu_id + 1  # ID从1开始
    node.name = f"Comp_Node_{npu_id}"
    node.type = COMP_NODE
    node.duration_micros = int(flops / (device_tflops * 1e6))  # 转换为微秒
    node.attr.extend([
        AttributeProto(name="device_id", int64_val=npu_id),
        AttributeProto(name="flops", double_val=flops)
    ])
    return node

def create_allreduce_node(npu_id, mesh_coord, comm_group, tensor_size, bandwidth):
    """创建All-Reduce通信节点"""
    node = Node()
    node.id = 100 + npu_id  # 通信节点ID范围
    node.name = f"AllReduce_Node_{npu_id}"
    node.type = COMM_COLL_NODE
    #node.ctrl_deps.append(npu_id)  # 依赖计算节点
    
    # 通信时间计算（数据量/带宽）
    comm_time = (tensor_size * 1e9) / (bandwidth * 1e9) * 1e6  # 转换为微秒
    node.duration_micros = int(comm_time)
    
    # 拓扑属性配置
    node.attr.extend([
        AttributeProto(name="collective_type", int32_val=ALL_REDUCE),
        AttributeProto(
            name="mesh_dimension", 
            sint32_list=Sint32List(values=[8,8])
        ),
        AttributeProto(
            name="mesh_coordinate",
            sint32_list=Sint32List(values=list(mesh_coord))
        ),
        AttributeProto(
            name="communication_group",
            uint64_list=Uint64List(values=comm_group)
        ),
        AttributeProto(
            name="tensor_config", 
            bytes_val=bytes(f"{tensor_size}GB", 'utf-8')
        ),
        AttributeProto(
            name="routing_policy",
            string_val="all-path-routing"
        )
    ])
    return node

def main():
    # 系统参数配置
    mesh_dim = (8, 8)
    num_npus = np.prod(mesh_dim)
    tensor_size = 1  # GB
    bandwidth = 25   # GB/s
    device_tflops = 1.25  # TFLOP/s
    
    # 生成网格坐标和通信组
    mesh_coords = generate_mesh_coordinates(mesh_dim)
    comm_group = list(range(num_npus))  # 所有节点参与通信

    for npu_id in range(num_npus):
        filename = f"/data/home/haochenhuang/deployment/evaluation/workload/chakra_trace.{npu_id}.et"
        with open(filename, "wb") as et:
            # 元数据头
            encodeMessage(et, GlobalMetadata(version="0.0.4"))
            
            # 计算节点（示例：1TFLOP计算）
            comp_node = create_computation_node(
                npu_id, 
                flops=1e12,
                device_tflops=device_tflops
            )
            #encodeMessage(et, comp_node)
        #filename = f"/data/home/haochenhuang/deployment/evaluation/workload/chakra_trace.{npu_id}_comm.et"
        #with open(filename, "wb") as et:
            # 通信节点
            comm_node = create_allreduce_node(
                npu_id,
                mesh_coords[npu_id],
                comm_group,
                tensor_size,
                bandwidth
            )
            encodeMessage(et, comm_node)

if __name__ == "__main__":
    main()