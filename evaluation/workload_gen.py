import sys
sys.path.append("/data/home/haochenhuang/deployment/astra-sim/extern/graph_frontend")
sys.path.append("/data/home/haochenhuang/deployment") 
from chakra.schema.protobuf.et_def_pb2 import Node, GlobalMetadata, AttributeProto, COMM_COLL_NODE, COMP_NODE, COMM_NODE, COMM_SEND_NODE, COMM_RECV_NODE, ALL_REDUCE, Int64List
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from node_allocation import MoE3DPNMOptimizer
import numpy as np
import json
import pdb
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast

num_experts = 64
num_devices = 64
num_layers = 26
#expert_flops=np.zeros((num_layers,num_experts))
h=2048
batch=50
 # 每个专家的计算量假设为1GFLOP
device_tflops = 1.25  # 单设备算力假设为100 TFLOPS
network_bandwidth = 25  # 网络带宽100GB/s

try:
    with open('/data/home/haochenhuang/deployment/experts_math.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("文件未找到，请检查文件路径和文件名。")
optimizer = MoE3DPNMOptimizer(E=64, h=2048, routing_trace=data)
expert_flops = optimizer.f*batch*h**2 
expert_output = optimizer.f*batch*h

def create_expert_task(expert_id, device_id, layer_id, lambda_val):
    # 计算持续时间（微秒）= (计算量 * λ) / (设备算力 * 1e6)
    duration = int((expert_flops[layer_id,expert_id] * lambda_val) / (device_tflops * 1e12) * 1e6)
    
    node = Node()
    node.name = f"Expert_{expert_id}_on_Device_{device_id}"
    node.type = COMP_NODE
    node.attr.extend([
        AttributeProto(name="expert_id", int64_val=expert_id),
        AttributeProto(name="assigned_device", int64_val=device_id),
        AttributeProto(name="compute_portion", double_val=lambda_val)
    ])
    node.duration_micros = duration
    return node

def create_expert_aggregation(expert_id, layer_id, devices):
    # 计算通信数据量（假设每个专家输出1GB）
    comm_size = expert_output[layer_id,expert_id]  # 1GB
    node = Node()
    node.name = f"AllReduce_Expert_{expert_id}"
    node.type = COMM_COLL_NODE
    node.attr.extend([
        AttributeProto(name="comm_type", int64_val=ALL_REDUCE),
        AttributeProto(name="participants", int64_list=Int64List(values=devices)),
        AttributeProto(name="data_size", int64_val=comm_size)
    ])
    
    # 通信时间 = 数据量/(带宽*1e6) 转换为微秒
    node.duration_micros = int(comm_size / (network_bandwidth * 1e9) * 1e6)
    return node

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

def select_aggregator(group):
    # 获取组内所有设备的位置
    devices = [expert_location[e] for e in group]
    
    # 计算最小包围矩形 (MBR)
    min_x = min(d[0] for d in devices)
    max_x = max(d[0] for d in devices)
    min_y = min(d[1] for d in devices)
    max_y = max(d[1] for d in devices)
    
    # 几何中心 (整数坐标)
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    
    # 找到离中心最近的设备作为聚合节点
    distances = [
        (abs(x - center_x) + abs(y - center_y), (x, y))
        for (x, y) in devices
    ]
    _, (agg_x, agg_y) = min(distances)
    
    return agg_x, agg_y

def route_path(src, dst):
    path = []
    x, y = src
    dx, dy = dst
    
    # X方向移动
    while x != dx:
        if x < dx:
            x += 1
        else:
            x -= 1
        path.append((x, y))
    
    # Y方向移动
    while y != dy:
        if y < dy:
            y += 1
        else:
            y -= 1
        path.append((x, y))
    
    return path
def create_comm_send_node(src_device: int, dst_device: int, bytes: int) -> Node:
    node = Node()
    node.type = COMM_SEND_NODE  # 关键：设置节点类型为发送
    node.attr.extend([
        AttributeProto(name="src", int64_val=src_device),
        AttributeProto(name="dst", int64_val=dst_device),
        AttributeProto(name="bytes", int64_val=bytes)
    ])
    return node

def create_comm_recv_node(src_device: int, dst_device: int, bytes: int) -> Node:
    node = Node()
    node.type = COMM_RECV_NODE  # 关键：设置节点类型为接收
    node.attr.extend([
        AttributeProto(name="src", int64_val=src_device),
        AttributeProto(name="dst", int64_val=dst_device),
        AttributeProto(name="bytes", int64_val=bytes)
    ])
    return node
'''
def generate_group_comm(group, frequency,optimizer):
    # 步骤1: 确定聚合点
    agg_x, agg_y = select_aggregator(group)
    #agg_device = [d for d in range(D) if M[d, agg_x, agg_y] == 1][0]
    
    # 步骤2: 生成通信节点
    comm_nodes = []
    for e in group:
        src_x, src_y = expert_location[e]
        src_device = [d for d in range(optimizer.D) if M[d, src_x, src_y] == 1][0]
        
        # 生成从src到agg的路径
        path = route_path((src_x, src_y), (agg_x, agg_y))
        for (x, y) in path:
            next_hop = [d for d in range(optimizer.D) if M[d, x, y] == 1][0]
            node = Node(
                type=COMM_NODE,
                src_device=src_device,
                dst_device=next_hop,
                bytes=expert_output_size * frequency  # 按频率加权
            )
            comm_nodes.append(node)
    
    return comm_nodes
'''
def generate_mesh_comm(src_x: int, src_y: int, dst_x: int, dst_y: int, bytes: int):
    # 获取路径上的所有跳
    path = route_path((src_x, src_y), (dst_x, dst_y))
    
    # 逐跳生成通信节点
    prev_node_id = None
    for (x, y) in path:
        current_device = get_device_at(x, y)  # 根据映射表M查找设备ID
        next_hop_device = get_device_at(x, y)  # 实际应为下一跳设备
        
        # 当前跳的发送节点
        send_node = create_comm_send_node(
            src_device=current_device,
            dst_device=next_hop_device,
            bytes=bytes
        )
        if prev_node_id is not None:
            send_node.data_deps.append(prev_node_id)
        
        # 下一跳的接收节点
        recv_node = create_comm_recv_node(
            src_device=current_device,
            dst_device=next_hop_device,
            bytes=bytes
        )
        recv_node.data_deps.append(send_node.id)
        
        prev_node_id = recv_node.id
para_strategy = optimizer.prune_search_space()
if para_strategy == 'TP':
    P=np.ones((optimizer.layer,optimizer.E,optimizer.D))/optimizer.D
else:
    P = optimizer.ilp_solver_gurobi(layer_id=1,time_limit=40)
if P is not None:
    layer_id=1
    mesh_shape=(8,8)
    Z=P[layer_id]>0
    M_init=generate_random_placement(optimizer.D, mesh_shape)
    #M=optimizer.optimize_placement_with_gurobi(Z,mesh_shape,layer_id,time_limit=360)
    M, cost_history =optimizer.optimize_placement_sa(M_init,Z,layer_id)
    #pdb.set_trace()
    dis=optimizer.evaluate_placement(M,Z,layer_id)
    print(f"Total communication distance is {dis:.2f} nodes")
    expert_location = {}
    X,Y=mesh_shape
    for e in range(optimizer.E):
        for d in range(optimizer.D):
            for x in range(X):
                for y in range(Y):
                    if M[d, x, y] == 1 and P[layer_id,e,d]>0:
                        if e in expert_location.keys():
                            expert_location[e].append((x,y))
                        else:
                            expert_location[e]=[(x,y)]

# 生成所有计算节点
compute_nodes = []
for l in range(num_layers):
    layer_nodes=[]
    for expert in range(num_experts):
        for device in range(num_devices):
            if P[expert, device] > 0:
                node = create_expert_task(expert, device, l,P[expert, device])
                layer_nodes.append(node)
    compute_nodes.append(layer_nodes)

all_comm_nodes = []
for group, freq in optimizer.fg[optimizer.e][layer_id].items():
    group=ast.literal_eval(group)
    nodes = generate_group_comm(group, freq,optimizer)
    all_comm_nodes.extend(nodes)

# 合并同一链路的通信量
merged_comm = defaultdict(int)
for node in all_comm_nodes:
    key = (node.src_device, node.dst_device)
    merged_comm[key] += node.bytes

# 生成最终通信节点
final_comm_nodes = [
    Node(
        type=COMM_NODE,
        src_device=src,
        dst_device=dst,
        bytes=total_bytes
    ) for (src, dst), total_bytes in merged_comm.items()
]
            
# 建立计算->聚合的依赖
for agg_node in final_comm_nodes:
    expert_id = int(agg_node.name.split('_')[2])
    dependent_devices = [d for d in range(num_devices) if P[expert_id, d] > 0]
    
    # 找到所有前置计算节点
    predecessors = [n.id for n in compute_nodes 
                    if f"Expert_{expert_id}_on_Device" in n.name]
    
    # 设置依赖关系
    agg_node.data_deps.extend(predecessors)
    
for device_id in range(num_devices):
    with open(f"device_{device_id}.et", "wb") as f:
        # 设备专属任务过滤
        device_tasks = [n for n in compute_nodes 
                       if n.attr_dict["assigned_device"] == device_id]
        
        # 相关聚合任务
        related_aggregation = [a for a in final_comm_nodes
                             if device_id in a.attr_dict["participants"]]
        
        # 写入文件
        encode_message(f, GlobalMetadata(version="0.0.4"))
        for task in device_tasks + related_aggregation:
            encode_message(f, task)