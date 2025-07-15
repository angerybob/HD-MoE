def generate_mesh_topology(rows=8, cols=8, bandwidth="200Gbps", delay="0.0001ms", loss="0", output_file="/data/home/haochenhuang/deployment/evaluation/mesh_topology.txt"):
    """
    生成任意规模的2D Mesh拓扑文件
    
    参数:
        rows (int): 行数（默认8）
        cols (int): 列数（默认8）
        bandwidth (str): 链路带宽（默认"200Gbps"）
        delay (str): 链路延迟（默认"0.0001ms"）
        loss (str): 丢包率（默认"0"）
        output_file (str): 输出文件名（默认"mesh_topology.txt"）
    """
    # 参数校验
    assert rows >= 2 and cols >= 2, "行数和列数必须≥2"
    assert bandwidth.endswith(("Gbps", "Mbps")), "带宽单位需为Gbps或Mbps"
    assert delay.endswith(("ms", "us", "ns")), "延迟单位需为ms/us/ns"
    
    num_nodes = rows * cols
    horizontal_links = rows * (cols - 1) * 2  # 每行(cols-1)条水平连接×双向
    vertical_links = cols * (rows - 1) * 2   # 每列(rows-1)条垂直连接×双向
    total_links = horizontal_links + vertical_links
    
    with open(output_file, 'w') as f:
        # 文件头
        f.write(f"{num_nodes} 0 {total_links}\n\n")
        
        # 水平链路生成
        #f.write(f"# ------------------- 水平链路（{horizontal_links}条） -------------------\n")
        for row in range(rows):
            start_node = row * cols
            #f.write(f"# Row {row} (节点{start_node}-{start_node + cols - 1})\n")
            for col in range(cols - 1):
                node_a = start_node + col
                node_b = node_a + 1
                f.write(f"{node_a} {node_b} {bandwidth} {delay} {loss}\n")
                f.write(f"{node_b} {node_a} {bandwidth} {delay} {loss}\n")
            #f.write("\n")
        
        # 垂直链路生成
        #f.write(f"# ------------------- 垂直链路（{vertical_links}条） -------------------\n")
        for col in range(cols):
            #f.write(f"# Column {col} (节点{col},{col + cols},...,{col + (rows - 1)*cols})\n")
            for row in range(rows - 1):
                node_a = col + row * cols
                node_b = node_a + cols
                f.write(f"{node_a} {node_b} {bandwidth} {delay} {loss}\n")
                f.write(f"{node_b} {node_a} {bandwidth} {delay} {loss}\n")
            #f.write("\n")
    
    print(f"已生成 {rows}x{cols} Mesh拓扑：\n"
          f"- 节点总数: {num_nodes}\n"
          f"- 总链路数: {total_links} (水平{horizontal_links} + 垂直{vertical_links})\n"
          f"- 带宽: {bandwidth}\n"
          f"- 延迟: {delay}\n"
          f"- 输出文件: {output_file}")
def generate_switched_mesh_topology(rows=8, cols=8, bandwidth="200Gbps", delay="0.0001ms", loss="0", output_file="/data/home/haochenhuang/deployment/evaluation/mesh_topology.txt"):
    """
    生成通过交换机连接的2D Mesh拓扑
    
    参数:
        rows (int): 行数（默认8）
        cols (int): 列数（默认8）
        bandwidth (str): 链路带宽（默认"200Gbps"）
        delay (str): 链路延迟（默认"0.0001ms"）
        loss (str): 丢包率（默认"0"）
        output_file (str): 输出文件名（默认"switched_mesh.txt"）
    """
    # 参数校验
    assert rows >= 2 and cols >= 2, "行数和列数必须≥2"
    assert bandwidth.endswith(("Gbps", "Mbps")), "带宽单位需为Gbps或Mbps"
    assert delay.endswith(("ms", "us", "ns")), "延迟单位需为ms/us/ns"
    
    num_nodes = rows * cols
    # 计算交换机数量：水平方向每行(cols-1)个，垂直方向每列(rows-1)个
    horizontal_switches = rows * (cols - 1)
    vertical_switches = cols * (rows - 1)
    total_switches = horizontal_switches + vertical_switches
    
    # 总链路数：每个交换机连接2个节点 → 总链路数 = 交换机数 × 2
    total_links = total_switches * 2
    
    with open(output_file, 'w') as f:
        # 文件头：节点数 交换机数 链路数
        f.write(f"{num_nodes} {total_switches} {total_links}\n")
        
        # 生成交换机ID列表（从num_nodes开始）
        switch_ids = list(range(num_nodes, num_nodes + total_switches))
        f.write(" ".join(map(str, switch_ids)) + "\n")
        
        # 生成水平链路（节点 ↔ 水平交换机）
        switch_counter = num_nodes  # 当前交换机ID
        for row in range(rows):
            for col in range(cols - 1):
                node_a = row * cols + col
                node_b = node_a + 1
                # 节点A连接交换机
                f.write(f"{node_a} {switch_counter} {bandwidth} {delay} {loss}\n")
                # 节点B连接交换机
                f.write(f"{node_b} {switch_counter} {bandwidth} {delay} {loss}\n")
                switch_counter += 1
        
        # 生成垂直链路（节点 ↔ 垂直交换机）
        for col in range(cols):
            for row in range(rows - 1):
                node_a = row * cols + col
                node_b = node_a + cols
                # 节点A连接交换机
                f.write(f"{node_a} {switch_counter} {bandwidth} {delay} {loss}\n")
                # 节点B连接交换机
                f.write(f"{node_b} {switch_counter} {bandwidth} {delay} {loss}\n")
                switch_counter += 1
    
    print(f"已生成 {rows}x{cols} 交换机式Mesh拓扑：\n"
          f"- 节点总数: {num_nodes}\n"
          f"- 交换机数: {total_switches}\n"
          f"- 总链路数: {total_links}\n"
          f"- 输出文件: {output_file}")


    
# 使用示例
if __name__ == "__main__":
    # 示例1：生成8x8默认拓扑
    generate_mesh_topology()
    #generate_switched_mesh_topology()
    
    # 示例2：生成4x4 400Gbps拓扑
    # generate_mesh_topology(rows=4, cols=4, bandwidth="400Gbps", output_file="4x4_mesh.txt")
    
    # 示例3：生成5x3 100Gbps+1us延迟拓扑
    # generate_mesh_topology(rows=5, cols=3, bandwidth="100Gbps", delay="1us", output_file="5x3_mesh.txt")