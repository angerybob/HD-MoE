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

H = 8
W = 16
rows_split = 1  # 行方向划分为2块
cols_split = 2  # 列方向划分为4块
i = 5          # 子区域内的索引

coords = get_subregion_coordinates(H, W, rows_split, cols_split, i)
print(f"2×4划分下，所有子区域第{i}个坐标：\n{coords}")