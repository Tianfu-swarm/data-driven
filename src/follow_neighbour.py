import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def generate_bat_positions_point_processed_uniform(num_bats=100, x_range=(0, 1), y_range=(0, 1)):
    # 生成均匀分布的坐标
    x_coords = np.random.uniform(x_range[0], x_range[1], num_bats)
    y_coords = np.random.uniform(y_range[0], y_range[1], num_bats)

    # 编号和坐标打包
    bats = [(i + 1, (x, y)) for i, (x, y) in enumerate(zip(x_coords, y_coords))]

    return bats

def generate_bat_positions(rows=10, cols=10):
    bats = []
    for i in range(rows):
        for j in range(cols):
            bat_number = i * cols + j + 1  # 蝙蝠编号，从 1 开始
            bats.append((bat_number, (i + 0.5, j + 0.5)))  # 记录编号和坐标

    return bats

def generate_follow_target(bats, range_r=1.0):
    bat_follow_target = {}

    # 将蝙蝠坐标映射到编号的字典
    position_map = {tuple(pos): bat_id for bat_id, pos in bats}

    # 遍历每个蝙蝠
    for bat_id, (x, y) in bats:
        neighbors = []  # 存储邻居蝙蝠编号

        # 遍历所有可能的邻居
        for neighbor_pos, neighbor_id in position_map.items():
            if neighbor_id == bat_id:
                continue  # 跳过自己

            neighbor_x, neighbor_y = neighbor_pos
            distance = math.sqrt((neighbor_x - x) ** 2 + (neighbor_y - y) ** 2)

            # 如果距离在范围内，添加到邻居列表
            if distance <= range_r:
                neighbors.append(neighbor_id)

        # 如果有邻居，随机选择一个跟随
        if neighbors:
            bat_follow_target[bat_id] = random.choice(neighbors)
        else:
            bat_follow_target[bat_id] = None  # 如果没有邻居，则不跟随

    return bat_follow_target


def calculate_subgroup_positions_size(bat_follow_target, bats):

    # 将蝙蝠编号和坐标映射到字典
    position_map = {bat_id: pos for bat_id, pos in bats}

    # 创建图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(position_map.keys())

    # 根据跟随关系添加边
    for bat_id, follow_target in bat_follow_target.items():
        if follow_target is not None:
            G.add_edge(bat_id, follow_target)

    # 获取所有子group（连通分量）
    connected_components = list(nx.connected_components(G))

    group_size_position = {}

    # 遍历每个子group
    for group_id, component in enumerate(connected_components, start=1):
        group_size = len(component)
        total_x = total_y = 0

        # 计算该子group的平均位置
        for bat_id in component:
            x, y = position_map[bat_id]
            total_x += x
            total_y += y

        avg_x = total_x / group_size
        avg_y = total_y / group_size

        # 存储该子group的大小和位置
        group_size_position[group_id] = (group_size, (avg_x, avg_y))

    return group_size_position

def plot_bat_follow_graph(bats, bat_follow_target):
    # 创建有向图
    G = nx.DiGraph()
    positions = {}  # 存储节点位置

    # 添加节点（bat）
    for bat_number, position in bats:
        G.add_node(bat_number)  # 添加节点
        positions[bat_number] = position  # 记录位置

    # 添加边（bat_follow_target）
    for bat_id, follow_target in bat_follow_target.items():
        if follow_target is not None:  # 跟随目标有效时添加边
            G.add_edge(bat_id, follow_target)

    # 绘制图
    plt.figure(figsize=(8, 6))

    # 绘制节点和边，使用实际位置
    nx.draw(G, pos=positions, with_labels=True, node_color='skyblue', node_size=200, font_size=12, font_weight='bold',
            arrowsize=20)

    # 显示图
    plt.title("Bat Follow Relationship Graph")
    plt.show()


def plot_group_position(group_size_position):
    group_positions = []
    sizes = []

    for group_id, (size, group_pos) in group_size_position.items():
        group_positions.append(group_pos)
        sizes.append(size)

    group_positions = np.array(group_positions)
    sizes = np.array(sizes)

    # 创建颜色映射
    norm = plt.Normalize(vmin=sizes.min(), vmax=sizes.max())
    cmap = plt.cm.viridis  # 选择一个颜色映射，viridis是常用的颜色映射之一

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(group_positions[:, 0], group_positions[:, 1], c=sizes, cmap=cmap, s=500, edgecolors='black',
                          norm=norm)

    # 在每个点内显示size数字
    for i, (x, y) in enumerate(group_positions):
        plt.text(x, y, str(sizes[i]), color='white', ha='center', va='center', fontweight='bold')

    # 添加颜色条
    plt.colorbar(scatter, label="Group Size")

    # 添加标题和标签
    plt.title("Group Positions and Sizes")
    plt.xlabel("Group X Position")
    plt.ylabel("Group Y Position")

    # 显示图形
    plt.show()

def plot_heatmap_position(cumulative_x, cumulative_y,range_r):
    # 网格分辨率设置
    grid_size = 10  # 网格大小
    x_max, y_max = 10, 10  # 坐标的最大值

    # 将坐标从 list 转换为 numpy 数组，以便进行数学运算
    cumulative_x = np.array(cumulative_x)
    cumulative_y = np.array(cumulative_y)

    # 将坐标映射到 0 到 grid_size-1 的整数范围 (0到899)
    x_indices = np.floor((cumulative_x / x_max) * (grid_size)).astype(int)
    y_indices = np.floor((cumulative_y / y_max) * (grid_size)).astype(int)

    # 创建一个 grid_size * grid_size 的二维数组 (900x900 网格)
    heatmap_data = np.zeros((grid_size, grid_size))

    # 将每个坐标点所在的小格的热力值加1
    for x_idx, y_idx in zip(x_indices, y_indices):
        heatmap_data[y_idx, x_idx] += 1
    print(heatmap_data)

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap='YlGnBu', origin='upper', interpolation='nearest')  # 设置 origin='upper'
    plt.colorbar(label='Heatmap Intensity')  # 显示热力图颜色条
    plt.title(f'Heatmap of Subgroup positions (nums = 10*10, range = {range_r})')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')

    # 设置x轴和y轴的刻度位置
    plt.xticks(np.arange(0, grid_size, 1), labels=[f"{i}" for i in range(grid_size)], fontsize=12)
    plt.yticks(np.arange(0, grid_size, 1), labels=[f"{i}" for i in range(grid_size)], fontsize=12)

    # 设置坐标轴的范围，确保刻度标签显示完整
    plt.xlim(-0.5, grid_size - 0.5)  # 使得x轴的范围适配
    plt.ylim(-0.5, grid_size - 0.5)  # 使得y轴的范围适配

    # 添加网格
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    # 显示图形
    plt.show()

def plot_heatmap_size_position(cumulative_group,range_r):
    # 网格分辨率设置
    grid_size = 10  # 10x10 网格
    x_max, y_max = 10, 10  # 坐标的最大范围

    # 创建一个 grid_size x grid_size 的二维数组，用于存储热力值
    heatmap_data = np.zeros((grid_size, grid_size))

    # 将位置和大小映射到网格
    for size, (x, y) in cumulative_group:
        # 映射到网格索引范围
        x_idx = int(np.floor((x / x_max) * grid_size))
        y_idx = int(np.floor((y / y_max) * grid_size))

        # 防止索引超出范围（避免边界点问题）
        x_idx = min(max(x_idx, 0), grid_size - 1)
        y_idx = min(max(y_idx, 0), grid_size - 1)

        # 累加 size 到对应的格子
        heatmap_data[y_idx, x_idx] += size

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap='YlGnBu', origin='lower', interpolation='nearest')  # origin='lower' 设置原点在左下角
    plt.colorbar(label='Cumulative Size Intensity')  # 显示热力图颜色条
    plt.title(f'Heatmap of the bats nums at subgroup positions (nums = 10*10, range = {range_r})')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')

    # 设置x轴和y轴的刻度位置
    plt.xticks(np.arange(0, grid_size, 1), labels=[f"{i}" for i in range(grid_size)], fontsize=12)
    plt.yticks(np.arange(0, grid_size, 1), labels=[f"{i}" for i in range(grid_size)], fontsize=12)

    # 设置坐标轴的范围，确保刻度标签显示完整
    plt.xlim(-0.5, grid_size - 0.5)  # 使得x轴的范围适配
    plt.ylim(-0.5, grid_size - 0.5)  # 使得y轴的范围适配

    # 添加网格
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.show()

def plot_group_nums(cumulative_group_nums,range_r):
    # 统计每个 size 出现的次数
    size_counts = Counter(cumulative_group_nums)
    print(size_counts)
    # 获取 size 和对应的出现次数
    x = list(size_counts.keys())
    y = list(size_counts.values())

    # 按照组大小 (x) 从小到大排序
    sorted_indices = sorted(range(len(x)), key=lambda i: x[i])  # 获取按 x 排序的索引
    x_sorted = [x[i] for i in sorted_indices]  # 按照排序的索引重新排序 x
    y_sorted = [y[i] for i in sorted_indices]

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(x_sorted, y_sorted, marker='o', linestyle='-', color='b')

    # 设置标题和标签
    plt.title(f"Frequency of Subgroup(nums = 10*10, range = {range_r})")
    plt.xlabel("Group nums")
    plt.ylabel("Frequency")

    plt.xticks([int(i) for i in x_sorted])

    # 显示图形
    plt.grid(True)
    plt.show()

# 初始化累积坐标
cumulative_x = []
cumulative_y = []
cumulative_group = []
cumulative_group_nums = []


# range_r = 1.415
# # 动态绘制热力图F
# for frame in range(100000):
#     # bats = generate_bat_positions(10,10)
#     bats = generate_bat_positions_point_processed_uniform(100, (0, 10), (0, 10))
#     bat_follow_target = generate_follow_target(bats,range_r)
#     group_size_position = calculate_subgroup_positions_size(bat_follow_target, bats)
#     # plot_bat_follow_graph(bats, bat_follow_target)
#     # plot_group_position(group_size_position)
#     ## 提取子组位置
#     group_positions = []
#     for group_id, (size, group_pos) in group_size_position.items():
#         group_positions.append(group_pos)  # 提取位置 (x, y)
#
#     # 转换为 NumPy 数组
#     new_x = [pos[0] for pos in group_positions]  # 提取所有 x 坐标
#     new_y = [pos[1] for pos in group_positions]  # 提取所有 y 坐标
#
#     # 将新位置加入累积数组
#     cumulative_x.extend(new_x)
#     cumulative_y.extend(new_y)
#
#     ##提取累计size与position
#     for size, group_pos in group_size_position.values():
#         cumulative_group.append((size, group_pos))
#
#     ##统计group_nums
#     num_groups = len(group_size_position)
#     cumulative_group_nums.append(num_groups)
#
# #
# plot_heatmap_position(cumulative_x, cumulative_y,range_r)
# plot_heatmap_size_position(cumulative_group,range_r)
# plot_group_nums(cumulative_group_nums,range_r)



# 创建一个字典来存储每个range_r的group数量
all_group_nums = {}
range_values = [1, np.sqrt(2), 2, np.sqrt(5), np.sqrt(8), 3,
                np.sqrt(10), np.sqrt(13), 4, np.sqrt(17), np.sqrt(18), np.sqrt(20), 5, np.sqrt(26),
                np.sqrt(30), np.sqrt(34), 6, np.sqrt(37), np.sqrt(40), np.sqrt(45), 7, np.sqrt(50)]

# range_values = [1.415]
# 遍历range_r从1到10
for range_r in range_values:
    cumulative_group_nums = []

    # 模拟 100000 个周期
    for frame in range(100000):
        # bats = generate_bat_positions(rows, cols)
        bats = generate_bat_positions_point_processed_uniform(100, (0, 10), (0, 10))
        bat_follow_target = generate_follow_target(bats, range_r)
        group_size_position = calculate_subgroup_positions_size(bat_follow_target, bats)

        # 统计每一帧的group数量
        num_groups = len(group_size_position)
        cumulative_group_nums.append(num_groups)

    # 将该range_r对应的group数量保存到字典
    all_group_nums[range_r] = cumulative_group_nums

# 绘制所有range_r对应的折线图
plt.figure(figsize=(20, 8))

for range_r, cumulative_group_nums in all_group_nums.items():
    size_counts = Counter(cumulative_group_nums)
    print(size_counts)
    # 获取 size 和对应的出现次数
    x = list(size_counts.keys())
    y = list(size_counts.values())

    # 按照组大小 (x) 从小到大排序
    sorted_indices = sorted(range(len(x)), key=lambda i: x[i])  # 获取按 x 排序的索引
    x_sorted = [x[i] for i in sorted_indices]  # 按照排序的索引重新排序 x
    y_sorted = [y[i] for i in sorted_indices]

    plt.plot(x_sorted, y_sorted, label=f"Range r = {range_r:.2f}", marker='o', linestyle='-')

# 设置标题和标签
plt.title("Number of Subgroups for Different Range r Values")
plt.xlabel("Number of Groups")
plt.ylabel("frequency")

# 显示图例
plt.legend()

# 设置横轴刻度，每隔 1 显示一个标签
plt.xticks(np.arange(0, 25, 1))

# 显示网格
plt.grid(True)
plt.show()