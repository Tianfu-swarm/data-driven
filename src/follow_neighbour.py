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

def generate_follow_target(bats, range_r=1.0, follow_prob=1.0):

    bat_follow_target = {}
    bat_neighbors_count = {}

    # 将蝙蝠坐标映射到编号的字典，方便快速索引
    position_map = {tuple(pos): bat_id for bat_id, pos in bats}

    # 遍历每只蝙蝠
    for bat_id, (x, y) in bats:
        neighbors = []  # 存储邻居蝙蝠编号

        # 遍历所有可能的邻居
        for (neighbor_x, neighbor_y), neighbor_id in position_map.items():
            if neighbor_id == bat_id:
                continue  # 跳过自己

            distance = math.sqrt((neighbor_x - x) ** 2 + (neighbor_y - y) ** 2)

            # 如果距离在范围内，添加到邻居列表
            if distance <= range_r:
                neighbors.append(neighbor_id)

        # 记录邻居数量
        bat_neighbors_count[bat_id] = len(neighbors)

        # 根据概率决定是否跟随邻居
        if neighbors and random.random() < follow_prob:
            # 随机选择一个邻居进行跟随
            bat_follow_target[bat_id] = random.choice(neighbors)
        else:
            # 没有邻居或选择不跟随
            bat_follow_target[bat_id] = None

    return bat_follow_target, bat_neighbors_count


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

def plot_heatmap():
    # 初始化累积坐标
    cumulative_x = []
    cumulative_y = []
    cumulative_group = []
    cumulative_group_nums = []
    cumulative_group_sizes = []

    range_r = 1
    # 动态绘制热力图F
    for frame in range(1):
        # bats = generate_bat_positions(10,10)
        bats = generate_bat_positions_point_processed_uniform(100, (0, 10), (0, 10))
        bat_follow_target, _ = generate_follow_target(bats,range_r)
        group_size_position = calculate_subgroup_positions_size(bat_follow_target, bats)
        plot_bat_follow_graph(bats, bat_follow_target)
        plot_group_position(group_size_position)
        ## 提取子组位置
        group_positions = []
        for group_id, (size, group_pos) in group_size_position.items():
            group_positions.append(group_pos)  # 提取位置 (x, y)

        # 转换为 NumPy 数组
        new_x = [pos[0] for pos in group_positions]  # 提取所有 x 坐标
        new_y = [pos[1] for pos in group_positions]  # 提取所有 y 坐标

        # 将新位置加入累积数组
        cumulative_x.extend(new_x)
        cumulative_y.extend(new_y)

        ##提取累计size与position
        for size, group_pos in group_size_position.values():
            cumulative_group.append((size, group_pos))

        ##统计group_nums
        num_groups = len(group_size_position)
        cumulative_group_nums.append(num_groups)


    plot_heatmap_position(cumulative_x, cumulative_y,range_r)
    plot_heatmap_size_position(cumulative_group,range_r)
    plot_group_nums(cumulative_group_nums,range_r)


def groupnum_of_different_range():

    # 创建一个字典来存储每个range_r的group数量
    all_group_nums = {}
    range_values = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    # 遍历range_r从1到10
    for range_r in range_values:
        cumulative_group_nums = []

        # 模拟 100000 个周期
        for frame in range(10000):
            # bats = generate_bat_positions(rows, cols)
            bats = generate_bat_positions_point_processed_uniform(100, (0, 10), (0, 10))
            bat_follow_target, _ = generate_follow_target(bats, range_r)
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
        # print(size_counts)
        # 获取 size 和对应的出现次数
        x = list(size_counts.keys())
        y = list(size_counts.values())

        # 按照组大小 (x) 从小到大排序
        sorted_indices = sorted(range(len(x)), key=lambda i: x[i])  # 获取按 x 排序的索引
        x_sorted = [x[i] for i in sorted_indices]  # 按照排序的索引重新排序 x
        y_sorted = [y[i] for i in sorted_indices]

        print(f"range_r = {range_r}", x_sorted, y_sorted)
        plt.plot(x_sorted, y_sorted, label=f"Range r = {range_r:.2f}", marker='o', linestyle='-')

    # 设置标题和标签
    plt.title("Number of Subgroups for Different Range r Values")
    plt.xlabel("Number of Groups")
    plt.ylabel("frequency")

    # 显示图例
    plt.legend()

    # 设置横轴刻度，每隔 1 显示一个标签
    plt.xticks(np.arange(0, max(x_sorted)+1, 1))
    # 显示网格
    plt.grid(True)
    plt.show()

def groupnum_of_different_density():

    # 创建一个字典来存储每个range_r的group数量
    all_group_nums = {}
    range_r = 3
    area_r_all = [5,6,7,8,9,10]
    # 遍历range_r从1到10
    for area_r in area_r_all:
        cumulative_group_nums = []

        # 模拟 100000 个周期
        for frame in range(10000):
            # bats = generate_bat_positions(rows, cols)
            bats = generate_bat_positions_point_processed_uniform(100, (0, area_r), (0, area_r))
            bat_follow_target, _ = generate_follow_target(bats, range_r)
            group_size_position = calculate_subgroup_positions_size(bat_follow_target, bats)

            # 统计每一帧的group数量
            num_groups = len(group_size_position)
            cumulative_group_nums.append(num_groups)

        # 将该range_r对应的group数量保存到字典
        all_group_nums[area_r] = cumulative_group_nums

    # 绘制所有range_r对应的折线图
    plt.figure(figsize=(20, 8))

    for area_r, cumulative_group_nums in all_group_nums.items():
        size_counts = Counter(cumulative_group_nums)
        # print(size_counts)
        # 获取 size 和对应的出现次数
        x = list(size_counts.keys())
        y = list(size_counts.values())

        # 按照组大小 (x) 从小到大排序
        sorted_indices = sorted(range(len(x)), key=lambda i: x[i])  # 获取按 x 排序的索引
        x_sorted = [x[i] for i in sorted_indices]  # 按照排序的索引重新排序 x
        y_sorted = [y[i] for i in sorted_indices]

        print(f"range_r = {range_r}", x_sorted, y_sorted)
        plt.plot(x_sorted, y_sorted, label=f"100 bats in {area_r}*{area_r}", marker='o', linestyle='-')

    # 设置标题和标签
    plt.title("Number of Subgroups for Different density Values")
    plt.xlabel("Number of Groups")
    plt.ylabel("frequency")

    # 显示图例
    plt.legend()

    # 设置横轴刻度，每隔 1 显示一个标签
    plt.xticks(np.arange(0, max(x_sorted) + 1, 1))
    # 显示网格
    plt.grid(True)
    plt.show()

def groupnum_of_different_neighbors():
    posibility = 0.9;
    # 你想要测试的一对参数 (range_r, area_size)
    pairs = [
        (2,10),
        (3,10),
        (4,10),
        (5,10),
        (6,10)
        ]

    # 创建一个字典来存储每个 (range_r, area_s) 的 group 数量分布
    all_group_nums = {}

    # 遍历每个参数对
    for (range_r, area_s) in pairs:
        cumulative_group_nums = []

        # 模拟 100000 个周期
        for frame in range(100000):
            # 注意根据你真实的 area_s 设置 x,y 范围，例如 (0, area_s)
            bats = generate_bat_positions_point_processed_uniform(
                100,  # 目标数
                (0, area_s),  # x 范围
                (0, area_s)  # y 范围
            )
            bat_follow_target, _ = generate_follow_target(bats, range_r, posibility)
            group_size_position = calculate_subgroup_positions_size(bat_follow_target, bats)

            # 统计每一帧的 group 数量
            num_groups = len(group_size_position)
            cumulative_group_nums.append(num_groups)

        # 将该 (range_r, area_s) 对应的 group 数量分布保存到字典
        all_group_nums[(range_r, area_s)] = cumulative_group_nums

    # 统一绘制所有曲线
    plt.figure(figsize=(15, 8))

    # 对字典中每个键值对绘图
    for (range_r, area_s), cumulative_group_nums in all_group_nums.items():
        size_counts = Counter(cumulative_group_nums)
        # 获取 group 数量 (x) 和对应的出现次数 (y)
        x = list(size_counts.keys())
        y = list(size_counts.values())

        # 按 group 数量从小到大排序
        sorted_indices = sorted(range(len(x)), key=lambda i: x[i])
        x_sorted = [x[i] for i in sorted_indices]
        y_sorted = [y[i] for i in sorted_indices]

        neighbor_count = range_r/area_s
        # 绘制曲线，添加有区分度的 label
        plt.plot(x_sorted, y_sorted,
                 label=f"neighbor count = {neighbor_count:.2f}, range:{range_r}, area:{area_s}×{area_s}",
                 marker='o', linestyle='-')

    # 设置标题和标签
    plt.title(f"Number of Subgroups for Different neighbors,P = {posibility}")
    plt.xlabel("Number of Groups")
    plt.ylabel("Frequency")

    # 显示图例
    plt.legend()

    # 设置横轴刻度，每隔 1 显示一个标签（根据最大 group 数量动态计算）
    max_groups = max(x_sorted) if x_sorted else 0
    plt.xticks(np.arange(0, max_groups + 1, 1))

    # 显示网格
    plt.grid(True)
    plt.show()

def count_neighbor_num():
    range_neighbors_distribution = {}

    # 生成 range_r 值，从 0 到 sqrt(200)，步长为 0.5
    range_values = np.arange(2, 7, 1)  # 从 0.5 开始，以避免 r = 0

    # 假设进行 10000 个 frame 计算
    for range_r in range_values:
        all_neighbors = []  # 用于存储所有蝙蝠的邻居数量

        for frame in range(10000):
            bats = generate_bat_positions_point_processed_uniform(100, (0, 10), (0, 10))  # 生成蝙蝠位置
            _, bat_neighbors_count = generate_follow_target(bats, range_r)  # 获取邻居个数

            # 将每个蝙蝠的邻居个数记录下来
            all_neighbors.extend(bat_neighbors_count.values())  # 使用 extend 将所有邻居个数合并到一个列表中

        # 使用 Counter 计算邻居数量的频率分布
        neighbor_counts = Counter(all_neighbors)

        # 将结果存储到字典中，按 range_r 分类
        range_neighbors_distribution[range_r] = neighbor_counts

    # 创建图形
    plt.figure(figsize=(15, 10))

    # 绘制每个 range_r 对应的邻居数量频率分布
    for range_r, neighbor_counts in range_neighbors_distribution.items():
        # 排序邻居数量并绘制频率分布
        sorted_neighbors = sorted(neighbor_counts.items())  # 按邻居数量排序
        x_vals, y_vals = zip(*sorted_neighbors)  # 解压排序后的数据为 x 和 y 值

        # 绘制折线图
        plt.plot(x_vals, y_vals, label=f'r = {range_r:.2f}', marker='o', linestyle='-')

    # 设置图形标签
    plt.title('Neighbor Count Frequency Distribution for Different r Values')
    plt.xlabel('Neighbor Count')
    plt.ylabel('Frequency (Count)')

    # 设置横轴每5个单位显示一个刻度
    plt.xticks(np.arange(0, max(x_vals) + 5, 5))

    # 启用网格线
    plt.grid(True)

    # 设置图例自动调整位置
    plt.legend(loc='best')

    # 显示图形
    plt.show()

def count_neighbor_num_density():
    """
    演示对多组 (range_r, area_s) 参数配对，
    分别计算邻居数频率分布并在同一张图中绘制。
    """

    # 1) 定义要测试的 (range_r, area_s) 配对列表
    pairs = [
        (5, 7),  # range=5, area=7×7
        (5, 6),  # range=5, area=6×6
        # 如果需要更多配对，比如 (3,8), (4,10) 等，也可在此添加
        # (3, 8),
        # (4, 10),
    ]

    # 用于存储不同 (range_r, area_s) 下的邻居数量分布
    neighbors_distribution_dict = {}

    # 计算的帧数（可自行调整）
    num_frames = 10000

    for (fixed_range_r, area_s) in pairs:
        # 用于累计所有蝙蝠的邻居数
        all_neighbors = []

        for _ in range(num_frames):
            # 在正方形 [0, area_s] × [0, area_s] 上随机产生 100 只蝙蝠
            bats = generate_bat_positions_point_processed_uniform(
                100,
                (0, area_s),
                (0, area_s)
            )

            # 获取每只蝙蝠的邻居数
            _, bat_neighbors_count = generate_follow_target(bats, fixed_range_r)

            # 将所有蝙蝠的邻居数汇总到 all_neighbors
            all_neighbors.extend(bat_neighbors_count.values())

        # 使用 Counter 计算邻居数量的出现频次
        neighbor_counts = Counter(all_neighbors)

        # 存储在字典中，以 (range_r, area_s) 作为 key
        neighbors_distribution_dict[(fixed_range_r, area_s)] = neighbor_counts

    # 2) 开始绘图
    plt.figure(figsize=(12, 8))

    # 找出所有结果里最大的邻居数量，用于设置横坐标范围
    max_neighbor_count = 0
    for counter_data in neighbors_distribution_dict.values():
        if counter_data:
            local_max = max(counter_data.keys())
            max_neighbor_count = max(max_neighbor_count, local_max)

    # 3) 绘制每个 (range_r, area_s) 的邻居数量频率分布
    for (fixed_range_r, area_s), neighbor_counts in neighbors_distribution_dict.items():
        if not neighbor_counts:
            continue

        # 将 (邻居数, 频次) 排序，便于按邻居数从小到大绘制
        sorted_neighbors = sorted(neighbor_counts.items())  # [(neighbor_num, count), ...]
        x_vals, y_vals = zip(*sorted_neighbors)

        # 绘制折线图，并在 label 中区分 (range_r, area_s)
        plt.plot(x_vals, y_vals,
                 label=f'Range={fixed_range_r}, Area={area_s}×{area_s}',
                 marker='o', linestyle='-')

    # 4) 设置图形的标题、坐标标签等
    plt.title('Neighbor Count Frequency Distribution for Different (Range, Area) Pairs')
    plt.xlabel('Neighbor Count')
    plt.ylabel('Frequency (Count)')

    # 设置横坐标刻度：可根据实际情况调整步长
    plt.xticks(np.arange(0, max_neighbor_count + 1, 5))
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def count_neighbor_num():
    range_neighbors_distribution = {}

    # 生成 range_r 值，从 0 到 sqrt(200)，步长为 0.5
    range_values = np.arange(2, 7, 1)  # 从 0.5 开始，以避免 r = 0

    # 假设进行 10000 个 frame 计算
    for range_r in range_values:
        all_neighbors = []  # 用于存储所有蝙蝠的邻居数量

        for frame in range(10000):
            bats = generate_bat_positions_point_processed_uniform(100, (0, 10), (0, 10))  # 生成蝙蝠位置
            _, bat_neighbors_count = generate_follow_target(bats, range_r)  # 获取邻居个数

            # 将每个蝙蝠的邻居个数记录下来
            all_neighbors.extend(bat_neighbors_count.values())  # 使用 extend 将所有邻居个数合并到一个列表中

        # 使用 Counter 计算邻居数量的频率分布
        neighbor_counts = Counter(all_neighbors)

        # 将结果存储到字典中，按 range_r 分类
        range_neighbors_distribution[range_r] = neighbor_counts

    # 创建图形
    plt.figure(figsize=(15, 10))

    # 绘制每个 range_r 对应的邻居数量频率分布
    for range_r, neighbor_counts in range_neighbors_distribution.items():
        # 排序邻居数量并绘制频率分布
        sorted_neighbors = sorted(neighbor_counts.items())  # 按邻居数量排序
        x_vals, y_vals = zip(*sorted_neighbors)  # 解压排序后的数据为 x 和 y 值

        # 绘制折线图
        plt.plot(x_vals, y_vals, label=f'r = {range_r:.2f}', marker='o', linestyle='-')

    # 设置图形标签
    plt.title('Neighbor Count Frequency Distribution for Different r Values')
    plt.xlabel('Neighbor Count')
    plt.ylabel('Frequency (Count)')

    # 设置横轴每5个单位显示一个刻度
    plt.xticks(np.arange(0, max(x_vals) + 5, 5))

    # 启用网格线
    plt.grid(True)

    # 设置图例自动调整位置
    plt.legend(loc='best')

    # 显示图形
    plt.show()


def count_neighbor_num_pairs():
    """
    演示对多组 (range_r, area_s) 参数配对，
    分别计算邻居数的频率分布并在同一张图中绘制。
    """

    # 1) 定义要测试的 (range_r, area_s) 配对列表
    #   这里举例定义了五对 (range_r, area_s)，可根据需要自行修改或扩展
    pairs = [
        (5, 10),
        (4, 8),
        (3, 6),
        (2, 4),
    ]

    # 用来保存每个 (range_r, area_s) 对应的 "邻居数量 -> 频次" 分布
    pairs_neighbors_distribution = {}

    # 设置模拟的帧数
    num_frames = 10000

    for (range_r, area_s) in pairs:
        # 用于累计所有蝙蝠在所有帧中的邻居数量
        all_neighbors = []

        # 进行多次模拟
        for frame in range(num_frames):
            # 在正方形 [0, area_s] × [0, area_s] 生成 100 只蝙蝠
            bats = generate_bat_positions_point_processed_uniform(
                100,
                (0, area_s),
                (0, area_s)
            )

            # 计算每只蝙蝠的邻居数量 (bat_neighbors_count 是 dict: index -> neighbor_count)
            _, bat_neighbors_count = generate_follow_target(bats, range_r)

            # 收集所有蝙蝠的邻居数量
            all_neighbors.extend(bat_neighbors_count.values())

        # 使用 Counter 统计“邻居数量出现的频次”
        neighbor_counts = Counter(all_neighbors)

        # 存储到字典，以 (range_r, area_s) 作为 Key
        pairs_neighbors_distribution[(range_r, area_s)] = neighbor_counts

    # 2) 开始绘图
    plt.figure(figsize=(15, 10))

    # 找到所有结果中最大的邻居数量，用于后续设置 X 轴刻度
    max_neighbor_count = 0
    for counter_data in pairs_neighbors_distribution.values():
        if counter_data:
            local_max = max(counter_data.keys())
            max_neighbor_count = max(max_neighbor_count, local_max)

    # 3) 逐对绘制不同 (range_r, area_s) 的邻居数频率分布
    for (range_r, area_s), neighbor_counts in pairs_neighbors_distribution.items():
        if not neighbor_counts:
            continue

        # 按邻居数量从小到大排序 (neighbor_num, count)
        sorted_neighbors = sorted(neighbor_counts.items())
        x_vals, y_vals = zip(*sorted_neighbors)

        # 绘制折线图
        plt.plot(
            x_vals, y_vals,
            label=f'r={range_r}, area={area_s}×{area_s}',
            marker='o', linestyle='-'
        )

    # 4) 设置图形的标题、坐标标签等
    plt.title('Neighbor Count Frequency Distribution for Different (range_r, area_s) Pairs')
    plt.xlabel('Neighbor Count')
    plt.ylabel('Frequency (Count)')

    # 设置横坐标刻度，这里每 5 个单位一个刻度，可根据需要调整
    plt.xticks(np.arange(0, max_neighbor_count + 5, 5))
    plt.grid(True)

    # 显示图例
    plt.legend(loc='best')

    # 最后显示图形
    plt.show()

def count_neighbor_num_bats():
    # 固定的搜索半径
    fixed_range_r = 5

    # 不同的蝙蝠数量
    area_size = [7,6]

    # 用于存储不同蝙蝠数量（num_bats）对应的邻居数量频率分布
    neighbors_distribution_dict = {}

    # 假设进行 1000 个 frame 计算（根据需求可自行调整）
    num_frames = 10000

    for size in area_size:
        all_neighbors = []

        for _ in range(num_frames):
            # 生成 num_bats 只蝙蝠的随机位置
            bats = generate_bat_positions_point_processed_uniform(100, (0, size), (0, size))

            # 获取邻居个数信息
            _, bat_neighbors_count = generate_follow_target(bats, fixed_range_r)

            # 将每个蝙蝠的邻居个数记录下来
            all_neighbors.extend(bat_neighbors_count.values())

        # 使用 Counter 计算邻居数量的频率分布
        neighbor_counts = Counter(all_neighbors)

        # 将结果存储到字典中，按 num_bats 分类
        neighbors_distribution_dict[size] = neighbor_counts

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 用于记录所有子图中最大的邻居数量，以便设置横坐标范围
    max_neighbor_count = 0

    # 找出整体最大邻居数量（避免在循环外调用 max(...) 导致范围不准确）
    for counter_data in neighbors_distribution_dict.values():
        if counter_data:
            local_max = max(counter_data.keys())
            if local_max > max_neighbor_count:
                max_neighbor_count = local_max

    # 绘制每个 size 对应的邻居数量频率分布
    for num_bats, neighbor_counts in neighbors_distribution_dict.items():
        # 排序邻居数量并绘制频率分布
        sorted_neighbors = sorted(neighbor_counts.items())  # [(neighbor_num, count), ...]
        if not sorted_neighbors:
            # 如果某个结果为空，可继续下一个
            continue
        x_vals, y_vals = zip(*sorted_neighbors)

        # 绘制折线图
        plt.plot(x_vals, y_vals, label=f'density: 100 bats in {num_bats}*{num_bats} area', marker='o', linestyle='-')

    # 设置图形标签
    plt.title(f'Neighbor Count Frequency Distribution (Range = {fixed_range_r})')
    plt.xlabel('Neighbor Count')
    plt.ylabel('Frequency (Count)')

    # 设置横轴刻度（例如每 5 个单位一个刻度，可根据实际分布调节）
    plt.xticks(np.arange(0, max_neighbor_count + 5, 5))

    # 启用网格线
    plt.grid(True)

    # 设置图例自动调整位置
    plt.legend(loc='best')

    # 显示图形
    plt.show()

def count_subgroup_size_distribution():
    range_values = np.arange(20, 21, 1)  # 半径范围 1 到 10
    posibility = 0.5
    subgroup_size_dict = {}  # 存储不同半径的子群大小

    for range_r in range_values:
        all_subgroup_sizes = []  # 单独存储当前半径的子群大小

        for frame in range(10000):  # 每个半径计算10次
            # 生成 100 只蝙蝠的随机位置
            bats = generate_bat_positions_point_processed_uniform(100, (0, 10), (0, 10))
            bat_follow_target, _ = generate_follow_target(bats, range_r, posibility)
            group_size_position = calculate_subgroup_positions_size(bat_follow_target, bats)

            # 提取所有群组大小
            subgroup_sizes = [size for size, _ in group_size_position.values()]
            all_subgroup_sizes.extend(subgroup_sizes)

        # 统计 group size 出现的次数
        subgroup_size_dict[range_r] = Counter(all_subgroup_sizes)

    # 绘制折线图
    plt.figure(figsize=(12, 8))

    for range_r, size_counts in subgroup_size_dict.items():
        sorted_sizes = sorted(size_counts.items())  # 排序群组大小
        x_size, y_frequency = zip(*sorted_sizes)  # 获取每个大小的出现次数

        plt.plot(x_size, y_frequency, label=f'r = {range_r:.2f}', marker='o', linestyle='-')

    plt.xlabel('Subgroup Size')
    plt.ylabel('Frequency')
    plt.title(f'Subgroup Size Frequency Distribution for Different Ranges (P={posibility})')
    # 设置横轴每5个单位显示一个刻度
    plt.xticks(np.arange(0, max(x_size) + 5, 5))  #

    # 启用网格线
    plt.grid(True)

    # 设置图例自动调整位置
    plt.legend(loc='best')

    # 显示图形
    plt.show()

def count_subgroup_size_distribution_posibility():
    posibility = np.arange(0.95, 1.0, 0.01)  # 半径范围 1 到 10
    range_r = 25
    subgroup_size_dict = {}  # 存储不同半径的子群大小

    for p in posibility:
        all_subgroup_sizes = []  # 单独存储当前半径的子群大小

        for frame in range(10000):  # 每个半径计算10次
            # 生成 100 只蝙蝠的随机位置
            bats = generate_bat_positions_point_processed_uniform(100, (0, 10), (0, 10))
            bat_follow_target, _ = generate_follow_target(bats, range_r, p)
            group_size_position = calculate_subgroup_positions_size(bat_follow_target, bats)

            # 提取所有群组大小
            subgroup_sizes = [size for size, _ in group_size_position.values()]
            all_subgroup_sizes.extend(subgroup_sizes)

        # 统计 group size 出现的次数
        subgroup_size_dict[p] = Counter(all_subgroup_sizes)

    # 绘制折线图
    plt.figure(figsize=(12, 8))

    for p, size_counts in subgroup_size_dict.items():
        sorted_sizes = sorted(size_counts.items())  # 排序群组大小
        x_size, y_frequency = zip(*sorted_sizes)  # 获取每个大小的出现次数

        plt.plot(x_size, y_frequency, label=f'p = {p:.2f}', marker='o', linestyle='-')

    plt.xlabel('Subgroup Size')
    plt.ylabel('Frequency')
    plt.title(f'Subgroup Size Frequency Distribution for Different P')
    # 设置横轴每5个单位显示一个刻度
    plt.xticks(np.arange(0, max(x_size) + 5, 5))  #

    # 启用网格线
    plt.grid(True)

    # 设置图例自动调整位置
    plt.legend(loc='best')

    # 显示图形
    plt.show()


def count_subgroup_size_distribution_fixed_range():
    # 固定的半径值（可根据需要修改）
    fixed_range_r = 5

    # 不同的蝙蝠数量（从 50 到 200，每次增加 50）
    num_bats_values = [50, 100, 150, 200]

    # 用于存储不同蝙蝠数量对应的子群大小分布
    subgroup_size_dict = {}

    for num_bats in num_bats_values:
        all_subgroup_sizes = []

        # 可以根据需要调整实验次数（例如 100 次、1000 次等）
        # 这里使用 1000 作为示例
        for _ in range(1000):
            # 生成 num_bats 只蝙蝠的随机位置
            bats = generate_bat_positions_point_processed_uniform(num_bats, (0, 10), (0, 10))

            # 计算蝙蝠的追随目标
            bat_follow_target, _ = generate_follow_target(bats, fixed_range_r)

            # 计算子群大小（返回形如 {group_id: (size, centroid_position), ...} 的字典）
            group_size_position = calculate_subgroup_positions_size(bat_follow_target, bats)

            # 提取所有群组的大小
            subgroup_sizes = [size for size, _ in group_size_position.values()]
            all_subgroup_sizes.extend(subgroup_sizes)

        # 统计子群大小出现的次数
        subgroup_size_dict[num_bats] = Counter(all_subgroup_sizes)

    # 开始绘制
    plt.figure(figsize=(12, 8))

    # 找到所有分布中最大的子群大小，用于设定横坐标刻度
    max_subgroup_size = 0
    for size_counter in subgroup_size_dict.values():
        if size_counter:
            local_max = max(size_counter.keys())
            if local_max > max_subgroup_size:
                max_subgroup_size = local_max

    # 绘制每个 num_bats 的子群大小分布
    for num_bats, size_counts in subgroup_size_dict.items():
        # 将子群大小和对应的频数按照子群大小排序
        sorted_sizes = sorted(size_counts.items())
        x_size, y_frequency = zip(*sorted_sizes) if sorted_sizes else ([], [])

        plt.plot(x_size, y_frequency, label=f'density: {num_bats} bats in 10*10 area', marker='o', linestyle='-')

    plt.xlabel('Subgroup Size')
    plt.ylabel('Frequency')
    plt.title(f'Subgroup Size Frequency Distribution (Range = {fixed_range_r})')

    # 设置横坐标刻度（步长可按需调整）
    plt.xticks(np.arange(0, max_subgroup_size + 5, 5))

    plt.grid(True)  # 启用网格线
    plt.legend(loc='best')  # 设置图例自动调整位置

    plt.show()


# count_subgroup_size_distribution_fixed_range()
# count_subgroup_size_distribution()
count_subgroup_size_distribution_posibility()
# count_neighbor_num()
# count_neighbor_num_pairs()
# count_neighbor_num_density()

# plot_heatmap()

# groupnum_of_different_density()
# groupnum_of_different_range()
# groupnum_of_different_neighbors()