from mpi4py import MPI
import random
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_bat_positions_point_processed_uniform(num_bats=100, x_range=(0, 10), y_range=(0, 10)):
    # 生成均匀分布的坐标
    x_coords = np.random.uniform(x_range[0], x_range[1], num_bats)
    y_coords = np.random.uniform(y_range[0], y_range[1], num_bats)

    # 编号和坐标打包
    bats = [(i + 1, (x, y)) for i, (x, y) in enumerate(zip(x_coords, y_coords))]

    return bats

def generate_follow_target(bats, range_r, follow_prob):
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

# 以下函数保持不变
def process_one_pair(args):
    range_r, area_s, n = args
    cumulative_group_nums = []
    for frame in range(10000):
        bats = generate_bat_positions_point_processed_uniform(
            n,  # 目标数
            (0, area_s),  # x 范围
            (0, area_s)  # y 范围
        )
        bat_follow_target, _ = generate_follow_target(bats, range_r, 1)
        group_size_position = calculate_subgroup_positions_size(bat_follow_target, bats)
        # 统计每一帧的 group 数量
        num_groups = len(group_size_position)
        cumulative_group_nums.append(num_groups)
    return (range_r, area_s, n), cumulative_group_nums


def save_to_hdf5(all_group_nums):
    rows = []
    for (range_r, area_s, n), group_nums in all_group_nums.items():
        for num in group_nums:
            rows.append((range_r, area_s, n, num))
    df = pd.DataFrame(rows, columns=["follow_range", "Area_range", "num_of_robots", "num_of_subgroups"])
    df.to_hdf("group_nums_results.h5", key="df", mode="w", complevel=5, complib="blosc")


def main():
    # 初始化MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    total_pairs = 10000

    # 由根进程生成所有任务
    if rank == 0:
        pairs = [
            (random.uniform(1, 15), random.uniform(1, 10), random.randint(10, 100))
            for _ in range(total_pairs)
        ]
        # 将任务均分为 size 个子列表
        chunk_size = (total_pairs + size - 1) // size  # 向上取整
        chunks = [pairs[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
    else:
        chunks = None

    # 分发任务，每个进程得到一个子列表
    local_pairs = comm.scatter(chunks, root=0)

    # 每个进程计算自己负责的任务
    local_results = []
    # 为避免多个进程同时打印干扰输出，仅在根进程显示进度条
    for pair in tqdm(local_pairs, desc=f"Rank {rank}", disable=(rank != 0)):
        result = process_one_pair(pair)
        local_results.append(result)

    # 收集所有进程的计算结果到根进程
    results = comm.gather(local_results, root=0)

    # 仅在根进程汇总并保存结果
    if rank == 0:
        # 将各个子列表合并为一个列表
        all_results = [item for sublist in results for item in sublist]
        all_group_nums = dict(all_results)
        print("Processing complete!")
        save_to_hdf5(all_group_nums)


if __name__ == '__main__':
    main()
