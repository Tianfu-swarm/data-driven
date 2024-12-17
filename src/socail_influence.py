import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time

# 定义函数：随机生成节点与节点的连接关系
def generate_bat_choices(num_bats, probability):
    bats = list(range(1, num_bats + 1))
    choices = {}
    for bat in bats:
        if random.random() < probability:
            possible_choices = [i for i in bats if i != bat]
            chosen_one = random.choice(possible_choices)
            choices[bat] = chosen_one
        else:
            choices[bat] = None
    return choices

# 定义函数：绘制图并返回每个子图的节点数量
def draw_and_count_subgraphs(num_bats, probability):

    # 生成节点的连接关系
    choices = generate_bat_choices(num_bats, probability)
    #print(choices)

    # 创建无向图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(range(1, num_bats + 1))

    # 添加边
    for node, connection in choices.items():
        if connection is not None:
            G.add_edge(node, connection)

    # 绘制图
    # plt.figure(figsize=(8, 6))
    # nx.draw(G, with_labels=True, node_color="lightblue", font_weight="bold")
    # plt.title("Randomly Generated Graph")
    # plt.show()

    # 获取所有子图的节点数
    connected_components = list(nx.connected_components(G))
    subgraph_sizes = [len(comp) for comp in connected_components]
    # 返回子图数量和子图节点数
    return len(connected_components), subgraph_sizes

def relation_numOfSubgroup_probabilities():
    # 参数设置
    num_bats = 42  # 节点数量
    probabilities = np.arange(0, 1.1, 0.1)  # 概率从0到1，每次增加0.1
    simulations = 10000  # 仿真次数

    # 用于存储不同概率下，各群体数量的比例
    groupnum_distribution = {prob: [] for prob in probabilities}


    # 对于每个概率值执行10000次模拟
    for prob in probabilities:
        groupnum_simulations = []  # 用于存储该概率下的所有群体数量
        for _ in range(simulations):
            groupnum, _ = draw_and_count_subgraphs(num_bats, prob)
            groupnum_simulations.append(groupnum)

        # 统计每个群体数量出现的频率
        unique, counts = np.unique(groupnum_simulations, return_counts=True)
        distribution = dict(zip(unique, counts / simulations))  # 计算比例
        groupnum_distribution[prob] = distribution

    # 绘制每个概率下群体数量的比例图
    plt.figure(figsize=(10, 6))

    for prob in probabilities:
        # 提取该概率下每个群体数量及其比例
        distribution = groupnum_distribution[prob]
        groupnums = sorted(distribution.keys())
        proportions = [distribution.get(g, 0) for g in groupnums]

        plt.plot(groupnums, proportions, marker='o', label=f'P = {prob:.3f}')

    plt.xlabel('Number of Subgroups', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.title(f'Proportion of Different Number of Subgroups (numOfBats = {num_bats})', fontsize=14)
    plt.legend()
    plt.grid(True)

    # 生成随机文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # 格式化当前时间：20240617_153045
    random_filename = f"../picture/numOfBats: {num_bats}--the relation of numSubGroup-probabilities_{timestamp}.png"

    # 保存图片
    plt.savefig(random_filename, dpi=300, bbox_inches='tight')

    plt.show()

def relation_numOfSubgroup_numOfBats():

    num_bats = np.arange(0, 110, 10)  # 节点数量
    probabilities = 1  # 概率从0到1，每次增加0.1
    simulations = 100000  # 仿真次数

    # 用于存储不同概率下，各群体数量的比例
    groupnum_distribution = {num: [] for num in num_bats}


    # 对于每个概率值执行10000次模拟
    for num in num_bats:
        groupnum_simulations = []  # 用于存储该概率下的所有群体数量
        for _ in range(simulations):
            groupnum, _ = draw_and_count_subgraphs(num, probabilities)
            groupnum_simulations.append(groupnum)

        # 统计每个群体数量出现的频率
        unique, counts = np.unique(groupnum_simulations, return_counts=True)
        distribution = dict(zip(unique, counts / simulations))  # 计算比例
        groupnum_distribution[num] = distribution

    # 绘制每个概率下群体数量的比例图
    plt.figure(figsize=(10, 6))

    for num in num_bats:
        # 提取该概率下每个群体数量及其比例
        distribution = groupnum_distribution[num]
        groupnums = sorted(distribution.keys())
        proportions = [distribution.get(g, 0) for g in groupnums]

        plt.plot(groupnums, proportions, marker='o', label=f'P = {num:.1f}')

    plt.xlabel('Number of Subgroups', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.title(f'Proportion of Different Number of Subgroups (P = {probabilities})', fontsize=14)
    plt.legend()
    plt.grid(True)

    # 生成随机文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # 格式化当前时间：20240617_153045
    random_filename = f"../picture/P: {probabilities}--the relation of numSubGroup-numOfBats_{timestamp}.png"

    # 保存图片
    plt.savefig(random_filename, dpi=300, bbox_inches='tight')

    plt.show()

def relation_numOfSubgroup_numOfBats_probabilities():
    # 参数设置
    probabilities = np.arange(0, 1.01, 0.01)  # 概率从0到1，每次增加0.1
    num_bats_values = np.arange(2, 101, 1)  # 节点数量从2到100，每次增加1
    simulations = 1000  # 仿真次数

    # 存储子群体数量为2的比例
    subgroup_2_proportions = np.zeros((len(probabilities), len(num_bats_values)))

    # 遍历所有概率和节点数量组合，计算子群体数量为2的比例
    for i, prob in enumerate(probabilities):
        for j, num_bats in enumerate(num_bats_values):
            groupnum_simulations = []  # 存储当前组合下的所有群体数量
            for _ in range(simulations):
                groupnum, _ = draw_and_count_subgraphs(num_bats, prob)
                groupnum_simulations.append(groupnum)

            # 统计子群体数量为2的比例
            unique, counts = np.unique(groupnum_simulations, return_counts=True)
            distribution = dict(zip(unique, counts / simulations))  # 计算比例
            subgroup_2_proportions[i, j] = distribution.get(2, 0)  # 获取子群体数量为2的比例

    # 创建3D图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 将数据转换为网格
    X, Y = np.meshgrid(probabilities, num_bats_values)
    Z = subgroup_2_proportions.T  # 转置以匹配网格维度

    # 绘制3D曲面图
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')

    # 设置轴标签和标题
    ax.set_xlabel('Probability (P)', fontsize=12)
    ax.set_ylabel('Number of Nodes (num_bats)', fontsize=12)
    ax.set_zlabel('Proportion of Subgroup = 2', fontsize=12)
    ax.set_title('Proportion of Subgroup Size = 2 vs P and num_bats', fontsize=14)

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # 显示图形
    plt.show()



relation_numOfSubgroup_numOfBats()
# relation_numOfSubgroup_probabilities()
# relation_numOfSubgroup_numOfBats_probabilities()