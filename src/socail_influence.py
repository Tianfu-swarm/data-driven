import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import binom
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time

# 定义函数：随机生成节点与节点的连接关系
def generate_bat_choices(num_bats, probability):
    bats = list(range(1, num_bats + 1))
    choices = {}
    none_count = 0  # Initialize a counter for None values

    for bat in bats:
        if random.random() < probability:
            possible_choices = [i for i in bats if i != bat]
            chosen_one = random.choice(possible_choices)
            choices[bat] = chosen_one
        else:
            choices[bat] = None
            none_count += 1  # Increment the counter when a None value is assigned

    return choices, none_count

# 定义函数：绘制图并返回每个子图的节点数量
def draw_and_count_subgraphs(num_bats, probability):

    # 生成节点的连接关系
    choices,none_count = generate_bat_choices(num_bats, probability)

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
    # 过滤掉只包含一个节点的子图
    filtered_components = [comp for comp in connected_components if len(comp) > 1]
    # 获取每个子图的节点数
    subgraph_sizes = [len(comp) for comp in filtered_components]
    # 返回子图数量和子图节点数（不统计大小1）
    # return len(filtered_components), subgraph_sizes
    # 返回子图数量和子图节点数（统计大小为1）
    return len(connected_components), subgraph_sizes, none_count

def relation_numOfSubgroup_probabilities(num_bats):
    # 参数设置
    num_bats = num_bats  # 节点数量
    probabilities = np.arange(0, 1.1, 0.1)  # 概率从0到1，每次增加0.1
    simulations = 100000  # 仿真次数

    # 用于存储不同概率下，各群体数量的比例
    groupnum_distribution = {prob: [] for prob in probabilities}


    # 对于每个概率值执行10000次模拟
    for prob in probabilities:
        groupnum_simulations = []  # 用于存储该概率下的所有群体数量
        for _ in range(simulations):
            groupnum, _, _ = draw_and_count_subgraphs(num_bats, prob)
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


def relation_numOfSubgroup_numOfBats(probabilities):
    num_bats = np.arange(100, 400, 100)  # Number of bats
    probabilities = probabilities
    simulations = 100000  # Number of simulations

    # Used to store the proportion of different group numbers for each number of bats
    groupnum_distribution = {num: [] for num in num_bats}
    noneingroupnum_distribution = {num: [] for num in num_bats}  # To store none count distributions

    # For each number of bats, run simulations
    for num in num_bats:
        groupnum_simulations = []  # Store all group numbers for this num of bats
        noneingroupnum_simulations = []  # Store all none counts for this num of bats
        for _ in range(simulations):
            groupnum, _, none_count = draw_and_count_subgraphs(num, probabilities)
            groupnum_simulations.append(groupnum)
            noneingroupnum_simulations.append(none_count)

        # Calculate the distribution of group numbers (proportions)
        unique_groupnums, groupnum_counts = np.unique(groupnum_simulations, return_counts=True)
        groupnum_distribution[num] = dict(zip(unique_groupnums, groupnum_counts / simulations))  # Proportion of groupnums

        # Calculate the distribution of none counts (proportions)
        unique_noneingroupnums, noneingroupnum_counts = np.unique(noneingroupnum_simulations, return_counts=True)
        noneingroupnum_distribution[num] = dict(zip(unique_noneingroupnums, noneingroupnum_counts / simulations))  # Proportion of none counts

    # Plot both groupnum and noneingroupnum distributions
    plt.figure(figsize=(12, 8))

    # Plot groupnum distribution
    for num in num_bats:
        distribution = groupnum_distribution[num]
        groupnums = sorted(distribution.keys())
        proportions = [distribution.get(g, 0) for g in groupnums]
        plt.plot(groupnums, proportions, marker='o', label=f'numOfBats = {num:.1f} - Groupnums')

    # Plot non-followers distribution
    for num in num_bats:
        distribution = noneingroupnum_distribution[num]
        noneingroupnums = sorted(distribution.keys())
        proportions = [distribution.get(n, 0) for n in noneingroupnums]
        plt.plot(noneingroupnums, proportions, marker='x', label=f'numOfBats = {num:.1f} - non-followers')

    #Plot Binomial distribution
    for num in num_bats:
        p = 1 - probabilities
        n = num
        x = np.arange(0, n + 1)
        pmf = binom.pmf(x, n, p)
        plt.plot(x, pmf, marker='*', alpha=0.5, label=f'numOfBats = {num:.1f} - binomial_distribution')


    # Labels and titleFp
    plt.xlabel('Number of Subgroups / non-followers', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.xlim(0, max(groupnums) + 15)
    plt.xticks(np.arange(0, max(groupnums) + 5, 20))
    plt.title(f'Proportion of Different Number of Subgroups and non-followers (P = {probabilities})', fontsize=14)
    plt.legend()
    plt.grid(True)

    # Generate a random filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Current timestamp: 20240617_153045
    random_filename = f"../picture/P_{probabilities}_binomial_distribution_{timestamp}.png"

    # Save the plot
    plt.savefig(random_filename, dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

def relation_numOfSubgroup_numOfBats_probabilities():
    # 参数设置
    probabilities = np.arange(0, 1.01, 0.01)  # 概率从0到1，每次增加0.1
    num_bats_values = np.arange(2, 101, 1)  # 节点数量从2到100，每次增加1
    simulations = 100000  # 仿真次数

    # 存储子群体数量为2的比例
    subgroup_2_proportions = np.zeros((len(probabilities), len(num_bats_values)))

    # 遍历所有概率和节点数量组合，计算子群体数量为2的比例
    for i, prob in enumerate(probabilities):
        for j, num_bats in enumerate(num_bats_values):
            groupnum_simulations = []  # 存储当前组合下的所有群体数量
            for _ in range(simulations):
                groupnum, _, _ = draw_and_count_subgraphs(num_bats, prob)
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


for prob in [i / 10 for i in range(9, 11)]:
    relation_numOfSubgroup_numOfBats(probabilities=prob)


# relation_numOfSubgroup_probabilities(num_bats=100)
# relation_numOfSubgroup_numOfBats_probabilities()

# group, subgraph_sizes, none_count = draw_and_count_subgraphs(42, 0.1)
# print(group, subgraph_sizes, none_count)