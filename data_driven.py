import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import datetime


def generate_bat_choices(num_bats=42):
    bats = list(range(1, num_bats + 1))
    choices = {}
    for bat in bats:
        possible_choices = [i for i in bats if i != bat]
        chosen_one = random.choice(possible_choices)
        choices[bat] = chosen_one
    return choices


def find_loops(choices):
    visited = set()  # 用于跟踪已访问的节点
    loops = []  # 用于存储所有找到的循环

    # 遍历所有 bat，从每个 bat 开始查找循环
    for start_bat in choices:
        if start_bat in visited:
            continue  # 如果 bat 已经被访问过，则跳过

        path = []  # 当前路径
        path_set = set()  # 当前路径的节点集合，用于快速查找环
        current_bat = start_bat

        # 遍历链直到遇到环或到达未被选择的节点
        while current_bat not in path_set and current_bat not in visited:
            path.append(current_bat)
            path_set.add(current_bat)
            current_bat = choices[current_bat]  # 跳到下一个 bat

        # 如果当前节点回到了起始节点，表示找到了一个环
        if current_bat in path_set:
            loop_start_index = path.index(current_bat)
            loop = path[loop_start_index:]  # 找到循环的所有节点
            loops.append(loop)  # 将循环添加到结果列表中
            visited.update(loop)  # 标记循环中的节点为已访问
        else:
            # 如果没有形成循环，则将当前路径的节点标记为已访问
            visited.update(path)

    return loops

def expand_loop_layers(choices, loop):
    # Initialize loop members and the first layer
    loop_set = set(loop)  # Keep track of all visited nodes
    current_layer = loop  # Nodes in the current layer
    layers = {1: list(loop)}  # Start with the given loop as the first layer

    layer_index = 1
    while current_layer:
        next_layer = []
        # Find all bats pointing to nodes in the current layer
        for bat, choice in choices.items():
            if bat not in loop_set and choice in current_layer:
                next_layer.append(bat)
                loop_set.add(bat)  # Mark as visited

        # Move to the next layer if any bats are found
        if next_layer:
            layer_index += 1
            layers[layer_index] = next_layer
        current_layer = next_layer

    return layers

def test_loop(num_bats,loop_num):
    group_sizes = []  # 用于保存每次循环的 group 大小

    # 运行多次过程
    for _ in range(loop_num):
        choices = generate_bat_choices(num_bats)

        # print("\nFinding loops:")
        loops = find_loops(choices)

        loop_layers = {}
        group_index = 1  # 用于命名 group1, group2, ...

        # 每次的 group 大小列表
        current_group_sizes = []

        for loop in loops:
            # print("\nLoop members:")
            # print(" -> ".join(f"Bat {bat}" for bat in loop))
            #
            # print("\nExpanding layers around the loop:")
            layers = expand_loop_layers(choices, loop)
            loop_layers.update(layers)

            # 计算当前 loop 和其相关层的总 bats 数量
            total_bats = sum(len(bats) for bats in layers.values())
            # print(f"Group{group_index}: Total bats connected to the loop: {total_bats}")
            current_group_sizes.append(total_bats)
            group_index += 1

            # for layer, bats in layers.items():
            #     print(f"Layer {layer}: {', '.join(f'Bat {bat}' for bat in bats)}")

        # 记录当前轮次的 group 总数
        group_sizes.append(current_group_sizes)


    return group_sizes


def plot_group_sizes(group):
    # 提取每个组大小的数量
    all_sizes = []

    for sizes in group:
        all_sizes.extend(sizes)  # 将每次循环的组大小展开到一个单一的列表

    # 统计每个组大小出现的次数
    size_counts = Counter(all_sizes)

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(size_counts.keys(), size_counts.values(), color='skyblue')

    # 设置图表的标题和标签
    plt.title('Frequency of Group Sizes', fontsize=14)
    plt.xlabel('Group Size', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    import os
    save_dir = "/Users/tianfu/Desktop"
    save_path = os.path.join(save_dir, f"1-frequency of group sizes.png")
    plt.savefig(save_path, format='png', dpi=300)

    # 显示图表
    plt.xticks(list(size_counts.keys()))  # 设置x轴的刻度为组大小
    plt.show()


def plot_group_numbers_frequency_by_size(group):
    # 统计每个 group 大小内数字的频率
    # 遍历所有可能的分组大小，从2到21
    for group_size in range(2, 22):  # 2 到 21
        # 提取所有大小为 group_size 的分组
        selected_numbers = [num for sizes in group if len(sizes) == group_size for num in sizes]

        if selected_numbers:  # 如果有大小为 group_size 的分组
            # 统计数字的频率
            number_counts = Counter(selected_numbers)

            # 绘制柱状图
            plt.figure(figsize=(12, 8))
            numbers = list(number_counts.keys())
            counts = list(number_counts.values())

            plt.bar(numbers, counts, color='skyblue')

            # 设置图表的标题和标签
            plt.title(f'Frequency of Numbers in Subgroups of Size {group_size}', fontsize=16)
            plt.xlabel('Number', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)


            import os
            save_dir = "/Users/tianfu/Desktop"
            save_path = os.path.join(save_dir, f"{group_size}-frequency of nums in subgroup of size {group_size}.png")
            plt.savefig(save_path, format='png', dpi=300)

            # 显示图表
            plt.tight_layout()
            plt.show()



def plot_group_size_frequency(group):
    # 统计每个分组的大小出现的频率
    group_size_counts = Counter(len(sizes) for sizes in group)

    # 计算总的分组数
    total_groups = sum(group_size_counts.values())

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    sizes = list(group_size_counts.keys())
    counts = list(group_size_counts.values())

    bars = plt.bar(sizes, counts, color='skyblue')

    # 在柱子上方显示百分比
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_groups) * 100  # 计算百分比
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{percentage:.4f}%',
                 ha='center', va='bottom', fontsize=12)

    # 设置图表的标题和标签ßß
    plt.title('Frequency of Number of Subgroup', fontsize=16)
    plt.xlabel('Number of Subgroup', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    import os
    save_dir = "/Users/tianfu/Desktop"
    save_path = os.path.join(save_dir, f"-frequency of num of subgroup.png")
    plt.savefig(save_path, format='png', dpi=300)

    # 显示图表
    plt.tight_layout()
    plt.show()

def main():
    num_bats = 1000
    loop_num = 1000000
    group = test_loop(num_bats,loop_num)

    # 输出每次循环的 group 数据
    # print("\nGroup sizes across all runs:")
    # for index, sizes in enumerate(group, start=1):
    #     print(f"Run {index}: {sizes}")

    plot_group_sizes(group)

    plot_group_numbers_frequency_by_size(group)

    plot_group_size_frequency(group)

    now = datetime.datetime.now()
    print("finish running at :", now)

if __name__ == "__main__":
    main()
