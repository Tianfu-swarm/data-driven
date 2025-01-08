import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


def plot_binomial_pmf_line(n, p):
    """
    可视化二项分布的概率质量函数（PMF）

    参数:
    - n: 试验次数
    - p: 单次试验成功的概率
    - k: 高亮显示的值（可以为 None）
    """
    # 生成可能的成功次数
    x = np.arange(0, n + 1)

    # 计算 PMF
    pmf = binom.pmf(x, n, p)

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(x, pmf, marker='o', color='blue', label="PMF", linewidth=2, markersize=8)

    # 设置图表标题和标签
    plt.title(f"Binomial Distribution (n={n}, p={1-p})", fontsize=16)
    plt.xlabel("Number of Not Successes", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.xticks(np.arange(0, n + 1, step=max(1, n // 20)))
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 显示图表
    plt.show()


# 示例用法
n = 100  # 试验次数
p = 0.1  # 单次成功的概率


# 调用函数
plot_binomial_pmf_line(n, p)
