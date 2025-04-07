import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib.ticker import MultipleLocator

# 读取数据
file_path = '/Users/tianfu/Desktop/phd/fission:fusion/fission_fusion/data/result_swarm/task_1-10.csv'
df = pd.read_csv(file_path, header=None)  # 假设数据没有列名
print(df)

# 解析数据并按键值分组
grouped_data = defaultdict(list)
timestamps = defaultdict(list)

# 按照时间戳排序
df = df.sort_values(by=[1])  # 按第二列（时间戳）排序

# 找到最小时间戳
min_timestamp = df[1].min()

for index, row in df.iterrows():
    key, timestamp, value = row.iloc[0], row.iloc[1], row.iloc[2]  # 第一列是组名，第二列是时间戳，第三列是值
    grouped_data[key].append(int(value))
    timestamps[key].append(timestamp - min_timestamp)  # 将时间戳归一化到从 0 开始

# 对 key 进行排序，按照 bot 后的数字大小排序
sorted_keys = sorted(grouped_data.keys(), key=lambda x: int(x.split('bot')[-1]))

# 生成排序后的 grouped_data 和 timestamps
sorted_grouped_data = {key: grouped_data[key] for key in sorted_keys}

# 更新 grouped_data 和 timestamps 以确保它们是按 key 的顺序存储的
grouped_data = sorted_grouped_data

# 计算每个组的数据个数
group_counts = {key: len(values) for key, values in grouped_data.items()}

# 打印每个组的数据个数
for key, count in group_counts.items():
    print(f"Group {key}: {count} data points")

# 找到最长的序列长度
max_length = max(len(v) for v in grouped_data.values())

#
# 统一长度，短的填充 NaN
for key in grouped_data:
    while len(grouped_data[key]) < max_length:
        grouped_data[key].append(None)
    while len(timestamps[key]) < max_length:
        timestamps[key].append(None)

# 转换为 DataFrame
df_values = pd.DataFrame({key: values[:max_length] for key, values in grouped_data.items()})
df_timestamps = pd.DataFrame({key: values[:max_length] for key, values in timestamps.items()})

# df_values = pd.DataFrame.from_dict(grouped_data, orient='index').transpose()
# df_timestamps = pd.DataFrame.from_dict(timestamps, orient='index').transpose()

print(df_values)

# 绘制折线图
plt.figure(figsize=(20, 10))
for column in df_values.columns:
    plt.plot(df_timestamps[column], df_values[column], marker='o', label=column)

# 添加值为 14 的虚线，设置粗细为 2
plt.axhline(y=14, color='red', linestyle='--', linewidth=2, label='Desired Group Size')

# 设置横坐标每 50 秒标注一次
ax = plt.gca()  # 获取当前坐标轴
ax.xaxis.set_major_locator(MultipleLocator(50))

y_min, y_max = plt.ylim()  # 获取当前 y 轴范围
yticks = np.arange(0, y_max + 1, 2)

# 设置横坐标从 0 开始
ax.set_xlim(left=0)  # 横坐标从 0 开始

plt.xlabel("Time (seconds)", fontsize=18)  # 横坐标标签字体大小
plt.ylabel("Size", fontsize=18)  # 纵坐标标签字体大小
# plt.title("Group Size Over Time", fontsize=16)  # 标题字体大小
plt.xticks(fontsize=16)  # 横坐标刻度字体大小
plt.yticks(yticks,fontsize=16)  # 纵坐标刻度字体大小
plt.legend(ncol=3, fontsize=16)  # 图例字体大小


plt.grid(True)
# plt.xticks(rotation=45)  # 旋转横坐标标签，避免重叠
plt.tight_layout()  # 调整布局
plt.show()

###############  video  ############
import shutil
def generate_vedio():
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(20, 10))
    lines = {}

    # 初始化每条折线
    for column in df_values.columns:
        (line,) = ax.plot([], [], marker='o', label=column)
        lines[column] = line

    # 添加红色参考线
    ax.axhline(y=14, color='red', linestyle='--', linewidth=2, label='Desired Group Size')

    # 设置轴参数
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.set_xlim(0, df_timestamps.max().max() + 10)
    ax.set_ylim(0, df_values.max().max() + 2)

    ax.set_xlabel("Time (seconds)", fontsize=14)
    ax.set_ylabel("Size", fontsize=14)
    ax.set_title("Group Size Over Time", fontsize=16)
    ax.legend(ncol=3, fontsize=12)
    ax.grid(True)

    # 更新函数
    def update(frame):
        for column in df_values.columns:
            x_data = df_timestamps[column][:frame + 1]
            y_data = df_values[column][:frame + 1]
            lines[column].set_data(x_data, y_data)
        return list(lines.values())

    # 创建动画
    fps = 10
    interval = 1000 / fps
    num_frames = max_length

    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)

    # 保存：自动判断使用 mp4 还是 gif
    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=fps)
    ani.save("../data/group_size_over_time.mp4", writer=writer)

    plt.close()
