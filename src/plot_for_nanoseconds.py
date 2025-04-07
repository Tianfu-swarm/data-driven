import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
file_path = '../data/groupsize_data_bot0_use_sim_timestamp.csv'
df = pd.read_csv(file_path, header=None, names=['group', 'timestamp', 'value'])

# 确保时间戳是整数
df['timestamp'] = df['timestamp'].astype(float).astype(int)  # 避免科学计数法

# 归一化时间戳（转换为秒）
df['timestamp_seconds'] = df['timestamp'] / 1e9  # 转换成秒
df['relative_timestamp_seconds'] = (df['timestamp'] - df['timestamp'].min()) / 1e9

# 按时间排序
df = df.sort_values(by=['timestamp'])

# 打印前几行，检查数据是否正确
print(df.head())

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(df['relative_timestamp_seconds'], df['value'], marker='o', label='Group /bot0')

plt.xlabel("Time (seconds)")
plt.ylabel("Size")
plt.title("Group Size Over Time")
plt.legend()
plt.grid(True)
plt.show()
