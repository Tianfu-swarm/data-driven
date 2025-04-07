import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==== 配置部分 ====
parent_path = '/Users/tianfu/Desktop/phd/fission:fusion/fission_fusion/data'  # CSV 文件的上层目录
folder_names = [
    'result_No_Communication',
    'result_Minimal_Communication',
    'result_Continuous_Communication'
]
desired_size = 14  # 希望的 group size

# ==== 主统计逻辑 ====
oversize_time_stats = []  # 每个 CSV 文件一条记录：{'folder': ..., 'avg_oversize_time': ...}

for folder in folder_names:
    folder_path = os.path.join(parent_path, folder)
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path, header=None, on_bad_lines='skip')  # [机器人ID, 时间戳, group size]
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")
            continue

        # ==== 剔除有时间跳跃的文件 ====
        has_time_jump = False
        for bot_id, group in df.groupby(0):
            group = group.sort_values(1)
            if (group[1].diff() > 10).any():
                has_time_jump = True
                break

        if has_time_jump:
            print(f"[SKIP] Time gap > 10s in file: {file} of folder: {folder}")
            continue

        # ==== 统计 oversize 时间 ====
        oversize_duration_total = 0
        for bot_id, group in df.groupby(0):
            group = group.sort_values(1)
            values = group[2].values
            timestamps = group[1].values

            for i in range(1, len(values)):
                if values[i - 1] > desired_size:
                    dt = timestamps[i] - timestamps[i - 1]
                    if 0 < dt < 1e5:
                        oversize_duration_total += dt

        robot_count = df[0].nunique()
        if robot_count > 0:
            avg_oversize_time = oversize_duration_total / robot_count
            oversize_time_stats.append({
                'folder': folder.replace('result_', '').replace('_', ' '),
                'avg_oversize_time': avg_oversize_time
            })
        else:
            print(f"[WARNING] No robots found in {file} of {folder}, skipping.")

# ==== 可视化（箱线图） ====
df_box = pd.DataFrame(oversize_time_stats)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x='folder',
    y='avg_oversize_time',
    data=df_box,
    palette='pastel',
    width=0.5,
    fliersize=6,
    flierprops=dict(marker='o', color='red', alpha=0.6),
    linewidth=2,
    boxprops=dict(edgecolor='black'),
    medianprops=dict(color='red', linewidth=2),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
)

plt.xlabel("Communication Strategy", fontsize=16)
plt.ylabel("Avg. Oversize Time per Robot (s)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# ==== 打印统计信息 ====
print("\n[INFO] Summary stats for boxplot:")
for folder in df_box['folder'].unique():
    vals = df_box[df_box['folder'] == folder]['avg_oversize_time']
    print(f"{folder}: mean = {vals.mean():.2f}, std = {vals.std():.2f}, count = {len(vals)}")

plt.show()
