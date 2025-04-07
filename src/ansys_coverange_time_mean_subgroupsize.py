import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ==== 配置部分 ====
parent_path = '/Users/tianfu/Desktop/phd/fission:fusion/fission_fusion/data'  # 所有子文件夹的上层目录
folder_names = [
    # 'result_No_Communication',
    # 'result_Minimal_Communication',
    'result_Continuous_Communication',
    'result_swarm'
]  # 你可以根据实际需要扩展这个列表
desired_size = 14  # 期望 group size

# ==== 数据处理 ====
all_data = []
not_reached_files = {}  # key: folder name, value: list of file names

for folder in folder_names:
    folder_path = os.path.join(parent_path, folder)
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, header=None, on_bad_lines='skip')  # [机器人ID, 时间戳, group size]

        # 检查每个机器人是否有时间戳跳跃 >10s
        has_time_jump = False
        for bot_id, group in df.groupby(0):
            time_diffs = group.sort_values(1)[1].diff()
            if (time_diffs > 10).any():
                has_time_jump = True
                break

        if has_time_jump:
            print(f"[SKIP] Time jump >10s found in {file} of {folder}")
            if folder not in not_reached_files:
                not_reached_files[folder] = []
            not_reached_files[folder].append(file)
            continue  # 跳过该文件

        start_time = df[1].min()
        reach_times = []
        all_bots_stable = True  # 标记这一组是否全部机器人都稳定

        for bot_id, group in df.groupby(0):
            group = group.sort_values(1)
            values = group[2].values
            timestamps = group[1].values

            # 如果最后一个数据不是 desired_size，则该机器人未稳定
            if values[-1] != desired_size:
                all_bots_stable = False
                break

            # 向前找到最后一个非 desired_size 的索引
            index = len(values) - 1
            while index >= 0 and values[index] == desired_size:
                index -= 1

            # 记录该 bot 达成稳定状态的起始时间（稳定段的起点）
            if index + 1 < len(timestamps):
                reach_times.append(timestamps[index + 1])
            else:
                # 万一整段都是 desired_size，直接从头就稳定了
                reach_times.append(timestamps[0])

        # 如果有 bot 不稳定，跳过整个文件
        if not all_bots_stable:
            print(f"[SKIP] At least one bot not stable in {file} of {folder}")
            if folder not in not_reached_files:
                not_reached_files[folder] = []
            not_reached_files[folder].append(file)
            continue

        if reach_times:
            last_time = max(reach_times)

            if last_time - start_time > 1200:
                print(f"[SKIP] Max reach time too long (>700s) in {file} of {folder}")
                if folder not in not_reached_files:
                    not_reached_files[folder] = []
                not_reached_files[folder].append(file)
                continue  # 跳过该文件

            stabilization_time = last_time - start_time
            if stabilization_time < 1e5:
                all_data.append({
                    'stabilization_time': stabilization_time,
                    'folder': folder.replace('result_', '').replace('_', ' ')
                })
            else:
                print(f"[WARNING] Unrealistically high time in {file} of {folder}")
        else:
            print(f"[SKIP] No bots reached desired size in {file} of {folder}")
            if folder not in not_reached_files:
                not_reached_files[folder] = []
            not_reached_files[folder].append(file)



df_plot = pd.DataFrame(all_data)

print("\n[INFO] Summary of files where not be convergence")
for folder, files in not_reached_files.items():
    print(f" - {folder}: {len(files)} files skipped")

# ==== 成功率统计（真正保留的成功文件）====
print("\n[INFO] Success Rate Summary (after filtering)")
for folder in folder_names:
    folder_path = os.path.join(parent_path, folder)
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 标记 gap>10s 的文件（不计入总数）
    skipped_due_to_gap = set()
    for file in not_reached_files.get(folder, []):
        # 如果跳过原因是 gap >10s，在 earlier loop 打印了 "[SKIP] Time jump >10s found"
        # 我们假设这一类跳过的文件为 gap 问题
        if f"[SKIP] Time jump >10s found in {file} of {folder}" in open(__file__).read():
            skipped_due_to_gap.add(file)

    total_valid_files = [f for f in all_files if f not in skipped_due_to_gap]
    total = len(total_valid_files)

    # 成功文件 = 出现在 all_data 中的文件数量（已达成 desired size）
    folder_label = folder.replace('result_', '').replace('_', ' ')
    success = sum(1 for d in all_data if d['folder'] == folder_label)

    failed = total - success
    rate = (success / total * 100) if total > 0 else 0

    print(f" - {folder_label}: {success}/{total} files succeeded ({rate:.1f}%)")

# ==== 箱线图数据统计 ====
print("\n[INFO] Boxplot Statistical Summary (per group)")
summary = df_plot.groupby('folder')['stabilization_time'].agg(['count', 'mean', 'std', 'median']).reset_index()
for _, row in summary.iterrows():
    print(f" - {row['folder']}: count={int(row['count'])}, mean={row['mean']:.2f}s, "
          f"std={row['std']:.2f}s, median={row['median']:.2f}s")

# ==== 数据可视化 ====
plt.figure(figsize=(12, 6))
sns.boxplot(
    x='folder',
    y='stabilization_time',
    hue='folder',
    data=df_plot,
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

# plt.title("Convergence Time Distribution", fontsize=16)
plt.xlabel("Communication Strategy", fontsize=18)
plt.ylabel("Convergence Time (s)", fontsize=18)
plt.xticks(fontsize=16, rotation=0)  # rotation=30 适用于长标签
plt.yticks(fontsize=16)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


