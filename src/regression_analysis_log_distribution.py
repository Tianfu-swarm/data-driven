import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 读取数据
file_path = 'group_nums_results.h5'
df = pd.read_hdf(file_path,key="df")
print(df)

# 计算 log(1/r), log(1/A), log(n), log(numsofsubgroup)
df['log_1_r'] = np.log(1 / df['follow_range'].clip(lower=1e-10))
df['log_1_A'] = np.log(1 / (df['Area_range'].clip(lower=1e-10) ** 2))
df['log_n'] = np.log(df['num_of_robots'].clip(lower=1e-10))
df['log_numsofsubgroup'] = np.log(df['num_of_subgroups'].clip(lower=1e-10))
print(df)

# 分组计算均值 (mu) 和标准差 (sigma)
grouped = df.groupby(['follow_range', 'Area_range', 'num_of_robots'])['log_numsofsubgroup'].agg(['mean', 'std']).reset_index()
grouped.rename(columns={'mean': 'mu', 'std': 'sigma'}, inplace=True)

# 调试：检查分组后的数据框
print("分组后的数据框：")
print(grouped)

# 合并回 df，确保 grouped 也有 log_1_r, log_1_A, log_n
grouped = grouped.merge(df[['follow_range', 'Area_range', 'num_of_robots', 'log_1_r', 'log_1_A', 'log_n']].drop_duplicates(),
                        on=['follow_range', 'Area_range', 'num_of_robots'], how='left')

# 调试：检查合并后的数据框
print("合并后的数据框：")
print(grouped)

# 删除 NaN 值，避免回归时报错
print("删除 NaN 值前的数据框：")
print(grouped.isnull().sum())
grouped.dropna(inplace=True)

# 调试：检查删除 NaN 值后的数据框
print("删除 NaN 值后的数据框：")
print(grouped)
print("grouped 的形状:", grouped.shape)

# 如果 grouped 为空，则退出程序
if grouped.shape[0] == 0:
    raise ValueError("grouped 数据框为空，请检查数据或分组逻辑。")

##检查这三个变量的相关性
print(grouped[['log_1_r', 'log_1_A', 'log_n']].corr())
#检查σ是否随着变量而变化
# print("σ 的唯一值:", np.unique(grouped['sigma']))


# 继续拟合模型
X_mu = grouped[['log_1_r', 'log_1_A', 'log_n']]
y_mu = grouped['mu']
model_mu = LinearRegression()
model_mu.fit(X_mu, y_mu)

# 拟合 μ 的线性回归模型
X_mu = grouped[['log_1_r', 'log_1_A', 'log_n']]
y_mu = grouped['mu']
model_mu = LinearRegression()
model_mu.fit(X_mu, y_mu)
y_mu_pred = model_mu.predict(X_mu)
r2_mu = r2_score(y_mu, y_mu_pred)
print(f"μ 模型的 R²: {r2_mu:.4f}")
print("μ 模型的系数:", model_mu.coef_)
print("μ 模型的截距:", model_mu.intercept_)

# 拟合 σ 的线性回归模型
X_sigma = grouped[['log_1_r', 'log_1_A', 'log_n']]
y_sigma = grouped['sigma']
model_sigma = LinearRegression()
model_sigma.fit(X_sigma, y_sigma)
y_sigma_pred = model_sigma.predict(X_sigma)
r2_sigma = r2_score(y_sigma, y_sigma_pred)
print(f"σ 模型的 R²: {r2_sigma:.4f}")
print("σ 模型的系数:", model_sigma.coef_)
print("σ 模型的截距:", model_sigma.intercept_)

# 输出 μ 和 σ 的关系公式
def print_relationship(model, target):
    coefficients = model.coef_
    intercept = model.intercept_
    formula = f"{target} = {intercept:.4f}"
    for i, coef in enumerate(coefficients):
        if i == 0:
            formula += f" + {coef:.4f} * log(1/r)"
        elif i == 1:
            formula += f" + {coef:.4f} * log(1/A)"
        elif i == 2:
            formula += f" + {coef:.4f} * log(n)"
    print(formula)

# 输出 μ 的关系
print("\nμ 的关系公式：")
print_relationship(model_mu, "μ")

# 输出 σ 的关系
print("\nσ 的关系公式：")
print_relationship(model_sigma, "σ")
