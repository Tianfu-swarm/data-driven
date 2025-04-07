import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 读取数据
file_path = '../data/group_nums_results_3.h5'
df = pd.read_hdf(file_path,key="df")
print(df)

