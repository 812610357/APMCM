import numpy as np
import pandas as pd

data = list([])
for i in range(1, 5):
    data[i-1] = np.array(pd.read_csv(
        f".\code\graph2{i}.csv", header=2))  # 从csv文件获取数据

print("ha")
