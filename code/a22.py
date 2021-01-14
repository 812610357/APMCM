import numpy as np
import pandas as pd

data = list([])
for i in range(1, 5):
    data.append(np.array(pd.read_csv(
        f".\code\graph2{i}.csv", header=2)))  # 从csv文件获取数据

a=np.max(data[0][1, :])

print(a)
