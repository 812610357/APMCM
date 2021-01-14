import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for i in range(1, 5):
    data = np.array(pd.read_csv(
        f".\code\graph2{i}.csv", header=2))  # 从csv文件获取数据
    plt.plot(data[:, 0], data[:, 1], '-o', color='r', markersize=3)

plt.axis("equal")
plt.xlabel("X(mm)")
plt.ylabel("Y(mm)")
plt.title("Graph1 contour", fontsize=16)
plt.show()
