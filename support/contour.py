import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data0 = np.array(pd.read_csv("graph1.csv", header=2))
plt.plot(data0[:, 0], data0[:, 1], '-o', markersize=1)
for i in range(1, 130):
    data = np.array(pd.read_csv(
        f"contour{i}.csv", header=0))  # 从csv文件获取数据
    plt.plot(data[:, 0], data[:, 1], '-', color='r')

print("Length of curve: 10060.3875423606 mm")
print("Number of turns: 129")
print("Number of dots:  78578")

plt.axis("equal")
plt.xlabel("X(mm)")
plt.ylabel("Y(mm)")
plt.title("Graph1 contour", fontsize=16)
plt.show()
