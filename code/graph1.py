import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

data = np.array(pd.read_csv("graph1.csv", header=2))  # 从csv文件获取数据
x0 = np.array(data[:, 0])
y0 = np.array(data[:, 1])
x1 = np.array(data[:, 2])
y1 = np.array(data[:, 3])
x2 = np.array(data[:, 4])
y2 = np.array(data[:, 5])


plt.figure(figsize=(4, 6))
plt.scatter(x0, y0, 1, marker='o',
            color='r',
            label="Graph1")  # 绘制散点图
# plt.scatter(x1, y1, 1, marker='o',
#            color='b',
#            label="Graph1")  # 绘制散点图
plt.scatter(x2, y2, 1, marker='o',
            color='g',
            label="Graph1")  # 绘制散点图

plt.axis("equal")
plt.xlabel("X(mm)")
plt.ylabel("Y(mm)")
plt.legend()
plt.title("Graph1", fontsize=16)
plt.show()
