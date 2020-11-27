import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签


def func(x, a1, a2, a3, a4):
    return a1*np.sin(a2*(x+a3))+a4  # 自定义要拟合的函数


data1 = np.array(pd.read_csv("graph21.csv", header=2))  # 从csv文件获取数据
data2 = np.array(pd.read_csv("graph22.csv", header=2))
data3 = np.array(pd.read_csv("graph23.csv", header=2))
data4 = np.array(pd.read_csv("graph24.csv", header=2))
x1 = np.array(data1[:, 0])
y1 = np.array(data1[:, 1])
x2 = np.array(data2[:, 0])
y2 = np.array(data2[:, 1])
x3 = np.array(data3[:, 0])
y3 = np.array(data3[:, 1])
x4 = np.array(data4[:, 0])
y4 = np.array(data4[:, 1])

plt.figure(figsize=(4, 6))
plt.scatter(x1, y1, 3, marker='o',
            color='r',
            label="Graph21")  # 绘制散点图
plt.scatter(x2, y2, 3, marker='o',
            color='r',
            label="Graph22")  # 绘制散点图
plt.scatter(x3, y3, 3, marker='o',
            color='r',
            label="Graph23")  # 绘制散点图
plt.scatter(x4, y4, 3, marker='o',
            color='r',
            label="Graph24")  # 绘制散点图

plt.axis("equal")
plt.xlabel("X(mm)")
plt.ylabel("Y(mm)")
plt.legend()
plt.title("Graph1", fontsize=16)
plt.show()
