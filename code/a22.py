import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import argmax, argmin  # 返回最大或最小值对应的序号
import pandas as pd
import math
import time

data = list([])
for i in range(1, 5):
    data.append(np.array(pd.read_csv(
        f".\code\graph2{i}.csv", header=2)))  # 从csv文件获取数据

first_x = list([np.max(data[0][:, 1]), np.min(data[0][:, 1])])
a = np.min(data[0][:, 1])
b = np.min(data[1][:, 1])
print(a)
print(b)


def _min(store, coor):
    return np.min(data[store][:, coor])


def _max(store, coor):
    return np.max(data[store][:, coor])


def divide(data):
    for i in range(1, 5):
        for j in range(1, 5):
