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
n = data.len()
first_x = list([np.max(data[0][:, 1]), np.min(data[0][:, 1])])
a = np.min(data[0][:, 1])
b = np.min(data[1][:, 1])
print(a)
print(b)


def _min(store, coor, data):
    return np.min(data[store][:, coor])


def _max(store, coor, data):
    return np.max(data[store][:, coor])


def divide(data):
    # sto will be out of range
    # sto[本名][0（爹名），1（辈分）]
    sto = list([[1000, n], [1000, n], [1000, n], [1000, n]])
    for i in range(1, 5):  # i,j都是爹名字 ，然后开始找爹
        for j in range(1, 5):
            if _max(i, 0, data) > _max(j, 0, data) and _min(i, 0, data) < _min(j, 0, data):
                # 小的人做儿子，去找爹，大的人坐享其成
                if _max(i, 1, data) > _max(j, 1, data) and _min(i, 1, data) < _min(j, 1, data):
                    if sto[i][1] < sto[j][1]:  # 先认第一个碰到的人做爹，如果碰到第二个人辈份比第一个人的辈分小，就认这个人做爹
                        sto[j] = [i, sto[j][1]-1]  # 自己的辈分-1
                    elif sto[i][1] > sto[j][1]:  # 如果碰到的人比已认做爹的辈分大，就当他老大，不管
                        continue
            elif _max(i, 0, data) > _max(j, 0, data) and _min(i, 0, data) > _min(j, 0, data):
                # 同级不管
                continue
            return sto
