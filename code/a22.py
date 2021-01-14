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

n = len(data)
for i in range(0, 4):
    print("小：", np.min(data[i][:, 1]), "大：", np.max(data[i][:, 1]))


def _min(store, cor, data):
    return np.min(data[store][:, cor])


def _max(store, cor, data):
    return np.max(data[store][:, cor])


def divide(data):
    # sto[本名][0（爹名），1（继承数）]
    sto = list([[1000, 1], [1000, 1], [1000, 1], [1000, 1]])
    for i in range(0, 4):  # i,j都是爹名字 ，然后开始找爹
        for j in range(0, 4):
            if _max(i, 0, data) > _max(j, 0, data) and _min(i, 0, data) < _min(j, 0, data):
                # 小的人做儿子，去找爹，大的人坐享其成
                if _max(i, 1, data) > _max(j, 1, data) and _min(i, 1, data) < _min(j, 1, data):
                    if sto[i][1] > sto[j][1]:  # 先认第一个碰到的人做爹，如果碰到第二个人继承数比第一个人的继承数小，就认这个人做爹
                        sto[j] = [i, sto[j][1]+1]  # 自己的继承数+1
                    elif sto[i][1] < sto[j][1]:  # 如果碰到的人比已认做爹的继承数大，就当他老大，不管
                        continue
            elif _max(i, 0, data) > _max(j, 0, data) and _min(i, 0, data) > _min(j, 0, data):
                # 同级不管
                continue
            return sto


print(divide(data))
