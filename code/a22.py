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


def _min(parentre, cor, data):
    return np.min(data[parentre][:, cor])


def _max(parentre, cor, data):
    return np.max(data[parentre][:, cor])


def divide(data):
    # parent[本名][0（爹名），1（继承数）]
    parent = list([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])
    for i in range(0, 4):  # i,j都是爹名字 ，然后开始找爹
        for j in range(0, 4):
            if i == j:
                continue
            if _max(i, 0, data) > _max(j, 0, data) and _min(i, 0, data) < _min(j, 0, data):
                # 小的人做儿子，去找爹，大的人坐享其成
                if _max(i, 1, data) > _max(j, 1, data) and _min(i, 1, data) < _min(j, 1, data):
                    # 先认第一个碰到的人做爹，如果碰到第二个人继承数比第一个人的继承数小，就认这个人做爹
                    if _max(parent[j][0], 0, data) > _max(i, 0, data) and _min(parent[j][0], 0, data) < _min(i, 0, data) and _max(parent[j][0], 1, data) > _max(i, 1, data) and _min(parent[j][0], 1, data) < _min(i, 1, data):
                        parent[j] = [i, parent[j][1]+1]  # 自己的继承数+1
                    else:  # 如果碰到的人比已认做爹的继承数大，就当他老大，不管
                        continue
            elif _max(i, 0, data) > _max(j, 0, data) and _min(i, 0, data) > _min(j, 0, data):
                # 同级不管
                continue
            return(parent)


print(divide(data))
