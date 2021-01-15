import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import time
data = list([])
for i in range(1, 5):
    data.append(np.array(pd.read_csv(
        f".\code\graph2{i}.csv", header=2)))  # 从csv文件获取数据

n = len(data)
for i in range(0, 4):
    print(i, "小：", np.min(data[i][:, 1]), "大：", np.max(data[i][:, 1]))


def _min(parentre, cor, data):
    return np.min(data[parentre][:, cor])


def _max(parentre, cor, data):
    return np.max(data[parentre][:, cor])


def range_judge(i, j, data):
    if _max(i, 0, data) > _max(j, 0, data) and _min(i, 0, data) < _min(j, 0, data) and _max(i, 1, data) > _max(j, 1, data) and _min(i, 1, data) < _min(j, 1, data):  # i和j比较，如果是包含关系，就返回小的那一个，如果是不是包含关系，就返回0
        return j
    elif _max(i, 0, data) < _max(j, 0, data) and _min(i, 0, data) > _min(j, 0, data) and _max(i, 1, data) < _max(j, 1, data) and _min(i, 1, data) > _min(j, 1, data):
        return i
    else:
        return -2


'''
for i in range(0, 4):
    for j in range(i+1, 4):
        print(i, j, range_judge(i, j, data))
'''


def divide(data):
    # parent[本名][0（爹名），1（继承数）]
    parent = list([])
    for i in range(n):
        parent.append([-1, 1])
    for i in range(0, n):  # i,j都是爹名字 ，然后开始找爹
        for j in range(i+1, n):
            if range_judge(i, j, data) != -2:  # 每两个人只会比较一次
                small_name = range_judge(i, j, data)
                big_name = (i if j == small_name else j)
                parent[small_name][1] += 1
                # 小的人做儿子，去找爹，大的人坐享其成
                # 先认第一个碰到的人做爹，如果碰到第二个人继承数比第一个人的继承数小，就认这个人做爹
                if range_judge(big_name, parent[small_name][0], data) == big_name or parent[small_name][0] == -1:
                    parent[small_name][0] = big_name  # 自己的继承数+1
                else:  # 如果碰到的人比已认做爹的继承数大，就当他老大，不管
                    continue
    return(parent)


print(divide(data))
