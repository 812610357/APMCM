import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import time

plt.axis("equal")
d = -0.1
path = ".\code\graph2.csv"


'''
第一部分
'''


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


def findparent(data):
    # parent[本名][0（父级），1（层数）]
    parent = list([])
    for i in range(len(data)):
        parent.append([-1, 1])
    for i in range(0, len(data)):  # i,j都是父级名字 ，然后开始找父级
        for j in range(i+1, len(data)):
            if range_judge(i, j, data) != -2:
                small_name = range_judge(i, j, data)
                big_name = (i if j == small_name else j)
                parent[small_name][1] += 1
                if range_judge(big_name, parent[small_name][0], data) == big_name or parent[small_name][0] == -1:
                    parent[small_name][0] = big_name  # 自己的层数+1
                else:
                    continue
    return(parent)


'''
第三部分
'''


def writecsv(data, num):  # 导出线条
    dataframe = pd.DataFrame(data={'x': data[:, 0], 'y': data[:, 1]})
    dataframe.to_csv(f".\code\\zigzag{num}.csv",
                     index=False, mode='w', sep=',')
    pass


def readcsv(path):  # 读取线条
    data = list([])
    data0 = pd.read_csv(
        path, index_col=False, header=2)
    j = 0
    if data0.dtypes.X != "float64":
        for i in range(len(data0.values)):
            if "MainCurve" in data0.values[i, 0]:
                data += list([np.array(data0.values[j:i, :], dtype='float64')])
                j = i+2
    data += list([np.array(data0.values[j:len(data0.values), :], dtype='float64')])
    for i in range(len(data)):
        plt.plot(data[i][:, 0], data[i][:, 1], '-o', color='b', markersize=1)
    return(data)


'''
主函数
'''


start = time.thread_time()

data = readcsv(path)
