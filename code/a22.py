import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import time

plt.axis("equal")
d = -0.1
path = ".\code\graph2.csv"
length = 0
times = 0
dots = 0


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
第二部分
'''


def unit(v):  # 单位化
    return(v/np.linalg.norm(v))


def angle(v):  # 取辐角
    return(math.atan2(v[1], v[0]))


def inangle(v1, v2):  # 向量夹角
    return(math.acos(round(np.dot(v1, np.transpose(v2)) / (np.linalg.norm(v1)*np.linalg.norm(v2)), 9)))


def cross(v1, v2):  # 平面内向量叉乘
    return(v1[0]*v2[1]-v2[0]*v1[1])


def ifwide(data, last):  # 与上一级间距控制
    i = 0
    while i < data.shape[0]:  # 遍历该级所有数据
        j = max([0, i-20])
        while j < min(last.shape[0], i+20):  # 遍历上级部分数据
            if np.linalg.norm(data[i, :]-last[j, :]) < abs(d)*0.999:  # 小于一个精度的直接删除
                data[i] = [0, 0]
                break
            else:
                j += 1
        i += 1
    i = 0
    while i < data.shape[0]:
        if data[i, 0] == data[i, 1] == 0:
            data = np.delete(data, i, axis=0)
            continue
        i += 1
    return(data)


def iflong(data):  # 同级点间距控制
    i = 0
    while i < data.shape[0]-1:  # 遍历所有数据
        if np.linalg.norm(data[i+1, :]-data[i, :]) > 2*abs(d):  # 两点间距过大的添加中点
            new = np.array([(data[i+1, 0]+data[i, 0])/2,
                            (data[i+1, 1]+data[i, 1])/2])
            data = np.insert(data, i+1, new,  axis=0)
            continue
        else:
            i = i+1
    return(data)


def draw(data):  # 画等高线
    global dots
    data = np.insert(data, data.shape[0], values=data[1, :], axis=0)
    data = np.insert(data, 0, values=data[data.shape[0]-3, :], axis=0)
    temp = np.array([0, 0])
    i = 0
    while i < data.shape[0]-2:
        v1 = data[i+1, :]-data[i, :]
        v2 = data[i+2, :]-data[i+1, :]
        if 0 < inangle(v1, v2) < math.pi*0.9:  # 一般情况在菱形中使用向量得到内缩点
            u = d/(math.sin(inangle(v1, v2)))
            if cross(v1, v2) > 0:
                new = data[i+1, :]+(unit(v2)-unit(v1))*u
            else:
                new = data[i+1, :]-(unit(v2)-unit(v1))*u
        else:
            if inangle(v1, v2) == 0:  # 两向量平行的特殊情况
                if angle(v1) > 0:
                    new = data[i+1, :] + unit([v1[1], -v1[0]])*abs(d)
                else:
                    new = data[i+1, :] - unit([-v1[1], v1[0]])*abs(d)
            else:  # 排除转角过大的点
                i += 1
                continue
        i += 1
        temp = np.row_stack((temp, new))
        dots += 1
    temp = np.delete(temp, 0, axis=0)
    temp = ifwide(temp, data)  # 与上一级间距控制
    temp = iflong(temp)  # 同级点间距控制
    return(temp)


def ifdivide(data):  # 判断区域划分
    for i in range(data.shape[0]-2):
        for j in range(i, data.shape[0]-2):
            x1 = data[i, 0]
            y1 = data[i, 1]
            x2 = data[j, 0]
            y2 = data[j, 1]
            if 0 < math.sqrt((x2-x1)**2+(y2-y1)**2) < abs(d) and j-i > 3:  # 间距过近且向量方向差超过90度
                v1 = data[i+2, :]-data[i, :]
                v2 = data[j+2, :]-data[j, :]
                if abs(angle(v1)-angle(v2)) > math.pi/2:
                    return(np.array([i, j]))
    return(np.array([0, 0]))


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
