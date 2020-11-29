from operator import truediv
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import array
from numpy.core.fromnumeric import shape
import pandas as pd
import os
import math
import time

start = time.thread_time()

data0 = np.array(pd.read_csv(".\code\graph1.csv", header=2))
data = data0  # 从csv文件获取数据
d = -0.1  # 精度
plt.axis("equal")
plt.plot(data[:, 0], data[:, 1], '-o', markersize=1)


def unit(v):  # 单位化
    return(v/np.linalg.norm(v))


def angle(v):  # 取辐角
    return(math.atan2(v[1], v[0]))


def inangle(v1, v2):  # 向量夹角
    return(math.acos(round(np.dot(v1, np.transpose(v2)) / (np.linalg.norm(v1)*np.linalg.norm(v2)), 9)))


def draw(data):  # 画线
    data = np.insert(data, data.shape[0], values=data[1, :], axis=0)
    data = np.insert(data, 0, values=data[data.shape[0]-3, :], axis=0)
    temp = np.array([0, 0])
    i = 0
    while i < data.shape[0]-2:
        v1 = data[i+1, :]-data[i, :]
        v2 = data[i+2, :]-data[i+1, :]
        if 0 < inangle(v1, v2) < math.pi*0.9:  # 一般情况在菱形中使用向量得到内缩点
            u = d/(math.sin(inangle(v1, v2)))
            if (angle(v2) > angle(v1) and not(angle(v2) > math.pi/2 and angle(v1) < -math.pi/2)) or (angle(v2) < -math.pi/2 and angle(v1) > math.pi/2):
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
    temp = np.delete(temp, 0, axis=0)
    temp = iflong(temp)  # 同级点间距控制
    temp = ifcross(temp)  # 交叉控制
    temp = ifwide(temp, data)  # 与上一级间距控制
    plt.plot(temp[:, 0], temp[:, 1], '-', color='r')
    return(temp)


def iflong(data):  # 同级点间距控制
    i = 0
    while i < data.shape[0]-1:
        if np.linalg.norm(data[i+1, :]-data[i, :]) > 2*abs(d):  # 两点间距过大的添加中点
            new = np.array([(data[i+1, 0]+data[i, 0])/2,
                            (data[i+1, 1]+data[i, 1])/2])
            data = np.insert(data, i+1, new,  axis=0)
            continue
        else:
            i = i+1
    return(data)


def ifwide(data, last):  # 与上一级间距控制
    i = 0
    while i < data.shape[0]:
        j = 0
        while j < last.shape[0]:
            if i >= data.shape[0]:
                break
            if np.linalg.norm(data[i, :]-last[j, :]) < abs(d)*0.999:  # 小于一个间距的直接删除
                data = np.delete(data, i, axis=0)
                j = 0
            else:
                j += 1
        i += 1
    return(data)


def ifcross(data):  # 交叉控制
    j = 1
    i = 0
    while j:
        j = 0
        while i < data.shape[0]-3:
            v1 = data[i+1, :]-data[i, :]
            v2 = data[i+2, :]-data[i+1, :]
            v3 = data[i+3, :]-data[i+2, :]
            if inangle(v1, v2)+inangle(v2, v3) > math.pi:  # 连续三个向量转角超过180度直接删除
                data = np.delete(data, [i+1, i+2], axis=0)
                j = 1
                break
            i += 1
    return(data)


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


def drawline(data):
    length = 0
    times = 0
    while True:
        temp = data[-1]
        if data[0].shape[0] < 10:
            break
        if temp.shape[0] < 10:
            del data[len(data)-1]
            print('-')
            continue
        index = ifdivide(temp)  # 分割点序号
        if index[0] == 0 and index[1] == 0:
            data[-1] = draw(temp)
            print(times)
            times += 1
            for j in range(data[-1].shape[0]-1):
                length = length + \
                    math.sqrt((data[-1][j+1, 0]-data[-1][j, 0])**2 +
                              (data[-1][j+1, 1]-data[-1][j, 1])**2)
            # plt.plot(data0[:, 0], data0[:, 1], '-o', color='b', markersize=1)
            # plt.show()
            # plt.axis("equal")
        else:
            data.append(temp[math.floor(index[0])+1: math.floor(index[1]), :])
            data[-1] = np.row_stack((data[-1], data[-1][0:1, :]))
            temp1 = temp[0:math.floor(index[0])+1, :]
            temp2 = temp[math.floor(index[1]):temp.shape[0], :]
            data[-2] = np.row_stack((temp1, temp2))
    return([length, times])


data = list([data])
data = drawline(data)

end = time.thread_time()
print('Length of curve: %s mm' % data[0])
print('Number of turns: %s' % data[1])
print('Running time:    %s Seconds' % (end-start))

plt.show()
