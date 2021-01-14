import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import argmax, argmin
import pandas as pd
import math
import time

start = time.thread_time()

data = np.array(pd.read_csv(".\code\graph1.csv", header=2))  # 从csv文件获取数据
d = -1
plt.axis("equal")
plt.plot(data[:, 0], data[:, 1], '-o', markersize=1)
dots = 0


def findex(data, i):  # 极值返回1
    if (data[i, 1] >= data[i-1, 1] and data[i, 1] > data[i+1, 1]) or (data[i, 1] <= data[i-1, 1] and data[i, 1] < data[i+1, 1]):
        return(1)
    else:
        return(0)


def unit(v):  # 单位化
    return(v/np.linalg.norm(v))


def inangle(v1, v2):  # 向量夹角
    return(math.acos(np.dot(v1, np.transpose(v2)) / (np.linalg.norm(v1)*np.linalg.norm(v2))))


def cross(v1, v2):
    return(v1[0]*v2[1]-v2[0]*v1[1])


def ifcross(p1, p2, q1, q2):
    v11 = q1-p1
    v12 = q2-p1
    v21 = q1-p2
    v22 = q2-p2
    if cross(v11, v12)*cross(v21, v22) < 0 and cross(v11, v21)*cross(v12, v22) < 0:
        return(1)
    else:
        return(0)


def drawborder(data):  # 内缩一次
    data = np.insert(data, data.shape[0], values=data[1, :], axis=0)
    data = np.insert(data, 0, values=data[data.shape[0]-3, :], axis=0)
    temp = np.array([0, 0])
    times = data.shape[0]-2
    i = 0
    while i < times:
        v1 = data[i+1, :]-data[i, :]
        v2 = data[i+2, :]-data[i+1, :]
        u = d/(math.sin(inangle(v1, v2)))
        if cross(v1, v2) > 0:
            new = data[i+1, :]+(unit(v2)-unit(v1))*u
        else:
            new = data[i+1, :]-(unit(v2)-unit(v1))*u
        temp = np.row_stack((temp, new))
        i += 1
    temp = np.delete(temp, 0, axis=0)
    i = 0
    while i < temp.shape[0]-3:
        j = i
        while j < temp.shape[0]-1:
            if ifcross(temp[i, :], temp[i+1, :], temp[j, :], temp[j+1, :]):
                temp = np.row_stack((temp[0:i, :], temp[j+1:, :]))
                continue
            else:
                j += 1
        i += 1
    return(temp)


def getint(data):  # 按精度离散化
    temp = new = np.array([0, 0.])
    for i in range(data.shape[0]-1):
        x1 = data[i, 0]
        y1 = data[i, 1]
        x2 = data[i+1, 0]
        y2 = data[i+1, 1]
        k = (y2-y1)/(x2-x1)  # 差分法
        if y1//abs(d) < y2//abs(d):
            for j in range(1, math.floor(y2//abs(d)-y1//abs(d)+1)):
                new[1] = round((y1//abs(d)+j)*abs(d), 1)
                new[0] = (new[1]-y1)/k+x1
                temp = np.row_stack((temp, new))
        else:
            if y1//abs(d) > y2//abs(d):
                for j in range(0, math.floor(y1//abs(d)-y2//abs(d))):
                    new[1] = round((y1//abs(d)-j)*abs(d), 1)
                    new[0] = (new[1]-y1)/k+x1
                    temp = np.row_stack((temp, new))
    temp = np.delete(temp, 0, axis=0)
    return(temp)


def getdline(data):  # 确定分割线，返回分割点序号
    temp = 0
    dline = np.array([0, 0, 0])
    for i in range(1, data.shape[0]-2):
        k = 0
        if findex(data, i) == 1:
            for j in range(data.shape[0]*2//3):
                if j+i >= data.shape[0]-1:
                    l = j-data.shape[0]
                else:
                    l = j
                if data[i+l+1, 1] == data[i, 1]:
                    temp = i+l+1
                    if data[i+l+1, 0] > data[i, 0]:
                        k += 2
                    else:
                        k += 1
                    break
            for j in range(data.shape[0]*2//3):
                if data[i-j-2, 1] == data[i, 1]:
                    if data[i-j-2, 0] > data[i, 0]:
                        k += 2
                    else:
                        k += 1
                    break
            if k == 3:
                dline = np.array([data[i, 1], i, temp])
                print(dline)
                dots = np.array([data[math.floor(dline[1]), :],
                                 data[math.floor(dline[2]), :]])
                plt.plot(dots[:, 0], dots[:, 1], '--o',
                         color='g', markersize=2)
                return(dline)
    print(dline)
    return(dline)


def divide(data):  # 获得分割区域，并导入序列
    i = 0
    while True:
        dline = getdline(data[i])
        if dline[1] == 0 and dline[2] == 0:
            i += 1
        else:
            temp = data[i]
            data.append(temp[math.floor(dline[1]): math.floor(dline[2])+1, :])
            temp1 = temp[0:math.floor(dline[1])+1, :]
            temp2 = temp[math.floor(dline[2]):temp.shape[0], :]
            data[i] = np.row_stack((temp1, temp2))
            continue
        if i == len(data):
            break
    return(data)


def writecsv(data, num):  # 导出线条
    dataframe = pd.DataFrame(data={'x': data[:, 0], 'y': data[:, 1]})
    dataframe.to_csv(f".\code\\zigzag{num}.csv",
                     index=False, mode='w', sep=',')
    pass


def drawline(data):  # 画平行线
    global dots
    length = 0  # 画线总长
    times = 0  # 平行线数量
    for i in range(len(data)):
        line = np.array([0, 0])
        area = data[i]
        maxy = round(max(area[:, 1]), 1)
        miny = round(min(area[:, 1]), 1)
        j = miny
        while j <= maxy:
            index = (np.where(area == j))[0]
            temp = area[index, 0]
            if round(j/abs(d)) % 2:
                line = np.row_stack((line, [j, min(temp)]))
                temp = np.delete(temp, argmin(temp))
                line = np.row_stack((line, [j, min(temp)]))
            else:
                line = np.row_stack((line, [j, max(temp)]))
                temp = np.delete(temp, argmax(temp))
                line = np.row_stack((line, [j, max(temp)]))
            j = round(j + abs(d), 1)
        line = np.delete(line, 0, axis=0)
        line = np.column_stack((line[:, 1], line[:, 0]))
        writecsv(line, i+1)
        plt.plot(line[:, 0], line[:, 1], '-', color='r')
        times = times+int(line.shape[0]/2)
        for j in range(line.shape[0]-1):
            length = length + \
                math.sqrt((line[j+1, 0]-line[j, 0])**2 +
                          (line[j+1, 1]-line[j, 1])**2)
            dots += 1
        i += 1
    return([length, times])


data = drawborder(data)
data = getint(data)
data = list([data])
data = divide(data)
data = drawline(data)

end = time.thread_time()
print('Length of curve:         %s mm' % data[0])
print('Number of parallel line: %s' % data[1])
print('Number of dots:          %s' % dots)
print('Running time:            %s Seconds' % (end-start))

plt.show()
