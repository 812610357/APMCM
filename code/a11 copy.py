import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
import time

start = time.thread_time()

data = np.array(pd.read_csv(".\code\graph1.csv", header=2))  # 从csv文件获取数据
d = -0.1
plt.axis("equal")
plt.plot(data[:, 0], data[:, 1], '-o', markersize=1)


def findex(data, i):  # 寻找y极值，极大返回1，极小返回2
    if (data[i, 1] >= data[i-1, 1] and data[i, 1] > data[i+1, 1]):
        return(1)
    else:
        if (data[i, 1] <= data[i-1, 1] and data[i, 1] < data[i+1, 1]):
            return(2)
        else:
            return(0)


def unit(v):  # 单位化
    return(v/np.linalg.norm(v))


def angle(v):  # 取辐角
    return(math.atan2(v[1], v[0]))


def inangle(v1, v2):  # 向量夹角
    return(math.acos(np.dot(v1, np.transpose(v2)) / (np.linalg.norm(v1)*np.linalg.norm(v2))))


def draw(data):
    data = np.insert(data, data.shape[0], values=data[1, :], axis=0)
    data = np.insert(data, 0, values=data[data.shape[0]-3, :], axis=0)
    temp = np.array([0, 0])
    times = data.shape[0]-2
    i = 0
    while i < times:
        k = 0
        v1 = data[i+1, :]-data[i, :]
        v2 = data[i+2, :]-data[i+1, :]
        u = d/(math.sin(inangle(v1, v2)))
        if (angle(v2) > angle(v1) and not(angle(v2) > math.pi/2 and angle(v1) < -math.pi/2)) or (angle(v2) < -math.pi/2 and angle(v1) > math.pi/2):
            new = data[i+1, :]+(unit(v2)-unit(v1))*u
        else:
            new = data[i+1, :]-(unit(v2)-unit(v1))*u
        for j in range(0, data.shape[0]-2):
            if np.linalg.norm(new-data[j, :]) < abs(d)*0.999:
                k = 1
                break
        if k == 0:
            temp = np.row_stack((temp, new))
        i += 1
    temp = np.delete(temp, 0, axis=0)

    plt.plot(temp[:, 0], temp[:, 1], '-o', color='r', markersize=2)
    return(temp)


def divide(data):
    op = np.array([0, 0])
    for i in range(1, data.shape[0]-2):
        k = 0
        if findex(data, i+1) == 1:
            for j in range(i+1, -3, -1):
                if data[j, 1] > data[i+1, 1]:
                    if data[j+1, 1] <= data[i+1, 1]:
                        k += 1
                    break
            for j in range(i+1, data.shape[0]-2):
                if data[j, 1] > data[i+1, 1]:
                    if data[j-1, 1] <= data[i+1, 1]:
                        k += 1
                    break
            if k == 2:
                op = np.row_stack(
                    (op, [i+1, math.floor((data[i+1, 1])//abs(d))*abs(d)+abs(d)]))
        else:
            if findex(data, i+1) == 2:
                for j in range(i+1, -100, -1):
                    if data[j, 1] < data[i+1, 1]:
                        if data[j+1, 1] >= data[i+1, 1]:
                            k += 1
                        break
                for j in range(i+1, data.shape[0]-2):
                    if data[j, 1] < data[i+1, 1]:
                        if data[j-1, 1] >= data[i+1, 1]:
                            k += 1
                        break
                if k == 2:
                    op = np.row_stack(
                        (op, [i+1, math.floor((data[i+1, 1])//abs(d))*abs(d)]))
    op = np.delete(op, 0, axis=0)
    print(op)
    return(op)


def getint(data):
    temp = new = np.array([0, 0.])
    for i in range(data.shape[0]-1):
        x1 = data[i, 0]
        y1 = data[i, 1]
        x2 = data[i+1, 0]
        y2 = data[i+1, 1]
        k = (y2-y1)/(x2-x1)  # 差分法取整点
        if y1//abs(d) < y2//abs(d):
            for j in range(1, math.floor(y2//abs(d)-y1//abs(d)+1)):
                new[1] = (y1//abs(d)+j)*abs(d)
                new[0] = (new[1]-y1)/k+x1
                temp = np.row_stack((temp, new))
        else:
            if y1//abs(d) > y2//abs(d):
                for j in range(0, math.floor(y1//abs(d)-y2//abs(d))):
                    new[1] = (y1//abs(d)-j)*abs(d)
                    new[0] = (new[1]-y1)/k+x1
                    temp = np.row_stack((temp, new))
    temp = np.insert(temp, temp.shape[0], values=temp[0, :], axis=0)
    plt.plot(temp[:, 0], temp[:, 1], '-o', color='g', markersize=2)
    return(temp)


data = draw(data)
data = getint(data)
op = divide(data)

end = time.thread_time()
print('Running time: %s Seconds' % (end-start))

plt.show()
