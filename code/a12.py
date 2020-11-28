import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import array
import pandas as pd
import os
import math
import time

start = time.thread_time()

data = np.array(pd.read_csv(".\code\graph1.csv", header=2))  # 从csv文件获取数据
d = -1
plt.axis("equal")
plt.plot(data[:, 0], data[:, 1], '-o', markersize=1)


def findex(v1, v2):
    x1 = v1[0]
    y1 = v1[1]
    x2 = (v1+v2)[0]
    y2 = (v1+v2)[1]
    if (x1 > 0 and x1 > x2) or (x1 < 0 and x1 < x2):
        return(1)
    else:
        if(y1 > 0 and y1 > y2) or (y1 < 0 and y1 < y2):
            return(2)
        else:
            return(0)


def unit(v):  # 单位化
    return(v/np.linalg.norm(v))


def angle(v):  # 取辐角
    return(math.atan2(v[1], v[0]))


def inangle(v1, v2):  # 向量夹角
    return(math.acos(np.dot(v1, np.transpose(v2)) / (np.linalg.norm(v1)*np.linalg.norm(v2))))


def roc(v1, v2):  # 外接圆半径
    x1 = v1[0]
    y1 = v1[1]
    x2 = v2[0]
    y2 = v2[1]
    if x1*y2 == x2*y1:
        return(0)
    else:
        xc = ((y1+y2)*y1*y1+x1 ** 2*y2+x2 ** 2*y1)/(x1*y2-x2*y1)
        yc = ((x1+x2)*x1*x2+y1 ** 2*x2+y2 ** 2*x1)/(x2*y1-x1*y2)
        return(math.sqrt(xc ** 2+yc ** 2)/2)


def icr(v1, v2):  # 内切圆半径
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    c = np.linalg.norm(v1+v2)
    r = math.sqrt((a+b-c)*(a-b+c)*(-a+b+c)/(a+b+c))/2
    theta = inangle(v1, v2)
    return(r/math.sin(theta/2))


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
        if 1:  # roc(v1, v2) > abs(d) and inangle(v1, v2) < 0.9*math.pi:
            u = d/(math.sin(inangle(v1, v2)))
            if (angle(v2) > angle(v1) and not(angle(v2) > math.pi/2 and angle(v1) < -math.pi/2)) or (angle(v2) < -math.pi/2 and angle(v1) > math.pi/2):
                new = data[i+1, :]+(unit(v2)-unit(v1))*u
            else:
                new = data[i+1, :]-(unit(v2)-unit(v1))*u
            for j in range(0, data.shape[0]-2):
                if np.linalg.norm(new-data[j, :]) < abs(d)*0.999:
                    k = 1
                    break
            # if np.linalg.norm(new-data[i+1, :]) > abs(d)*5 or (temp.shape[0] > 1 and np.linalg.norm(new-temp[temp.shape[0]-1]) < abs(d)*0.1):
            #    k = 1
#            if np.linalg.norm(new-temp[temp.shape[0]-2]) < abs(d)*1.5:
#                temp = np.delete(temp, temp.shape[0]-1, axis=0)
            if k == 0:
                temp = np.row_stack((temp, new))
        else:
            data = np.delete(data, i+1, axis=0)
            times -= 1
        i += 1
    temp = np.delete(temp, 0, axis=0)

    j = 1
    i = 0
    while j:
        j = 0
        while i < temp.shape[0]-3:
            v1 = temp[i+1, :]-temp[i, :]
            v2 = temp[i+2, :]-temp[i+1, :]
            v3 = temp[i+3, :]-temp[i+2, :]
            i += 1
            if inangle(v1, v2)+inangle(v2, v3) > math.pi:
                temp = np.delete(temp, [i, i+1], axis=0)
                j = 1
                break

    plt.plot(temp[:, 0], temp[:, 1], '-o', color='r', markersize=2)
    return(temp)


for m in range(3):
    data = draw(data)
    print(m)

end = time.thread_time()
print('Running time: %s Seconds' % (end-start))

plt.show()
