import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
import time

start = time.thread_time()

data0 = np.array(pd.read_csv(".\code\graph1.csv", header=2))  # 从csv文件获取数据
d = -0.1
plt.axis("equal")
plt.plot(data0[:, 0], data0[:, 1], '-o', markersize=1)


def unit(v):  # 单位化
    return(v/np.linalg.norm(v))


def angle(v):  # 取辐角
    return(math.atan2(v[1], v[0]))


def inangle(v1, v2):
    return(math.acos(np.dot(v1, np.transpose(v2)) / (np.linalg.norm(v1)*np.linalg.norm(v2))))


def roc(v1, v2):  # 求曲率半径
    x1 = v1[0]
    y1 = v1[1]
    x2 = v2[0]
    y2 = v2[1]
    if 1*y2 == x2*y1:
        return(0)
    else:
        xc = ((y1+y2)*y1*y1+x1 ** 2*y2+x2 ** 2*y1)/(x1*y2-x2*y1)
        yc = ((x1+x2)*x1*x2+y1 ** 2*x2+y2 ** 2*x1)/(x2*y1-x1*y2)
        return(math.sqrt(xc ** 2+yc ** 2)/2)


def icr(v1, v2):
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    c = np.linalg.norm(v1+v2)
    r = math.sqrt((a+b-c)*(a-b+c)*(-a+b+c)/(a+b+c))/2
    theta = inangle(v1, v2)
    return(r/math.sin(theta/2))


def draw(data):
    temp = np.array([0, 0])
    delrow = np.array([0])
    for i in range(data.shape[0]-2):
        k = 0
        v1 = data[i+1, :]-data[i, :]
        v2 = data[i+2, :]-data[i+1, :]
        if roc(v1, v2) > abs(d):
            u = d/(math.sin(inangle(v1, v2)))
            if (angle(v2) > angle(v1) and not(angle(v2) > math.pi/2 and angle(v1) < -math.pi/2)) or (angle(v2) < -math.pi/2 and angle(v1) > math.pi/2):
                new = data[i+1, :]+(unit(v2)-unit(v1))*u
            else:
                new = data[i+1, :]-(unit(v2)-unit(v1))*u
            for j in range(0, data.shape[0]):
                if np.linalg.norm(new-data[j, :]) < abs(d)*0.999:
                    k = 1
                    break
            if (temp.shape[0] > 1 and np.linalg.norm(new-temp[temp.shape[0]-1]) < abs(d)*0.1):
                k = 1
            if k == 0:
                temp = np.row_stack((temp, new))
        else:
            delrow = np.append(delrow, [i+1])
    temp = np.delete(temp, 0, axis=0)
    delrow = np.delete(delrow, 0, axis=0)
    data = np.delete(data, delrow, axis=0)
    plt.plot(temp[:, 0], temp[:, 1], '-o', color='r', markersize=2)
    return()


for m in range(10):
    draw(data0)
    d = d-0.1
    print(m)

end = time.thread_time()
print('Running time: %s Seconds' % (end-start))

plt.show()
