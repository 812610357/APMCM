import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import time

start = time.thread_time()

data0 = np.array(pd.read_csv(".\code\graph1.csv", header=2))
data = data0  # 从csv文件获取数据
d = -0.1  # 精度
plt.axis("equal")
plt.plot(data[:, 0], data[:, 1], '-o', markersize=1)
dots = 0
times = 0


def unit(v):  # 单位化
    return(v/np.linalg.norm(v))


def angle(v):  # 取辐角
    return(math.atan2(v[1], v[0]))


def inangle(v1, v2):  # 向量夹角
    return(math.acos(round(np.dot(v1, np.transpose(v2)) / (np.linalg.norm(v1)*np.linalg.norm(v2)), 9)))


def cross(v1, v2):  # 平面内向量叉乘
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


def delcross(temp):
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

    temp = iflong(temp)  # 同级点间距控制
    '''
    temp = ifcross(temp)  # 交叉控制
    '''
    temp = ifwide(temp, data)  # 与上一级间距控制
    return(temp)


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


def ifwide(data, last):  # 与上一级间距控制
    i = 0
    while i < data.shape[0]:  # 遍历该级所有数据
        j = 0
        while j < last.shape[0]:  # 遍历上级所有数据
            if i >= data.shape[0]:
                break
            if np.linalg.norm(data[i, :]-last[j, :]) < abs(d)*0.999:  # 小于一个精度的直接删除
                data = np.delete(data, i, axis=0)
                if j > 20:
                    j -= 20
                else:
                    j = 0
            else:
                j += 1
        i += 1
    return(data)


'''
def ifcross(data):  # 交叉控制
    i = 0
    while i < data.shape[0]-3:  # 遍历该级所有数据
        v1 = data[i+1, :]-data[i, :]
        v2 = data[i+2, :]-data[i+1, :]
        v3 = data[i+3, :]-data[i+2, :]
        if inangle(v1, v2)+inangle(v2, v3) > math.pi:  # 连续三个向量转角超过180度直接删除
            data = np.delete(data, [i+1, i+2], axis=0)
        else:
            i += 1
    return(data)
'''


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


def drawline(data):  # 判断是否需要分割
    length = 0
    global times
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
            if times > 0:
                temp = delcross(temp)
                writecsv(temp)
            plt.plot(temp[:, 0], temp[:, 1], '-o', color='r', markersize=3)
            data[-1] = draw(temp)
            times += 1
            print(times)
            for j in range(data[-1].shape[0]-1):
                length = length + \
                    math.sqrt((data[-1][j+1, 0]-data[-1][j, 0])**2 +
                              (data[-1][j+1, 1]-data[-1][j, 1])**2)
            '''
            plt.plot(data0[:, 0], data0[:, 1], '-o', color='b', markersize=1)
            plt.show()
            plt.axis("equal")
            '''
        else:
            data.append(temp[math.floor(index[0])+1: math.floor(index[1]), :])
            data[-1] = np.row_stack((data[-1], data[-1][0:1, :]))
            temp1 = temp[0:math.floor(index[0])+1, :]
            temp2 = temp[math.floor(index[1]):temp.shape[0], :]
            data[-2] = np.row_stack((temp1, temp2))
    return([length, times])


def writecsv(data):
    global times
    dataframe = pd.DataFrame(data={'x': data[:, 0], 'y': data[:, 1]})
    dataframe.to_csv(f".\code\contour{times}.csv",
                     index=False, mode='w', sep=',')
    pass


data = list([data])
data = drawline(data)

end = time.thread_time()
print('Length of curve: %s mm' % data[0])
print('Number of turns: %s' % data[1])
print('Number of dots: %s' % dots)
print('Running time:    %s Seconds' % (end-start))

plt.show()
