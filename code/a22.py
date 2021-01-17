import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import time

plt.axis("equal")
d = -0.1
path = ".\code\graph1.csv"
length = 0
storeys = 0
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


def dot(v1, v2):  # 向量点乘
    return(v1[0]*v2[0]+v1[1]*v2[1])


def cross(v1, v2):  # 平面内向量叉乘
    return(v1[0]*v2[1]-v2[0]*v1[1])


def vertical(v, d):  # 求垂直向量
    return(unit(np.array([v[1], -v[0]]))*abs(d))


def divide1(data):  # 对复连通区域进行划分
    parent = np.array(findparent(data))
    for i in range(1, (max(parent[:, 1]+1))//2+1):  # 填充 i 层
        for j in range(parent.shape[0]):  # 搜索 i 层的外边界
            index = list([])
            if parent[j, 1] == 2*i-1:
                index.append(j)
                for k in range(parent.shape[0]):  # 搜索 j 作为外边界的对应内边界
                    if parent[k, 0] == j:
                        index.append(k)
            data = link(data, index)
            k = 0
            while k < len(data):
                if data[k].all() == data[k].all() == 0:
                    del data[k]
                    continue
                k += 1
    return(data)


def link(data, index):  # 按需要对内外层连接
    i = 0
    while i < len(index)-1:
        j = i+1
        while j < len(index):
            linkpoint = ifclose(data[index[i]], data[index[j]])
            if linkpoint.shape[0] == 0:
                j += 1
            elif linkpoint[1] > linkpoint[3]:
                temp = np.row_stack((data[index[i]][:linkpoint[0]], data[index[j]][linkpoint[1]:],
                                     data[index[j]][1:linkpoint[3]], data[index[i]][linkpoint[2]:]))
                data[index[i]] = temp
                data[index[j]] = np.array([0, 0])
                del index[j]
            else:
                temp = np.row_stack((data[index[i]][:linkpoint[0]],
                                     data[index[j]][linkpoint[1]:linkpoint[3]],
                                     data[index[i]][linkpoint[2]:]))
                data[index[i]] = temp
                data[index[j]] = np.array([0, 0])
                del index[j]
        i += 1
    return(data)


def ifclose(data1, data2):  # 连接点序号
    index = np.array([], dtype="int64")
    point1 = -1  # 第一个连接点，指向data2
    point2 = -2  # 第二个连接点，指向data2
    for i in range(data1.shape[0]):
        for j in range(data2.shape[0]):
            if np.linalg.norm(data1[i, :]-data2[j, :]) < abs(d):
                if point2 == -2:
                    point1 = j
                elif point1 == -2:
                    point2 = j
                    break
            elif point2 < -1 < point1:
                index = np.append(index, [i+1, point1], axis=0)
                point2 = -1
                point1 = -2
                break
            elif j == data2.shape[0]-1 and point1 < -1 < point2:
                index = np.append(index, [i-1, point2+1], axis=0)
                return(index)
    if point2 == -1:
        index = np.append(index, [index[0]+1, index[1]-1], axis=0)
    return(index)


def divide2(data):
    i = s = 0
    while i < len(data):
        dividepoint = ifnear(data[i], s)
        if dividepoint.shape[0] == 0:
            i += 1
            s = 0
        else:
            temp = np.array(data[i][dividepoint[0]:dividepoint[1]+1, :])
            temp = np.row_stack((temp, [data[i][dividepoint[0]]]))
            data.append(temp)
            data[i] = np.delete(data[i], range(
                dividepoint[0], dividepoint[1]+1), axis=0)
            s = dividepoint[0]
            continue
    return(data)


def ifnear(data, s):  # 分割点序号
    for i in range(s, data.shape[0]-2):
        point = -1  # 第二分割点指向
        for j in range(min(i+5, data.shape[0]-2), data.shape[0]-2):
            if np.linalg.norm(data[i, :]-data[j, :]) < 1.5*abs(d):  # 间距过近且向量方向差超过90度
                v1 = data[i+2, :]-data[i, :]
                v2 = data[j+2, :]-data[j, :]
                if dot(v1, v2) < 0:
                    point = j
            elif point > -1:
                return(np.array([i, point]))
    return(np.array([]))


def delcross(data):
    i = 0
    while i < data.shape[0]-3:
        j = i
        while j < i+(data.shape[0]-i)//4:
            if ifcross(data[i, :], data[i+1, :], data[j, :], data[j+1, :]):
                data = np.row_stack((data[:i, :], data[j+1:, :]))
                continue
            else:
                j += 1
        i += 1
    return(data)


def ifcross(p1, p2, q1, q2):
    v11 = q1-p1
    v12 = q2-p1
    v21 = q1-p2
    v22 = q2-p2
    if cross(v11, v12)*cross(v21, v22) < 0 and cross(v11, v21)*cross(v12, v22) < 0:
        return(1)
    else:
        return(0)


def addlong(data):  # 同级点间距控制
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


def delnarrow(data, last):  # 与上一级间距控制
    i = 0
    while i < data.shape[0]:  # 遍历该级所有数据
        j = 0
        while j < last.shape[0]:  # 遍历上级部分数据
            if i >= data.shape[0]:
                break
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


def drawline(data):  # 画等高线
    global dots
    data = np.insert(data, data.shape[0], values=data[1, :], axis=0)
    data = np.insert(data, 0, values=data[data.shape[0]-3, :], axis=0)
    temp = np.array([0, 0])
    i = 0
    while i < data.shape[0]-2:
        v1 = data[i+1, :]-data[i, :]
        v2 = data[i+2, :]-data[i+1, :]
        if cross(v1, v2) > 0 and dot(v1, v2) < 0 and storeys == 0:  # 外折时用圆弧补充
            vv1 = vertical(v1, d)
            vv2 = vertical(v2, d)
            new = vv1+data[i+1, :]
            ang = angle(vv1)+0.1
            while ang < angle(vv2):
                new = np.row_stack(
                    (new, [math.cos(ang)*abs(d), math.sin(ang)*abs(d)]+data[i+1, :]))
                ang += 0.1
        elif 0 < inangle(v1, v2) < math.pi*0.9:  # 一般情况在菱形中使用向量得到内缩点
            u = d/(math.sin(inangle(v1, v2)))
            if cross(v1, v2) > 0:
                new = data[i+1, :]+(unit(v2)-unit(v1))*u
            else:
                new = data[i+1, :]-(unit(v2)-unit(v1))*u
        elif inangle(v1, v2) == 0:  # 两向量平行的特殊情况
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
    temp = delcross(temp)
    temp = addlong(temp)
    temp = delnarrow(temp, data)  # 与上一级间距控制
    return(temp)


def orderline(data):
    data = np.delete(data, -1, axis=0)
    data = np.row_stack((data, data[:10, :]))
    data = np.delete(data, [range(9)], axis=0)
    return(data)


'''
第三部分
'''


def writecsv(data):  # 导出线条
    dataframe = pd.DataFrame(data={'x': data[:, 0], 'y': data[:, 1]})
    dataframe.to_csv(f".\code\contour{times}.csv",
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
        plt.plot(data[i][:, 0], data[i][:, 1],
                 '-o', color='b', markersize=1)
    return(data)


def getlength(data):
    global length
    for i in range(data.shape[0]-1):
        length += np.linalg.norm(data[i+1, :]-data[i, :])
    pass


def draw(data):
    global length
    global storeys
    global times
    while True:
        data = divide1(data)
        data = divide2(data)
        i = 0
        while i < len(data):
            if data[i].shape[0] < 12:
                del data[i]
                print('-')
                continue
            if storeys > 0:
                times += 1
                plt.plot(data[i][:, 0], data[i][:, 1],
                         '-', color='r', markersize=3)
                writecsv(data[i])
                getlength(data[i])
            data[i] = drawline(data[i])
            if data[i].shape[0] < 12:
                del data[i]
                continue
            data[i] = orderline(data[i])
            i += 1
        if len(data) == 0:
            break
        storeys += 1
        print(storeys)
    pass


'''
主函数
'''

start = time.thread_time()

data = readcsv(path)
data = draw(data)

end = time.thread_time()
print('Length of curve: %s mm' % length)
print('Number of turns: %s' % times)
print('Number of dots:  %s' % dots)
print('Running time:    %s Seconds' % (end-start))

plt.show()
