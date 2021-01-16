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
第二部分
'''


def unit(v):  # 单位化
    return(v/np.linalg.norm(v))


def inangle(v1, v2):  # 向量夹角
    return(math.acos(np.dot(v1, np.transpose(v2)) / (np.linalg.norm(v1)*np.linalg.norm(v2))))


def cross(v1, v2):  # 叉乘
    return(v1[0]*v2[1]-v2[0]*v1[1])


def ifcross(p1, p2, q1, q2):  # 判断有无交叉
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
        if x1 == x2:
            k = math.inf
        else:
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
    plt.plot(temp[:, 0], temp[:, 1], '-o', color='g', markersize=2)
    return(temp)


def findmax(data):  # 搜索极大值
    index = np.array([], dtype="int64")
    for i in range(-1, data.shape[0]-1):
        if data[i, 1] > data[i-1, 1] and data[i, 1] >= data[i+1, 1]:
            index = np.append(index, [i], axis=0)
    return(index)


def findmin(data):  # 搜索极小值
    index = np.array([], dtype="int64")
    for i in range(-1, data.shape[0]-1):
        if data[i, 1] <= data[i-1, 1] and data[i, 1] < data[i+1, 1]:
            index = np.append(index, [i], axis=0)
    return(index)


def findex(data):  # 获取极值序号
    index = list([])
    for i in range(len(data)):
        index.append(np.array([findmax(data[i]), findmin(data[i])]))
    return(index)


def findm(data, index):  # 获取最大值序号
    temp = list([])
    for i in range(len(index)):
        if index[i].shape[1] == 1:
            temp.append(np.array([index[i][0, 0], index[i][1, 0]]))
            continue
        maxy = np.max(data[i][:, 1])
        miny = np.min(data[i][:, 1])
        m = [[], []]
        for j in range(index[i].shape[1]):
            if data[i][index[i][0, j], 1] == maxy:
                m[0] = index[i][0, j]
        for j in range(index[i].shape[1]):
            if data[i][index[i][1, j], 1] == miny:
                m[1] = index[i][0, j]
        temp.append(np.array(m))
    return(temp)


def divideout(data_out, data_in, divide_in):  # 获取外层分割点
    ym = np.array([data_in[divide_in[0], 1],
                   data_in[divide_in[1], 1]])
    divide_out = np.array([], dtype='int16')
    for i in [0, 1]:
        for j in range(data_out.shape[0]):
            if data_out[j, 1] == ym[i] and data_out[j, 0] > data_in[divide_in[0], 0]:
                divide_out = np.append(divide_out, [j], axis=0)
                break
    return(divide_out)


def stackline(data_out, data_in, divide_out, divide_in):  # 复连通区域分割连接
    temp1 = np.row_stack(
        (data_out[:divide_out[0]+1], data_in[divide_in[0]:divide_in[1]+1], data_out[divide_out[1]:]))
    temp2 = np.row_stack(
        (data_in[:divide_in[0]], data_out[divide_out[0]+1:divide_out[1]], data_in[divide_in[1]+1:]))
    return(list([temp1, temp2]))


def divide1(data, index, parent):  # 对复连通区域进行划分
    temp = list([])
    for i in range(1, (max(parent[:, 1]+1))//2+1):  # 填充 i 层
        for j in range(parent.shape[0]):  # 搜索 i 层的外边界
            if parent[j, 1] == 2*i-1:
                data_out = data[j]
                for k in range(parent.shape[0]):  # 搜索 j 作为外边界的对应内边界
                    if parent[k, 0] == j:
                        data_in = data[k]
                        divide_in = index[k]  # 内层分割点
                        divide_out = divideout(data_out,  # 外层分割点
                                               data_in, divide_in)
                        line = stackline(data_out, data_in,  # 交叉连接分割点
                                         divide_out, divide_in)
                        data_out = line[0]  # 更新外层
                        temp.append(line[1])  # 写入内层
                temp.append(data_out)  # 写入外层
    return(temp)


def divideline(data, index):  # 获取单连通区域分割点
    line = np.array([0, 0])
    for n in [0, 1]:
        for i in index[n]:
            judge = 0
            j = i-2
            while j > -0.02*data.shape[0]:
                if data[j, 1] == data[i, 1]:
                    judge += 1
                    break
                j -= 1
            if judge == 0:
                continue
            k = i+2
            while k < 0.98*data.shape[0]:
                if data[k, 1] == data[i, 1]:
                    judge += 1
                    break
                k += 1
            if judge == 1:
                continue
            elif n == 0:
                line = np.row_stack((line, [j, i]))
            else:
                line = np.row_stack((line, [i, k]))
    line = np.delete(line, 0, axis=0)
    return(line)


def dividesub(data, line):  # 复连通区域分割连接
    temp = list([])
    while line.shape[0]:
        judge = 0
        for i in range(1, line.shape[0]):
            if line[0, 0] < line[i, 0] < line[0, 1]:
                line = np.row_stack((line[i, :], line))
                line = np.delete(line, i+1, axis=0)
                judge = 1
                break
        if judge == 0:
            temp.append(np.array(data[line[0, 0]+1:line[0, 1], :]))
            for j in range(line[0, 0]+1, line[0, 1]):
                data[j] = [0, 0]
            line = np.delete(line, 0, axis=0)
    temp.append(np.array(data[:, :]))
    for i in range(len(temp)):
        j = 0
        while j < temp[i].shape[0]:
            if temp[i][j, 0] == temp[i][j, 1] == 0:
                temp[i] = np.delete(temp[i], j, axis=0)
                continue
            j += 1
    return(temp)


def divide2(data, index):  # 对单连通区域进行划分
    temp = list([])
    for i in range(len(data)):
        if index[i].shape[1] > 1:
            line = divideline(data[i], index[i])
            temp += dividesub(data[i], line)
        else:
            temp += list([data[i]])
    return(temp)


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


def drawline(data):  # 画平行线
    dots = 0
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
            if round(j/abs(d)+1) % 2:
                line = np.row_stack((line, [j, min(temp)]))
                line = np.row_stack((line, [j, max(temp)]))
            else:
                line = np.row_stack((line, [j, max(temp)]))
                line = np.row_stack((line, [j, min(temp)]))
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
    return([length, times, dots])


'''
主函数
'''


start = time.thread_time()

data = readcsv(path)
for i in range(len(data)):
    data[i] = drawborder(data[i])
    data[i] = getint(data[i])
parent = np.array(findparent(data))
index = findex(data)
index = findm(data, index)
data = divide1(data, index, parent)
index = findex(data)
data = divide2(data, index)
data = drawline(data)

end = time.thread_time()

print('Length of curve:         %s mm' % data[0])
print('Number of parallel line: %s' % data[1])
print('Number of dots:          %s' % data[2])
print('Running time:            %s Seconds' % (end-start))

plt.show()
