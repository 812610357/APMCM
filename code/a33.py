import numpy as np
import matplotlib.pyplot as plt
import math

d = 1   # 修改向量（圆半径）长度
rad = 1   # 画圆密度


def unit(v):  # 单位化
    return(v/np.linalg.norm(v))


def cross(v1, v2):  # 平面内向量叉乘
    return(v1[0]*v2[1]-v2[0]*v1[1])


def dot(v1, v2):  # 向量点乘
    return(v1[0]*v2[0]+v1[1]*v2[1])


def angle(v):  # 取辐角
    return(math.atan2(v[1], v[0]))


def vertical(v, d):  # 求垂直向量
    vt = np.array([v[1], -v[0]])
    return(unit(vt)*d)


def f_e_dot(v, head_dot):  # 根据向量和起点求末点
    end_dot = [v[0]+head_dot[0], v[1]+head_dot[1]]
    return end_dot


def vec_sita(v, ang, head_dot):  # 根据初向量、α角、▲θ、原向量起始点
    v_r = [d*math.cos(ang), d*math.sin(ang)]
    return f_e_dot(v_r, head_dot)


plt.axis("equal")
data = np.array([[-3, 2], [-1, 2], [0, 0], [1, 2], [3, 2]])
plt.plot(data[:, 0], data[:, 1], "-")
i = 0
while i < data.shape[0]-2:
    temp = np.array([])
    v1 = data[i+1, :]-data[i, :]
    v2 = data[i+2, :]-data[i+1, :]
    if cross(v1, v2) > 0 and dot(v1, v2) < 0:
        # 已知向量和距离求其垂直向量
        vl = vertical(v1, d)  # 左边垂直向量
        vr = vertical(v2, d)  # 右边垂直向量

        a = data[i+1, :]                 # 顶点记录
        c1 = [vl[0]+a[0], vl[1]+a[1]]   # 左边第一点记录

        temp = np.array(c1)
        for ang in np.linspace(angle(vl), angle(vr), num=10):
            # 将每一个圆上的点记录在temp中
            temp = np.row_stack((temp, vec_sita(vl, ang, a)))
    if temp.shape[0] > 1:
        plt.plot(temp[:, 0], temp[:, 1], "-")
    i += 1
plt.show()
