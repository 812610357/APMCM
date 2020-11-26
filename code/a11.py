import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math

data = np.array(pd.read_csv(".\code\graph1.csv", header=2))  # 从csv文件获取数据
d = -1
plt.axis("equal")
plt.plot(data[:, 0], data[:, 1], '-o', markersize=1)


def unit(v):
    return(v/np.linalg.norm(v))


def slope(v):
    return(math.atan2(v[1], v[0]))


for i in range(data.shape[0]-2):
    v1 = data[i+1, :]-data[i, :]
    v2 = data[i+2, :]-data[i+1, :]
    if -math.inf < slope(v2) < math.inf:
        u = d/(math.sin(math.acos(np.dot(v1, np.transpose(v2)) /
                                  (np.linalg.norm(v1)*np.linalg.norm(v2)))))
        if (slope(v2) > slope(v1) and not(slope(v2) > 3 and slope(v1) < -3)) or (slope(v2) < -3 and slope(v1) > 3):
            a = data[i+1, :]+(unit(v2)-unit(v1))*u
        else:
            a = data[i+1, :]-(unit(v2)-unit(v1))*u
        plt.plot(a[0], a[1], 'o', markersize=1)
plt.show()
