import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import argmax, argmin  # 返回最大或最小值对应的序号
import pandas as pd
import math
import time

data = list([])
for i in range(1, 5):
    data.append(np.array(pd.read_csv(
        f".\code\graph2{i}.csv", header=2)))  # 从csv文件获取数据
n = data.len()

parent = list([np.array([3, 1]), np.array([2, 2]),
               np.array([1, -1]), np.array(2, 2)])
