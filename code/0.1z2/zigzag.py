import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = list([])
data0 = pd.read_csv(
    f".\code\graph2.csv", index_col=False, header=2)
j = 0
for i in range(len(data0.values)):
    if "MainCurve" in data0.values[i, 0]:
        data += list([np.array(data0.values[j:i, :], dtype='float64')])
        j = i+2
data += list([np.array(data0.values[j:len(data0.values), :], dtype='float64')])
for i in range(len(data)):
    plt.plot(data[i][:, 0], data[i][:, 1], '-o', color='b', markersize=1)

for i in range(1, 10):
    data = np.array(pd.read_csv(
        f".\code\\0.1z2\zigzag{i}.csv", header=0))
    plt.plot(data[:, 0], data[:, 1], '-', color='r')

print("Length of curve:         9054.233810174643 mm")
print("Number of parallel line: 1059")
print("Number of dots:          2109")
print("Running time:            4125 ms")

plt.axis("equal")
plt.xlabel("X(mm)")
plt.ylabel("Y(mm)")
plt.title("Graph1 zigzag", fontsize=16)
plt.show()
