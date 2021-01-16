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

for i in range(1, 109):
    data = np.array(pd.read_csv(
        f".\code\contour{i}.csv", header=0))
    plt.plot(data[:, 0], data[:, 1], '-o', color='r', markersize=3)

print("Length of curve: 955.359178738121 mm")
print("Number of turns: 10")
print("Number of dots:  3583")
print("Running time:    21765.625 ms")

plt.axis("equal")
plt.xlabel("X(mm)")
plt.ylabel("Y(mm)")
plt.title("Graph1 contour", fontsize=16)
plt.show()
