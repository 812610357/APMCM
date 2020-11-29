import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data0 = np.array(pd.read_csv("graph1.csv", header=2))
plt.plot(data0[:, 0], data0[:, 1], '-o', markersize=1)
for i in range(1, 7):
    data = np.array(pd.read_csv(
        f"zigzag{i}.csv", header=0))
    plt.plot(data[:, 0], data[:, 1], '-', color='r')

print("Length of curve:         10022.714003590447 mm")
print("Number of parallel line: 911")
print("Number of dots:          1816")

plt.axis("equal")
plt.xlabel("X(mm)")
plt.ylabel("Y(mm)")
plt.title("Graph1 zigzag", fontsize=16)
plt.show()
