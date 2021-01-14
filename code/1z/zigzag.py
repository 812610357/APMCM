import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data0 = np.array(pd.read_csv(".\code\graph1.csv", header=2))
plt.plot(data0[:, 0], data0[:, 1], '-o', markersize=1)
for i in range(1, 5):
    data = np.array(pd.read_csv(
        f".\code\\1z\zigzag{i}.csv", header=0))
    plt.plot(data[:, 0], data[:, 1], '-', color='r')

print("Length of curve:         918.9751087943642 mm")
print("Number of parallel line: 89")
print("Number of dots:          174")
print("Running time:            3250 ms")

plt.axis("equal")
plt.xlabel("X(mm)")
plt.ylabel("Y(mm)")
plt.title("Graph1 zigzag", fontsize=16)
plt.show()
