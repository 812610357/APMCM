import numpy as np

a = np.array([[0, 10], [0, 0], [2, 3]])
i = 0
while i < a.shape[0]:
    if a[i, 0] == a[i, 1] == 0:
        a = np.delete(a, i, axis=0)
        continue
    i += 1
print(a)
