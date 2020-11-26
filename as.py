import matplotlib.pyplot as plt
import numpy as np
import math

x = np.linspace(0, 2*math.pi)
plt.plot(x, math.atan(x), '-')
plt.show()
