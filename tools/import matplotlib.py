import matplotlib.pyplot as plt
import numpy as np

Qc = 5
kappa = 4
ROBOT_RADIUS = 0.5

d = np.linspace(0, 5, 100)  # distance between robot and obstacle
cost = Qc / (1 + np.exp(kappa * (d - 2*ROBOT_RADIUS)))

plt.figure()
plt.plot(d, cost)
plt.xlabel('Distance (d) between robot and obstacle')
plt.ylabel('Collision cost')
plt.title('Nonlinear collision cost function')
plt.grid(True)
plt.show()
