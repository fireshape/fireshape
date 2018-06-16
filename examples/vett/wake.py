import numpy as np
from numpy import sinh

def sinh_wake_function(y, R, N):
    def U1(y, N):
        return 1./(1+sinh(y/sinh(1))**(2*N))
    return 1 - R + 2*R*U1(y, N)

import matplotlib.pyplot as plt

y = np.linspace(0, 1.0, 100)
fy = sinh_wake_function(2 * y, -0.6, 2)
plt.figure()
plt.plot(y, fy)
plt.show()
