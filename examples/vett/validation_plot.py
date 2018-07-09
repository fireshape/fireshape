import matplotlib.pyplot as plt
import numpy as np

Z = np.zeros((31, 31))
Z[:,:] = np.nan
X = range(0, 31)
Y = range(0, 31)
for x1 in range(1, 26):
    for x2 in range(x1+4, 30):
        try:
            with open(f"validation-cg3cg2-clscale0.5/{x1}-{x2}.csv", "r") as fi:
                s = fi.readline()
                Z[x1, x2] = float(s.split("-")[1])
        except:
            pass
plt.figure()
plt.contourf(X, Y, Z, 40)
plt.contour(X, Y, Z, 40, colors="black")
amax = np.unravel_index(np.nanargmax(Z, axis=None), Z.shape)
plt.scatter([amax[1]], [amax[0]], marker="*", s=500)
plt.savefig("validation.pdf")
plt.show()
