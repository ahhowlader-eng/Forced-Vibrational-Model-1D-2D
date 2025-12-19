import numpy as np
import matplotlib.pyplot as plt
from math import gcd
import time

t0 = time.time()

# -----------------------------
# Initialization
# -----------------------------
n = 999
m = np.zeros((n+1, n+1), dtype=int)

# Percolation network (bond probability = 1)
for i in range(n):
    for j in range(n):
        d = np.random.rand()
        if d <= 1.0:
            m[i, j] = 1

# Defect pattern
m[1:n:2, 0:n:3] = 0
m[0:n:2, 2:n:3] = 0

# -----------------------------
# Lattice geometry
# -----------------------------
acc = 1.0
aa = np.sqrt(3) * acc
a = np.sqrt(3) / 2

x, y = np.meshgrid(np.arange(1, n+2), np.arange(1, n+2))
nn = y.shape[0]

shift = np.tile(np.array([[0], [0.5]]), (nn // 2, nn))
x = (x + shift) * acc
y = (a * y) * acc

# Mask inactive sites
mask0 = (m == 0)
x[mask0] = 0.0
y[mask0] = 0.0

# -----------------------------
# Plot lattice
# -----------------------------
plt.figure(1)
plt.plot(x, y, 'k.', markersize=5)
plt.axis('equal')
plt.axis([0, 150, 0, 150])
plt.grid(True)

# -----------------------------
# Chiral vector / tube geometry
# -----------------------------
ch1 = 10
ch2 = 10
tc = 250

ch = aa * np.sqrt(ch1**2 + ch1*ch2 + ch2**2)
dr = gcd(2*ch2 + ch1, 2*ch1 + ch2)
t = tc * (np.sqrt(3) / dr) * ch
r = ch / (2 * np.pi)

# Polygon region
xv = np.array([x[0,0],
               x[0,0] + ch - 0.005,
               x[0,0] + ch - 0.005,
               x[0,0],
               x[0,0]])

yv = np.array([y[0,0],
               y[0,0],
               y[0,0] + t - 0.005,
               y[0,0] + t - 0.005,
               y[0,0]])

# Point-in-polygon
from matplotlib.path import Path
poly = Path(np.column_stack((xv, yv)))
pts = np.column_stack((x.flatten(), y.flatten()))
inside = poly.contains_points(pts).reshape(x.shape)

plt.figure(2)
plt.plot(xv, yv, 'k-')
plt.plot(x[inside], y[inside], 'k.', markersize=5)
plt.axis('equal')
plt.axis([0, 150, 0, 150])
plt.grid(True)

# -----------------------------
# Count lattice nodes
# -----------------------------
LN = np.sum(inside[:n, :n])

# -----------------------------
# Unit cell size
# -----------------------------
lnn = 2 * (2 * ch * ch) / (aa * aa * dr)
lnn = int(round(lnn))

xx = x[:(2*tc)+8, :lnn-10]
yy = y[:(2*tc)+8, :lnn-10]

mi = m[:(2*tc)+8, :lnn-10]

# Boundary removal
mi[:4, :] = 0
mi[(2*tc)+4:(2*tc)+8, :] = 0

# -----------------------------
# Neighbor marking rules
# -----------------------------
for i in range(1, (2*tc)+7):
    for j in range(1, (lnn-10)-1):
        if j % 3 == 1 and i % 2 == 0:
            if mi[i,j] == 1 and mi[i+1,j] == 0: mi[i+1,j] = 2
            if mi[i,j] == 1 and mi[i-1,j] == 0: mi[i-1,j] = 2
            if mi[i,j] == 1 and mi[i,j-1] == 0: mi[i,j-1] = 2

for i in range(1, (2*tc)+7):
    for j in range(1, (lnn-10)-1):
        if j % 3 == 1 and i % 2 == 1:
            if mi[i,j] == 1 and mi[i+1,j] == 0: mi[i+1,j] = 2
            if mi[i,j] == 1 and mi[i-1,j] == 0: mi[i-1,j] = 2
            if mi[i,j] == 1 and mi[i,j+1] == 0: mi[i,j+1] = 2

for i in range(1, (2*tc)+7):
    for j in range(1, (lnn-10)-1):
        if j % 3 == 0 and i % 2 == 0:
            if mi[i,j] == 1 and mi[i+1,j+1] == 0: mi[i+1,j+1] = 2
            if mi[i,j] == 1 and mi[i-1,j+1] == 0: mi[i-1,j+1] = 2
            if mi[i,j] == 1 and mi[i,j-1] == 0: mi[i,j-1] = 2

for i in range(1, (2*tc)+7):
    for j in range(1, (lnn-10)-1):
        if j % 3 == 2 and i % 2 == 1:
            if mi[i,j] == 1 and mi[i+1,j-1] == 0: mi[i+1,j-1] = 2
            if mi[i,j] == 1 and mi[i-1,j-1] == 0: mi[i-1,j-1] = 2
            if mi[i,j] == 1 and mi[i,j+1] == 0: mi[i,j+1] = 2

# -----------------------------
# Replication and masking
# -----------------------------
M = np.tile(mi, (1, 3))
M[:, :(lnn-10)-3] = 0
M[:, (2*(lnn-10))+3 : 3*(lnn-10)] = 0

# -----------------------------
# Save data
# -----------------------------
np.savez("n1010_per_10000_0_exact.npz", M=M, LN=LN)

print("Elapsed time:", time.time() - t0)
