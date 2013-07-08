import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

import planemesh
import halfedge
import igarashi



t = 0
n = 10
scale = 10
xy, triangles = planemesh.build(n, scale)
halfedges = halfedge.build(triangles)
pins = np.asarray([0, n, (n+1)**2-1])
rPi = 1.0 / np.pi
pinPoses = np.asarray(( (0, 0), (10, 0),
    (10 + 5 * np.cos(5 * t * rPi), 10 + 5 * np.sin(5 * t * rPi)) ))
w = 1000.0
b, A = igarashi.build0(xy, halfedges, pins, pinPoses, w)
trA = A.transpose()
v = spla.spsolve(trA * A, trA * b).reshape(-1, 2)

plt.figure()
plt.ion()
plt.show()

tStart = time.time()
while t < 10:
    pinPoses = np.asarray(( (0, 0), (10, 0),
        (10 + 5 * np.cos(5 * t * rPi), 10 + 5 * np.sin(5 * t * rPi)) ))
    b[-(pinPoses.size):] = w * pinPoses.reshape(-1, 1)    
    v = spla.spsolve(trA * A, trA * b).reshape(-1, 2)
    
    plt.clf()
    plt.gca().set_aspect('equal')
    plt.axis([-5, 15, -5, 15])
    plt.triplot(v[:,0], v[:,1], triangles)
    plt.draw()
    t = time.time() - tStart
    plt.pause(0.01)
