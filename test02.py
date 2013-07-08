import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
import cProfile

import planemesh
import halfedge
import igarashi


def main():
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
    b, A = igarashi.build1(xy, halfedges, pins, pinPoses, w)
    trA = A.transpose()
    v = spla.spsolve(trA * A, trA * b).reshape(-1, 2)
    
    plt.figure()
    plt.ion()
    plt.show()
    
    tStart = time.clock()
    while t < 10:
        benchStart = time.clock()
        pinPoses = np.asarray(( (0, 0), (10, 0),
            (10 + 5 * np.cos(5 * t * rPi), 10 + 5 * np.sin(5 * t * rPi)) ))
        b, A = igarashi.build1(xy, halfedges, pins, pinPoses, w)
        trA = A.transpose()
        v = spla.spsolve(trA * A, trA * b).reshape(-1, 2)
        print 'tick = %e' % (time.clock() - benchStart, )
        
        #plt.clf()
        #plt.gca().set_aspect('equal')
        #plt.axis([-5, 15, -5, 15])
        #plt.triplot(v[:,0], v[:,1], triangles)
        #plt.draw()
        t = time.clock() - tStart
        #plt.pause(0.01)

if __name__ == '__main__':
    cProfile.run('main()')
