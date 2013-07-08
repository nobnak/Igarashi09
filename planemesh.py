# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def build(n, scale):
    dx = scale / n
    xy = []
    triangles = []
    for row in xrange(0, n):
        ioffset = row * (n + 1)
        for col in xrange(0, n):
            xy.append((col * dx, row * dx))
            i = ioffset + col
            triangles.append((i, i + n + 2, i + 1))
            triangles.append((i, i + n + 1, i + n + 2))
        xy.append((n * dx, row * dx))
    for col in xrange(0, n + 1):
        xy.append((col * dx, n * dx))
    return (np.asarray(xy), np.asanyarray(triangles))


if __name__ == '__main__':
    n = 10
    scale = 10
    xy, triangles = build(n, scale)
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.triplot(xy[:,0], xy[:,1], triangles)
    plt.show()
