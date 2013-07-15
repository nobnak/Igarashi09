import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import cProfile
import pstats

import planemesh
import halfedge
import igarashi

n = 10
scale = 200
w = 1000.0
pins = np.asarray(( n+2, ((n+1)**2-1)/2, (n+1)*n-2 ))
mouseIsDown = False
movePin = -1


def init():
    registerIgarashi()
    compileIgarashi()
    executeIgarashi()

def registerIgarashi():
    global xy, triangles, pins, pinPoses, nVertices, nEdges, edges, heVectors, heIndices
    global A1top, A2top, G
    xy, triangles = planemesh.build(n, scale)
    xy -= 0.5*scale
    pinPoses = xy[pins, :]
    halfedges = halfedge.build(triangles)
    heVectors = np.asarray([xy[he.ivertex, :] - xy[he.prev().ivertex, :] for he in halfedges])
    edges, heIndices = halfedge.toEdge(halfedges)
    nVertices = xy.shape[0]
    nEdges = edges.shape[0]
    A1top, G = igarashi.buildA1top(heVectors, halfedges, edges, heIndices, nVertices)
    A2top = igarashi.buildA2top(edges, nVertices)

def compileIgarashi():
    global A1bottom, A2bottom, tA1, tA2, sqA1, sqA2, fctA1, fctA2
    A1bottom = igarashi.buildA1bottom(pins, w, nVertices)
    A2bottom = igarashi.buildA2bottom(pins, w, nVertices)
    A1 = sp.vstack((A1top, A1bottom))
    tA1 = A1.transpose()
    A2 = sp.vstack((A2top, A2bottom))
    tA2 = A2.transpose()
    sqA1 = tA1 * A1
    sqA2 = tA2 * A2

def executeIgarashi():
    global v2
    b1 = igarashi.buildB1(pins, pinPoses, w, nEdges)
    v1 = spla.spsolve(sqA1, tA1 * b1)
    b2 = igarashi.buildB2(heVectors, heIndices, edges, pinPoses, w, G, v1)
    v2x = spla.spsolve(sqA2, tA2 * b2[:, 0])
    v2y = spla.spsolve(sqA2, tA2 * b2[:, 1])
    v2 = np.vstack((v2x, v2y)).T
    
def main(nLoops):
    init()
    for i in xrange(0, nLoops):
        executeIgarashi()
    
if __name__ == '__main__':
    cProfile.run("main(100)", "profile.log")
    pstats.Stats("profile.log").sort_stats('time').print_stats()