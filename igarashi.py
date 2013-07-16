import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt

import planemesh
import halfedge


threeVertices2twoEdges = np.asarray(( (-1.,  0., 1., 0., 0., 0.),
                                      ( 0., -1., 0., 1., 0., 0.),
                                      (-1.,  0., 0., 0., 1., 0.),
                                      ( 0., -1., 0., 0., 0., 1.) ))
fourVertices2threeEdges = np.asarray(( (-1.,  0., 1., 0., 0., 0., 0., 0.),
                                       ( 0., -1., 0., 1., 0., 0., 0., 0.),
                                       (-1.,  0., 0., 0., 1., 0., 0., 0.),
                                       ( 0., -1., 0., 0., 0., 1., 0., 0.),
                                       (-1.,  0., 0., 0., 0., 0., 1., 0.),
                                       ( 0., -1., 0., 0., 0., 0., 0., 1.) ))


def buildA1top(heVectors, halfedges, edges, heIndices, nVertices):
    Arows = []
    Acols = []
    Adata = []
    Grows = []
    Gcols = []
    Gdata = []
    for row in xrange(0, edges.shape[0]):
        v0, v1 = edges[row, :]
        Arows.append(2 * row); Acols.append(2 * v0); Adata.append(-1.0)
        Arows.append(2 * row); Acols.append(2 * v1); Adata.append(1.0)
        Arows.append(2 * row + 1); Acols.append(2 * v0 + 1); Adata.append(-1.0)
        Arows.append(2 * row + 1); Acols.append(2 * v1 + 1); Adata.append(1.0)

        vertices = [v0, v1]
        he = halfedges[heIndices[row]]
        edgeVectors = [heVectors[he.iself], ]
        vertices.append(halfedges[he.inext].ivertex)
        edgeVectors.append(-heVectors[he.prev().iself])
        verts2edges = threeVertices2twoEdges
        if he.ipair != -1:
            pair = halfedges[he.ipair]
            vertices.append(halfedges[pair.inext].ivertex)
            edgeVectors.append(heVectors[pair.inext])
            verts2edges = fourVertices2threeEdges
        g = []
        for v in edgeVectors:
            g.extend(( (v[0], v[1]), (v[1], -v[0]) ))
        g = np.asarray(g)
        e = heVectors[heIndices[row], :]
        e = np.asarray(( (e[0], e[1]), (e[1], -e[0]) ))
        g = np.dot( la.inv(np.dot(g.T, g)), g.T )
        g = np.dot( g, verts2edges )
        h = - np.dot(e, g)
        rows = []
        cols = []
        for i in xrange(0, len(vertices)):
            rows.append(2 * row); cols.append(2 * vertices[i])
            rows.append(2 * row); cols.append(2 * vertices[i] + 1)
        for i in xrange(0, len(vertices)):
            rows.append(2 * row + 1); cols.append(2 * vertices[i])
            rows.append(2 * row + 1); cols.append(2 * vertices[i] + 1)

        data = h.flatten()
        Arows.extend(rows)
        Acols.extend(cols)
        Adata.extend(data)
        Grows.extend(rows)
        Gcols.extend(cols)
        Gdata.extend(g.flatten())
    spA1top = sp.csr_matrix((Adata, (Arows, Acols)), shape=(edges.size, nVertices * 2))
    spG = sp.csr_matrix((Gdata, (Grows, Gcols)), shape=(edges.size, nVertices * 2))
    return spA1top, spG



def buildA1bottom(pins, w, nVertices):
    Arows = []
    Acols = []
    Adata = []
    for row in xrange(0, len(pins)):
        pin = pins[row]
        Arows.append(2 * row); Acols.append(2 * pin); Adata.append(w)
        Arows.append(2 * row + 1); Acols.append(2 * pin + 1); Adata.append(w)
    spA1bottom = sp.csr_matrix((Adata, (Arows, Acols)), shape=(pins.size * 2, nVertices * 2))    
    return spA1bottom



def buildB1(pins, pinPositions, w, nEdges):
    brows = range(nEdges * 2, nEdges * 2 + pinPositions.size)
    bcols = [0 for i in xrange(0, len(brows))]
    bdata = (w * pinPositions).flatten()
    bshape = (nEdges * 2 + pinPositions.size, 1)
    b1 = sp.csr_matrix((bdata, (brows, bcols)), shape=bshape).tolil()
    return b1



def buildA2top(edges, nVertices):
    Arow = []
    Acol = []
    Adata = []
    for row in xrange(0, edges.shape[0]):
        v0, v1 = edges[row, :]
        Arow.append(row); Acol.append(v0); Adata.append(-1)
        Arow.append(row); Acol.append(v1); Adata.append(1)
    shape = (edges.shape[0], nVertices)
    spA2top = sp.csr_matrix((Adata, (Arow, Acol)), shape=shape)
    return spA2top



def buildA2bottom(pins, w, nVertices):
    Arow = []
    Acol = []
    Adata = []
    for row in xrange(0, pins.size):
        pin = pins[row]
        Arow.append(row); Acol.append(pin); Adata.append(w)
    shape = (pins.size, nVertices)
    spA2bottom = sp.csr_matrix((Adata, (Arow, Acol)), shape=shape)
    return spA2bottom 



def buildB2(heVectors, heIndices, edges, pinPoses, w, G, v1):
    T1 = G * v1
    b2 = []
    for row in xrange(0, edges.shape[0]):
        e0 = heVectors[heIndices[row], :]
        c = T1[2 * row]; s = T1[2 * row + 1]
        rScale = 1.0 / np.sqrt(c * c + s * s)
        c *= rScale; s *= rScale
        T2 = np.asarray(( (c, s), (-s, c) ))
        e1 = np.dot(T2, e0)
        b2.extend(e1)
    for row in xrange(0, pinPoses.shape[0]):
        pinPos = pinPoses[row, :]
        b2.extend(w * pinPos)
    b2 = np.asarray(b2).reshape(-1, 2)
    return b2



def test2():
    n = 10
    scale = 10
    xy, triangles = planemesh.build(n, scale)
    halfedges = halfedge.build(triangles)
    heVectors = np.asarray([xy[he.ivertex, :] - xy[he.prev().ivertex, :] for he in halfedges])
    pins = np.asarray([0, n])
    #pinPoses = xy[pins, :]
    pinPoses = np.asarray(( (-scale, 0), (0, 0) ))
    w = 1000.0
    nVertices = xy.shape[0]
    edges, heIndices = halfedge.toEdge(halfedges)
    nEdges = edges.shape[0]
    
    A1top, G = buildA1top(heVectors, halfedges, edges, heIndices, nVertices)
    A1bottom = buildA1bottom(pins, w, nVertices)
    b1 = buildB1(pins, pinPoses, w, nEdges)
    A1 = sp.vstack((A1top, A1bottom))
    tA1 = A1.transpose()
    v1 = spla.spsolve(tA1 * A1, tA1 * b1)
    
    A2top = buildA2top(edges, nVertices)
    A2bottom = buildA2bottom(pins, w, nVertices)
    b2 = buildB2(heVectors, heIndices, edges, pinPoses, w, G, v1)
    A2 = sp.vstack((A2top, A2bottom))
    tA2 = A2.transpose()
    v2x = spla.spsolve(tA2 * A2, tA2 * b2[:, 0])
    v2y = spla.spsolve(tA2 * A2, tA2 * b2[:, 1])
    v2 = np.vstack((v2x, v2y)).T
    
    if n == 1:
        answerA1top = np.asarray(( (     0,      0, -0.25,  0.25, -0.25, -0.25,    0.5,      0),
                                   (     0,      0, -0.25, -0.25,  0.25, -0.25,      0,    0.5),
                                   (-0.333,  0.333, 0.666,     0,     0,     0, -0.333, -0.333),
                                   (-0.333, -0.333,     0, 0.666,     0,     0,  0.333, -0.333),
                                   (   0.5,      0,  -0.5,  -0.5,     0,     0,      0,    0.5),
                                   (     0,    0.5,   0.5,  -0.5,     0,     0,   -0.5,      0),
                                   (-0.333, -0.333,     0,     0, 0.666,     0, -0.333,  0.333),
                                   ( 0.333, -0.333,     0,     0,     0, 0.666, -0.333, -0.333),
                                   (     0,    0.5,     0,     0,  -0.5,  -0.5,    0.5,      0),
                                   (  -0.5,      0,     0,     0,   0.5,  -0.5,      0,    0.5)
                                  ))
        print "Error of G : %e" % la.norm(A1top - answerA1top, np.Inf)
        #print "Full A1top : ", A1top.todense()

    v1 = v1.reshape(-1, 2)
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.triplot(v2[:,0], v2[:,1], triangles)
    plt.show()
    
        
if __name__ == '__main__':
    test2()
