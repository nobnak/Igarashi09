import sys
import optparse
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import planemesh
import halfedge
import igarashi

winSize = (600, 400)
winTitle = 'Test OpenGL'
winOrigin = (-winSize[0] * 0.5, -winSize[1] * 0.5)

n = 10
scale = 300
w = 1000.0


def init():
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    registerIgarashi()
    compileIgarashi()
    executeIgarashi()

def registerIgarashi():
    global xy, triangles, pins, pinPoses, nVertices, edges
    global A1top, A2top, G
    xy, triangles = planemesh.build(n, scale)
    halfedges = halfedge.build(triangles)
    pins = np.asarray([0, n])
    pinPoses = np.asarray( ((0, 0), (scale, 0)) )
    edges, heIndices = halfedge.toEdge(halfedges)
    A1top, G = igarashi.buildA1top(xy, halfedges, edges, heIndices)
    nVertices = xy.shape[0]
    A2top = igarashi.buildA2top(edges, nVertices)

def compileIgarashi():
    global A1bottom, A2bottom
    A1bottom = igarashi.buildA1bottom(xy, pins, w)
    A2bottom = igarashi.buildA2bottom(pins, w, nVertices)

def executeIgarashi():
    global v2
    b1 = igarashi.buildB1(xy, edges, pins, pinPoses, w)
    A1 = sp.vstack((A1top, A1bottom))
    tA1 = A1.transpose()
    v1 = spla.spsolve(tA1 * A1, tA1 * b1)
    b2 = igarashi.buildB2(xy, edges, pinPoses, w, G, v1)
    A2 = sp.vstack((A2top, A2bottom))
    tA2 = A2.transpose()
    v2x = spla.spsolve(tA2 * A2, tA2 * b2[:, 0])
    v2y = spla.spsolve(tA2 * A2, tA2 * b2[:, 1])
    v2 = np.vstack((v2x, v2y)).T
    
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glPushAttrib(GL_CURRENT_BIT)
    
    glColor3f(0.5, 0.5, 0.5)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glBegin(GL_TRIANGLES)
    for row in xrange(0, triangles.shape[0]):
        tri = triangles[row, :]
        glVertex(*v2[tri[0], :])
        glVertex(*v2[tri[1], :])
        glVertex(*v2[tri[2], :])
    glEnd()
    glColor3f(1, 0, 0)
    glPointSize(5.0)
    glBegin(GL_POINTS)
    for row in xrange(0, pinPoses.shape[0]):
        pos = pinPoses[row, :]
        glVertex(*pos)
    glEnd()
    
    glPopAttrib()
    glPopMatrix()
    
    glutSwapBuffers()
    
def reshape(width, height):
    global winSize, winOrigin
    winSize = (width, height)
    winOrigin = (-width * 0.5, -height * 0.5)
    
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(winOrigin[0], winOrigin[0] + width, winOrigin[1], winOrigin[1] + height, -1.0, 1.0)
    
def keyboard(key, x, y):
    pass
def mouse(button, state, x, y):
    global pinPoses
    x, y = mouse2camera(x, y)
    print "mouse button=%s state=%s (%.1f,%.1f)" % (button, state, x, y)
    if state == GLUT_DOWN:
        pinPoses[1, :] = np.asarray((x, y))
        executeIgarashi()

    
def motion(x, y):
    x, y = mouse2camera(x, y)
    #print "motion (%.1f, %.1f)" % (x, y)
    
def mouse2camera(x, y):
    y = winSize[1] - y
    return winOrigin[0] + x, winOrigin[1] + y

def main(options, args):
    glutInit(sys.argv)

    glutInitDisplayMode(GLUT_DOUBLE)
    glutInitWindowSize(*winSize)
    glutCreateWindow(winTitle)
    init()
        
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutMainLoop()



if __name__ == '__main__':
    parser = optparse.OptionParser(__doc__)
    options, args = parser.parse_args()
    main(options, args)