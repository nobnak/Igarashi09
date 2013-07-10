import sys
import optparse
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np

import planemesh

winSize = (600, 400)
winTitle = 'Test OpenGL'
winOrigin = (-winSize[0] * 0.5, -winSize[1] * 0.5)

n = 10
scale = 300
vertices = None
triangles = None


def init():
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    
    global vertices, triangles 
    vertices, triangles = planemesh.build(n, scale)
    half = scale * 0.5
    vertices -= half
          
    
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glPushAttrib(GL_CURRENT_BIT)
    
    glColor3f(1, 0, 0)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glBegin(GL_TRIANGLES)
    for row in xrange(0, triangles.shape[0]):
        tri = triangles[row, :]
        glVertex(*vertices[tri[0], :])
        glVertex(*vertices[tri[1], :])
        glVertex(*vertices[tri[2], :])
    glEnd()
    
    glPopAttrib()
    glPopMatrix()
    
    glutSwapBuffers()
    
def reshape(width, height):
    winSize = (width, height)
    winOrigin = (-width * 0.5, -height * 0.5)
    
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(winOrigin[0], winOrigin[0] + width, winOrigin[1], winOrigin[1] + height, -1.0, 1.0)
    
def keyboard(key, x, y):
    pass
def mouse(button, state, x, y):
    x, y = mouse2camera(x, y)
    print "mouse button=%s state=%s (%.1f,%.1f)" % (button, state, x, y)
    
def motion(x, y):
    x, y = mouse2camera(x, y)
    print "motion (%.1f, %.1f)" % (x, y)
    
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