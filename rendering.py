import sys
import optparse
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

winSize = (600, 400)
winTitle = 'Test OpenGL'


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glPushAttrib(GL_CURRENT_BIT)
    
    glColor3f(1, 0, 0)
    glBegin(GL_POLYGON)
    glVertex3f(0, 0, 0)
    glVertex3f(1, 0, 0)
    glVertex3f(0, 1, 0)
    glEnd()
    
    glPopAttrib()
    glPopMatrix()
    
    glutSwapBuffers()

def main(options, args):
    glutInit(sys.argv)

    glutInitDisplayMode(GLUT_DOUBLE)
    glutInitWindowSize(*winSize)
    glutCreateWindow(winTitle)
    init()
        
    glutDisplayFunc(display)
    glutMainLoop()


if __name__ == '__main__':
    parser = optparse.OptionParser(__doc__)
    options, args = parser.parse_args()
    main(options, args)