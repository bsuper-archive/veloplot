import random
import shared
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

randomize = 0

verticies = (
    (3, -1, -0.1),
    (3, 1, -0.1),
    (-3, 1, -0.1),
    (-3, -1, -0.1),
    (3, -1, 0.1),
    (3, 1, 0.1),
    (-3, -1, 0.1),
    (-3, 1, 0.1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )

surfaces = (
    (0,1,2,3),
    (4,5,7,6),
    (0,3,6,4),
    (1,2,7,5),
    (0,1,5,4),
    (2,3,6,7)
    )


def ReflectorUpdate():
    global reflectors
    reflector0 = (
        (-2.6, -0.25, z0),
        (-2.6, 0.25, z0),
        (-2.1, 0.25, z0),
        (-2.1, -0.25, z0)
        )
    reflector1 = (
        (-2, 0.35, z1),
        (-2, 0.85, z1),
        (-1.5, 0.85, z1),
        (-1.5, 0.35, z1)
        )
    reflector2 = (
        (-1.4, -0.25, z2),
        (-1.4, 0.25, z2),
        (-0.9, 0.25, z2),
        (-0.9, -0.25, z2)
        )
    reflector3 = (
        (-2, -0.85, z3),
        (-2, -0.35, z3),
        (-1.5, -0.35, z3),
        (-1.5, -0.85, z3)
        )
    reflector4 = (
        (2.6, -0.25, z4),
        (2.6, 0.25, z4),
        (2.1, 0.25, z4),
        (2.1, -0.25, z4)
        )
    reflector5 = (
        (2, -0.35, z5),
        (2, -0.85, z5),
        (1.5, -0.85, z5),
        (1.5, -0.35, z5)
        )
    reflector6 = (
        (1.4, -0.25, z6),
        (1.4, 0.25, z6),
        (0.9, 0.25, z6),
        (0.9, -0.25, z6)
        )
    reflector7 = (
        (2, 0.85, z7),
        (2, 0.35, z7),
        (1.5, 0.35, z7),
        (1.5, 0.85, z7)
        )
    reflectors = (
        (reflector0),
        (reflector1),
        (reflector2),
        (reflector3),
        (reflector4),
        (reflector5),
        (reflector6),
        (reflector7)
        )

def RandomizeZ():
    global z0
    global z1
    global z2
    global z3
    global z4
    global z5
    global z6
    global z7
    scale = 1
    offset = 0.2
    z0 = random.random()*scale+offset
    z1 = random.random()*scale+offset
    z2 = random.random()*scale+offset
    z3 = random.random()*scale+offset
    z4 = random.random()*scale+offset
    z5 = random.random()*scale+offset
    z6 = random.random()*scale+offset
    z7 = random.random()*scale+offset
    
def UpdateZ():
    global z0
    global z1
    global z2
    global z3
    global z4
    global z5
    global z6
    global z7
    scale = 1
    offset = 0.2
    z0 = (1-shared.zvals[0])*scale+offset
    z1 = (1-shared.zvals[1])*scale+offset
    z2 = (1-shared.zvals[2])*scale+offset
    z3 = (1-shared.zvals[3])*scale+offset
    z4 = (1-shared.zvals[4])*scale+offset
    z5 = (1-shared.zvals[5])*scale+offset
    z6 = (1-shared.zvals[6])*scale+offset
    z7 = (1-shared.zvals[7])*scale+offset

def DrawBoard():
    glBegin(GL_QUADS)
    glColor3fv((0,1,0))
    for surface in surfaces:
        for vertex in surface:
            glVertex3fv(verticies[vertex])
    glEnd()
    
    glBegin(GL_LINES)
    glColor3fv((0,0,0))
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()
    
def DrawReflectors():
    glBegin(GL_LINES)
    glColor3fv((0,0,0))
    for reflector in reflectors:
        glVertex3fv(reflector[0])
        glVertex3fv(reflector[1])
        glVertex3fv(reflector[0])
        glVertex3fv(reflector[3])
        glVertex3fv(reflector[2])
        glVertex3fv(reflector[1])
        glVertex3fv(reflector[2])
        glVertex3fv(reflector[3])
    glEnd()
    
    glBegin(GL_QUADS)
    glColor3fv((1,1,1))
    for reflector in reflectors:
        for vertex in reflector:
            glVertex3fv(vertex)
    glEnd()

def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    glTranslatef(0,0, -10)

    glRotatef(60, -1, 0, 0.7)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    glTranslatef(-0.5,0,0)
                if event.key == pygame.K_RIGHT:
                    glTranslatef(0.5,0,0)

                if event.key == pygame.K_UP:
                    glTranslatef(0,1,0)
                if event.key == pygame.K_DOWN:
                    glTranslatef(0,-1,0)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    #glTranslatef(0,0,1.0)
                    glRotatef(-10,0,0,1)

                if event.button == 5:
                    #glTranslatef(0,0,-1.0)
                    glRotatef(10,0,0,1)
        
        if randomize:
            RandomizeZ()
        else:
            UpdateZ()
        ReflectorUpdate()
        #glRotatef(1, 3, 1, 1)
        #glRotatef(1, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        DrawBoard()
        DrawReflectors()
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == '__main__':
    randomize = 0
    main()
