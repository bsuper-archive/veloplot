import random
import shared
import pygame
from pygame.locals import *
import numpy as np
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
        (-2.6+dy0, -0.85-dx0, z0),
        (-2.6+dy0, 0.85-dx0, z0),
        (-0.9+dy0, 0.85-dx0, z0),
        (-0.9+dy0, -0.85-dx0, z0)
        )
    reflector1 = (
        (2.6+dy1, -0.85-dx1, z1),
        (2.6+dy1, 0.85-dx1, z1),
        (0.9+dy1, 0.85-dx1, z1),
        (0.9+dy1, -0.85-dx1, z1)
        )
    reflectors = (
        (reflector0),
        (reflector1)
        )

def Randomize():
    global dx0
    global dx1
    global dy0
    global dy1
    global z0
    global z1
    scale = 1
    offset = 0.2
    dx0 = random.random()
    dx1 = random.random()
    dy0 = random.random()
    dy1 = random.random()
    z0 = random.random()*scale+offset
    z1 = random.random()*scale+offset
    
def Update():
    global dx0
    global dx1
    global dy0
    global dy1
    global z0
    global z1
    scale = 1 #cm/mm
    dx0 = shared.xyzvals[0]*scale
    dy0 = shared.xyzvals[1]*scale
    z0 = shared.xyzvals[2]*scale-1.5
    dx1 = shared.xyzvals[3]*scale
    dy1 = shared.xyzvals[4]*scale
    z1 = shared.xyzvals[5]*scale-1.5

#arrays for moving average
full = False
index = 0
count = 10
dx0_saved = np.zeros(count)
dx1_saved = np.zeros(count)
dy0_saved = np.zeros(count)
dy1_saved = np.zeros(count)
z0_saved = np.zeros(count)
z1_saved = np.zeros(count)

def MovingAverage():
    global dx0
    global dx1
    global dy0
    global dy1
    global z0
    global z1
    global dx0_saved
    global dx1_saved
    global dy0_saved
    global dy1_saved
    global z0_saved
    global z1_saved
    global index
    global full
    if index >= count:
        index = 0
    dx0_saved[index] = dx0
    dx1_saved[index] = dx1
    dy0_saved[index] = dy0
    dy1_saved[index] = dy1
    z0_saved[index] = z0
    z1_saved[index] = z1
    index = index + 1
    if full:
        dx0 = np.mean(dx0_saved)
        dx1 = np.mean(dx1_saved)
        dy0 = np.mean(dy0_saved)
        dy1 = np.mean(dy1_saved)
        z0 = np.mean(z0_saved)
        z1 = np.mean(z1_saved)
    else:
        dx0 = np.mean(dx0_saved[:index])
        dx1 = np.mean(dx1_saved[:index])
        dy0 = np.mean(dy0_saved[:index])
        dy1 = np.mean(dy1_saved[:index])
        z0 = np.mean(z0_saved[:index])
        z1 = np.mean(z1_saved[:index])
        if index >= count:
            full = True

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
            Randomize()
        else:
            Update()
            MovingAverage()
        ReflectorUpdate()
        #glRotatef(1, 3, 1, 1)
        #glRotatef(1, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        DrawBoard()
        DrawReflectors()
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == '__main__':
    randomize = 1
    main()
