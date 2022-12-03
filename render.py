import numpy as np
import pygame
from pygame.locals import *
import glm
from OpenGL.GL import *
from OpenGL.GLU import *


#######
# This pygame renderer is based off of the renderer in this tutorial
# https://pythonprogramming.net/opengl-pyopengl-python-pygame-tutorial/
# We learned how to render basic shapes from this tutorial
# and created the rest of the render functionality on our own to work with our open ai gym model.
#######
def render_init():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.01, 100.0)
    glTranslatef(0.0, -10.0, -45)


def render(env):
    # Uncomment to rotate the scene
    # glRotatef(-1.0, 0, 1, 0)

    # RENDER Robotic Arm
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glColor((1, 1, 1))
    cube_centers = []
    end_effectors = []

    for r in range(env.num_robots):
        cubes = []
        m = glm.mul(glm.rotate(-glm.pi() / 2, (0, 0, 1)), glm.translate(env.robot_roots[r]))
        m = glm.mul(m, glm.rotate(env.theta[0 + env.num_arms * r], (0, 1, 0)))
        RenderCube(env, m)
        cubes.append(m * glm.vec4(0, 0, 0, 1))
        m = glm.mul(m, glm.rotate(env.phi[0 + env.num_arms * r], (0, 0, 1)))
        RenderArm(env, m)

        for i in range(1, env.num_arms):
            m = glm.mul(glm.mul(glm.mul(m, glm.translate((env.arm_length, 0, 0,))),
                                glm.rotate(env.theta[i + env.num_arms * r], (1, 0, 0))),
                        glm.rotate(-glm.pi(), (0, 0, 1)))
            RenderCube(env, m)
            cubes.append(m * glm.vec4(0, 0, 0, 1))
            m = glm.mul(m, glm.rotate(env.phi[i + env.num_arms * r], (0, 0, 1)))
            RenderArm(env, m)

        m = glm.mul(glm.mul(glm.mul(m, glm.translate((env.arm_length, 0, 0,))), glm.rotate(0, (1, 0, 0))),
                    glm.rotate(-glm.pi(), (0, 0, 1)))
        RenderCube(env, m)
        cubes.append(m * glm.vec4(0, 0, 0, 1))
        cube_centers.append(cubes)
        end_effectors.append(glm.vec3(cubes[-1]))

    # RENDER DESTINATION
    # glColor((1, 1, 0))
    # for d in range(env.num_robots):
    #     RenderSphere(env, (env.dest[3*d:3*d+3], env.destSize))

    for d in range(env.num_robots):
        RenderLine(env, end_effectors[d], env.dest[3*d:3*d+3])

    # RENDER OBSTACLES
    # for obstacle in env.obstacles:
    #     glColor((1, 0.2, 0.2))
    #     env.RenderSphere(obstacle)

    pygame.display.flip()

def RenderArm(env, transform):
        box_edges = [
            (0, 1),
            (0, 3),
            (0, 4),
            (2, 1),
            (2, 3),
            (2, 6),
            (7, 3),
            (7, 4),
            (7, 6),
            (5, 1),
            (5, 4),
            (5, 6)
        ]
        arm_vertices = [
            glm.vec4(env.arm_length, -env.arm_width, -env.arm_width, 1),
            glm.vec4(env.arm_length, env.arm_width, -env.arm_width, 1),
            glm.vec4(0, env.arm_width, -env.arm_width, 1),
            glm.vec4(0, -env.arm_width, -env.arm_width, 1),
            glm.vec4(env.arm_length, -env.arm_width, env.arm_width, 1),
            glm.vec4(env.arm_length, env.arm_width, env.arm_width, 1),
            glm.vec4(0, env.arm_width, env.arm_width, 1),
            glm.vec4(0, -env.arm_width, env.arm_width, 1)
        ]
        glBegin(GL_LINES)
        for edge in box_edges:
            for vertex in edge:
                v = glm.mul(transform, arm_vertices[vertex])
                v_ = (v.x, v.y, v.z)
                glVertex3fv(v_)
        glEnd()

def RenderCube(env, transform):
    box_edges = [
        (0, 1),
        (0, 3),
        (0, 4),
        (2, 1),
        (2, 3),
        (2, 6),
        (7, 3),
        (7, 4),
        (7, 6),
        (5, 1),
        (5, 4),
        (5, 6)
    ]
    box_vertices = [
        glm.vec4(2 * env.arm_width, -2 * env.arm_width, -2 * env.arm_width, 1),
        glm.vec4(2 * env.arm_width, 2 * env.arm_width, -2 * env.arm_width, 1),
        glm.vec4(-2 * env.arm_width, 2 * env.arm_width, -2 * env.arm_width, 1),
        glm.vec4(-2 * env.arm_width, -2 * env.arm_width, -2 * env.arm_width, 1),
        glm.vec4(2 * env.arm_width, -2 * env.arm_width, 2 * env.arm_width, 1),
        glm.vec4(2 * env.arm_width, 2 * env.arm_width, 2 * env.arm_width, 1),
        glm.vec4(-2 * env.arm_width, 2 * env.arm_width, 2 * env.arm_width, 1),
        glm.vec4(-2 * env.arm_width, -2 * env.arm_width, 2 * env.arm_width, 1)
    ]
    glBegin(GL_LINES)
    for edge in box_edges:
        for vertex in edge:
            v = glm.mul(transform, box_vertices[vertex])
            v_ = (v.x, v.y, v.z)
            glVertex3fv(v_)
    glEnd()

def RenderLine(env, a0, a1):
    a0 = np.array(a0)
    a1 = np.array(a1)
    dist = np.sqrt(np.sum(np.square(a1 - a0))) / env.arm_length / env.num_arms / 2
    glColor((dist, 1-dist, 0))
    glBegin(GL_LINES)
    glVertex3fv(a0)
    glVertex3fv(a1)
    glEnd()

def RenderSphere(env, obstacle):
    glPushMatrix()
    glTranslatef(obstacle[0][0], obstacle[0][1], obstacle[0][2])  # Move to the place
    glScale(obstacle[1], obstacle[1], obstacle[1])
    sphere_base = gluNewQuadric()
    gluSphere(sphere_base, 1.0, 32, 16)  # Draw sphere
    glPopMatrix()