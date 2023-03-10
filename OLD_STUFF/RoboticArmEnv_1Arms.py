import numpy as np
import gym
import pygame
import OpenGL
import glm
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random

class RoboticArmEnv_1Arms_V0(gym.Env):
    def __init__(self, render=True):
        super().__init__()
        # Sim Parameters
        self.arm_length = 5
        self.arm_width = 0.25
        self.num_arms = 1
        self.destSize = 2
        # gym init
        reach_dist = self.num_arms * self.arm_length
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(np.array([0.0, 0.0, 0.0, -reach_dist, -reach_dist, -reach_dist, -reach_dist, -reach_dist, -reach_dist]), np.array([2*np.pi, np.pi, 2 * reach_dist, reach_dist, reach_dist, reach_dist, reach_dist, reach_dist, reach_dist]), dtype=np.float32)
        # self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.state = np.zeros(9)
        # renderer init
        if render: # prevent UI from getting stuck during training
            pygame.init()
            display = (800, 600)
            pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
            gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
            glTranslatef(0.0, 0, -45)
            # glTranslatef(0.0, 0, -25)
            # glRotatef(90, 1, 0, 0)

        # state init
        self.done = False
        self.theta0 = np.pi # [0.0, 2pi]
        self.phi0 = 0 # [0, pi]
        destination_x = random.random()
        destination_y = random.random()
        destination_z = random.random()
        self.dest = glm.vec3(destination_x, destination_y, destination_z)
        self.dest = glm.normalize(self.dest)
        self.dest = self.arm_length * self.dest
        end_effector = glm.vec3(0,0,self.arm_length)
        self.dist2Dest = (glm.length(self.dest - end_effector))

        # Obstacles
        self.obstacles = [
            # (glm.vec3(5, 0, 0), 2)
        ]

        pass

    def step(self, action):
        # TAKE ACTION
        increment = 0.1
        if action == 0:
            self.theta0 += increment
        elif action == 1:
            self.theta0 -= increment
        elif action == 2:
            self.phi0 += increment
        elif action == 3:
            self.phi0 -= increment

        if self.theta0 < 0:
            self.theta0 += 2*np.pi
        elif self.theta0 > 2*np.pi:
            self.theta0 -= 2 * np.pi
        self.theta0 = np.clip(self.theta0, 0, np.pi * 2)
        self.phi0 = np.clip(self.phi0, 0.0, np.pi)

        # Calculate Robotic Arm Positions
        cube_centers = []
        m = glm.mul(glm.rotate(self.theta0, (0, 1, 0)), glm.rotate(-glm.pi() / 2, (0, 0, 1)))
        cube_centers.append(m * glm.vec4(0, 0, 0, 1))
        m = glm.mul(m, glm.rotate(self.phi0, (0, 0, 1)))
        m = glm.mul(glm.mul(glm.mul(m, glm.translate((self.arm_length, 0, 0,))), glm.rotate(0, (1, 0, 0))), glm.rotate(-glm.pi(), (0, 0, 1)))
        cube_centers.append(m * glm.vec4(0, 0, 0, 1))
        end_effector = glm.vec3(cube_centers[-1])

        # spheres through the arms
        collision_detected = False
        for c in range(len(cube_centers) - 1):
            for a in np.arange(0.0, 1.0, 0.1):
                for obstacle in self.obstacles:
                    box1 = glm.vec3(cube_centers[c].x, cube_centers[c].y, cube_centers[c].z)
                    box2 = glm.vec3(cube_centers[c + 1].x, cube_centers[c + 1].y, cube_centers[c + 1].z)
                    b = 1 - a
                    arm_sphere = (a * box1.x + b * box2.x, a * box1.y + b * box2.y, a * box1.z + b * box2.z)
                    collision_detected = collision_detected or self.checkSphereCollision(arm_sphere, self.arm_width, obstacle[0], obstacle[1])

        self.dist2Dest = (glm.length(self.dest - end_effector))
        self.state = np.array([self.theta0, self.phi0, self.dist2Dest, end_effector.x, end_effector.y, end_effector.z, self.dest.x, self.dest.y, self.dest.z])

        # CALCULATE REWARD
        hit_destination = self.checkSphereCollision(glm.vec3(end_effector), self.arm_width, self.dest, self.destSize)
        if collision_detected:
            reward = -1000
            done = True
        elif hit_destination:
            reward = 0
            done = True
        else:
            reward = -1 * self.dist2Dest
            done = False

        info = {"End Effector":end_effector}
        return self.state, reward, done, info

    def reset(self, seed=None, options=None):
        self.done = False
        self.theta0 = np.pi # [0.0, 2pi]
        self.phi0 = np.pi / 2 # [-pi/2, pi/2]

        # a random location on the surface of the sphere
        while(True):
            destination_x = random.random() - 0.5
            destination_y = random.random() - 0.5
            destination_z = random.random() - 0.5
            self.dest = glm.vec3(destination_x, destination_y, destination_z)
            self.dest = glm.normalize(self.dest)
            self.dest = self.arm_length * self.dest
            destTooClose = False
            for obstacle in self.obstacles:
                destTooClose = destTooClose or self.checkSphereCollision(obstacle[0], obstacle[1], self.dest, self.destSize)
            if not destTooClose:
                break

        end_effector = glm.vec3(0,0, self.num_arms * self.arm_length-0.01)
        self.dist2Dest = (glm.length(self.dest - end_effector))

        self.state = np.array([self.theta0, self.phi0, self.dist2Dest, end_effector.x, end_effector.y, end_effector.z, self.dest.x, self.dest.y, self.dest.z])

        return self.state

    def render(self, mode='human'):
        # RENDER Robotic Arm
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        cube_centers = []
        glColor((1, 1, 1))
        m = glm.mul(glm.rotate(self.theta0, (0, 1, 0)), glm.rotate(-glm.pi() / 2, (0, 0, 1)))
        self.RenderCube(m)
        cube_centers.append(m * glm.vec4(0, 0, 0, 1))
        m = glm.mul(m, glm.rotate(self.phi0, (0, 0, 1)))
        self.RenderArm(m)
        m = glm.mul(glm.mul(glm.mul(m, glm.translate((self.arm_length, 0, 0,))), glm.rotate(0, (1, 0, 0))), glm.rotate(-glm.pi(), (0, 0, 1)))
        self.RenderCube(m)
        cube_centers.append(m * glm.vec4(0, 0, 0, 1))

        # RENDER DESTINATION
        glColor((1, 1, 0))
        self.RenderSphere(((self.dest.x, self.dest.y, self.dest.z), self.destSize))

        # RENDER OBSTACLES
        for obstacle in self.obstacles:
            glColor((1, 0.2, 0.2))
            self.RenderSphere(obstacle)

        pygame.display.flip()

    def close(self):
        pygame.quit()

    def RenderArm(self, transform):
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
            glm.vec4(self.arm_length, -self.arm_width, -self.arm_width, 1),
            glm.vec4(self.arm_length, self.arm_width, -self.arm_width, 1),
            glm.vec4(0, self.arm_width, -self.arm_width, 1),
            glm.vec4(0, -self.arm_width, -self.arm_width, 1),
            glm.vec4(self.arm_length, -self.arm_width, self.arm_width, 1),
            glm.vec4(self.arm_length, self.arm_width, self.arm_width, 1),
            glm.vec4(0, self.arm_width, self.arm_width, 1),
            glm.vec4(0, -self.arm_width, self.arm_width, 1)
        ]
        glBegin(GL_LINES)
        for edge in box_edges:
            for vertex in edge:
                v = glm.mul(transform, arm_vertices[vertex])
                v_ = (v.x, v.y, v.z)
                glVertex3fv(v_)
        glEnd()

    def RenderCube(self, transform):
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
            glm.vec4(2 * self.arm_width, -2 * self.arm_width, -2 * self.arm_width, 1),
            glm.vec4(2 * self.arm_width, 2 * self.arm_width, -2 * self.arm_width, 1),
            glm.vec4(-2 * self.arm_width, 2 * self.arm_width, -2 * self.arm_width, 1),
            glm.vec4(-2 * self.arm_width, -2 * self.arm_width, -2 * self.arm_width, 1),
            glm.vec4(2 * self.arm_width, -2 * self.arm_width, 2 * self.arm_width, 1),
            glm.vec4(2 * self.arm_width, 2 * self.arm_width, 2 * self.arm_width, 1),
            glm.vec4(-2 * self.arm_width, 2 * self.arm_width, 2 * self.arm_width, 1),
            glm.vec4(-2 * self.arm_width, -2 * self.arm_width, 2 * self.arm_width, 1)
        ]
        glBegin(GL_LINES)
        for edge in box_edges:
            for vertex in edge:
                v = glm.mul(transform, box_vertices[vertex])
                v_ = (v.x, v.y, v.z)
                glVertex3fv(v_)
        glEnd()

    def RenderSphere(self, obstacle):
        glPushMatrix()
        glTranslatef(obstacle[0][0], obstacle[0][1], obstacle[0][2])  # Move to the place
        glScale(obstacle[1], obstacle[1], obstacle[1])
        sphere_base = gluNewQuadric()
        gluSphere(sphere_base, 1.0, 32, 16)  # Draw sphere
        glPopMatrix()

    def checkSphereCollision(self, p1, r1, p2, r2):
        return glm.length(p2 - p1) < r1 + r2
