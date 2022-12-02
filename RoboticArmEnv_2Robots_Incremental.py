import numpy as np
import gym
import pygame
import OpenGL
import glm
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random
import time

def bound(num):
    if num < 0:
        num += 2*np.pi
    elif num > 2*np.pi:
        num -= 2*np.pi
    return num


class RoboticArmEnv_V1(gym.Env):
    def __init__(self, training=True, num_arms=2, arm_length=10, arm_width=0.5, destSize=10, increment=0.1, alpha_reward=0.25):
        super().__init__()
        # Sim Parameters
        self.arm_length = arm_length
        self.arm_width = arm_width
        self.num_arms = num_arms
        self.num_robots = 2
        self.robot_roots = [glm.vec3(0.0, self.arm_length, 0.0), glm.vec3(0.0, -self.arm_length, 0.0)]
        self.destSize = destSize
        self.increment = np.pi / 12.0
        self.num_increments = round(np.pi * 2 / self.increment)
        self.total_correct_reward = 10000

        self.alpha_reward = alpha_reward
        self.first_time_hit = np.array([False]*self.num_robots,dtype=bool)

        self.num_increments_dest = 20
        self.incremental_destSize = 0.5
        self.destSizes = np.array(np.arange(self.destSize - self.incremental_destSize + self.incremental_destSize * self.num_increments_dest, self.destSize - self.incremental_destSize, -self.incremental_destSize),dtype=float)
        # print(self.destSizes[4])
        # print(self.destSizes)
        self.destSizeIndex = np.array([0]*self.num_robots, dtype=int)

        # gym init
        reach_dist = self.arm_length
        self.action_space = gym.spaces.Discrete(4*self.num_robots*self.num_arms)
        # num_arms*[self.theta, self.phi], self.dest.x, self.dest.y, self.dest.z])
        self.observation_space = gym.spaces.Tuple(tuple(self.num_robots * self.num_arms * 2 * [gym.spaces.Discrete(self.num_increments)]) +
                                                   tuple([gym.spaces.Box(np.array(3*self.num_robots*[-reach_dist*3]), np.array(3*self.num_robots*[reach_dist*3]), dtype=np.float32)]))
        print(self.observation_space)
        self.rendered = False
        self.training = training

        # state init
        self.reset()

        # Obstacles
        self.obstacles = [
            # (glm.vec3(5, 0, 0), 2)
        ]
        self.name = "RobotArmEnv"
        random.seed(time.time())

    def step(self, action):
        if action < self.num_arms * self.num_robots:
            self.theta[action//2] += (-1)**action*self.increment
        else:
            self.phi[action//2-self.num_arms * self.num_robots] += (-1)**action*self.increment
        self.theta = np.array(list(map(bound, self.theta)))
        self.phi = np.array(list(map(bound, self.phi)))

        # Calculate Robotic Arm Positions
        cube_centers = []
        end_effectors = []
        for r in range(self.num_robots):
            cubes = []
            m = glm.mul(glm.rotate(-glm.pi() / 2, (0, 0, 1)), glm.translate(self.robot_roots[r]))
            m = glm.mul(m, glm.rotate(self.theta[0+self.num_arms*r], (0, 1, 0)))
            cubes.append(m * glm.vec4(0, 0, 0, 1))
            m = glm.mul(m, glm.rotate(self.phi[0+self.num_arms*r], (0, 0, 1)))

            for i in range(1, self.num_arms):
                m = glm.mul(glm.mul(glm.mul(m, glm.translate((self.arm_length, 0, 0,))), glm.rotate(self.theta[i+self.num_arms*r], (1, 0, 0))), glm.rotate(-glm.pi(), (0, 0, 1)))
                cubes.append(m * glm.vec4(0, 0, 0, 1))
                m = glm.mul(m, glm.rotate(self.phi[i+self.num_arms*r], (0, 0, 1)))

            m = glm.mul(glm.mul(glm.mul(m, glm.translate((self.arm_length, 0, 0,))), glm.rotate(0, (1, 0, 0))), glm.rotate(-glm.pi(), (0, 0, 1)))
            cubes.append(m * glm.vec4(0, 0, 0, 1))
            cube_centers.append(cubes)
            end_effectors.append(glm.vec3(cubes[-1]))

        # spheres through the arms
        collision_detected = False
        # Collision detection between arms
        for c1 in range(self.num_robots):
            centers1 = cube_centers[c1]
            for c2 in range(c1+1, self.num_robots):
                centers2 = cube_centers[c2]
                for j1 in range(len(centers1) - 1):
                    box1_robot1 = glm.vec3(centers1[j1].x, centers1[j1].y, centers1[j1].z)
                    box2_robot1 = glm.vec3(centers1[j1 + 1].x, centers1[j1 + 1].y, centers1[j1 + 1].z)
                    for j2 in range(len(centers2) - 1):
                        box1_robot2 = glm.vec3(centers2[j2].x, centers2[j2].y, centers2[j2].z)
                        box2_robot2 = glm.vec3(centers2[j2 + 1].x, centers2[j2 + 1].y, centers2[j2 + 1].z)
                        dist = self.closestDistanceBetweenLines(box1_robot1, box2_robot1, box1_robot2, box2_robot2)
                        if dist < self.arm_width*2.0:
                            collision_detected = True

        # self.state = tuple(np.round(self.theta / self.increment)) + tuple(np.round(self.phi / self.increment)) + tuple(np.array(self.dest, dtype=np.float32),)
        self.state = np.concatenate((np.round(self.theta / self.increment), np.round(self.phi / self.increment),
                                     np.array(self.dest, dtype=np.float32)))

        # CALCULATE REWARD
        dest1 = glm.vec3(self.dest[0:3])
        dest2 = glm.vec3(self.dest[3:6])
        robot_1_hit_destination = self.checkSphereCollision(glm.vec3(end_effectors[0]), self.arm_width, dest1, self.destSize)
        robot_2_hit_destination = self.checkSphereCollision(glm.vec3(end_effectors[1]), self.arm_width, dest2, self.destSize)

        # general Rewards
        if collision_detected:
            reward = -10000
            done = True
        elif robot_1_hit_destination and robot_2_hit_destination:
            reward = self.total_correct_reward * self.alpha_reward
            done = True
        elif robot_1_hit_destination:
            reward = -1
            done = False
        elif robot_2_hit_destination:
            reward = -1
            done = False
        else:
            reward = -2
            done = False

        dist_robot1 = self.getDistanceToPoints(glm.vec3(end_effectors[0]), dest1)
        dist_robot2 = self.getDistanceToPoints(glm.vec3(end_effectors[1]), dest2)
        # Gives incremental rewards as the robotic arms approach the destination
        while self.destSizeIndex[0] < self.num_increments_dest:
            if dist_robot1 < self.destSizes[self.destSizeIndex[0]]:
                self.destSizeIndex[0] += 1
                # print(self.destSizeIndex[0])
                reward += (1 - self.alpha_reward) / self.num_robots * self.total_correct_reward / self.num_increments_dest
            else:
                break
        while self.destSizeIndex[1] < self.num_increments_dest:
            if dist_robot2 < self.destSizes[self.destSizeIndex[1]]:
                self.destSizeIndex[1] += 1
                # print(self.destSizeIndex[1])
                reward += (1 - self.alpha_reward) / self.num_robots * self.total_correct_reward / self.num_increments_dest
            else:
                break

        info = {"End Effector": end_effectors}
        return self.state, reward, done, info

    def reset(self, seed=None, options=None):
        # state init
        self.done = False
        self.theta = np.array(self.num_arms * self.num_robots * [random.randint(0, self.num_increments) * self.increment]) # [0, 2*pi]
        self.phi = np.array(self.num_arms * self.num_robots * [random.randint(0, self.num_increments) * self.increment]) # [0, 2*pi]
        self.dest = []
        for r in range(self.num_robots):
            destination_x = random.random()
            destination_y = random.random()
            destination_z = random.random()
            dest = glm.vec3(destination_x, destination_y, destination_z)
            # dest = glm.vec3(0, 0, 0)
            dest = glm.normalize(dest)
            dest = self.robot_roots[r] + dest * self.num_arms * self.arm_length * random.random()
            self.dest.append(dest.y)
            self.dest.append(dest.x)
            self.dest.append(dest.z)

        self.state = self.observation_space.sample()
        print(self.state)
        self.state = np.concatenate((np.round(self.theta / self.increment), np.round(self.phi / self.increment), np.array(self.dest, dtype=np.float32)))
        # self.state = tuple(np.round(self.theta / self.increment)) + tuple(np.round(self.phi / self.increment)) + tuple(np.array(self.dest, dtype=np.float32),)
        print(self.state)
        return self.state

    def render_init(self):
        if self.rendered:
            return
        pygame.init()
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        gluPerspective(45, (display[0] / display[1]), 0.01, 100.0)
        glTranslatef(0.0, -10.0, -45)
        self.rendered = True

    def render(self, mode='human'):
        if self.training:
            return

        self.render_init()

        # Uncomment to rotate the scene
        glRotatef(-1.0, 0, 1, 0)

        # RENDER Robotic Arm
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor((1, 1, 1))
        cube_centers = []
        end_effectors = []

        for r in range(self.num_robots):
            cubes = []
            m = glm.mul(glm.rotate(-glm.pi() / 2, (0, 0, 1)), glm.translate(self.robot_roots[r]))
            m = glm.mul(m, glm.rotate(self.theta[0 + self.num_arms * r], (0, 1, 0)))
            self.RenderCube(m)
            cubes.append(m * glm.vec4(0, 0, 0, 1))
            m = glm.mul(m, glm.rotate(self.phi[0 + self.num_arms * r], (0, 0, 1)))
            self.RenderArm(m)

            for i in range(1, self.num_arms):
                m = glm.mul(glm.mul(glm.mul(m, glm.translate((self.arm_length, 0, 0,))),
                                    glm.rotate(self.theta[i + self.num_arms * r], (1, 0, 0))),
                            glm.rotate(-glm.pi(), (0, 0, 1)))
                self.RenderCube(m)
                cubes.append(m * glm.vec4(0, 0, 0, 1))
                m = glm.mul(m, glm.rotate(self.phi[i + self.num_arms * r], (0, 0, 1)))
                self.RenderArm(m)

            m = glm.mul(glm.mul(glm.mul(m, glm.translate((self.arm_length, 0, 0,))), glm.rotate(0, (1, 0, 0))),
                        glm.rotate(-glm.pi(), (0, 0, 1)))
            self.RenderCube(m)
            cubes.append(m * glm.vec4(0, 0, 0, 1))
            cube_centers.append(cubes)
            end_effectors.append(glm.vec3(cubes[-1]))

        # RENDER DESTINATION
        # glColor((1, 1, 0))
        # for d in range(self.num_robots):
        #     if(self.destSizeIndex[d] < self.num_increments_dest):
        #         self.RenderSphere((self.dest[3*d:3*d + 3], self.destSizes[self.destSizeIndex[d]]))
        #     else:
        #         self.RenderSphere((self.dest[3*d:3*d+3], self.destSize))

        for d in range(self.num_robots):
            self.RenderLine(end_effectors[d], self.dest[3*d:3*d+3])

        # RENDER OBSTACLES
        # for obstacle in self.obstacles:
        #     glColor((1, 0.2, 0.2))
        #     self.RenderSphere(obstacle)

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

    def RenderLine(self, a0, a1):
        a0 = np.array(a0)
        a1 = np.array(a1)
        dist = np.sqrt(np.sum(np.square(a1 - a0))) / self.arm_length / self.num_arms / 2
        glColor((dist, 1-dist, 0))
        glBegin(GL_LINES)
        glVertex3fv(a0)
        glVertex3fv(a1)
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

    def getDistanceToPoints(self, p1, p2):
        return glm.length(p2 - p1)

    ############
    # Finding the closest distance between two line segments (used in the collision calculation) uses a solution
    # posted to stack overflow. This is a minor part of the assignment and replacing our collision detection with
    # this solution sped up our training by a lot.
    # https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
    # The solution found on stack overflow has been modified (simplified) to better match our use case.
    #############
    def closestDistanceBetweenLines(self, a0, a1, b0, b1):
        ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
            Return the closest points on each segment and their distance
        '''

        # Calculate denomitator
        A = a1 - a0
        B = b1 - b0
        magA = np.linalg.norm(A)
        magB = np.linalg.norm(B)

        _A = A / magA
        _B = B / magB

        cross = np.cross(_A, _B)
        denom = np.linalg.norm(cross) ** 2

        # If lines are parallel (denom=0) test if lines overlap.
        # If they don't overlap then there is a closest point solution.
        # If they do overlap, there are infinite closest positions, but there is a closest distance
        if not denom:
            d0 = np.dot(_A, (b0 - a0))

            # Overlap only possible with clamping
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if np.absolute(d0) < np.absolute(d1):
                    return a0, b0, np.linalg.norm(a0 - b0)
                return np.linalg.norm(a0 - b1)

            # Is segment B after A?
            elif d0 >= magA <= d1:
                if np.absolute(d0) < np.absolute(d1):
                    return a1, b0, np.linalg.norm(a1 - b0)
                return np.linalg.norm(a1 - b1)

            # Segments overlap, return distance between parallel segments
            return np.linalg.norm(((d0 * _A) + a0) - b0)

        # Lines criss-cross: Calculate the projected closest points
        t = (b0 - a0)
        detA = np.linalg.det([t, _B, cross])
        detB = np.linalg.det([t, _A, cross])

        t0 = detA / denom
        t1 = detB / denom

        pA = a0 + (_A * t0)  # Projected closest point on segment A
        pB = b0 + (_B * t1)  # Projected closest point on segment B

        # Clamp projections
        if t0 < 0:
            pA = a0
        elif t0 > magA:
            pA = a1

        if t1 < 0:
            pB = b0
        elif t1 > magB:
            pB = b1

        # Clamp projection A
        if (t0 < 0) or (t0 > magA):
            dot = np.dot(_B, (pA - b0))
            if dot < 0:
                dot = 0
            elif dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (t1 < 0) or (t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if dot < 0:
                dot = 0
            elif dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

        return np.linalg.norm(pA - pB)
