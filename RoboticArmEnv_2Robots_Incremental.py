import numpy as np
import gym
import glm
import random
import time

def bound(num):
    if num < 0:
        num += 2*np.pi
    elif num > 2*np.pi:
        num -= 2*np.pi
    return num


class RoboticArmEnv_V1(gym.Env):
    def __init__(self, training=True, num_arms=2, arm_length=10, arm_width=0.5, destSize=5, alpha_reward=0.5):
        super().__init__()
        # Sim Parameters
        self.arm_length = arm_length
        self.arm_width = arm_width
        self.num_arms = num_arms
        self.num_robots = 1
        self.robot_roots = [glm.vec3(0.0, self.arm_length, 0.0), glm.vec3(0.0, -self.arm_length, 0.0)]
        self.destSize = destSize
        self.increment = np.pi / 12.0
        self.num_increments = round(np.pi * 2 / self.increment)
        self.total_correct_reward = 1000

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
        self.observation_space = gym.spaces.Box(
            np.array(2 * self.num_robots * self.num_arms * [0] + 3 * self.num_robots * [-reach_dist*3]),
            np.array(2 * self.num_robots * self.num_arms * [2 * np.pi] + 3 * self.num_robots * [reach_dist*3]),
            dtype=np.float32)

        # print(self.observation_space)

        self.rendered = False
        self.training = training

        # state init
        self.reset()

        # Obstacles
        self.obstacles = [
            # (glm.vec3(5, 0, 0), 2)
        ]
        self.name = "RobotArmEnv"
        self.previous_end_effectors = []
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

        self.state = np.concatenate((self.theta, self.phi, self.dest))

        # print(self.state)

        # CALCULATE REWARD
        prev_dist_destinations = []
        dist_destinations = []
        num_hits = 0
        num_dist_farther = 0
        num_dist_closer = 0
        for r in range(self.num_robots):
            prev_dist_destinations.append(self.getDistanceToPoints(glm.vec3(self.dest[3*r:3*r+3]), self.previous_end_effectors[r]))
            dist_destinations.append(self.getDistanceToPoints(glm.vec3(self.dest[3*r:3*r+3]), end_effectors[r]))

            if (dist_destinations[r] < self.destSize):
                num_hits += 1

            if prev_dist_destinations[r] - dist_destinations[r] > 0.001:
                num_dist_closer += 1
            elif prev_dist_destinations[r] - dist_destinations[r] < 0.001:
                num_dist_farther += 1


        # general Rewards
        if collision_detected:
            reward = -10000
            done = True
        elif num_hits == self.num_robots:
            reward = self.total_correct_reward * self.alpha_reward
            done = True
        else:
            reward = -2 * (num_dist_farther + 1) + num_dist_closer
            done = False


        for r in range(self.num_robots):
            dist_robot = dist_destinations[r]
            # Gives incremental rewards as the robotic arms approach the destination
            while self.destSizeIndex[r] < self.num_increments_dest:
                if dist_robot < self.destSizes[self.destSizeIndex[r]]:
                    self.destSizeIndex[r] += 1
                    # print(self.destSizeIndex[0])
                    reward += (1 - self.alpha_reward) / self.num_robots * self.total_correct_reward / self.num_increments_dest
                else:
                    break

        info = {"End Effector": end_effectors}
        return self.state, reward, done, info

    def reset(self, seed=None, options=None):
        # state init
        self.done = False
        self.theta = np.random.rand(self.num_arms * self.num_robots)*2*np.pi  # [0, 2*pi]
        self.phi = np.random.rand(self.num_arms * self.num_robots)*2*np.pi  # [0, 2*pi]
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

        # Calculate Robotic Arm Positions
        self.previous_end_effectors = []
        for r in range(self.num_robots):
            m = glm.mul(glm.rotate(-glm.pi() / 2, (0, 0, 1)), glm.translate(self.robot_roots[r]))
            m = glm.mul(m, glm.rotate(self.theta[0 + self.num_arms * r], (0, 1, 0)))
            m = glm.mul(m, glm.rotate(self.phi[0 + self.num_arms * r], (0, 0, 1)))

            for i in range(1, self.num_arms):
                m = glm.mul(glm.mul(glm.mul(m, glm.translate((self.arm_length, 0, 0,))),
                                    glm.rotate(self.theta[i + self.num_arms * r], (1, 0, 0))),
                            glm.rotate(-glm.pi(), (0, 0, 1)))
                m = glm.mul(m, glm.rotate(self.phi[i + self.num_arms * r], (0, 0, 1)))

            m = glm.mul(glm.mul(glm.mul(m, glm.translate((self.arm_length, 0, 0,))), glm.rotate(0, (1, 0, 0))),
                        glm.rotate(-glm.pi(), (0, 0, 1)))
            self.previous_end_effectors.append(glm.vec3(m * glm.vec4(0, 0, 0, 1)))


        self.destSizeIndex = np.array([0]*self.num_robots, dtype=int)
        self.state = np.concatenate((self.theta, self.phi, self.dest))
        # print(self.state)
        return self.state


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
