import gym
import numpy as np
from gym.envs.registration import register
from stable_baselines3 import *
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

import RoboticArmEnv_2Robots_Incremental as RAE
import time
import argparse
import random

# initialize CLI
parser = argparse.ArgumentParser(description='CLI for train module')
parser.add_argument('-nr', '--N_ROBOTS', help='# of robot', default=1, type=int)
parser.add_argument('-na', '--N_ARMS', help='# of arms per robot', default=2, type=int)
parser.add_argument('-a', '--ALPHA', help='alpha value', default=0.5, type=float)
args = vars(parser.parse_args())

# number of arm segments
register(
    id="RoboticArmEnv-v1",
    entry_point=RAE.RoboticArmEnv_V1,
    max_episode_steps=1000,
    kwargs={'num_arms': args['N_ARMS'], 'alpha_reward': args['ALPHA'], 'num_robots': args['N_ROBOTS']}
)

if __name__ == '__main__':
    # Get current timestamp as JobID
    JobID = str(round(time.time()))+str(random.randrange(1e5, 1e6))  # include random seed to prevent collision
    print('args:', args)
    print('JobID:', JobID)

    # Parallel environments
    env = SubprocVecEnv([lambda: gym.make("RoboticArmEnv-v1")]*4)
    env = VecMonitor(env, './output/ppo/'+JobID)

    # Single Threaded Env
    # env = RAE.RoboticArmEnv_V1(training=True, num_arms=args['N_ARMS'], alpha_reward=args['ALPHA'], num_robots=args['N_ROBOTS'])
    # env = Monitor(env,'log')

    # It will check your custom environment and output additional warnings if needed
    # check_env(env)
    # model = A2C("MlpPolicy", env, verbose=2)
    # model.learn(total_timesteps=100000)
    # model.save("a2c")

    model = PPO("MlpPolicy", env, verbose=2)
    model.learn(total_timesteps=1e7)
    model.save("./output/ppo/"+JobID)

    # model = DQN("MlpPolicy", env, verbose=2, exploration_fraction=0.70)
    # model.learn(total_timesteps=100)
    # model.save("dqn")

    # del model # remove to demonstrate saving and loading
