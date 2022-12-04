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
parser.add_argument('-d', '--DESTSIZE', help='destination size', default=5, type=int)
parser.add_argument('-m', '--MAX_STEP', help='maximum timesteps', default=1000, type=int)
parser.add_argument('-e', '--TIMESTEPS', help='total timesteps to train', default=1e5, type=int)
parser.add_argument('-i', '--input', help='input model path', default=None, type=str)
parser.add_argument('-o', '--output', help='output model path', default=None, type=str)
args = vars(parser.parse_args())

# number of arm segments
register(
    id="RoboticArmEnv-v1",
    entry_point=RAE.RoboticArmEnv_V1,
    max_episode_steps=args['MAX_STEP'],
    kwargs={'num_arms': args['N_ARMS'], 'alpha_reward': args['ALPHA'],
            'num_robots': args['N_ROBOTS'], 'destSize': args['DESTSIZE']
            }
)

if __name__ == '__main__':
    # Get current timestamp as JobID
    JobID = str(round(time.time()))+str(random.randrange(1e5, 1e6))  # include random seed to prevent collision
    fname = 'output/ppo/'+'_'.join(map(str, list(args.values())[:4]))+'/'+JobID
    if args['output']:
        fname = args['output']
    print('args:', args)
    print('JobID:', JobID)

    # Parallel environments
    env = SubprocVecEnv([lambda: gym.make("RoboticArmEnv-v1")]*4)
    env = VecMonitor(env, fname)

    # Single Threaded Env
    # env = RAE.RoboticArmEnv_V1(training=True, num_arms=args['N_ARMS'], alpha_reward=args['ALPHA'], num_robots=args['N_ROBOTS'])
    # env = Monitor(env,'log')

    # It will check your custom environment and output additional warnings if needed
    # check_env(env)

    model = PPO("MlpPolicy", env, verbose=2)
    if args['input']:
        model = PPO.load(args['input'], env)
    model.learn(total_timesteps=args['TIMESTEPS'])
    model.save(fname)