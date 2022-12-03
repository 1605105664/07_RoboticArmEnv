import gym
import numpy as np
from gym.envs.registration import register
from stable_baselines3 import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

import RoboticArmEnv_2Robots_Incremental as RAE


#number of arm segments
N_ARMS=2
register(
    id="RoboticArmEnv-v1",
    entry_point=RAE.RoboticArmEnv_V1,
    max_episode_steps=1000,
    kwargs={'num_arms': N_ARMS}
)

# Parallel environments
# env = make_vec_env("RoboticArmEnv-v1", n_envs=8)

# Single Threaded Env
env = RAE.RoboticArmEnv_V1(training=True, num_arms=N_ARMS)

# It will check your custom environment and output additional warnings if needed
# check_env(env)
# model = A2C("MlpPolicy", env, verbose=2)
# model.learn(total_timesteps=100000)
# model.save("a2c")

# model = PPO("MlpPolicy", env, verbose=2)
# model.learn(total_timesteps=1000000)
# model.save("ppo")

model = DQN("MlpPolicy", env, verbose=2, exploration_fraction=0.70)
model.learn(total_timesteps=100)
model.save("dqn")

# del model # remove to demonstrate saving and loading

