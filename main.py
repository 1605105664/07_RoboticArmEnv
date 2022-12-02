import gym
import numpy as np
from gym.envs.registration import register
from stable_baselines3 import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

import RoboticArmEnv_2Robots_Incremental as RAE
import time

#number of arm segments
N_ARMS=2
register(
    id="RoboticArmEnv-v1",
    entry_point=RAE.RoboticArmEnv_V1,
    max_episode_steps=500,
    kwargs={'num_arms': N_ARMS}
)

# Parallel environments
# env = make_vec_env("RoboticArmEnv-v1", n_envs=8)

# Single Threaded Env
env = RAE.RoboticArmEnv_V1(training=True, num_arms=N_ARMS)

# It will check your custom environment and output additional warnings if needed
# check_env(env)

# model = A2C("MlpPolicy", env, verbose=2)
# model.learn(total_timesteps=1000000)
# model.save("a2c")

# model = PPO("MlpPolicy", env, verbose=3)
# model.learn(total_timesteps=1000000)
# model.save("ppo")

model = DQN("MlpPolicy", env, verbose=3, exploration_fraction=0.70)
model.learn(total_timesteps=1000000)
model.save("dqn")
# #
del model # remove to demonstrate saving and loading

# model = A2C.load("a2c")
# model = PPO.load("ppo")
model = DQN.load("dqn")

episodes = 100
env = RAE.RoboticArmEnv_V1(training=False, num_arms=N_ARMS)
for episode in range(episodes):
    done = False
    obs = env.reset()
    cumReward = 0
    while not done:
        # Training Model
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        # Random Action
        # random_action = env.action_space.sample()
        # obs, reward, done, info = env.step(random_action)

        # print('obs',obs,'reward', reward)
        cumReward+=reward
        env.render()
        time.sleep(0.05) #slow down the animation
    print('Reward:', cumReward)

