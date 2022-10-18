import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

import RoboticArmEnv as RAE
import time


# Parallel environments
env = RAE.RoboticArmEnvV0(render=False)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=15000)
model.save("a2c")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c")
episodes = 10

score=0
env = RAE.RoboticArmEnvV0(render=True)
for episode in range(episodes):
    done = False
    obs = env.reset()
    cumReward = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        cumReward+=reward
        env.render()
    print('Reward:', cumReward)
    if reward>0:
        score+=1
print(score,'/',episodes)

