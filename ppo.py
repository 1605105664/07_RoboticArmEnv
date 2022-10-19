import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import RoboticArmEnv_1Arms as RAE1
import time

training = False

if training:
    # Parallel environments
    env = RAE1.RoboticArmEnv_1Arms_V0(render=False)
    model = PPO("MlpPolicy", env, verbose=2)
    model.gamma = 0.999
    # model.n_steps = 10000
    # model.n_epochs = 200
    model.learning_rate = .5
    # model.batch_size = 5
    model.gae_lambda = 0.05
    model.learn(total_timesteps=10000)
    model.save("ppo_1arm3")
    del model # remove to demonstrate saving and loading

model = PPO.load("ppo_1arm3")
episodes = 1000

score=0
env = RAE1.RoboticArmEnv_1Arms_V0(render=True)
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
    if cumReward>-100:
        score+=1
print(score,'/',episodes)

