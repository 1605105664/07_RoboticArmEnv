from stable_baselines3 import *
from gym.envs.registration import register
import RoboticArmEnv_2Robots_Incremental as RAE
import render
import time

#number of arm segments
N_ARMS=2
register(
    id="RoboticArmEnv-v1",
    entry_point=RAE.RoboticArmEnv_V1,
    max_episode_steps=1000,
    kwargs={'num_arms': N_ARMS}
)

# model = A2C.load("a2c")
model = PPO.load("ppo")
# model = DQN.load("dqn")

render.render_init()
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
        render.render(env)
        time.sleep(0.05) #slow down the animation
    print('Reward:', cumReward)