from stable_baselines3 import *
import RoboticArmEnv_2Robots_Incremental as RAE
import render
import time
import glob

# number of arm segments
N_ARMS = 2
N_ROBOTS = 2

model = PPO.load(max(glob.glob('output/ppo/*.zip')))

render.render_init()
episodes = 100
env = RAE.RoboticArmEnv_V1(training=False, num_arms=N_ARMS, num_robots=N_ROBOTS)
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
        cumReward += reward
        render.render(env)
        time.sleep(0.05)  # slow down the animation
    print('Reward:', cumReward)
