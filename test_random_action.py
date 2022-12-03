import time
import glob
import render
import RoboticArmEnv_2Robots_Incremental as RAE


# number of arm segments
N_ARMS = 2
N_ROBOTS = 2

render.render_init()
episodes = 100
env = RAE.RoboticArmEnv_V1(training=False, num_arms=N_ARMS, num_robots=N_ROBOTS)
for episode in range(episodes):
    done = False
    obs = env.reset()
    cumReward = 0
    while not done:
        # Training Model
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        # Random Action
        # random_action = env.action_space.sample()
        # obs, reward, done, info = env.step(random_action)

        # print('obs',obs,'reward', reward)
        cumReward += reward
        render.render(env)
        time.sleep(0.05)  # slow down the animation
    print('Reward:', cumReward)