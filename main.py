from stable_baselines3.common.env_checker import check_env
import RoboticArmEnv_3Arms as RAE3

env = RAE3.RoboticArmEnv_3Arms_V0()

# It will check your custom environment and output additional warnings if needed
check_env(env)


episodes = 10

for episode in range(episodes):
    done = False
    obs = env.reset()
    reward_epi = 0
    while not done:
        random_action = env.action_space.sample()
        # print("action",random_action)
        obs, reward, done, info = env.step(random_action)
        env.render()
        reward_epi += reward
        # print(info.get("End Effector"))
    print('reward',reward_epi)

