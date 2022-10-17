from stable_baselines3.common.env_checker import check_env
import RoboticArmEnv as RAE

env = RAE.RoboticArmEnvV0()

# It will check your custom environment and output additional warnings if needed
check_env(env)


episodes = 10

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        random_action = env.action_space.sample()
        # print("action",random_action)
        obs, reward, done, info = env.step(random_action)
        env.render()
        # print(info.get("End Effector"))
    print('reward',reward)

