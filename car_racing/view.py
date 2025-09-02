import gymnasium as gym

env = gym.make("CarRacing-v3", render_mode="human", continuous=False)

obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    if done:
        break

env.close()
