import gym
from stable_baselines3 import PPO

# Load the trained model
model = PPO.load("ppo_mountaincar_initial")

# Create the environment
env = gym.make('MountainCarContinuous-v0')

# If you used any wrappers during training (e.g., a custom reward wrapper),
# make sure to wrap the environment in the same way here.

# Visualize the performance of the trained model
obs = env.reset()
for t in range(1000):  # adjust the range for the number of steps you want
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # print(t, obs, reward, done, info)
    env.render()
    if done:
        print("Finished after {} timesteps".format(t+1))
        obs = env.reset()

# Close the environment
env.close()
