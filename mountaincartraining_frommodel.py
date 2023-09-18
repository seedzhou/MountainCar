import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
         # Define the custom reward based on potential and kinetic energy
        potential_energy = 0 #abs(obs[0] + 0.5)
        kinetic_energy = obs[1]**2*200

        # Give a large bonus when the car is close to the goal
        goal_position = 0.6
        if obs[0] >= goal_position - 0.2:  # adjust the threshold as needed
            potential_energy += obs[0]*500  # adjust the bonus value as needed

        # The reward is the sum of the potential and kinetic energy
        reward = kinetic_energy + potential_energy 

        #print(f"Position: {obs[0]}, Velocity: {obs[1]}, Reward: {reward}")

        return obs, reward, done, info


# Create the environment
env = gym.make('MountainCarContinuous-v0')
env = Monitor(env)
env = CustomRewardWrapper(env)

tensorboard_log_dir = "./logs/"

# Load the trained model
model = PPO.load("ppo_mountaincar_initial")

model.set_env(env)  # Set the environment for the loaded model
model.learn(total_timesteps=50000)  # Adjust the timesteps as needed

# Save the trained agent
model.save("ppo_mountaincar_trained_frommodel")

env.close()
