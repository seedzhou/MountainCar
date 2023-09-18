import gym
import pygame
from gym.utils.play import play

class CustomWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomWrapper, self).__init__(env)
        
    def step(self, action):
        if isinstance(action, int):
            action = [action]
        return super(CustomWrapper, self).step(action)

mapping = {(pygame.K_LEFT,): -1, (pygame.K_RIGHT,): 1}
env = CustomWrapper(gym.make("MountainCarContinuous-v0"))
play(env, keys_to_action=mapping)
