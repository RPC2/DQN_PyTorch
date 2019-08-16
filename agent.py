import torch
import numpy as np
import gym
import os
import random

from config import AgentConfig, EnvConfig
from network import MlpPolicy


class Agent(AgentConfig, EnvConfig):
    def __init__(self):
        self.env = gym.make(self.env_name)
        self.action_size = self.env.action_space.n
        if torch.cuda.is_available():
            if self.train_cartpole:
                self.policy_network = MlpPolicy(action_size=self.action_size)
                self.target_network = MlpPolicy(action_size=self.action_size)

    def new_random_game(self):
        self.env.reset()
        action = self.env.action_space.sample()
        screen, reward, terminal, info = self.env.step(action)
        return screen, reward, action, terminal

    def train(self):
        screen, reward, action, terminal = self.new_random_game()
        print(screen)
        print(reward)
        print(action)
        print(terminal)
        self.env.render()
