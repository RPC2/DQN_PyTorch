import torch
import numpy as np
import gym
import os
import random

from config import AgentConfig, EnvConfig
from memory import ReplayMemory
from network import MlpPolicy
from ops import *


class Agent(AgentConfig, EnvConfig):
    def __init__(self):
        self.env = gym.make(self.env_name)
        self.action_size = self.env.action_space.n  # 2 for cartpole
        self.memory = ReplayMemory(action_size=self.action_size, per=self.per)
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
        episode = 0
        step = 0
        last_episode_reward = 0

        if not os.path.exists("./GIF/"):
            os.makedirs("./GIF/")

        # A new episode
        while step < self.max_step:
            start_step = step
            episode += 1
            episode_length = 0
            total_episode_reward = 0
            frames_for_gif = []

            if episode % self.gif_every == 0:
                self.gif = True
            else:
                self.gif = False

            # Get initial state
            screen, reward, action, terminal = self.new_random_game()
            current_state = set_init_state(screen)

            # A step in an episode
            while episode_length < self.max_episode_length:
                step += 1
                episode_length += 1

                # Choose action
                if random.uniform(0, 1) < self.epsilon:
                    action = random.randrange(self.action_size)
                else:
                    q_values = self.policy_network(current_state)
                    action = np.argmax(q_values)

                # Act
                new_state, reward, terminal, info = self.env.step(action)

                if self.gif:
                    frames_for_gif.append(new_state)

                if self.train_cartpole:
                    next_state = np.append(current_state[:3], [new_state], axis=0)

                action_one_hot = [1 if i == action else 0 for i in range(self.action_size)]

                self.memory.add(current_state, reward, action_one_hot, terminal, next_state)

                current_state = next_state
                total_episode_reward += reward

                # train
                if step > self.start_learning and step % self.train_freq == 0:
                    if self.per:
                        state_batch, reward_batch, action_batch, terminal_batch, next_state_batch, leaf_index_batch, \
                            is_weights = self.memory.sample(self.batch_size)
                    else:
                        state_batch, reward_batch, action_batch, terminal_batch, \
                            next_state_batch = self.memory.sample(self.batch_size)


                if terminal:
                    last_episode_reward = total_episode_reward
                    last_episode_length = step - start_step

                    self.env.reset()

                    if self.gif:
                        generate_gif(last_episode_length, frames_for_gif, total_episode_reward, "./GIF/", episode)

            self.env.render()

    def minibatch_learning(self, state_batch, reward_batch, action_batch, terminal_batch, next_state_batch,
                           leaf_index_batch=None, is_weights=None):
        # Get target network Q value
        target_q_values = self.target_network(state_batch)

