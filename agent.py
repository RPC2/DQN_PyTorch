import torch
import numpy as np
import gym
import os
import random
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from config import AgentConfig, EnvConfig
from memory import ReplayMemory
from network import MlpPolicy
from ops import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(AgentConfig, EnvConfig):
    def __init__(self):
        self.env = gym.make(self.env_name)
        self.action_size = self.env.action_space.n  # 2 for cartpole
        self.memory = ReplayMemory(action_size=self.action_size, per=self.per)
        if self.train_cartpole:
            self.policy_network = MlpPolicy(action_size=self.action_size).to(device)
            # self.target_network = MlpPolicy(action_size=self.action_size).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.005)
        self.loss = 0
        self.criterion = nn.MSELoss()

    def new_random_game(self):
        self.env.reset()
        action = self.env.action_space.sample()
        screen, reward, terminal, info = self.env.step(action)
        return screen, reward, action, terminal

    def train(self):
        episode = 0
        step = 0
        last_episode_reward = 0
        last_episode_length = 0
        y = []

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
            current_state, reward, action, terminal = self.new_random_game()
            # print("current_state: " + str(current_state)) # [-0.00239923  0.23727428 -0.01354857 -0.29871889]
            # current_state = set_init_state(screen)

            # A step in an episode
            while episode_length < self.max_episode_length:
                step += 1
                episode_length += 1

                # Choose action
                if random.uniform(0, 1) < self.epsilon:
                    action = random.randrange(self.action_size)
                else:
                    q_values = self.policy_network(current_state.to(device))
                    action = np.argmax(q_values)

                # Act
                new_state, reward, terminal, info = self.env.step(action)

                if self.gif:
                    frames_for_gif.append(new_state)

                self.memory.add(current_state, reward, action, terminal, new_state)

                current_state = new_state
                total_episode_reward += reward

                if step > self.start_learning and step % self.train_freq == 0:
                    if self.per:
                        state_batch, reward_batch, action_batch, terminal_batch, next_state_batch, leaf_index_batch, \
                            is_weights = self.memory.sample(self.batch_size)
                    else:
                        state_batch, reward_batch, action_batch, terminal_batch, \
                            next_state_batch = self.memory.sample(self.batch_size)
                        self.minibatch_learning(state_batch, reward_batch, action_batch, terminal_batch,
                                                next_state_batch)

                if step % 100 == 0:
                    print(
                        'episode: %.2f, total step: %.2f, last_episode length: %.2f, last_episode_reward: %.2f, loss: %.4f'
                        % (episode, step, last_episode_length, last_episode_reward, self.loss))

                # if step % self.reset_step == 0:
                #     self.target_network.load_state_dict(self.policy_network.state_dict())

                # if step % self.plot_every == 0:
                #     y.append(last_episode_reward)
                #     x = range(len(y))
                #     plt.plot(x, y)
                #     plt.ylabel('reward')
                #     plt.show()

                if terminal:
                    last_episode_reward = total_episode_reward
                    last_episode_length = step - start_step

                    self.env.reset()

                    if self.gif:
                        generate_gif(last_episode_length, frames_for_gif, total_episode_reward, "./GIF/", episode)

                    break

            self.env.render()

        self.env.close()

    def minibatch_learning(self, state_batch, reward_batch, action_batch, terminal_batch, next_state_batch,
                           leaf_index_batch=None, is_weights=None):

        y_batch = torch.FloatTensor()
        for i in range(self.batch_size):
            if terminal_batch[i]:
                y_batch = torch.cat((y_batch, torch.FloatTensor([reward_batch[i]]).to(device)), 0)
            else:
                next_state_q = torch.max(self.policy_network(torch.FloatTensor(next_state_batch[i]).to(device)))
                y = torch.FloatTensor([reward_batch[i] + self.gamma * next_state_q])
                y_batch = torch.cat((y_batch, y), 0)

        policy_q = torch.max(self.policy_network(torch.FloatTensor(state_batch)), dim=1)[0]

        self.loss = self.criterion(policy_q, y_batch)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

