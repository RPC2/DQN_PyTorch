import numpy as np
import random


class ReplayMemory:
    def __init__(self, memory_size=100000, action_size=4, cartpole_env=True, per=False):
        if cartpole_env:
            self.states = np.zeros(shape=(memory_size, 4))
            self.next_states = np.zeros(shape=(memory_size, 4))

        self.actions = np.zeros(memory_size)
        self.rewards = np.zeros(memory_size)
        self.terminals = np.zeros(memory_size)

        self.count = 0
        self.current = 0
        self.memory_size = memory_size
        self.per = per  # Use prioritized experience replay

    def add(self, state, reward, action, terminal, next_state):
        self.states[self.current] = state
        self.rewards[self.current] = reward
        self.actions[self.current] = action
        self.terminals[self.current] = terminal
        self.next_states[self.current] = next_state

        self.current = (self.current + 1) % self.memory_size
        self.count += 1

    def sample(self, batch_size):
        state_batch = []
        reward_batch = []
        action_batch = []
        terminal_batch = []
        next_state_batch = []

        if self.per:
            a = 1  # TODO: implement PER
        else:  # randomly select samples from memory
            for i in range(batch_size):
                data_index = random.randint(0, self.current-1 if self.count < self.memory_size else self.memory_size-1)
                state_batch.append(self.states[data_index])
                reward_batch.append(self.rewards[data_index])
                action_batch.append(self.actions[data_index])
                terminal_batch.append(self.terminals[data_index])
                next_state_batch.append(self.next_states[data_index])

            return state_batch, reward_batch, action_batch, terminal_batch, next_state_batch
