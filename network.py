import torch
import torch.nn as nn


class MlpPolicy(nn.Module):
    def __init__(self, action_size, input_size=4):
        super(MlpPolicy, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, self.action_size)
        # torch.nn.init.kaiming_uniform(self.fc1.weight)
        # torch.nn.init.kaiming_uniform(self.fc2.weight)
        # torch.nn.init.kaiming_uniform(self.fc3.weight)
        # torch.nn.init.kaiming_uniform(self.fc4.weight)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

