import torch
import torch.nn as nn


class MlpPolicy(nn.Module):
    def __init__(self, action_size=4, input_size=4):
        super(MlpPolicy, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, self.action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.softmax(self.fc1(x))
        x = self.softmax(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

