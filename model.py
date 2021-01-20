import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


class CNNNet(nn.Module):
    def __init__(self, input_dim=0, output_dim=0, hidden_size=0, lr=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.lr = lr
        self.bn = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 6, 5)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.fc1 = nn.Linear(16 * 33 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # x=self.bn(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.bn2(x)
        x = x.reshape(-1, 16 * 33 * 61)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x









