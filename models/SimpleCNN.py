import torch
from torch import nn
import torch.nn.functional as F

"""
input: 1 x 28 x 28
conv1: 6 x 28 x 28 -> relu
pool:  6 x 14 x 14 -> relu
conv2: 16x 10 x 10
pool:  16x 5  x 5
fc1:   16x 5  x 5 -> 120 -> relu
fc2:   120 -> 84 -> relu
fc3:   84  -> 10
"""
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    net = SimpleCNN()
    print(net)