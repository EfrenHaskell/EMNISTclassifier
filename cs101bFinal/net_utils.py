import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.connected = nn.Linear(in_features=(32 * 7 * 7), out_features=47)

    def forward(self, data):
        data = self.pool(self.conv_layer1(data))
        data = self.pool(F.relu(self.conv_layer2(data)))
        data = torch.flatten(data, 1)
        data = F.relu(self.connected(data))
