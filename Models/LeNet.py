import torch.nn as nn
import torch
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self, num_classes=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels=16, kernel_size=5, stride=1, padding=0)
        
        self.fc1 = nn.Linear(53 * 53 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        #output layer
        self.fc3 = nn.Linear(84, num_classes)

        self.sig = nn.Sigmoid()

    def forward(self, t):
        t = t

        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = torch.flatten(t, 1)

        t = self.fc1(t)
        t = F.relu(t)
        t = F.dropout(t, p=0.5)

        t = self.fc2(t)
        t = F.relu(t)
        t = F.dropout(t, p=0.5)

        t = self.fc3(t)
        t = self.sig(t)

        return t