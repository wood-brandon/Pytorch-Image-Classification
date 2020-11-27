import torch.nn as nn
import torch
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, t):
        # hidden conv layer 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=2)

        # hidden conv layer 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=2)

        # hidden conv layer 3
        t = self.conv3(t)
        t = F.relu(t)
        t = self.conv4(t)
        t = F.relu(t)
        t = self.conv5(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=2)

        t = torch.flatten(t, 1)

        # hidden fully connected layer 1
        t = self.fc1(t)
        t = F.relu(t)
        t = F.dropout(t, p=0.5)

        # hidden fully connected layer 2
        t = self.fc2(t)
        t = F.relu(t)
        t = F.dropout(t, p=0.5)

        # hidden fully connected layer 3
        t = self.fc3(t)
        t = self.sig(t)
        return t
