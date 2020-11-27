import torch.nn as nn
import torch
import torch.nn.functional as F

class VGG(nn.Module):
    
    def __init__(self, num_classes=3):
        super().__init__()
        #Convolutional block 1
        self.conv164 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv264 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        #Convolutional block 2
        self.conv1128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2128 = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=3, stride=1, padding=1)
        #Convolutional block 3
        self.conv1256 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2256 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        #Convolutional block 4
        self.conv1512 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv2512 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        #Fully connected layer
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, t):
        t = t
        #Conv layer 1 + ReLU
        t = self.conv164(t)
        t = F.relu(t)
        t = self.conv264(t)
        t = F.relu(t)
        
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)

        #Conv layer 2 + ReLU
        t = self.conv1128(t)
        t = F.relu(t)
        t = self.conv2128(t)
        t = F.relu(t)

        t = F.max_pool2d(t, kernel_size = 2, stride = 2)

        #Conv layer 3 + ReLU
        t = self.conv1256(t)
        t = F.relu(t)
        t = self.conv2256(t)
        t = F.relu(t)
        t = self.conv2256(t)
        t = F.relu(t)

        t = F.max_pool2d(t, kernel_size = 2, stride = 2)

        #Conv layer 4 + ReLU
        t = self.conv1512(t)
        t = F.relu(t)
        t = self.conv2512(t)
        t = F.relu(t)
        t = self.conv2512(t)
        t = F.relu(t)

        t = F.max_pool2d(t, kernel_size = 2, stride = 2)

        #Conv layer 5 + ReLU
        t = self.conv2512(t)
        t = F.relu(t)
        t = self.conv2512(t)
        t = F.relu(t)
        t = self.conv2512(t)
        t = F.relu(t)

        t = F.max_pool2d(t, kernel_size = 2, stride = 2)

        t = torch.flatten(t, 1)

        #Fully connected layer 1
        t = self.fc1(t)
        t = F.relu(t)
        t = F.dropout(t, p = 0.5)

        #Fully connected layer 2
        t = self.fc2(t)
        t = F.relu(t)
        t = F.dropout(t, p = 0.5)

        #Fully connected layer 3
        t = self.fc3(t)
        t = self.sig(t)

        return t