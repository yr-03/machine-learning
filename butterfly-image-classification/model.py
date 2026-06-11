import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1: 224x224x3 -> 112x112x32
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Block 2: 112x112x32 -> 56x56x128
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Block 3: 56x56x128 -> 14x14x256
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1, bias=False) # Capped at 256 to save memory
        self.bn6 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.1) # Dropout2d is better for images than standard Dropout

        # Global Adaptive Pooling ensures it collapses cleanly regardless of final size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Streamlined Classifier Head
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.dropout_fc = nn.Dropout(0.5)
        self.output = nn.Linear(512, 75)

    def forward(self, x):
        # Block 1 (One pool total: 224 -> 112)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        
        # Block 2 (Two pools total: 112 -> 56 -> 28)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout_conv(x)

        # Block 3 (Two pools total: 28 -> 14 -> 7)
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.dropout_conv(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.output(x)
        return x