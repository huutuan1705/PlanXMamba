import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels=128):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 96, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)
        return torch.cat([b1, b2, b3, b4], dim=1)  # Tổng: 128 + 128 + 128 + 64 = 384