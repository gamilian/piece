import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = F.relu(x)
        return x

class RoIAlign(nn.Module):
    def __init__(self, output_size):
        super(RoIAlign, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        # Placeholder for the RoIAlign operation, as it is not natively supported by PyTorch
        # A real implementation should be used in practice, this is just a placeholder
        return F.adaptive_avg_pool2d(x, self.output_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(8, 8),
            ResidualBlock(8, 8),
            ResidualBlock(8, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(16, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128),
            RoIAlign(output_size=(4, 4))
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.residual_blocks(x)
        x = self.fc_layers(x)
        return x

# Instantiate the network
net = NeuralNetwork()
print(net)
