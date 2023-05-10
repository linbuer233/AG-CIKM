import torch
import torch.nn as nn


class mergeBlock(nn.Module):
    def __init__(self, input_channels, reduction):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction)
        self.fc2 = nn.Linear(input_channels // reduction, input_channels)
        self.convout = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.size()
        x1 = self.avg_pool(x)
        x1 = x1.reshape(B, C)
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.sigmoid(x1).reshape(B, C, 1, 1)
        x1 = x * x1
        out = self.convout(x1)
        return out


class mergeBlock1(nn.Module):
    def __init__(self, input_channels, reduction):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction)
        self.fc2 = nn.Linear(input_channels // reduction, input_channels)
        self.convout = nn.Conv3d(input_channels, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C1, C2, _, _ = x.size()
        x1 = self.avg_pool(x)
        x1 = x1.reshape(B, C1, C2)
        x1 = x1.permute(0, 2, 1)
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = x1.permute(0, 2, 1)
        x1 = self.sigmoid(x1).reshape(B, C1, C2, 1, 1)
        x1 = x * x1
        out = self.convout(x1)
        return out


if __name__ == "__main__":
    model = mergeBlock1(4, 1)
    # mergeBlock1
    x = torch.rand(2, 4, 32, 16, 16)
    # y = torch.rand(2, 1, 32, 16, 16)
    # mergeBlock
    # x = torch.rand(2, 4, 16, 16)
    # y = torch.rand(2, 32, 16, 16)
    out = model(x)
    print(out.shape)
