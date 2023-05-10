import sys

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm

sys.path.append("..")
from config import config as cfg


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction, spatial_kernel=7, sn_eps=0.0001):
        """
        :param channel: inputx 的通道数，即 C
        :param reduction:
        :param spatial_kernel:
        """
        super().__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel // reduction, 1, bias=False), eps=sn_eps),
            # inplace=True直接替换，节省内存
            nn.ReLU(),
            # nn.Linear(channel // reduction, channel,bias=False)
            spectral_norm(nn.Conv2d(channel // reduction, channel, 1, bias=False), eps=sn_eps)
        )
        # spatial attention
        self.conv = spectral_norm(nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                            padding=spatial_kernel // 2, bias=False), eps=sn_eps)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputx: Tensor) -> Tensor:
        """
        :param inputx: B | C | H | W
        :return:
        """
        # Channel attention module（CAM）
        max_out = self.mlp(self.max_pool(inputx))
        avg_out = self.mlp(self.avg_pool(inputx))
        channel_out = self.sigmoid(max_out + avg_out).to(cfg().device)
        inputx = channel_out * inputx

        # Spatial attention module（SAM）
        max_out, _ = torch.max(inputx, dim=1, keepdim=True)  # _ 为max_out中的值在dim=1的位置矩阵
        avg_out = torch.mean(inputx, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1))).to(cfg().device)
        inputx = spatial_out * inputx
        return inputx


if __name__ == '__main__':
    x = torch.randn(1, 8, 3, 3)
    net = CBAMLayer(8, 4)
    y = net.forward(x)
    print(y.shape)
