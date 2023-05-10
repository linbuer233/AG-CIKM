import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pixelshuffle import PixelUnshuffle
from torch.nn.utils.parametrizations import spectral_norm
from CommonFun import RandomList


# 鉴别器 discriminator
class Discriminator(nn.Module):
    """
    分为 时间鉴别器 TemporaDiscriminator 和 空间鉴别器 SpatialDiscriminator
    输入的 Tensor 的维度 step|batch|channel|height|width
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 output_shapes: int,
                 output_channels_n: int,
                 time_steps: int,
                 forecast_steps: int,
                 n: int,
                 batch_size: int,
                 T_kernel_size: int = 2,
                 Spa_kernel_size: int = 2):
        super().__init__()
        # self.tempora_discriminator = TemporaDiscriminator(
        #     input_channels, output_channels, time_steps + forecast_steps,
        #     output_shapes, T_kernel_size)
        self.spatial_discriminator = SpatialDiscriminator(
            input_channels, output_channels_n, forecast_steps, output_shapes,
            batch_size, n, Spa_kernel_size)

    def forward(self, inputx, outputx):
        # print(inputx.shape, outputx.shape)
        # tempora_x = self.tempora_discriminator(inputx, outputx)
        # print(tempora_x.shape)
        spatial_x = self.spatial_discriminator(outputx)
        # print(spatial_x.shape)
        return torch.sigmoid(spatial_x)  # torch.sigmoid(tempora_x + spatial_x)


class TemporaDiscriminator(nn.Module):
    """
    时间鉴别器 利用三维卷积神经网络提取二维场时序上的信息，旨在区分到观测到的和生成的雷达序列
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 time_channels: int,
                 output_shapes: int,
                 kernel_size: int = 2):
        """
        :param input_channels: 输入的通道数 (不是时间的通道数)
        :param output_channels: 输出的通道数
        :param time_channels: 输入的时间通道数 等同于 time_steps + forecast_steps | Ct_in
        :param output_shapes: 输入图片的宽和高
        :param kernel_size: 时间维卷积核大小 (Ct) 输出的时间通道数 Ct_out = (Ct_in+2*padding-kernel_size[0])/stride+1
        """
        super().__init__()
        # space_to_depth 部分
        self.S2D = PixelUnshuffle(2)
        # 3D 卷积部分 + 下采样部分
        # time_channels -> time_channels/2 | W & H -> W/2 & H/2
        self.Conv3d1 = spectral_norm(
            nn.Conv3d(4, 8, (kernel_size, 2, 2), stride=(2, 2, 2), padding=0))
        # time_channels/2 -> time_channels/4 | W/2 & H/2 -> W/4 & H/4
        self.Conv3d2 = spectral_norm(
            nn.Conv3d(8, 16, (kernel_size, 2, 2), stride=(2, 2, 2), padding=0))
        # time_channels/4 -> time_channels/8 | W/4 & H/4 -> W/8 & H/8
        self.Conv3d3 = spectral_norm(
            nn.Conv3d(16, 32, (kernel_size, 2, 2), stride=(2, 2, 2),
                      padding=0))
        self.Convout = spectral_norm(nn.Conv2d(32, 1, kernel_size=3,
                                               padding=1))

        self.BN3d1 = nn.BatchNorm3d(4)
        self.BN3d2 = nn.BatchNorm3d(8)
        self.BN3d3 = nn.BatchNorm3d(16)
        self.BN3d4 = nn.BatchNorm3d(32)

    def forward(self, inputx, outputx):
        """time_steps 序列的图片和 forecast_steps 序列的图片和为一个序列
        # time_steps+forecast_steps | batch_size | channel | height | width"""
        all_time = torch.cat([inputx, outputx], dim=0)
        all_time = self.S2D(
            all_time
        )  # T | B | input_channels | H | W -> # T | B | input_channels*4 | H/2 | W/2
        all_time = all_time.permute(1, 2, 0, 3, 4)
        all_time = self.BN3d1(all_time)
        # batch_size | channels | T | height | width
        all_time = self.Conv3d1(
            all_time
        )  # B | input_channels*4 | T | H/2 | W/2 -> # B | input_channels*8 | T//2 | H/4 | W/4
        all_time = self.BN3d2(all_time)
        all_time = F.relu(all_time)

        all_time = self.Conv3d2(
            all_time
        )  # B | input_channels*8 | T//2 | H/4 | W/4 -> # B | input_channels*16 | T//4 | H/8 | W/8
        all_time = self.BN3d3(all_time)
        all_time = F.relu(all_time)
        # batch_size | channel | T | height | width
        all_time = self.Conv3d3(
            all_time
        )  # B | input_channels*16 | T//4 | H/8 | W/8 -> # B | input_channels*32 | T//8 | H/16 | W/16
        all_time = self.BN3d4(all_time)
        all_time = F.relu(all_time)

        all_time = all_time.permute(
            2, 0, 1, 3, 4)  # 1 | B | input_channels*32 | H/16 | W/16

        all_time = self.Convout(
            all_time[0, :, :, :, :]
        )  # B | input_channels*32 | H/16 | W/16 -> # B | 1 | H/16 | W/16

        return all_time


class SpatialDiscriminator(nn.Module):
    """
    空间鉴别器分辨率减半通道数加倍，旨在区分单个观测到的雷达场和生成的场，确保空间的一致性，避免模糊的预测
    """

    def __init__(self,
                 input_channels: int,
                 output_channels_n: int,
                 forecast_steps: int,
                 output_shape: int,
                 batch_size: int,
                 n: int,
                 kernel_size: int = 2):
        super().__init__()
        self.output_shape = output_shape
        self.batch_size = batch_size
        # self.suilist = [0, 2, 4, 6, 8]
        self.suilist = RandomList().randomlist(n)
        # space_to_depth 部分
        self.S2D = PixelUnshuffle(2)

        # 卷积 + 下采样部分
        # channels -> channels*output_channels_n | W & H -> W/2 & H/2
        self.Conv2d1 = spectral_norm(
            nn.Conv2d(input_channels * 4,
                      input_channels * 4 * output_channels_n,
                      kernel_size=kernel_size,
                      stride=2,
                      padding=0))
        # channels*output_channels_n -> channels*output_channels_n**2 | W/2 & H/2 -> W/4 & H/4
        self.Conv2d2 = spectral_norm(
            nn.Conv2d(input_channels * 4 * output_channels_n,
                      input_channels * 4 * output_channels_n**2,
                      kernel_size=kernel_size,
                      stride=2,
                      padding=0))
        # channels*output_channels_n**2 -> channels*output_channels_n**3 | W/4 & H/4 -> W/8 & H/8
        self.Conv2d3 = spectral_norm(
            nn.Conv2d(input_channels * 4 * output_channels_n**2,
                      input_channels * 4 * output_channels_n**3,
                      kernel_size=kernel_size,
                      stride=2,
                      padding=0))
        self.Convout = spectral_norm(
            nn.Conv2d(input_channels * 4 * output_channels_n**3,
                      1,
                      kernel_size=3,
                      padding=1))
        self.ConvTmerge = spectral_norm(
            nn.Conv2d(n, 1, kernel_size=3, padding=1))

        self.BN1 = nn.BatchNorm2d(input_channels * 4 * output_channels_n)
        self.BN2 = nn.BatchNorm2d(input_channels * 4 * output_channels_n**2)
        self.BN3 = nn.BatchNorm2d(input_channels * 4 * output_channels_n**3)
        self.BNout = nn.BatchNorm2d(1)

        self.fc = nn.Sequential(
            nn.Linear(output_shape // 16 * output_shape // 16, 1024),
            nn.LeakyReLU(0.2, True), nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, outputx):
        """随机挑选输出的图片中的几张
        # n in forecast_steps | batch_size | channel | height | width"""
        xall = []
        for t_i in self.suilist:
            x = outputx[t_i]  # B | C | H | W
            x = self.S2D(x)  # (H/W -> H/2 | W/2) C -> 4

            x = self.Conv2d1(x)  # B | C | H | W -> B | 4*2(8) | H/4 | W/4
            x = self.BN1(x)
            x = F.relu(x)

            x = self.Conv2d2(x)  # B | C | H | W -> B | 4*2*2(16) | H/8 | W/8
            x = self.BN2(x)
            x = F.relu(x)

            x = self.Conv2d3(
                x)  # B | C | H | W -> B | 4*2*2*2(32) | H/16 | W/16
            x = self.BN3(x)
            x = F.relu(x)

            x = self.Convout(x)
            x = self.BNout(x)
            x = F.relu(x)
            xall.append(x)
        xall = torch.stack(xall,
                           dim=0)  # n in forecast_steps | B | 1 | H/16 | W/16
        # xall = xall.mean(axis=0)
        # 之前是 n in forecast_steps | B | 1 | H/16 | W/16 转变为 B | 1 | H/16 | W/16
        # 下两行把  n in forecast_steps | B | 1 | H/16 | W/16 转变为 B | 1
        xall = self.ConvTmerge(xall[:, :, 0, :, :].permute(1, 0, 2, 3)).reshape(self.batch_size, -1)
        xall = self.fc(xall)
        # xall = xall.mean(axis=[2, 3])
        return xall


class Discriminator1(nn.Module):

    def __init__(self):
        super().__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            # nn.Sigmoid()
        )

    def forward(self, input):
        xall = []
        for x_i in range(0, 10, 2):
            x = self.dis(input[x_i])
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            xall.append(x)
        # print(xall)
        xall = torch.stack(xall,
                           dim=0)  # n in forecast_steps | B | 1 | H/16 | W/16
        # print(xall)
        xall = xall.mean(axis=0)
        return xall


########## test ##########
if __name__ == '__main__':
    # 模型参数
    input_channels: int = 1
    output_channels: int = 1
    output_shapes: int = 128
    output_channels_n: int = 2
    time_steps: int = 5
    forecast_steps: int = 10
    n: int = 8
    batch_size: int = 2

    model = Discriminator1()

    x = torch.randn(10, 2, 1, 128, 128)
    out = model(x)
    print(out.shape)
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     # 模型参数
#     input_channels: int = 1
#     output_channels: int = 1
#     output_shapes: int = 128
#     output_channels_n: int = 2
#     time_steps: int = 5
#     forecast_steps: int = 10
#     n: int = 8
#     batch_size: int = 2
#
#     model = Discriminator(
#         input_channels,
#         output_channels,
#         output_shapes,
#         output_channels_n,
#         time_steps,
#         forecast_steps,
#         n,
#         batch_size,
#     )
#     device = torch.device("cpu")
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     loss_function = nn.MSELoss()
#     lossal = []
#
#     # data T | B | C | H | W
#     mnist = np.load('../../data/mnist_test_seq.npy')  # (20,10000,64,64)
#     # print(mnist.shape)
#     trainData = mnist[:15, 0:2, :, :]
#     inputxReal = torch.zeros(5, 2, 1,output_shapes,output_shapes)
#     inputxReal[:, :, 0, 64:, 64:] = torch.Tensor(trainData[:5, :, :, :])
#
#     outputxReal = torch.zeros(10, 2, 1,output_shapes,output_shapes)
#     outputxReal[:, :, 0, 64:, 64:] = torch.Tensor(trainData[:10, :, :, :])
#
#     outputxFake = torch.rand(10, 2, 1,output_shapes,output_shapes) * 100
#
#     # 定义真实的图片为 1，假的图片为 0 batch_size | 1
#     real_label = torch.ones(2, 1, output_shapes//16, output_shapes//16)
#     fake_label = torch.zeros(2, 1, output_shapes//16, output_shapes//16)
#     reallossall = []
#     fakelossall = []
#     lossall = []
#     for _ in range(100):
#         optimizer.zero_grad()
#         outReal = model(inputxReal, outputxReal)
#         # print('outreal', outReal)
#         realloss = loss_function(outReal, real_label)
#         # print("realloss_score", realloss.mean().item())
#         reallossall.append(realloss.mean().item())
#         realloss.backward()
#
#         outFake = model(inputxReal, outputxFake)
#         # print('outfake', outFake)
#         fakeloss = loss_function(outFake, fake_label)
#         # print('fakeloss_score', fakeloss.mean().item())
#         fakelossall.append(fakeloss.mean().item())
#         fakeloss.backward()
#         # a=input()
#         # 总的误差
#         lossall.append(fakeloss.mean().item() + realloss.mean().item())
#         optimizer.step()
#     print(lossall)
#     plt.scatter(range(len(lossall)),
#                 lossall,
#                 color='k',
#                 label='lossal',
#                 alpha=0.5)
#     plt.plot(range(len(fakelossall)),
#              fakelossall,
#              color='r',
#              label='fakeloss',
#              alpha=0.5)
#     plt.plot(range(len(reallossall)),
#              reallossall,
#              color='b',
#              label='realloss',
#              alpha=0.5)
#     plt.legend()
#     plt.show()
