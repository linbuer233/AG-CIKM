import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm

sys.path.append("..")
from config import config as cfg


class TrajGRUCell(nn.Module):
    def __init__(self, in_channels: int, output_channels: int, kernel_size: int = 5, L: int = 5, sn_eps=0.0001):
        """
        :param in_channels:
        :param kernel_size:
        :param L:
        公式：https://gitee.com/linziyang233/imgs/raw/master/img/TrajGRU.webp
        """
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = in_channels
        self.kernel_size = kernel_size
        self.L = L
        if kernel_size == 3:
            Wx_padding = 1
        else:
            Wx_padding = 2

        # x -> x_flow
        self.gamma_x = spectral_norm(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=5, padding=2, stride=1),
            eps=sn_eps
        )
        # hid -> hid_flow
        self.gamma_h = spectral_norm(
            nn.Conv2d(self.hid_channels, out_channels=32, kernel_size=5, padding=2, stride=1),
            eps=sn_eps
        )
        # 生成flow
        self.generate_flow = spectral_norm(
            nn.Conv2d(in_channels=32, out_channels=2 * self.L, kernel_size=5, padding=2, stride=1),
            eps=sn_eps
        )

        self.W_xz = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=self.hid_channels,
                                            kernel_size=kernel_size, padding=Wx_padding, stride=1),
                                  eps=sn_eps)

        self.W_xr = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=self.hid_channels,
                                            kernel_size=kernel_size, padding=Wx_padding, stride=1),
                                  eps=sn_eps)

        self.W_xh = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=self.hid_channels,
                                            kernel_size=kernel_size, padding=Wx_padding, stride=1),
                                  eps=sn_eps)

        self.W_hz = nn.ModuleList([])
        self.W_hr = nn.ModuleList([])
        self.W_hh = nn.ModuleList([])
        for i in range(self.L):
            self.W_hz.append(spectral_norm(nn.Conv2d(in_channels=self.hid_channels, out_channels=self.hid_channels,
                                                     kernel_size=1, padding=0, stride=1),
                                           eps=sn_eps))

            self.W_hr.append(spectral_norm(nn.Conv2d(in_channels=self.hid_channels, out_channels=self.hid_channels,
                                                     kernel_size=1, padding=0, stride=1),
                                           eps=sn_eps))

            self.W_hh.append(spectral_norm(nn.Conv2d(in_channels=self.hid_channels, out_channels=self.hid_channels,
                                                     kernel_size=1, padding=0, stride=1),
                                           eps=sn_eps))

    def warp(self, inputx: Tensor, flow: Tensor) -> Tensor:
        """
        :param inputx:
        :param flow:
        :return:
        """
        B, C, H, W = inputx.size()

        # 创建坐标格点矩阵
        # H | W
        x_grid = torch.arange(0, W).reshape(1, W).repeat(H, 1)
        y_grid = torch.arange(0, H).reshape(H, 1).repeat(1, W)

        # B | 1 | H | W
        x_grid = x_grid.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
        y_grid = y_grid.reshape(1, 1, H, W).repeat(B, 1, 1, 1)

        # B | 2 | H | W
        xy_grid = torch.cat([x_grid, y_grid], dim=1).to(cfg().device)
        # UV 格点坐标
        uvgrid = xy_grid + flow
        uvgrid[:, 0, :, :] = 2.0 * uvgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        uvgrid[:, 1, :, :] = 2.0 * uvgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        # B | 2 | H | W -> B | H | W | 2
        uvgrid = uvgrid.permute(0, 2, 3, 1)

        output = F.grid_sample(inputx, uvgrid, align_corners=True)

        return output

    def flow_generator(self, inputx: Tensor, hid_state: Tensor) -> Tensor:
        """
        :param input:
        :param hid_state:
        :return:
        """
        # inputx to flow
        x2f = self.gamma_x(inputx)
        # hid to flow
        h2f = self.gamma_h(hid_state)
        flow = x2f + h2f
        flow = F.leaky_relu(flow, negative_slope=0.1)

        flows = self.generate_flow(flow)
        flows = torch.split(flows, 2, dim=1)

        return flows

    def forward(self, inputx: Tensor = None, hid_state: Tensor = None) -> (Tensor, Tensor):
        """
        :param inputx:
        :param hid_state:
        :return:
        公式：https://gitee.com/linziyang233/imgs/raw/master/img/TrajGRU.webp
        read_gate: 输入门 rt
        update_gate: 更新门 zt
        h_tilde: 隐藏层预选值 hid_temp
        h_t: 隐藏层 h^t
        """
        if inputx is None and hid_state is None:
            raise ValueError(f"输入 {inputx} 和隐藏状态 {hid_state} 不能同时为空")

        if inputx is None:
            inputx = torch.zeros(hid_state.size(0), self.in_channels, hid_state.size(2), hid_state.size(3)).to(
                cfg().device)

        if hid_state is None:
            hid_state = torch.zeros(inputx.size(0), self.hidden_channels, inputx.size(2), inputx.size(3)).to(
                inputx.device)

        flows = self.flow_generator(inputx, hid_state)
        warped_data = []
        for flow in flows:
            warped_data.append(self.warp(hid_state, flow))
        temp_zt = torch.tensor(0, device=inputx.device)
        temp_rt = torch.tensor(0, device=inputx.device)
        temp_ht = torch.tensor(0, device=inputx.device)
        for i in range(self.L):
            temp_zt = temp_zt + self.W_hz[i](warped_data[i])
            temp_rt = temp_rt + self.W_hr[i](warped_data[i])
            temp_ht = temp_ht + self.W_hh[i](warped_data[i])
        # print(self.W_xz(inputx).shape,temp_rt.shape)
        zt = torch.sigmoid(self.W_xz(inputx) + temp_zt)
        rt = torch.sigmoid(self.W_xr(inputx) + temp_rt)

        temp = rt * temp_ht
        hid_temp = F.leaky_relu(self.W_xh(inputx) + temp, negative_slope=0.1)
        newhid_state = zt * hid_temp + (1 - zt) * hid_temp

        return newhid_state, newhid_state


class TrajGRU(nn.Module):
    """
    由 TrajGRUCell 组合形成 TrajGRU
    """

    def __init__(self, in_channels: int, output_channels: int, kernel_size: int = 5):
        super().__init__()
        self.cell = TrajGRUCell(in_channels, output_channels)

    def forward(self, inputx: Tensor, hid_state: Tensor) -> (Tensor, Tensor):
        output, hid_state = self.cell(inputx, hid_state)
        return output, hid_state


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = TrajGRU(1, 1)
    device = torch.device("cpu")
    model.to(device)
    inputx = torch.rand(4, 1, 32, 32).to(device)
    y = torch.rand(4, 1, 32, 32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    lossal = []
    for _ in range(100):
        optimizer.zero_grad()
        hid = torch.randn(4, 1, 32, 32).to(device)
        out, hid = model(inputx, hid)
        # print(out.shape)
        loss = loss_function(y, out)
        lossal.append(loss.item())
        loss.backward()

        torch.nn.utils.clip_grad_value_(parameters=model.parameters(),
                                        clip_value=1.)

        optimizer.step()
    plt.plot(range(len(lossal)), lossal)
    plt.show()
