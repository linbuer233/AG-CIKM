# from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class ConvGRUCell(nn.Module):
    """ConvGRU 实现"""

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 sn_eps=0.0001):
        """
        :param input_channels: 输入的通道数
        :param output_channels: 输出的通道数
        :param kernel_size: 卷积核大小 Default:3
        :param stride: 卷积核步长 Default:1
        :param padding: 周围补 0 Default:0
        :param sn_eps: 谱归一化常值 Default:1e-4
        """
        super().__init__()
        # 输入门
        self.read_gate_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                # stride=stride,
                padding=padding,
            ),
            eps=sn_eps)
        # 更新门
        self.update_gate_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                # stride=stride,
                padding=padding,
            ),
            eps=sn_eps)
        # 输出门
        self.output_gate_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                # stride=stride,
                padding=padding,
            ),
            eps=sn_eps)

    def forward(self, x, prev_state):
        """
        ConvGRU forward
        :param x: 输入的数据 (Tensor) (x_t)
        :param prev_state: 以前的隐藏层 (h_{t-1})
        Return:
        新输出 + 新隐藏层
        ---公式 (https://gitee.com/linziyang233/imgs/raw/master/img/ConvGRU.png)
        read_gate: 输入门 r^t
        update_gate: 更新门 z^t
        h_tilde: 隐藏层预选值 h^~
        h_t: 隐藏层 h^t
        """
        # 沿通道轴连接输入和先前的隐藏层
        # print(x.shape,prev_state.shape)
        xh = torch.cat([x, prev_state], dim=1)  # 数据按列拼接
        # print('xh', xh.shape)
        # GRU 的输入门
        read_gate = torch.sigmoid(self.read_gate_conv(xh))
        # print('read_gate', read_gate.shape)
        # GRU 的更新门
        update_gate = torch.sigmoid(self.update_gate_conv(xh))
        # print('update_gate', update_gate.shape)
        # 隐藏层预选值
        gated_input = torch.cat([x, read_gate * prev_state], dim=1)
        # print('gated_input', gated_input.shape)
        h_tilde = F.relu(self.output_gate_conv(gated_input))

        out = update_gate * prev_state + (1.0 - update_gate) * h_tilde
        # out=update_gate * c + (1.0-update_gate) * prev_state
        new_state = out

        # 输出 新的隐藏层 和 输出
        return out, new_state


class ConvGRU(torch.nn.Module):
    """
    由 ConvGRUCell 组合形成 ConvGRU
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int = 3,
                 sn_eps: float = 0.00001):
        super().__init__()
        self.cell = ConvGRUCell(input_channels, output_channels, kernel_size,
                                sn_eps)

    def forward(self, x, hidden_state=None):
        # if hidden_state==None:
        #     hidden_state=torch.rand(x.shape[1:])
        # for step in range(len(x)):
        #     # 计算 正确的 timestep
        #     output,hidden_state=self.cell(x[step],hidden_state)
        #     outputs.append(output)
        outputs, hidden_state = self.cell(x, hidden_state)
        # 使之以张量的方式输出
        # outputs =torch.stack(outputs,dim=0)
        return outputs, hidden_state


if __name__ == '__main__':
    input_channels = 4
    hidden_channels = 4
    output_channels = 4
    time_step = 1
    batch_size = 1
    output_shape = 21

    model = ConvGRU(input_channels + hidden_channels, output_channels)

    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    lossal = []

    hidden_state = torch.rand(
        (batch_size, hidden_channels, output_shape, output_shape))
    # print(hidden_state)
    # data
    inputx = torch.rand(
        (time_step, batch_size, input_channels, output_shape, output_shape))
    y = torch.rand(
        (time_step, batch_size, input_channels, output_shape, output_shape))
    for _ in range(100):
        optimizer.zero_grad()
        out, hid = model(inputx[0], hidden_state)
        # print(out.shape)
        # print('hid',hid.shape)
        # 误差计算
        loss = loss_function(y[0], out)
        lossal.append(loss.item())
        print('loss', loss.item())
        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_value_(parameters=model.parameters(),
                                        clip_value=1.)
        optimizer.step()  # 权重更新

    plt.plot(range(len(lossal)), lossal)
    plt.show()
