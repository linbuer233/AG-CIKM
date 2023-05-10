import h5py
import torch
import torch.nn as nn
from torch.autograd import Variable

from CommonFun import RandomList
from config import config as cfg
from layers.Attention import CBAMLayer
from layers.TrajGRU import TrajGRU
from layers.mergeBlock import mergeBlock1 as mergeBlock


# 编码器 encoder
class Encoder(nn.Module):

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int = 3):
        super().__init__()
        # self.time_steps = time_steps
        # self.forecast_steps = forecast_steps
        self.output_channels = output_channels

        # 处理输入的模块
        self.convin = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels,
                      out_channels=4,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=1), nn.ReLU())
        # 第一层 TrajGRU 后的下采样模块
        self.downsample1 = nn.Sequential(
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, kernel_size=2, stride=2, padding=0),  # H/W||64->32
            nn.ReLU())
        # 第二层 TrajGRU 后的下采样模块
        self.downsample2 = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=2, stride=2,
                      padding=0),  # H/W||32->16
            nn.ReLU())
        # 第三层 TrajGRU 后的下采样模块
        self.downsample3 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=2, stride=2,
                      padding=0),  # H/W||16->8
            nn.ReLU())
        # 第三层 TrajGRU 后的上采样模块
        self.upsample3 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')  # H/W||8 ->16
        )
        # 第二层 TrajGRU 后的上采样模块
        self.upsample2 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')  # H/W||16->32
        )
        # 第一层 TrajGRU 后的上采样模块
        self.upsample1 = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')  # H/W||32->64
        )
        # 输出处理模块
        self.convout = nn.Sequential(
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4,
                      out_channels=output_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=1), nn.ReLU())

        # self.BN4 = nn.BatchNorm2d(32)

        # TrajGRU 模块
        self.TrajGRUd1 = TrajGRU(4, 4)
        self.TrajGRUd2 = TrajGRU(8, 8)
        self.TrajGRUd3 = TrajGRU(16, 16)
        self.TrajGRUd4 = TrajGRU(32, 32)

        self.TrajGRUu1 = TrajGRU(4, 4)
        self.TrajGRUu2 = TrajGRU(8, 8)
        self.TrajGRUu3 = TrajGRU(16, 16)
        self.TrajGRUu4 = TrajGRU(32, 32)

        # Attention 层 加在 TrajGRU 层之后
        self.Attd1 = CBAMLayer(4, 4)
        self.Attd2 = CBAMLayer(8, 8)
        self.Attd3 = CBAMLayer(16, 16)
        self.Attd4 = CBAMLayer(32, 32)

        self.Attu1 = CBAMLayer(4, 4)
        self.Attu2 = CBAMLayer(8, 8)
        self.Attu3 = CBAMLayer(16, 16)
        self.Attu4 = CBAMLayer(32, 32)

        # self.mergeBlock=mergeBlock()
        # 参数初始化
        # he initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def forward(self,
                inputx,
                hid1,
                hid2,
                hid3,
                hid4,
                mB,
                time_steps: int = 5,
                forecast_steps: int = 10
                ):  # x -> time_step|batch_size|channel|height|width
        outcode = Variable(
            torch.zeros(forecast_steps, inputx.shape[1], self.output_channels,
                        inputx.shape[3], inputx.shape[4]))
        dex = Variable(
            torch.zeros(inputx.shape[1], time_steps, 32, inputx.shape[3] // 8,
                        inputx.shape[4] // 8))
        # tezhenghid1=torch.zeros(15,10,4,128,128)
        # tezhenghid2=torch.zeros(15,10,8,64,64)
        # tezhenghid3=torch.zeros(15,10,16,32,32)
        # tezhenghid4=torch.zeros(15,10,32,16,16)
        # tezheng_i=0
        for step in range(time_steps):
            x = inputx[step]
            x = self.convin(x)  # C | 1 -> 4
            # -----------------------------#
            x, hid1 = self.TrajGRUd1(x, hid1)  # C|H|W|| 4*64*64 -> 4*64*64
            # tezhenghid1[tezheng_i,:,:,:,:]=hid1
            # -----------------------------#
            x = self.Attd1(x)
            # 第一层下采样
            x = self.downsample1(x)  # C|H|W|| 4*64*64 -> 8*32*32
            # -----------------------------#
            x, hid2 = self.TrajGRUd2(x, hid2)  # C|H|W|| 8*32*32 -> 8*32*32
            # tezhenghid2[tezheng_i,:,:,:,:]=hid2
            # -----------------------------#
            x = self.Attd2(x)
            # 第二层下采样
            x = self.downsample2(x)  # C|H|W|| 8*32*32 -> 16*16*16
            # -----------------------------#
            x, hid3 = self.TrajGRUd3(x, hid3)  # C|H|W|| 16*16*16 -> 16*16*16
            # tezhenghid3[tezheng_i,:,:,:,:]=hid3
            # -----------------------------#
            x = self.Attd3(x)
            # 第三层下采样
            x = self.downsample3(x)  # C|H|W|| 16*16*16 ->32*8*8
            # -----------------------------#
            x, hid4 = self.TrajGRUd4(x, hid4)  # C|H|W|| 32*8*8 -> 32*8*8
            # tezhenghid4[tezheng_i,:,:,:,:]=hid4
            # -----------------------------#
            x = self.Attd4(x)

            # dex[:, 0 + 32 * step:32 + step * 32, :, :] = x
            dex[:, step, :, :, :] = x
            # tezheng_i+=1
            # dex[:, :, :, :] += x
        # 融合模块
        # B|T|C|H|W -> B|1|C|H|W
        # print(dex.shape)
        dex = mB(dex)
        dex = dex.reshape(inputx.shape[1], 32, inputx.shape[3] // 8,
                          inputx.shape[4] // 8)
        # print(dex.shape)
        # dex = dex / time_steps  # 另外的做法是创建一个融合器
        for step in range(forecast_steps):
            # -----------------------------#
            x, hid4 = self.TrajGRUu4(dex, hid4)  # C|H|W|| 32*8*8
            # tezhenghid4[tezheng_i,:,:,:,:]=hid4
            # -----------------------------#
            x = self.Attu4(x)
            # 第一层上采样
            x = self.upsample3(x)  # C|H|W|| 32*8*8 -> 16*16*16
            # -----------------------------#
            x, hid3 = self.TrajGRUu3(x, hid3)  # C|H|W|| 16*16*16
            # tezhenghid3[tezheng_i,:,:,:,:]=hid3
            # -----------------------------#
            x = self.Attu3(x)
            # 第二层上采样
            x = self.upsample2(x)  # C|H|W|| 16*16*16 -> 8*32*32
            # -----------------------------#
            x, hid2 = self.TrajGRUu2(x, hid2)  # C|H|W|| 8*32*32
            # tezhenghid2[tezheng_i,:,:,:,:]=hid2
            # -----------------------------#
            x = self.Attu2(x)
            # 第三层上采样
            x = self.upsample1(x)  # C|H|W|| 8*32*32 -> 4*64*64
            # -----------------------------#
            x, hid1 = self.TrajGRUu1(x, hid1)  # C|H|W|| 4*64*64
            # tezhenghid1[tezheng_i,:,:,:,:]=hid1
            # -----------------------------#
            x = self.Attu1(x)

            x = self.convout(x)  # C|H|W|| 4*64*64 -> output_channels*64*64

            outcode[step, :, :, :, :] = x
            # tezheng_i+=1
        return outcode #,tezhenghid1,tezhenghid2,tezhenghid3,tezhenghid4 # forecast_steps|batch|channel|height|width


################# test #################
if __name__ == '__main__':
    from tqdm import tqdm
    import wandb

    train_dict = dict(
        # 训练参数
        epochs=20,
        bilv=[0.1, 0.3, 0.3, 0.3],
        # 模型参数
        input_channels=1,
        output_channels=1,
        time_steps=5,
        forecast_steps=10,
        batch_size=10,
        output_shape=128,
        # Encoder_lr
        En_lr=0.0005,
        # mergeBlock_lr
        mB_lr=0.05,
    )

    wandb.init(config=train_dict, project="RainDL_Encoder")
    wcfg = wandb.config

    dataset = h5py.File(cfg().datatest, 'r')
    dataset = dataset['train']

    model = Encoder(wcfg.input_channels, wcfg.output_channels)
    model.to(cfg().device)
    optimizer = torch.optim.Adam(model.parameters(), lr=wcfg.En_lr)

    # global mB
    mB = mergeBlock(wcfg.time_steps, 1)
    mB.to(cfg().device)
    optimizermB = torch.optim.Adam(mB.parameters(), lr=wcfg.mB_lr)

    loss_function = nn.MSELoss()
    rdlist = RandomList()
    randomlist = rdlist.randomlist_B(2000, wcfg.batch_size, 10)
    for epoch_i in tqdm(range(wcfg.epochs)):
        for step_i in range(len(randomlist)):
            st = randomlist[step_i]
            ed = randomlist[step_i] + wcfg.batch_size
            hid1 = torch.randn(wcfg.batch_size, 4, wcfg.output_shape,
                               wcfg.output_shape)
            hid2 = torch.randn(wcfg.batch_size, 8, wcfg.output_shape // 2,
                               wcfg.output_shape // 2)
            hid3 = torch.randn(wcfg.batch_size, 16, wcfg.output_shape // 4,
                               wcfg.output_shape // 4)
            hid4 = torch.randn(wcfg.batch_size, 32, wcfg.output_shape // 8,
                               wcfg.output_shape // 8)

            train_data = torch.zeros(wcfg.batch_size, 15, 4, 128, 128)
            train_data[:, :, :, 13:114,
                       13:114] = torch.Tensor(dataset[st:ed, :, :, :, :])
            train_data = train_data * 95 / 255 - 10
            train_data = torch.swapaxes(train_data, 0, 1)
            train_data = (train_data[:, :, 0, :, :] * wcfg.bilv[0] +
                          train_data[:, :, 1, :, :] * wcfg.bilv[1] +
                          train_data[:, :, 2, :, :] * wcfg.bilv[2] +
                          train_data[:, :, 3, :, :] * wcfg.bilv[3]).reshape(
                              15, wcfg.batch_size, 1, wcfg.output_shape,
                              wcfg.output_shape)
            # print(train_data.shape)
            inputx = train_data[:5, :, :, :, :]
            y = train_data[5:, :, :, :, :]
            out = model(inputx, hid1, hid2, hid3, hid4, mB, wcfg.time_steps,
                        wcfg.forecast_steps)
            # print(out.shape)
            # 误差计算
            loss = loss_function(y, out)
            wandb.log({"loss": loss.item(), "epoch": epoch_i})
            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_value_(parameters=model.parameters(),
                                            clip_value=1.)
            optimizermB.step()
            optimizermB.zero_grad()
            optimizer.step()  # 权重更新
            optimizer.zero_grad()
            #break
        #break
    PATH = './en.pth'
    torch.save(model.state_dict(), PATH)
    wandb.finish()