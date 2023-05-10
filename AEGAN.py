import datetime

import h5py
import torch
import torch.nn as nn

from CommonFun import RandomList
from Discriminators import Discriminator
from EncoderTrajGRUAtt import Encoder
from config import config as cfg
from layers.mergeBlock import mergeBlock1 as mergeBlock


# VAE
class AE(nn.Module):

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 output_shape: int,
                 batch_size: int,
                 time_steps: int = 5,
                 forecast_steps: int = 10):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.output_shapes = output_shape
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.forecast_steps = forecast_steps

        self.hid1 = torch.randn(batch_size, 4, output_shape, output_shape)
        self.hid2 = torch.randn(batch_size, 8, output_shape // 2,
                                output_shape // 2)
        self.hid3 = torch.randn(batch_size, 16, output_shape // 4,
                                output_shape // 4)
        self.hid4 = torch.randn(batch_size, 32, output_shape // 8,
                                output_shape // 8)

        # en_de = Encoder(input_channels, output_channels)
        # self.encoder = en_de
        # self.decoder = en_de
        en = Encoder(input_channels, output_channels)
        de = Encoder(input_channels, output_channels)
        self.encoder = en
        self.decoder = de

    def noise_reparameterize(self, mean, std):
        eps = torch.rand(mean.shape)
        z = mean + eps * torch.exp(std)
        return z

    def forward(self, inputx, mBen, mBde):
        #
        x1 = self.encoder(inputx, self.hid1, self.hid2, self.hid3, self.hid4,
                          mBen, self.time_steps, self.forecast_steps)

        # nosie
        # mean = torch.swapaxes(x1, 0, 1).reshape(self.batch_size, -1).to(cfg().device)
        # std = torch.swapaxes(x1, 0, 1).reshape(self.batch_size, -1).to(cfg().device)
        # std = torch.sigmoid(std / 100)
        # z = self.noise_reparameterize(mean, std)
        #
        # z = z.reshape(self.batch_size, self.forecast_steps,
        #               self.input_channels, self.output_shapes,
        #               self.output_shapes)
        # z = torch.swapaxes(z, 0, 1).to(cfg().device)
        z = x1
        # --------------------------DECODER---------------------------- #
        # 反转，相当于用之后的时刻的图像往前推之前的图像
        z = self.decoder(torch.flip(z, (0, )), self.hid1, self.hid2, self.hid3,
                         self.hid4, mBde, self.forecast_steps, self.time_steps)
        # 再反转回来，方便计算误差
        z = torch.flip(z, (0, ))
        return z, x1


if __name__ == '__main__':
    import wandb
    from tqdm import tqdm
    import os
    os.environ["WANDB_API_KEY"] = '3a151c2b55636dcc7dac58caca5d42ec61e0f661'
    os.environ["WANDB_MODE"] = "offline"

    hyperparameter_defaults = dict(
        project_name="RainDL_AEGANfin",
        # 训练参数
        epochs=3,
        bilv=[0.1, 0.3, 0.3, 0.3],
        # 模型参数
        input_channels=1,
        output_channels=1,
        time_steps=5,
        forecast_steps=10,
        batch_size=10,
        output_shape=128,
        Enoptim_type='SGD',
        Disoptim_type='SGD',
        mBoptim_type='Adam',
        # Encoder_lr
        En_lr=0.0001,
        # Discriminator_lr
        Dis_lr=0.0001,
        # mergeBlock_lr
        mB_lr=0.05,
        n=8,
        output_channels_n=2,
    )
    wandb.init(config=hyperparameter_defaults,
               project="RainDL_VAEGANfin",
               name=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               save_code=True)
    wcfg = wandb.config

    AEmodel = AE(wcfg.input_channels, wcfg.output_channels, wcfg.output_shape,
                 wcfg.batch_size, wcfg.time_steps, wcfg.forecast_steps)
    # 加载预训练好的 encoder 模型
    # PATH = './en.pth'
    # AEmodel.encoder.load_state_dict(torch.load(PATH, map_location='cpu'))
    AEmodel.to(cfg().device)

    # 融合模块 (mergeBlock)
    mBen = mergeBlock(wcfg.time_steps, 16)
    # PATH = './mBen.pth'
    # mBen.load_state_dict(torch.load(PATH, map_location='cpu'))
    mBen.to(cfg().device)
    mBde = mergeBlock(wcfg.forecast_steps, 16)
    mBde.to(cfg().device)

    # 鉴别器 discriminator
    Dis = Discriminator(wcfg.input_channels, wcfg.output_channels,
                        wcfg.output_shape, wcfg.output_channels_n,
                        wcfg.time_steps, wcfg.forecast_steps, wcfg.n,
                        wcfg.batch_size)
    Dis.to(cfg().device)

    optimizerVAE = torch.optim.__dict__[wcfg.Enoptim_type](
        params=AEmodel.parameters(), lr=wcfg.En_lr)
    optimizermBen = torch.optim.__dict__[wcfg.mBoptim_type](
        params=mBen.parameters(), lr=wcfg.mB_lr)
    optimizermBde = torch.optim.__dict__[wcfg.mBoptim_type](
        params=mBde.parameters(), lr=wcfg.mB_lr)
    optimizerDis = torch.optim.__dict__[wcfg.Disoptim_type](
        params=Dis.parameters(), lr=wcfg.Dis_lr)

    loss_function = nn.MSELoss()
    lossdis_function = nn.BCELoss()

    rdlist = RandomList()
    randomlist = rdlist.randomlist_B(5000, wcfg.batch_size, wcfg.batch_size)
    count_step = 0
    for epoch_i in tqdm(range(wcfg.epochs)):
        for file_i in [cfg().datatrain1, cfg().datatrain2]:
            # data
            dataset = h5py.File(file_i, 'r')
            dataset = dataset['train']
            for step_i in range(len(randomlist)):

                st = randomlist[step_i]
                ed = randomlist[step_i] + wcfg.batch_size
                # data 预处理
                train_data = torch.zeros(wcfg.batch_size, 15, 4, 128, 128)
                train_data[:, :, :, 13:114,
                           13:114] = torch.Tensor(dataset[st:ed, :, :, :, :])
                train_data = train_data * 95 / 255 - 10
                train_data = torch.swapaxes(train_data, 0, 1)
                train_data = (
                    train_data[:, :, 0, :, :] * wcfg.bilv[0] +
                    train_data[:, :, 1, :, :] * wcfg.bilv[1] +
                    train_data[:, :, 2, :, :] * wcfg.bilv[2] +
                    train_data[:, :, 3, :, :] * wcfg.bilv[3]).reshape(
                        15, wcfg.batch_size, 1, wcfg.output_shape,
                        wcfg.output_shape)

                inputx = train_data[:5, :, :, :, :].to(cfg().device)
                y = train_data[5:, :, :, :, :].to(cfg().device)

                real_label = torch.ones(wcfg.batch_size, 1).to(cfg().device)
                fake_label = torch.zeros(wcfg.batch_size, 1).to(cfg().device)
                out, pre = AEmodel(inputx, mBen, mBde)
                # Training Discriminator
                if (epoch_i >= 1):
                    for _ in range(6):
                        optimizerDis.zero_grad()
                        # real
                        out_real = Dis(inputx, y)
                        realloss = lossdis_function(out_real, real_label)
                        realloss.backward(retain_graph=True)
                        # fake
                        out_fake = Dis(inputx, pre)
                        fakeloss = lossdis_function(out_fake, fake_label)
                        fakeloss.backward(retain_graph=True)
                        optimizerDis.step()

                # Training AE
                optimizermBen.zero_grad()
                optimizermBde.zero_grad()
                optimizerVAE.zero_grad()

                loss = loss_function(inputx, out)
                loss1 = loss_function(y, pre)
                wandb.log(
                    {
                        "loss": loss.item(),
                        "preloss": loss1.item(),
                        "epoch": epoch_i
                    },
                    step=count_step)
                loss.backward(retain_graph=True)  # 反向传播
                torch.nn.utils.clip_grad_value_(
                    parameters=AEmodel.parameters(), clip_value=1.)

                loss = loss_function(y, pre)
                wandb.log({"prelossen": loss.item()}, step=count_step)
                loss.backward()  # 反向传播
                torch.nn.utils.clip_grad_value_(
                    parameters=AEmodel.parameters(), clip_value=1.)

                optimizermBen.step()  # 权重更新
                optimizermBde.step()  # 权重更新
                optimizerVAE.step()  # 权重更新

                # Train generater of GAN
                if (epoch_i >= 1):
                    optimizermBen.zero_grad()
                    optimizerVAE.zero_grad()

                    _, outf = AEmodel(inputx, mBen, mBde)
                    outf_label = Dis(inputx, outf)
                    outfloss = -torch.mean(outf_label)
                    loss2 = loss_function(y, outf)
                    wandb.log(
                        {
                            "Dis": outf_label.detach().numpy().sum() /
                            wcfg.batch_size,
                            "Disloss": outfloss.item(),
                            "preloss2": loss2.item()
                        },
                        step=count_step)
                    outfloss.backward()

                    optimizermBen.step()  # 权重更新
                    optimizerVAE.step()  # 权重更新

                count_step += 1
    PATH = './en.pth'
    torch.save(AEmodel.encoder.state_dict(), PATH)
    # PATH = './de.pth'
    # torch.save(AEmodel.decoder.state_dict(), PATH)
    PATH = './mBen.pth'
    torch.save(mBen.state_dict(), PATH)
    '''
    训练模型时的测试部分，方便查看模型的训练效果
    # ----------------------TEST----------------------#
    train_data = torch.zeros(wcfg.batch_size, 15, 4, 128, 128)
    train_data[:, :, :, 13:114, 13:114] = torch.Tensor(dataset[:wcfg.batch_size, :, :, :, :])
    train_data = train_data * 95 / 255 - 10
    train_data = torch.swapaxes(train_data, 0, 1)
    train_data = (train_data[:, :, 0, :, :] * wcfg.bilv[0] +
                  train_data[:, :, 1, :, :] * wcfg.bilv[1] +
                  train_data[:, :, 2, :, :] * wcfg.bilv[2] +
                  train_data[:, :, 3, :, :] * wcfg.bilv[3]).reshape(
        15, wcfg.batch_size, 1, wcfg.output_shape, wcfg.output_shape)

    inputx = train_data[:5, :, :, :, :].to(cfg().device)
    y = train_data[5:, :, :, :, :].to(cfg().device)
    with torch.no_grad():
        out = AEmodel.encoder(inputx, AEmodel.hid1, AEmodel.hid2, AEmodel.hid3, AEmodel.hid4, mBen,
                               AEmodel.time_steps,
                               AEmodel.forecast_steps)
        for i in range(0, 10, 2):
            wandb.log({"preimg": [wandb.Image(out[i, 0, 0, :, :], caption="preimg")]}, step=count_step)
            wandb.log({"realimg": [wandb.Image(y[i, 0, 0, :, :], caption="realimg")]}, step=count_step)
            count_step += 1
    # ----------------------TEST----------------------#
    '''
    wandb.finish()
