import numpy as np
import pandas as pd
import h5py
import torch

from CommonFun import model_eva
# mymodel
from config import config as cfg
from layers.mergeBlock import mergeBlock1 as mergeBlock
from AEGAN import AE

# pysteps
from pysteps import nowcasts
from pysteps.motion.lucaskanade import dense_lucaskanade




"""
██████╗ ██╗     ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗         
██╔══██╗██║     ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║         
██║  ██║██║     ██╔████╔██║██║   ██║██║  ██║█████╗  ██║         
██║  ██║██║     ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║         
██████╔╝███████╗██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗    
╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝   
"""
bilv = [0.1, 0.3, 0.3, 0.3]
# 模型参数
input_channels = 1
output_channels = 1
time_steps = 5
forecast_steps = 10
batch_size = 10
output_shape = 128

model = AE(input_channels, output_channels, output_shape, batch_size,
           time_steps, forecast_steps)
# 加载预训练好的 encoder 模型
# PATH='./en.pth'
PATH = 'D:\\Project\\DL_leida_rain\\pth\\313\\en.pth'
model.encoder.load_state_dict(torch.load(PATH, map_location='cpu'))
model.to(cfg().device)

# 融合模块 (mergeBlock)
mBen = mergeBlock(time_steps, 16)
# PATH = './mBen.pth'
PATH = 'D:\\Project\\DL_leida_rain\\pth\\313\\mBen.pth'
mBen.load_state_dict(torch.load(PATH, map_location='cpu'))
mBen.to(cfg().device)


"""
██████╗  █████╗ ████████╗ █████╗ 
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
██║  ██║███████║   ██║   ███████║
██║  ██║██╔══██║   ██║   ██╔══██║
██████╔╝██║  ██║   ██║   ██║  ██║
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
"""
dataset = h5py.File(cfg().datatest, 'r')
dataset = dataset['train']


# 数据读取和预处理
def datapre(st, ed, dataset=dataset):
    train_data = torch.zeros(batch_size, 15, 4, 128, 128)
    train_data[:, :, :, 13:114, 13:114] = torch.Tensor(dataset[st:ed, :, :, :, :])
    # train_data[:, :, :, 13:114,13:114] = torch.Tensor(dataset[10:batch_size+10, :, :, :, :])
    train_data = train_data * 95 / 255 - 10
    train_data = torch.swapaxes(train_data, 0, 1)
    train_data = (train_data[:, :, 0, :, :] * bilv[0] +
                  train_data[:, :, 1, :, :] * bilv[1] +
                  train_data[:, :, 2, :, :] * bilv[2] +
                  train_data[:, :, 3, :, :] * bilv[3]).reshape(
        15, batch_size, 1, output_shape, output_shape)

    inputx = train_data[:5, :, :, :, :].to(cfg().device)
    y = train_data[5:, :, :, :, :].to(cfg().device)
    return inputx, y  # T | B | C | H | W

"""
██████╗ ██╗   ██╗███████╗████████╗███████╗██████╗ ███████╗
██╔══██╗╚██╗ ██╔╝██╔════╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
██████╔╝ ╚████╔╝ ███████╗   ██║   █████╗  ██████╔╝███████╗
██╔═══╝   ╚██╔╝  ╚════██║   ██║   ██╔══╝  ██╔═══╝ ╚════██║
██║        ██║   ███████║   ██║   ███████╗██║     ███████║
╚═╝        ╚═╝   ╚══════╝   ╚═╝   ╚══════╝╚═╝     ╚══════╝
"""

nowcast_method = nowcasts.get_method("steps")


def steppre(inputx):
    """
    inputx: T | B | C | H | W 
    """
    _, B, C, _, _ = inputx.size()
    # set nowcast parameters
    n_leadtimes = 10
    seed = 10
    R_fout = np.zeros((n_leadtimes, B, C, 101, 101))
    for i in range(B):
        V = dense_lucaskanade(inputx[:, i, 0, 13:114, 13:114].numpy())  # T | H | W
        R_f = nowcast_method(
            inputx[:, i, 0, 13:114, 13:114].numpy(),  # T | H | W 
            V,
            timesteps=n_leadtimes,
            n_ens_members=1,
            # n_cascade_levels=4,
            precip_thr=-10.0,
            kmperpixel=1.0,
            timestep=6,
            # decomp_method="fft",
            # bandpass_filter_method="gaussian",
            # noise_method="nonparametric",
            # vel_pert_method="bps",
            mask_method="incremental",
            seed=seed,
        )  # 1 | forecast_step | H | W
        R_fout[:, i, 0, :, :] = R_f[0, :, :, :]
    return R_fout  # forecast_step | B | C | H | W


"""
███╗   ███╗ ██████╗ ██████╗ ███████╗██╗         ███████╗██╗   ██╗ █████╗ 
████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║         ██╔════╝██║   ██║██╔══██╗
██╔████╔██║██║   ██║██║  ██║█████╗  ██║         █████╗  ██║   ██║███████║
██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║         ██╔══╝  ╚██╗ ██╔╝██╔══██║
██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗    ███████╗ ╚████╔╝ ██║  ██║
"""
modeleva = model_eva()
MSElist = []
SSIMlist = []
CSIlist = []
PODlist = []
FARlist = []
for st in range(0, len(dataset), 10):
    inputx, y = datapre(st, st + 10)
    inputxstep = torch.where(inputx == -10, 0, inputx)
    # pysteps
    outstep = steppre(inputxstep)
    # model
    with torch.no_grad():
        out = model.encoder(inputx, model.hid1, model.hid2, model.hid3, model.hid4, mBen, model.time_steps,
                            model.forecast_steps)  # forecast_step | B | C | H | W
    realradar = y[:, :, :, 13:114, 13:114].numpy()
    realradar = np.where(realradar<=0,0,realradar)
    outradar = out[:, :, :, 13:114, 13:114].numpy()
    outstep = np.where(np.isnan(outstep), 0, outstep)
    for B_i in range(10):
        # MSE
        MSElist.append((modeleva.MSEzT(realradar[:, B_i, 0, :, :], outradar[:, B_i, 0, :, :]),
                        modeleva.MSEzT(realradar[:, B_i, 0, :, :], outstep[:, B_i, 0, :, :])))
        # # SSIM
        SSIMlist.append((modeleva.SSIMzT(realradar[:, B_i, 0, :, :], outradar[:, B_i, 0, :, :]),
                         modeleva.SSIMzT(realradar[:, B_i, 0, :, :], outstep[:, B_i, 0, :, :])))
        # CSI
        CSIlist.append((modeleva.CSIzT(realradar[:, B_i, 0, :, :], outradar[:, B_i, 0, :, :], 5),
                        modeleva.CSIzT(realradar[:, B_i, 0, :, :], outradar[:, B_i, 0, :, :], 20),
                        modeleva.CSIzT(realradar[:, B_i, 0, :, :], outradar[:, B_i, 0, :, :], 30),
                        modeleva.CSIzT(realradar[:, B_i, 0, :, :], outradar[:, B_i, 0, :, :], 40),
                        #
                        modeleva.CSIzT(realradar[:, B_i, 0, :, :], outstep[:, B_i, 0, :, :], 5),
                        modeleva.CSIzT(realradar[:, B_i, 0, :, :], outstep[:, B_i, 0, :, :], 20),
                        modeleva.CSIzT(realradar[:, B_i, 0, :, :], outstep[:, B_i, 0, :, :], 30),
                        modeleva.CSIzT(realradar[:, B_i, 0, :, :], outstep[:, B_i, 0, :, :], 40),
                        ))
        # POD
        PODlist.append((modeleva.PODzT(realradar[:, B_i, 0, :, :], outradar[:, B_i, 0, :, :], 5),
                        modeleva.PODzT(realradar[:, B_i, 0, :, :], outradar[:, B_i, 0, :, :], 20),
                        modeleva.PODzT(realradar[:, B_i, 0, :, :], outradar[:, B_i, 0, :, :], 30),
                        modeleva.PODzT(realradar[:, B_i, 0, :, :], outradar[:, B_i, 0, :, :], 40),
                        #
                        modeleva.PODzT(realradar[:, B_i, 0, :, :], outstep[:, B_i, 0, :, :], 5),
                        modeleva.PODzT(realradar[:, B_i, 0, :, :], outstep[:, B_i, 0, :, :], 20),
                        modeleva.PODzT(realradar[:, B_i, 0, :, :], outstep[:, B_i, 0, :, :], 30),
                        modeleva.PODzT(realradar[:, B_i, 0, :, :], outstep[:, B_i, 0, :, :], 40),
                        ))
        # FAR
        FARlist.append((modeleva.FARzT(realradar[:, B_i, 0, :, :], outradar[:, B_i, 0, :, :], 5),
                        modeleva.FARzT(realradar[:, B_i, 0, :, :], outradar[:, B_i, 0, :, :], 20),
                        modeleva.FARzT(realradar[:, B_i, 0, :, :], outradar[:, B_i, 0, :, :], 30),
                        modeleva.FARzT(realradar[:, B_i, 0, :, :], outradar[:, B_i, 0, :, :], 40),
                        #
                        modeleva.FARzT(realradar[:, B_i, 0, :, :], outstep[:, B_i, 0, :, :], 5),
                        modeleva.FARzT(realradar[:, B_i, 0, :, :], outstep[:, B_i, 0, :, :], 20),
                        modeleva.FARzT(realradar[:, B_i, 0, :, :], outstep[:, B_i, 0, :, :], 30),
                        modeleva.FARzT(realradar[:, B_i, 0, :, :], outstep[:, B_i, 0, :, :], 40),
                        ))

"""
████████╗ ██████╗     ██████╗  █████╗ ████████╗ █████╗ ███████╗██████╗  █████╗ ███╗   ███╗███████╗
╚══██╔══╝██╔═══██╗    ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝
   ██║   ██║   ██║    ██║  ██║███████║   ██║   ███████║█████╗  ██████╔╝███████║██╔████╔██║█████╗  
   ██║   ██║   ██║    ██║  ██║██╔══██║   ██║   ██╔══██║██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝  
   ██║   ╚██████╔╝    ██████╔╝██║  ██║   ██║   ██║  ██║██║     ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗
   ╚═╝    ╚═════╝     ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝
"""
MSEdf = pd.DataFrame(MSElist, columns=['model', 'step'])
MSEdf.to_csv('D:\\Project\\DL_leida_rain\\evacsv\\zTMSE.csv')

SSIMdf = pd.DataFrame(SSIMlist, columns=['model', 'step'])
SSIMdf.to_csv('D:\\Project\\DL_leida_rain\\evacsv\\zTSSIM.csv')

CSIdf = pd.DataFrame(CSIlist, columns=['5dBzPOD','20dBzPOD', '30dBzPOD', '40dBzPOD', '5dBzPODstep','20dBzPODstep', '30dBzPODstep',
                                       '40dBzPODstep'])
CSIdf.to_csv('D:\\Project\\DL_leida_rain\\evacsv\\zTCSI.csv')

PODdf = pd.DataFrame(PODlist, columns=['5dBzPOD','20dBzPOD', '30dBzPOD', '40dBzPOD', '5dBzPODstep', '20dBzPODstep', '30dBzPODstep',
                                       '40dBzPODstep'])
PODdf.to_csv('D:\\Project\\DL_leida_rain\\evacsv\\zTPOD.csv')

FARdf = pd.DataFrame(FARlist, columns=['5dBzPOD','20dBzPOD', '30dBzPOD', '40dBzPOD', '5dBzPODstep','20dBzPODstep', '30dBzPODstep',
                                       '40dBzPODstep', ])
FARdf.to_csv('D:\\Project\\DL_leida_rain\\evacsv\\zTFAR.csv')