import random

import numpy as np
from skimage.metrics import structural_similarity as ssim


class RandomList(object):

    def randomlist(self, n: int) -> list:
        """
        创建不重复元素的随机列表
        """
        randlist = []
        while True:
            temp = np.random.randint(n)
            if temp not in randlist:
                randlist.append(temp)
                if len(randlist) == n:
                    break
        return randlist

    def randomlistnew(self, n: int) -> list:
        """
        randomlist 进化版
        """
        return random.sample(range(n), n)

    def randomlist_B(self, n: int, batch_size: int, step: int) -> list:
        """
        在小于或等于 batch_size 的间隔随机采样，
        例如总样本数为 10,batch_size=2,step=3 该函数实现的效果为：随机排序 [0,3,6].
        """
        return random.sample(range(0, n - batch_size, step),
                             len(range(0, n - batch_size, step)))

    def randomlist_v2(self, n: int, batch_size: int) -> list:
        """
        返回一个含有 n//batch_size 个的子列表的大列表，子列表大小为 batch_size，为 0-n 随机挑取 batch_size 个得到.
        如:\n
        # >>> randomlist_v2(10,2)\n
        # >>> [[5, 2], [1, 3], [4, 0], [7, 6], [8, 9]]
        """
        random_array = random.sample(range(n), n)
        biglist = []
        st = 0
        ed = batch_size
        for _ in range(n // batch_size):
            biglist.append(random_array[st:ed])
            st += batch_size
            ed += batch_size
        return biglist


# 模型评价
class model_eva():
    """
    模型评分函数
    real : 真实的雷达回波 T|H|W
    pre : 不同方法预测的雷达回波 T|H|W
    """

    # def __init__(self,):
    # 均方根误差 MSE，Mean Square Error
    def MSE(self, real, pre):
        return round(((real - pre) ** 2).sum() / real.size, 5)

    # 结构相似性指数 SSIM，Structural Similarity
    def SSIM(self, real, pre):
        T = real.shape[0]
        temp = 0
        for t_i in range(T):
            temp += ssim(real[t_i, :, :], pre[t_i, :, :], win_size=11, gaussian_weights=True)
        return round(temp / T, 5)

    # 雷达回波分级
    def radarsift(self, radar, k):
        # >=k
        radar_geqk = np.where(radar < k, 0, radar)
        radar_geqk = np.where(radar_geqk >= k, 1, radar_geqk)
        # radar_geqk=radar_geqk.sum()
        # < k
        radar_lesk = np.where(radar >= k, np.nan, radar)
        radar_lesk = np.where(radar_lesk < k, 1, radar_lesk)
        # radar_lesk=np.nansum(radar_lesk)
        return radar_geqk, radar_lesk

    def jianyan(self, real, pre, k):
        """
        Hits_k 为预报正确格点数，
        FalseAlarms_k 为误报格点数，
        Misses_k 为漏报格点数，
        """
        # real k
        real_geqk, real_lesk = self.radarsift(real, k)
        # pre k
        pre_geqk, pre_lesk = self.radarsift(pre, k)

        # Hits_k
        Hits_k = ((real_geqk + pre_geqk) == 2).sum()
        # FalseAlarms_k
        FalseAlarms_k = ((real_lesk + pre_geqk) == 2).sum()
        # Misses_k
        Misses_k = ((real_geqk + pre_lesk) == 2).sum()
        return Hits_k, FalseAlarms_k, Misses_k

    # 临界成功指数 CSI，Critical Success Index
    def CSI(self, real, pre, k):
        T = real.shape[0]
        temp = 0
        for t_i in range(T):
            Hits_k, FalseAlarms_k, Misses_k = self.jianyan(real[t_i, :, :], pre[t_i, :, :], k)
            if (Hits_k + FalseAlarms_k + Misses_k) == 0:
                temp += 0
            else:
                temp += Hits_k / (Hits_k + FalseAlarms_k + Misses_k)
        return round(temp / T, 5)

    # 虚警率 FAR，False Alarm Rate
    def FAR(self, real, pre, k):
        T = real.shape[0]
        temp = 0
        for t_i in range(T):
            Hits_k, FalseAlarms_k, Misses_k = self.jianyan(real[t_i, :, :], pre[t_i, :, :], k)
            if (Hits_k + FalseAlarms_k) == 0:
                temp += 0
            else:
                temp += FalseAlarms_k / (Hits_k + FalseAlarms_k)
        return round(temp / T, 5)

    # 命中率 POD，Probability Of Detection
    def POD(self, real, pre, k):
        T = real.shape[0]
        temp = 0
        for t_i in range(T):
            Hits_k, FalseAlarms_k, Misses_k = self.jianyan(real[t_i, :, :], pre[t_i, :, :], k)
            if (Hits_k + FalseAlarms_k) == 0:
                temp += 0
            else:
                temp += Hits_k / (Hits_k + FalseAlarms_k)
        return round(temp / T, 5)
    # 逐时刻
    # 均方根误差 MSE，Mean Square Error
    def MSEzT(self, real, pre):
        T = real.shape[0]
        temp = []
        for t_i in range(T):
            temp.append(round(((real[t_i,:,:] - pre[t_i,:,:]) ** 2).sum() / real[t_i,:,:].size, 5))
        return temp

    # 结构相似性指数 SSIM，Structural Similarity
    def SSIMzT(self, real, pre):
        T = real.shape[0]
        temp = []
        for t_i in range(T):
            temp.append(round(ssim(real[t_i, :, :], pre[t_i, :, :], win_size=11, gaussian_weights=True),5))
        return temp
    # 临界成功指数 CSI，Critical Success Index
    def CSIzT(self, real, pre, k):
        T = real.shape[0]
        temp = []
        for t_i in range(T):
            Hits_k, FalseAlarms_k, Misses_k = self.jianyan(real[t_i, :, :], pre[t_i, :, :], k)
            # print(Hits_k, FalseAlarms_k, Misses_k)
            if (Hits_k + FalseAlarms_k + Misses_k) == 0:
                temp.append(0)
            else:
                temp.append(round(Hits_k / (Hits_k + FalseAlarms_k + Misses_k),5))
        return temp

    # 虚警率 FAR，False Alarm Rate
    def FARzT(self, real, pre, k):
        T = real.shape[0]
        temp =[]
        for t_i in range(T):
            Hits_k, FalseAlarms_k, Misses_k = self.jianyan(real[t_i, :, :], pre[t_i, :, :], k)
            if (Hits_k + FalseAlarms_k) == 0:
                temp.append(0)
            else:
                temp.append(round( FalseAlarms_k / (Hits_k + FalseAlarms_k),5))
        return temp

    # 命中率 POD，Probability Of Detection
    def PODzT(self, real, pre, k):
        T = real.shape[0]
        temp = []
        for t_i in range(T):
            Hits_k, FalseAlarms_k, Misses_k = self.jianyan(real[t_i, :, :], pre[t_i, :, :], k)
            if (Hits_k + FalseAlarms_k) == 0:
                temp.append( 0)
            else:
                temp.append(round( Hits_k / (Hits_k + FalseAlarms_k),5))
        return temp

########## test ##########
if __name__ == '__main__':
    rdlist = RandomList()
    print("randomlist_B:", rdlist.randomlist_B(10, 2, 2))
    print("randomlist_v2", rdlist.randomlist_v2(10, 2))
    a = rdlist.randomlist_v2(10, 2)
    print(a)
    print(a[0])
