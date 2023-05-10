import torch


class config():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = 'D:\\Project\\DL_leida_rain\\data\\CIKM'
        self.datatest = self.path+'\\datatest.h5'
        self.datatrain1 = self.path+'\\dataset1.h5'
        self.datatrain2 = self.path+'\\dataset2.h5'
        self.datayanzeng = self.path+'\\dataseyan.h5'
