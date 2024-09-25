import torch
# from net.ITA import JNet, TNet, GNet, TBNet
# from net.ITA import JNet, XXNet
from net.ITA import XXNet
# from net.mamba import JNet_mamba

class net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.Jnet = JNet()
        self.XXnet = XXNet()

    def forward(self, data):
        x_j = self.XXnet(data)

        return x_j



