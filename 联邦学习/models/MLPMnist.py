import torch.nn as nn


class MLPMnist(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim_in, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, dim_out),
                                    nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = x.view(-1, x.nelement()//x.shape[0])
        x = self.layers(x)
        return x

