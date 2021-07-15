from torch import nn


class BPNet(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim):
        super(BPNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden,bias=True),
            nn.BatchNorm1d(n_hidden), nn.Sigmoid())
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden,bias=True),
            nn.BatchNorm1d(n_hidden), nn.Sigmoid())
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden,bias=True),
            nn.BatchNorm1d(n_hidden), nn.Sigmoid())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden, out_dim,bias=True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
