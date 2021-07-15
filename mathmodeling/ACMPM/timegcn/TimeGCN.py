import torch
import torch.nn as nn
import torch.nn.functional as f
import math
from torchvision import transforms
from torch.nn.parameter import Parameter
import numpy as np


class TimeGCN(nn.Module):
    def __init__(self, in_features=3, out_features=4):
        print("GCN-init")
        # in_features=256, hid_features, out_features
        super(TimeGCN, self).__init__()
        self.gcn1 = GraphConvolution(in_features, 10)
        self.gcn2 = GraphConvolution(10, 30)
        self.gcn3 = GraphConvolution(30, 60)
        self.fc = nn.Linear(in_features=60, out_features=out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj):
        # print("GCN-forward")
        x = self.normalizer(x)
        x = self.sigmoid(self.gcn1(x, adj))
        x = self.sigmoid(self.gcn2(x, adj))
        x = self.sigmoid(self.gcn3(x, adj))
        output = self.fc(x)
        return output

    @staticmethod
    def normalizer(x):
        x = x.numpy()
        min_values = x.min(0)
        max_values = x.max(0)
        x = (x - np.tile(min_values, (x.shape[0], 1))) / np.tile(max_values - min_values, (x.shape[0], 1))
        return torch.from_numpy(x)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.zeros(in_features, out_features), requires_grad=True)
        self.bias = Parameter(torch.zeros(out_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # print("GraphConvolution-forward")
        # 传播规则:AHW+B
        output = torch.mm(adj, x)
        output = torch.mm(output, self.weight)
        output = output + self.bias
        return output
