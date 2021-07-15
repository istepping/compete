import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from BPNet import BPNet
import numpy as np

model = BPNet(4, 10, 5)
model.load_state_dict(torch.load("model.pth"))

X1 = torch.from_numpy(np.loadtxt(r"telangpu.txt", delimiter=" ", dtype=np.float32))
X2 = torch.from_numpy(np.loadtxt(r"baideng.txt", delimiter=" ", dtype=np.float32))

Y1 = model(X1)
Y2 = model(X2)

print(Y1)
print(Y2)

