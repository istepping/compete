import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from BPNet import BPNet
import numpy as np

learning_rate = 0.05
weight_decay = 0.0001
epochs = 1000
in_features = 4
hidden_features = 10
out_features = 5
# 加载数据
data = torch.from_numpy(np.loadtxt(r"data.txt", delimiter="    ", dtype=np.float32))
X = data[:, 0:4]
Y = data[:, 4:]
print(X)
print(Y)
# 训练模型
model = BPNet(in_features, hidden_features, out_features)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
model.train()
loss_value = []
mini_loss = 100
for i in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss_value.append(loss)
    loss.backward()
    optimizer.step()
    if loss < mini_loss:
        torch.save(model.state_dict(), "model.pth")
    mini_loss = loss
print(mini_loss)
plt.plot(range(epochs), loss_value, '-')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
