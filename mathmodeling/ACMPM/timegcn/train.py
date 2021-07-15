import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from timegcn.TimeGCN import TimeGCN
import numpy as np

# 参数设置
learning_rate = 0.1  # 学习率
weight_decay = 0.001  # 学习率衰减
epochs = 2000  # 训练周期

# 生成图结构-> 加载数据-> 训练模型-> 保存模型
# 加载特征数据和生成图结构
# 加载数据
data = np.loadtxt(r"data.txt", delimiter="  ", dtype=np.float32)
X_ = data[:, 0:3]
Y = torch.from_numpy(data[:, 3:])
X = torch.from_numpy(
    np.insert(X_, -1, np.loadtxt(r"telangpu.txt", delimiter="  ", dtype=np.float32), axis=0))  # 预测不同的人使用不同数据
print(Y)
# 生成图结构
N = X.shape[0]  # 图节点数
adj = np.zeros([N, N])  # 初始化邻接矩阵
for i in range(N - 1):
    adj[i][i + 1] = 1
    adj[i + 1][i] = 1
    if i + 2 < N:
        adj[i][i + 2] = 1
        adj[i + 2][i] = 1
# 归一化操作
d = np.sum(adj, axis=0)
d = torch.tensor(np.diag(d), dtype=torch.float)
d = torch.inverse(d)
adj = d.mm(torch.tensor(adj, dtype=torch.float))
# 训练模型
model = TimeGCN(X.shape[1], Y.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
model.train()
loss_value = []
mini_loss = 1000
for i in range(epochs):
    optimizer.zero_grad()
    output = model(X, adj)
    loss = criterion(output[0:Y.shape[0], :], Y)
    loss_value.append(loss)
    loss.backward()
    optimizer.step()
    if loss < mini_loss:
        torch.save(model.state_dict(), "TimeGCNeva.pth")
    mini_loss = loss
print(loss_value[-1])
model.load_state_dict(torch.load("TimeGCNeva.pth"))
pre_y = model(X, adj)
print(pre_y)
print(pre_y[Y.shape[0]:])
Y = Y.numpy()
pre_y = pre_y.detach().numpy()
plt.plot(range(len(Y[:, 1])), Y[:, 1], label="Y1")
plt.plot(range(len(pre_y[:, 1])), pre_y[:, 1], label="predict_Y1")
plt.legend()
plt.show()
# 可视化结果
# plt.plot(range(epochs), loss_value, '-')
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.show()
# plt.plot(range())
