import numpy as np
from BPNet import BPNet
import torch
import matplotlib.pyplot as plt

data = np.loadtxt(r"data.txt", delimiter="    ", dtype=np.float32)
X = torch.from_numpy(data[:, 0:4])
Y = data[:, 4:]
model = BPNet(4, 10, 5)
model.load_state_dict(torch.load("model.pth"))
predict_Y = model(X)
predict_Y=predict_Y.detach().numpy()
plt.plot(range(len(Y[:, 0])), Y[:, 0], label="Y0")
plt.plot(range(len(predict_Y[:, 0])), predict_Y[:, 0], label="predict_Y0")
plt.plot(range(len(Y[:, 1])), Y[:, 1], label="Y1")
plt.plot(range(len(predict_Y[:, 1])), predict_Y[:, 1], label="predict_Y1")
plt.plot(range(len(Y[:, 2])), Y[:, 2], label="Y2")
plt.plot(range(len(predict_Y[:, 2])), predict_Y[:, 2], label="predict_Y2")
plt.plot(range(len(Y[:, 3])), Y[:, 3], label="Y3")
plt.plot(range(len(predict_Y[:, 3])), predict_Y[:, 3], label="predict_Y3")
plt.legend()
plt.show()

# 加载不同数据

# 同一个图上可视化结果
