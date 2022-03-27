import torch
import numpy as np
# Dim=0,用于loss
a = torch.tensor(2.2)
print(a.shape)    # torch.Size([])
print(len(a.shape))    # 0
print(a.size())    # torch.Size([])

# Dim=1,用于Bias/Linear input
b = torch.tensor([2])    # 直接这样写，里面的数据类型跟着里面数据变化
print(b)
print(b.type())
c = torch.tensor([1.1, 2.2])
print(c)
print(c.type())
d = torch.FloatTensor(2)
print(d)
e = torch.IntTensor([2.2])
print(e)

data = np.ones(3)
print(data)
f = torch.from_numpy(data)    # 将numpy转换成tensor
print(f)

# Dim=2,Linear input/batch
g = torch.randn(2, 3)    # 随机正太分布
print(g)
print(g.shape)
print(g.size())
print(g.size(0))
print(g.size(1))
print(g.shape[1])


# Dim=3 RNN input/Batch
h = torch.rand(3, 2, 3)    # 随机均匀分布
print(h)
print(h.shape)
print(h[0])
print(h[1])
print(list(h.shape))


# Dim=4 CNN:[b,c,h,w]
# 下面解释为2张照片，每张照片通道数为3，长宽为28×28
i = torch.rand(2, 3, 28, 28)    # 照片数 通道数(彩色图片为3) 图片长 图片宽
print(i)