import torch
import numpy as np

# 从numpy中导入
a = np.array([2, 3.3])
data = torch.from_numpy(a)
# print(data)
b = np.ones([3, 4])
dd = torch.from_numpy(b)
# print(dd)

# 从list中导入
# 大写的Tensor():与FloatTensor类似，接受shape作为参数，小写的接受现有的数据
c = torch.tensor([2., 3.2])
d = torch.FloatTensor([2., 3.2])    # 也可接受现有数据，但是数据必须用一个list来表示。如果接受shape：（2, 3）
e = torch.tensor([[2., 3.2], [1., 22.3]])
# print(c)
# print(d)
# print(e)

# 生成未初始化数据:只是作为一个容器，后面会把数据写进来
# torch.empty() : 给shape
# torch.FloatTensor(d1, d2, d3)
# torch.IntTensor(d1, d2, d3)

f = torch.empty(2, 3)
print(f)
print(torch.Tensor(2, 3))    # 数据大小相差大，记得覆盖否则可能出现torch.not number或torch.infinity
print(torch.IntTensor(2, 3))
print(torch.FloatTensor(2, 3))