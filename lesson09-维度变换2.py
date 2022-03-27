import torch

# expand/repeat
# expand: broadcasting  改变理解方式
# repeat: memory copied  实实在在的增加数据
a = torch.rand(4, 32, 14, 14)
b = torch.rand(1, 32, 1, 1)
print(b)
print(b.expand(4, 32, 14, 14))  # torch.Size([4, 32, 14, 14])

print(b.expand(-1, 32, -1, -1).shape)  # -1表示该维度不变 torch.Size([1, 32, 1, 1])
print(b.expand(-1, 32, -1, -4).shape)  # 写-4变-4    RuntimeError: invalid shape dimension -128 torch.Size([1, 32, 1,
# -4])

# repeat:不建议使用
print(b.repeat(4, 32, 1, 1).shape)  # 第二个拷贝32次 torch.Size([4, 1024, 1, 1])
print(b.repeat(4, 1, 1, 1).shape)  # torch.Size([4, 32, 1, 1])
print(b.repeat(4, 1, 32, 32).shape)  # torch.Size([4, 32, 32, 32])

# t():转置 只适合2D tensor
x = torch.randn(3, 4)
print(x)
print(x.t().shape)

# Transpose: 维度变换 (1,1,2) 到（1，2，1）
a = torch.rand(4, 3, 32, 32)
print(a.shape)  # torch.Size([4, 3, 32, 32])
"""
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
"""
a1 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 3, 32, 32)  # 要加contigous
a2 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 3, 32, 32).transpose(1, 3)
print(a1.shape)  # torch.Size([4, 3, 32, 32]) 错的
print(a2.shape)  # torch.Size([4, 32, 32, 3])

# permute:可以直接排位置，可以使用任意多次的transpose来达到他的目的
a = torch.rand(4, 3, 28, 28)
print(a.transpose(1, 3).shape)    # torch.Size([4, 28, 28, 3])
b = torch.rand(4, 3, 28, 32)
print(b.transpose(1, 3).shape)    # torch.Size([4, 32, 28, 3])
print(b.transpose(1, 3).transpose(1, 3).shape)    # torch.Size([4, 3, 28, 32])
print(b.permute(0, 2, 3, 1).shape)    # torch.Size([4, 28, 32, 3])
