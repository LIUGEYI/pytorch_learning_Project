#
#   view/reshape # 将一个shape转换成例一个shape
#   squeeze(减少维度)/unsqueeze(增加维度)
#   transpose(单维变换)/t(转置)/permute(多维变换)
#   expand(改变理解方式)/repeat(实实在在增加数据 memory copied)
import torch

# view: lost dim information
a = torch.rand(4, 1, 28, 28)
print(a)
print(a.shape)  # torch.Size([4, 1, 28, 28])
print(a.view(4, 28 * 28).shape)  # torch.Size([4, 784])
print(a.view(4 * 28, 28).shape)  # torch.Size([112, 28])
print(a.view(4 * 1, 28, 28).shape)  # torch.Size([4, 28, 28])
b = a.view(4, 784)
b.view(4, 28, 28, 1)  # logic bug

# flexible but prone to corrupt, 维度不匹配
# print(a.view(4, 783))    # RuntimeError: shape '[4, 783]' is invalid for input of size 3136

"""
范围:
    [-a.dim()-1, a.dim()+1]
    [-5, 5)
"""
a = torch.rand(4, 1, 28, 28)
print(a.shape)  # torch.Size([4, 1, 28, 28])
print(a.unsqueeze(0).shape)  # torch.Size([1, 4, 1, 28, 28])
print(a.unsqueeze(-1).shape)  # torch.Size([4, 1, 28, 28, 1])
print(a.unsqueeze(4).shape)  # torch.Size([4, 1, 28, 28, 1])
print(a.unsqueeze(-5).shape)  # torch.Size([1, 4, 1, 28, 28])
# print(a.unsqueeze(5).shape)    # IndexError: Dimension out of range (expected to be in range of [-5, 4], but got 5)

a = torch.tensor([1.2, 2.3])
print(a)
print(a.unsqueeze(-1))  # [2,1]
print(a.unsqueeze(0))  # [1,2]

# 案例:
b = torch.rand(32)
f = torch.rand(4, 32, 14, 14)
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print(b.shape) # torch.Size([1, 32, 1, 1])

# squeeze
b = torch.rand(1, 32, 1, 1)
print(b.squeeze().shape)  # 能压缩的都压缩 torch.Size([32])
print(b.squeeze(0).shape)  # 压缩第0个元素 torch.Size([32, 1, 1])
print(b.squeeze(-1).shape)  # torch.Size([1, 32, 1])
print(b.squeeze(1).shape)  # 32不能压缩就不压缩 torch.Size([1, 32, 1, 1])
print(b.squeeze(-4).shape)  # torch.Size([32, 1, 1])
