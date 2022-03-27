import torch
# set default type: torch中默认的类型是torch.FloatTensor
print(torch.tensor([1.2, 3]).type())
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1.2, 3]).type())


# rand/rand_like, randint
# rand : [0, 1]    均匀分布
# rand_like: [min, max)    最大值不包含在里面
# randint *_like
print(torch.rand(3, 3))    # 比较均匀的采样出来
a = torch.rand(3, 3)
print(torch.rand_like(a))    # rand_like接受的参数是一个tensor,相当于把a.shape读出来再送给rand函数
print(torch.randint(1, 10, [3, 3]))

# randn: 正态分布
# N(0, 1) 用在bias比较多
# N(u, std)
print(torch.randn(3, 3))
# full函数生成长度为10都为0的list    反差从1到0慢慢减小
print(torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1)))

# full 全部填充一个数
print(torch.full([2, 3], 7))
print(torch.full([], 7))    # dim=0
print(torch.full([1], 7))    # dim=1

# range
print(torch.arange(0, 10))    # 不包括10
print(torch.arange(0, 10, 2))

# linspace/logspace
print(torch.linspace(0, 10, steps=4))    # 等分的切，包括10
print(torch.logspace(0, 1, steps=10))    # 切10等分，每个取指数0**10~1**10

# ones:生成全是0的，直接给出shape
# zeros:生成全是1的
# eye: 生成对角线全是1的,只接受1个参数或2个参数
print(torch.ones(3, 3))
print(torch.zeros(3, 3))
print(torch.eye(3, 4))
data = torch.ones(3, 3)
print(torch.ones_like(data))


# randperm:随机打散 randomshuffle
print(torch.randperm(10))
a = torch.rand(2, 3)
b = torch.rand(2, 2)
idx = torch.randperm(2)
print(idx)
print(a)
print(b)
print(a[idx])    # 达到协同shuffle的功能
print(b[idx])