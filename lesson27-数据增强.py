# Data argumentation
# ---------------------------------------#
# 这些操作在torchvision包里面
# 1. Flip：翻转
# 2. Rotate
# 3. Random Move & Crop
# 4. GAN : 生成更多的样本
# 5. Noise: N(0, 0.001)加高斯白噪声
# ---------------------------------------#

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

batch_size=200
learning_rate=0.01
epochs=10

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([    # Compose的操作类似于nn.Sequential里面
                       transforms.RandomHorizontalFlip(),    # 水平角度的翻转    （随机翻转-可能翻转也有可能不翻转）
                       transforms.RandomVerticalFlip(),    # 垂直方向
                       transforms.RandomRotation(15),    # 旋转方向，参数为旋转的度数
                       transforms.RandomRotation([90, 180, 270]),    # 随机的从90度180度270度中挑一个角度旋转
                       transforms.Resize([32, 32]),    # 传入的参数为list
                       transforms.RandomCrop([28, 28]),    # 裁剪
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),    # x 转换成x'
    batch_size=batch_size, shuffle=True)