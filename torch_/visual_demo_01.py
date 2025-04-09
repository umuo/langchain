#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/4/10 00:34
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from simple_nn import SimpleNN  # 自定义的简单神经网络
import torch
import random

# 数据预处理配置
transform = transforms.ToTensor()  # 将PIL图像转换为张量
# 初始化测试数据集
test_dataset = torchvision.datasets.MNIST(
    root='./data',  # 数据集存储路径
    train=False,  # 加载测试集
    download=True,  # 自动下载数据集
    transform=transform  # 应用数据预处理
)

# 模型初始化与加载
model = SimpleNN()  # 实例化神经网络
model.load_state_dict(torch.load("simple_nn.pth"))  # 加载预训练权重
model.eval()


# 用于抓取中间层输出
def get_intermediate_output(model, image):
    x = image.view(-1, 28 * 28)
    x = model.fc1(x)
    x_relu = model.relu(x)
    return x_relu.squeeze().detach().numpy()


# 显示前 10 个神经元的权重
plt.figure(figsize=(10, 4))
weights = model.fc1.weight.data  # shape: (128, 784)
for i in range(10):
    w = weights[i].reshape(28, 28)
    plt.subplot(2, 5, i + 1)
    plt.imshow(w, cmap='seismic')
    plt.title(f'Neuron {i}')
    plt.axis('off')
plt.suptitle("第一层前10个神经元的权重")
plt.tight_layout()
plt.show()

# 选择一个样本
image, label = test_dataset[random.randint(0, len(test_dataset) - 1)]
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"原图 Label: {label}")
plt.show()

# 获取中间层输出
features = get_intermediate_output(model, image)
plt.plot(features)
plt.title("中间层特征向量 (128维)")
plt.grid()
plt.show()
