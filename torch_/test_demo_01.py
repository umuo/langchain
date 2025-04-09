#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/4/10 00:13

# 导入依赖库
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from simple_nn import SimpleNN  # 自定义的简单神经网络

# 数据预处理配置
transform = transforms.ToTensor()  # 将PIL图像转换为张量

# 初始化测试数据集
test_dataset = torchvision.datasets.MNIST(
    root='./data',         # 数据集存储路径
    train=False,           # 加载测试集
    download=True,         # 自动下载数据集
    transform=transform    # 应用数据预处理
)

# 创建测试数据加载器
test_loader = DataLoader(
    test_dataset,
    batch_size=64,         # 每批加载64个样本
    shuffle=False          # 测试集不需要打乱顺序
)

# 模型初始化与加载
model = SimpleNN()  # 实例化神经网络
model.load_state_dict(torch.load("simple_nn.pth"))  # 加载预训练权重
model.eval()

# 准确率计算初始化
correct = 0  # 正确预测数
total = 0    # 总样本数

# 禁用梯度计算以提升推理性能
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"✅ 测试集准确率: {accuracy * 100:.2f}%")