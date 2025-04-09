#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/4/9 23:48
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from simple_nn import SimpleNN

# 1. 准备数据
# 定义数据预处理：将PIL图像转换为Tensor，并自动归一化到[0,1]
transform = transforms.ToTensor()

# 加载MNIST手写数字数据集（自动下载如果不存在）
train_dataset = torchvision.datasets.MNIST(
    root='./data',       # 数据集存储路径
    train=True,          # 使用训练集
    download=True,       # 自动下载
    transform=transform  # 应用预处理
)
# 创建数据加载器：自动分批次、打乱数据
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,       # 每批64张图片
    shuffle=True         # 训练时打乱顺序
)

model = SimpleNN()  # 实例化神经网络

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（含Softmax）
optimizer = optim.Adam(            # Adam优化器
    model.parameters(),
    lr=0.001                      # 学习率
)
# 4. 训练模型
for epoch in range(5):  # 训练5轮
    for images, labels in train_loader:
        # 前向传播：计算预测值
        outputs = model(images)
        # 计算损失值（预测结果 vs 真实标签）
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()  # 清空历史梯度
        loss.backward()        # 反向传播计算梯度
        optimizer.step()       # 参数更新

    # 打印本轮训练结果（保留4位小数）
    print(f"Epoch [{epoch + 1}/5], Loss: {loss.item():.4f}")

# 5. 保存模型
torch.save(model.state_dict(), 'simple_nn.pth')  # 只保存模型参数
print("✅ 训练完成，模型已保存！")
