#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/4/10 00:15
import torch.nn as nn


# 2. 定义神经网络结构
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 全连接层1：28x28输入 -> 128个神经元
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层 -> 隐藏层
        # ReLU激活函数（非线性变换）
        self.relu = nn.ReLU()
        # 全连接层2：128个神经元 -> 10个输出（对应0-9数字）
        self.fc2 = nn.Linear(128, 10)  # 隐藏层 -> 输出层

    def forward(self, x):
        # 将二维图像数据展平成一维向量（保持批次维度）
        x = x.view(-1, 28 * 28)  # 把图像展平成一维向量
        x = self.fc1(x)     # 第一层线性变换
        x = self.relu(x)    # 应用激活函数
        x = self.fc2(x)     # 第二层线性变换（输出logits）
        return x
