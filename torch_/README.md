# 和AI一起学PyTorch

## 神经网络

1、什么是神经网络？
输入一张图片，它经过一系列“变换”和“激活”，最终输出一个结果（比如“这是只猫”）

2、前馈神经网络的组成
- 输入层（Input Layer）
- 隐藏层（Hidden Layers）--> 可有多层，包含神经元 + 激活函数
- 输出层（Output Layer）

每一层的神经元都和下一层的神经元连接，每个连接都有一个权重（weight），每个神经元还有一个偏置（bias），
这些权重和偏置都是神经网络的参数，需要通过训练来学习。

一句话概括：把输入不断乘权重，加偏置，然后经过激活函数，最后输出结果；然后通过梯度下降不断优化这些参数，让预测越来越准。

### 神经元

每个神经元就是一个小函数
```text
输出 = 激活函数（权重 * 输入 + 偏置）= activation(w * x + b)
```
常用的激活函数：
- ReLU函数：f(x) = max(0, x) 深度网络最常见
- Sigmoid函数：f(x) = 1 / (1 + exp(-x)) 二分类问题常用
- Tanh函数：f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) 归一化到[-1, 1]

神经元的输入是一个向量，输出是一个标量，
神经元的计算过程是：
1、将输入向量和权重向量进行点乘，得到一个标量
2、将这个标量加上偏置，得到一个标量
3、将这个标量作为激活函数的输入，得到一个标量
4、将这个标量作为神经元的输出

### 激活函数
激活函数是一个非线性函数，它的作用是将输入的标量映射到一个新的标量，
激活函数的作用是：
1、增加模型的非线性能力
2、增加模型的表达能力
3、增加模型的泛化能力
常用的激活函数有：
1、sigmoid函数
2、tanh函数
3、ReLU函数
4、LeakyReLU函数
5、Softmax函数 --> 用于多分类问题

### 前向传播（Forward Propagation）

把输入数据传到网络，计算出预测输出。

### 损失函数（Loss Function）
损失函数是一个函数，它的作用是计算预测输出和真实输出之间的误差。
损失函数的作用是：
1、计算预测输出和真实输出之间的误差
2、用于反向传播，更新网络的参数
常用的损失函数有：
1、均方误差（MSE）
2、交叉熵（Cross Entropy）
3、Hinge Loss

### 反向传播（Backward Propagation）
利用链式法则，计算出每个参数对损失的影响（梯度）

### 梯度下降（Gradient Descent）
利用梯度下降算法，更新网络的参数，使得损失函数最小化

## 代码示例
```python
#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/4/9 23:48
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. 准备数据
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


# 2. 定义神经网络结构
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层 -> 隐藏层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 隐藏层 -> 输出层

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 把图像展平成一维向量
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleNN()

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
for epoch in range(5):  # 训练5轮
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/5], Loss: {loss.item():.4f}")

# 5. 保存模型
torch.save(model.state_dict(), 'simple_nn.pth')
print("✅ 训练完成，模型已保存！")
```

输出
```text
100%|██████████| 9.91M/9.91M [00:02<00:00, 4.28MB/s]
100%|██████████| 28.9k/28.9k [00:00<00:00, 216kB/s]
100%|██████████| 1.65M/1.65M [00:04<00:00, 347kB/s]
100%|██████████| 4.54k/4.54k [00:00<00:00, 55.1kB/s]
Epoch [1/5], Loss: 0.1131
Epoch [2/5], Loss: 0.0403
Epoch [3/5], Loss: 0.1430
Epoch [4/5], Loss: 0.0361
Epoch [5/5], Loss: 0.0419
✅ 训练完成，模型已保存！
```
