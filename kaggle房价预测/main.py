import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt


# Extract data, process data
train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')
# 提取标签，只取最后一列
train_label = torch.tensor(train_csv.iloc[:, -1], dtype=torch.float32).reshape(-1, 1)#reshape(-1,1)的作用是将其转化成一列
# 提取特征，从第二列到倒数第二列
all_feature = pd.concat((train_csv.iloc[:, 1:-1], test_csv.iloc[:, 1:]))
"""
在读取 CSV 文件时，pandas 默认会将第一行作为列名，并不会将其包含在数据中进行处理或训练。
具体来说：
pd.read_csv 函数默认会自动识别并使用第一行作为列名。
在后续的数据处理和特征提取过程中，列名仅用于索引和操作数据，而不会被用作实际的训练数据。
"""
number = all_feature.dtypes[all_feature.dtypes != 'object'].index  #筛选出数值列
all_feature[number] = all_feature[number].apply(
    lambda x: (x - x.mean()) / (x.std())
)
"""
这段代码的功能是对 all_feature[number] 列中的每个元素进行标准化处理，
即将每个元素减去该列的均值后除以该列的标准差，
从而将数据转换为均值为0、标准差为1的标准正态分布。
"""
# 填充数值列缺失值为均值，之前直接置为0，但是缺失不意味着0，直接填0可能导致误差较大，所以用均值填充
all_feature[number] = all_feature[number].fillna(all_feature[number].mean())

# 将类别型特征转换为 one-hot 向量
all_feature = pd.get_dummies(all_feature, dummy_na=True)

# 分离数据,

train_n = train_csv.shape[0]
all_feature = all_feature.astype(np.float32)  # 转换数据
train_data = torch.tensor(all_feature[:train_n].values, dtype=torch.float32)
test_data = torch.tensor(all_feature[train_n:].values, dtype=torch.float32)


class data(Dataset):
    def __init__(self, tensor1, tensor2):
        super().__init__()
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def __len__(self):
        return len(self.tensor1)

    def __getitem__(self, idex):
        train_one_data = self.tensor1[idex, :]
        label = self.tensor2[idex, -1]
        return train_one_data, label
"""
初始化时保存传入的两个张量。
返回数据集长度，即第一个张量的长度。
根据索引返回单个训练样本及其标签。
"""

data_1 = data(train_data, train_label)
data_2 = DataLoader(data_1, batch_size=32)

loss = nn.MSELoss()
in_feature = train_data.shape[1]


def loss_function(net, infeature, labels):
    final_y = torch.clamp(net(infeature), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(final_y), torch.log(labels)))
    return rmse.item()  # 将张量变为数值输出
"""
infeature 和 in_feature 是两个不同的变量名。in_feature 是在定义神经网络时使用的变量，表示输入特征的数量。而 infeature 是在 loss_function 函数和训练循环中使用的参数名，表示输入的特征数据。
具体来说：
in_feature 是一个整数，表示输入特征的数量，用于定义神经网络的输入层大小。
infeature 是一个张量，表示输入的特征数据，用于传递给神经网络模型进行前向传播。
因此，infeature 的命名是合理的，只是与 in_feature 的命名不同，但它们代表的意义和用途也不同。


使用 net(infeature) 获取网络预测值，并通过 torch.clamp 将其限制在 [1, +∞) 范围内。
计算预测值和标签的对数，然后使用均方误差损失函数 loss 计算损失。
取损失的平方根得到 RMSE，并返回其数值形式。
"""


def get_net():
    net = nn.Sequential(
        nn.Linear(in_feature, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.BatchNorm1d(16),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.BatchNorm1d(8),
        nn.Linear(8, 1)
    )
    return net


net = get_net()


def train(net, train_data, train_label, num_epoch, learning_rate, weight_decay):
    train_ls = []
    optimer = Adam(net.parameters(),
                   lr=learning_rate,
                   weight_decay=weight_decay
                   )

    for epoch in range(num_epoch):
        for x, y in data_2:
            optimer.zero_grad()
            loss_value = loss(net(x).squeeze(), y)
            #上面定义了loss=nn.MSELoss()
            loss_value.backward()
            print(epoch, " ",loss_value.item())
            optimer.step()
        train_ls.append(loss_function(net, train_data, train_label))

    return train_ls
"""
去掉维度为1的维度（即使用 squeeze() 方法）是为了确保张量的形状与后续计算中的预期形状匹配。具体来说：
网络输出形状：net(x) 的输出可能是一个形状为 (batch_size, 1) 的张量，表示每个样本的预测值。
损失函数要求：损失函数 loss 需要两个相同形状的张量作为输入。如果标签 y 的形状是 (batch_size,)，那么为了匹配标签的形状，需要将预测值的形状从 (batch_size, 1) 转换为 (batch_size,)。
通过调用 squeeze() 方法，可以去掉维度为1的维度，使得预测值的形状变为 (batch_size,)，从而与标签 y 的形状一致，避免在计算损失时出现形状不匹配的错误net。
"""

train(net, train_data, train_label, 30, learning_rate=0.1, weight_decay=1)

pred = net(test_data).detach().numpy()
"""net(test_data).detach().numpy() 的整体作用是：
使用训练好的模型对测试数据进行预测，得到一个 PyTorch 张量。
将该张量从计算图中分离出来，停止梯度跟踪。
将分离后的张量转换为 NumPy 数组，以便后续处理和保存。
"""

test_csv['SalePrice'] = pd.Series(pred.reshape(1, -1)[0])
submission = pd.concat((test_csv['Id'], test_csv['SalePrice']), axis=1)
submission.to_csv('submission.csv', index=False)
