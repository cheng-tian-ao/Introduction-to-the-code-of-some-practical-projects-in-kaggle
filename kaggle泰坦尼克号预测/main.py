import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam

# 数据加载
train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')

# 处理训练和测试数据
train_label = torch.tensor(train_csv['Survived'].values, dtype=torch.float32).reshape(-1, 1)  # 将 Survived 作为标签，当切片方法不太容易操作时可以考虑直接用列标签作为下表赋值

all_feature = pd.concat((train_csv.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1),
                         test_csv.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)))

"""
这段代码的功能是将训练集和测试集中的特征列合并成一个数据框。具体步骤如下：
从训练集中移除 'Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin' 列。
从测试集中移除 'PassengerId', 'Name', 'Ticket', 'Cabin' 列。
将处理后的训练集和测试集按列合并。
test_csv 在这行代码执行后不会发生变化。所有的操作都是基于副本进行的，原始的 train_csv 和 test_csv 保持不变
axis=1 在 pandas 中表示操作是沿着列的方向进行的。
具体来说，axis=1 意味着对 DataFrame 的列进行操作，而 axis=0 则表示对行进行操作。
"""
# 数值型特征处理
number = all_feature.dtypes[all_feature.dtypes != 'object'].index
all_feature[number] = all_feature[number].apply(lambda x: (x - x.mean()) / x.std())

# 填充缺失值（数值型用均值，类别型用众数）
all_feature[number] = all_feature[number].fillna(all_feature[number].mean())
all_feature = pd.get_dummies(all_feature, dummy_na=True)
"""
因此，all_feature[number] 是指所有数值型列，并且允许有缺失值
允许存在缺失值：在标准化之前，数值型列中可能存在缺失值（NaN）。这些缺失值不会影响标准化操作，因为 pandas 的 mean() 和 std() 方法会自动忽略缺失值进行计算。
填充缺失值：在标准化之后，使用每列的均值来填充缺失值，确保所有数值型列没有缺失值，以便后续建模和训练。
"""

# 确保为浮点数
all_feature = all_feature.astype(np.float32)
"""
all_feature = all_feature.astype(np.float32) 这行代码的作用是将 all_feature 数据框中的所有数值转换为 32 位浮点数（float32）。然而，这并不会直接将字符型（字符串）数据转换为浮点数。具体来说：
数值型列：对于已经是数值型的列（如 int, float），它们会被转换为 float32 类型。
字符型列：对于字符型（字符串）列，astype(np.float32) 会引发错误，因为无法将字符串直接转换为浮点数。
在你提供的代码中，all_feature 数据框在执行 astype(np.float32) 之前已经进行了以下处理：
使用 pd.get_dummies(all_feature, dummy_na=True) 将类别型特征转换为独热编码（One-Hot Encoding），生成的是数值型列（通常是 int 或 float）。
因此，在执行 astype(np.float32) 时，数据框中已经没有字符型列了，所有列都是数值型的。
示例说明
假设原始数据框中有以下列：
数值型列：Age, Fare
字符型列：Sex, Embarked
经过 pd.get_dummies 处理后，Sex 和 Embarked 列会被转换为多个数值型列（例如 Sex_male, Sex_female, Embarked_C, Embarked_Q, Embarked_S），这些新列的值为 0 或 1。
因此，执行 astype(np.float32) 是安全的，不会导致错误，并且只会将数值型列转换为 float32。
"""
# 分割训练和测试数据
"""
train_n = train_csv.shape[0] 这行代码的作用是获取训练集 train_csv 的行数，即训练样本的数量。
随后在分割数据时，使用这个值来区分训练数据和测试数据。
在之前的合并是纵向合并，行数增加，列数不变
"""
train_n = train_csv.shape[0]
train_data = torch.tensor(all_feature[:train_n].values, dtype=torch.float32)
test_data = torch.tensor(all_feature[train_n:].values, dtype=torch.float32)

# 数据集
class TitanicDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

dataset = TitanicDataset(train_data, train_label)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 损失函数和模型
loss_func = nn.BCEWithLogitsLoss()
#这段代码的功能是从训练数据中获取输入特征的数量。train_data.shape[1] 返回训练数据的列数，即每个样本的特征数量，并将其赋值给变量 in_feature。
in_feature = train_data.shape[1]
print(in_feature)
def get_net():
    net = nn.Sequential(
        nn.Linear(in_feature, 10),
        nn.ReLU(),
        nn.BatchNorm1d(10),
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.BatchNorm1d(8),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.BatchNorm1d(4),
        nn.Linear(4, 1)
    )
    return net
"""
批量归一化（Batch Normalization，简称BN）是一种深度学习中常用的技巧，旨在加速训练过程并提高模型的稳定性。它通过规范化每一层的输入来减少内部协变量偏移（Internal Covariate Shift），即在训练过程中由于参数更新导致的各层输入分布的变化。
批量归一化的作用
加速训练：通过将每层的输入规范化到均值为0、方差为1的标准正态分布，使得网络更容易收敛，从而加速训练过程。
允许使用更高的学习率：规范化后的数据分布更加稳定，可以使用更大的学习率而不易导致梯度爆炸或消失。
减少对初始化的依赖：减少了对初始权重的敏感性，使得模型对不同初始化方案更具鲁棒性。
提供正则化效果：类似于Dropout，批量归一化引入了一定程度的噪声，有助于防止过拟合。
"""
net = get_net()

# 训练函数
def train(net, train_loader, num_epochs, learning_rate, weight_decay):
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loss = []

    for epoch in range(num_epochs):
        net.train()
        """
        net.train() 设置网络为训练模式。在训练模式下，某些层（如Dropout、BatchNorm）会表现得不同，以确保模型能够更好地学习数据的特征。
        """
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = net(x).squeeze()
            loss = loss_func(outputs, y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')
        train_loss.append(total_loss / len(train_loader))
    return train_loss

# 训练模型
train_loss = train(net, data_loader, num_epochs=21, learning_rate=0.001, weight_decay=0.01)

# 预测
net.eval()
with torch.no_grad():
    test_preds = torch.sigmoid(net(test_data)).numpy()
    test_preds = (test_preds > 0.5).astype(int).flatten()
"""
net.eval()：将模型设置为评估模式。在评估模式下，模型中的某些层（如 Dropout 和 BatchNorm）会改变行为，以确保推理时的稳定性。
with torch.no_grad():：禁用梯度计算。这不仅节省了内存，还加快了推理速度，因为不需要计算和存储中间梯度。
test_preds = torch.sigmoid(net(test_data)).numpy()：使用训练好的模型对测试数据进行预测，并通过 sigmoid 函数将输出转换为概率值，最后将结果转换为 numpy 数组。
test_preds = (test_preds > 0.5).astype(int).flatten()：将预测的概率值大于 0.5 的设为 1，否则设为 0，并将结果展平为一维数组。
"""
# 准备提交文件
submit = pd.DataFrame({
    "PassengerId": test_csv["PassengerId"],
    "Survived": test_preds
})

submit.to_csv('submission.csv', index=False)