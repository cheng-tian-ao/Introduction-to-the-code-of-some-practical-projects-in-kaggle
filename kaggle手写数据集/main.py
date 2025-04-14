import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from torch import nn
from torch.optim import Adam

train_csv=pd.read_csv('..\\kaggle手写数据集数据\\train.csv')
test_csv=pd.read_csv('..\\kaggle手写数据集数据\\test.csv')

#train_label=torch.tensor(train_csv['label'].values,dtype=torch.float32).reshape(-1,1)
#train_label = torch.tensor(train_csv['label'].values, dtype=torch.long).unsqueeze(1)
train_label = torch.tensor(train_csv['label'].values, dtype=torch.long)
#all_feature=pd.concat((train_csv.drop(['label'],axis=1),test_csv.drop([],axis=1)))
all_feature = pd.concat((train_csv.drop(['label'], axis=1), test_csv))

number =all_feature.dtypes[all_feature.dtypes != 'object'].index
#all_feature[number]=all_feature[number].apply(lambda x:(x-x.mean())/(x.std()))
"""
这里千万不要归一化，因为会将其转换成0-1的数，而他需要long型

"""


all_feature =all_feature.astype(np.float32)


train_n=train_csv.shape[0]
train_data =torch.tensor(all_feature.values[:train_n],dtype=torch.float32)
test_data =torch.tensor(all_feature.values[train_n:],dtype=torch.float32)

class TitanicDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
dataset = TitanicDataset(train_data, train_label)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

loss_func=nn.CrossEntropyLoss()

in_feature=train_data.shape[1]
print(in_feature)
def get_net():
    net=nn.Sequential(
        nn.Linear(in_feature,512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,10),
        #nn.Softmax(dim=1)
    )
    return net
net =get_net()
def train(net,train_loader,number_epochs,learning_rate,weight_decay):
    optimizer=Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)
    train_loss =[]
    for epoch in range(number_epochs):
        net.train()
        total_loss =0.0
        for x,y in train_loader:
            optimizer.zero_grad()
            outputs = net(x).squeeze()

            y = y.long()
            """
            cross_entropy_loss函数要求目标标签必须是整型（Long）
            """
            loss = loss_func(outputs,y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')
        train_loss.append(total_loss / len(train_loader))
    return train_loss
train_loss = train(net, data_loader, number_epochs=10, learning_rate=0.001, weight_decay=0.01)

# 预测
net.eval()
with torch.no_grad():
    test_pred = net(test_data).numpy()
    test_pred = np.argmax(test_pred, axis=1)

submission = pd.DataFrame({
    'ImageId': range(1, len(test_pred) + 1),
    'Label': test_pred
})
submission.to_csv('submission.csv', index=False)
