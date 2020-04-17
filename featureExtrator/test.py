import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 导入数据
device_no = 1
# feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
# label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')
feature_matrix = np.load('feature_matrix' + str(device_no) + '.npy')
label_matrix = np.load('label_matrix' + str(device_no) + '.npy')
# print(feature_matrix.shape)  #(2830, 2)
# print(label_matrix.shape)  #(2830,)

# 定义训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.2, random_state=0)
# print(X_train.shape)  #(2264, 2)
# print(X_test.shape)  #(566, 2)

# 归一化数据
from sklearn import preprocessing
X_train = preprocessing.scale(X_train)  # 标准化，均值为0，方差为1
X_test = preprocessing.scale(X_test)  # 标准化，均值为0，方差为1

# 设置参数
sequence_length = 1  # rnn 时间步数 / 图片高度
input_size = 2   # rnn 每步输入值 / 图片每行像素
hidden_size = 128
num_layers = 2
num_classes = 9
batch_size = 100
num_epochs = 2
learning_rate = 0.003

# np转换成张量
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(X_train)

for epoch in range(num_epochs):
    for i, voltage in enumerate(X_train):
        voltage = voltage.reshape(-1, sequence_length, input_size).to(device)
        label = y_train[i]
        label = label.to(device)

        # Forward pass
        output = model(voltage)
        loss = criterion(output, label)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))