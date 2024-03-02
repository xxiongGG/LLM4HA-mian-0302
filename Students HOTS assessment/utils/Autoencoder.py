import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from data.dp_Text import DataPreprocess

# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, encoding_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 定义训练函数
def train_autoencoder(data, encoding_dim, num_epochs, batch_size):
    input_dim = data.shape[1]
    autoencoder = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # 将数据转换为Tensor
    data_tensor = torch.from_numpy(data).float()

    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(data_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 开始训练
    for epoch in range(num_epochs):
        for batch_data in dataloader:
            inputs = batch_data[0]
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    return autoencoder

# 示例数据
w2v_path = '../params/keywords_list.model'
sen_len = 10
path = '../data/q_list.xlsx'
data = pd.read_excel(path)
data_Key = data['Key'].to_list()
data_x = []
for key in data_Key:
    key_list = key[1:-1].split(',')
    data_x.append(key_list)
data_y = data['HOT_tag_id'].to_list()
preprocess = DataPreprocess(data_x, sen_len, q_path=path, w2v_path=w2v_path)

embedding = preprocess.make_embedding()
data_x = preprocess.sentence_word2idx()
data_x = preprocess.data_x_fusing(data_x, 3)
print(data_x.shape)

# 训练自编码器降维到32维
encoding_dim = 32
num_epochs = 1000
batch_size = 1

autoencoder = train_autoencoder(data_x.numpy(), encoding_dim, num_epochs, batch_size)

# 提取编码器的输出作为降维后的表示
encoded_data = autoencoder.encoder(torch.from_numpy(data_x).float()).detach().numpy()
print(encoded_data.shape)