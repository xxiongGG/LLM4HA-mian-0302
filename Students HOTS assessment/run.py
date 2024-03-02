from modules.HOT_20231215 import HOT

import logging
import numpy as np
import torch
import torch.utils.data as Data

# lstm or rnn or gru
EPOCH = 300
# NET_NAME = 'lstm'
NUM_QUESTIONS = 5
BATCH_SIZE = 2
HIDDEN_SIZE = 10
NUM_LAYERS = 1


def get_data_loader(data_path, batch_size, shuffle=False):
    data = torch.FloatTensor(np.load(data_path))
    data_loader = Data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader


train_loader = get_data_loader('data/train_data.npy', BATCH_SIZE, True)
test_loader = get_data_loader('data/test_data.npy', BATCH_SIZE, False)

logging.getLogger().setLevel(logging.INFO)
net_list = ['lstm', 'rnn', 'gru']
for NET_NAME in net_list:

    hot = HOT(NUM_QUESTIONS, HIDDEN_SIZE, NUM_LAYERS, net_name=NET_NAME)
    hot.train(train_loader, epoch=EPOCH)

    hot.save('params/HOT_' + NET_NAME.upper() + '.params')
    hot.load('params/HOT_' + NET_NAME.upper() + '.params')

    auc, acc = hot.eval(test_loader)
    print('[' + NET_NAME.upper() + ' Test] AUC:%.6f, ACC:%.6f' % (auc, acc))
