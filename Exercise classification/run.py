import torch.utils.data
from torch.optim import Adam
from tqdm import tqdm
from data.Dataset_loader import Dataset
import pandas as pd
import numpy as np
from models.model import BertClassifier
from transformers import BertTokenizer
from layers.focal_loss import FocalLoss
from utils.util import EarlyStopping
from utils.util import get_acc_curve
from utils.util import get_loss_curve
from torch import nn


def train(model, train_data, val_data, learning_rate, epochs, tokenizer, labels):
    early_stopping = EarlyStopping(patience=3, verbose=True)
    train, val = Dataset(train_data, tokenizer, labels), Dataset(val_data, tokenizer, labels)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2, shuffle=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = FocalLoss(class_num=5, gamma=4)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    train_log = []
    for epoch_num in range(epochs):
        total_acc_train, total_loss_train, total_acc_val = 0, 0, 0
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            batch_loss = criterion(output, train_label.long()) * 0.1
            total_loss_train += batch_loss.item()
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()


        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'''Epochs: {epoch_num + 1} 
                | Train Loss: {total_loss_train / len(train_data): .3f} 
                | Train Accuracy: {total_acc_train / len(train_data): .3f} 
                | Val Accuracy: {total_acc_val / len(val_data): .3f}''')
        train_log.append([epoch_num, total_loss_train / len(train_data),
                          total_acc_train / len(train_data), total_acc_val / len(val_data)])
        early_stopping(total_acc_val / len(val_data), model, 'results')
    train_log = pd.DataFrame(data=train_log, columns=['Epochs', 'Train Loss', 'Train Accuracy', 'Val Accuracy'])
    train_log.to_excel('results/logs/train_log_1221.xlsx', index=False)
    print('The train_log save succeed!')
    return model



def evaluate(model, test_data):
    test = Dataset(test_data, tokenizer, labels)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = model.cuda()

    total_acc_test = 0
    outputs = []
    label_lists = []
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

            output = output.argmax(dim=1).detach().cpu().numpy()
            test_label = test_label.detach().cpu().numpy()

            label_lists += test_label.tolist()
            outputs += output.tolist()

        acc = total_acc_test / len(test_data)
        print('Test ACC:{:.3f}'.format(acc))

        test_log = [acc]
        test_log = pd.DataFrame(data=[test_log], columns=['Test Accuracy'])

        test_log.to_excel('results/logs/test_log_1221.xlsx', index=False)
        print('The test_log save succeed!')
        return acc



if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-english')
    labels = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4
    }
    EPOCHS = 100
    model = BertClassifier()
    print(model)
    LR = (1e-6) * 0.5
    df = pd.read_excel('data/q_list_en_2.xlsx')
    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])
    #
    print(len(df_train), len(df_val), len(df_test))
    train_model = train(model, df_train, df_val, LR, EPOCHS, tokenizer, labels)
    model.load_state_dict(torch.load('results/checkpoint.pth'))
    acc = evaluate(model, df_test)

    df = pd.read_excel("results/logs/train_log_1221.xlsx")
    epochs = len(df)
    acc_list = df['Val Accuracy'].tolist()
    loss_list = df['Train Loss'].tolist()
    get_loss_curve(epochs, loss_list)
    get_acc_curve(epochs, acc_list, acc, title='v')
