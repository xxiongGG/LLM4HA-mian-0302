import logging
import torch
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score
from layers.HOT_Nets import RNN_Net
from layers.HOT_Nets import LSTM_Net
from layers.HOT_Nets import GRU_Net
import pandas as pd
from utils.utils import get_acc


def process_raw_pred(raw_question_matrix, raw_pred, num_questions: int) -> tuple:
    questions = torch.nonzero(raw_question_matrix)[1:, 1] % num_questions
    length = questions.shape[0]
    pred = raw_pred[: length]
    pred = pred.gather(1, questions.view(-1, 1)).flatten()
    truth = torch.nonzero(raw_question_matrix)[1:, 1] // num_questions
    return pred, truth


class HOT():
    def __init__(self, num_questions, hidden_size, num_layers, net_name='rnn'):
        super(HOT, self).__init__()
        self.num_questions = num_questions
        self.net_name = net_name

        if self.net_name == 'rnn':
            self.hot_model = RNN_Net(num_questions, hidden_size, num_layers)
        elif self.net_name == 'lstm':
            self.hot_model = LSTM_Net(num_questions, hidden_size, num_layers)
        elif self.net_name == 'gru':
            self.hot_model = GRU_Net(num_questions, hidden_size, num_layers)

        print('Current net is {}.'.format(self.net_name.upper()))

    def train(self, train_data, test_data=None, *, epoch: int, lr=0.002) -> ...:
        print('Train begging.')
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.hot_model.parameters(), lr)
        train_log = []
        for e in range(epoch):
            all_pred, all_target = torch.Tensor([]), torch.Tensor([])
            # for batch in tqdm(train_data):
            for batch in train_data:
                integrated_pred = self.hot_model(batch)
                batch_size = batch.shape[0]
                for student in range(batch_size):
                    pred, truth = process_raw_pred(batch[student], integrated_pred[student], self.num_questions)
                    all_pred = torch.cat([all_pred, pred])
                    all_target = torch.cat([all_target, truth.float()])

            preds = all_pred.detach().numpy()
            targets = all_target.detach().numpy()
            auc = roc_auc_score(targets, preds)
            acc = get_acc(preds, targets)
            loss = loss_function(all_pred, all_target)
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_log.append([epoch, loss.item(), auc, acc])
            print("[Epoch {}] Loss: {:.6f}, AUC:{:.3f}, ACC:{:.3f}".format(e, loss, auc, acc))
        train_log = pd.DataFrame(data=train_log, columns=['Epochs', 'Train Loss', 'Train AUC', 'Train ACC'])
        train_log.to_excel('results/' + self.net_name.upper() + '/1215/train_log.xlsx', index=False)
        print('The train_log save succeed!')

    def eval(self, test_data) -> float:
        self.hot_model.eval()
        y_pred = torch.Tensor([])
        y_truth = torch.Tensor([])
        for batch in tqdm(test_data, "evaluating"):
            integrated_pred = self.hot_model(batch)
            batch_size = batch.shape[0]
            for student in range(batch_size):
                pred, truth = process_raw_pred(batch[student], integrated_pred[student], self.num_questions)
                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth])
        preds = y_pred.detach().numpy()
        targets = y_truth.detach().numpy()
        auc = roc_auc_score(targets, preds)
        acc = get_acc(preds, targets)
        test_log = [auc, acc]
        test_log = pd.DataFrame(data=[test_log], columns=['Test AUC', 'Test ACC'])
        test_log.to_excel('results/' + self.net_name.upper() + '/1215/test_log.xlsx', index=False)
        print('The test_log save succeed!')
        return auc, acc

    def save(self, filepath):
        torch.save(self.hot_model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.hot_model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
