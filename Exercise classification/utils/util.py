from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch


def find_max_position(lst):
    if type(lst) != list:
        lst = list(lst)
    max_index = lst.index(max(lst))
    result = [1 if i == max_index else 0 for i in range(len(lst))]
    return result


def get_acc(outputs, labels):
    zero = torch.zeros_like(outputs)
    one = torch.ones_like(outputs)
    y_score = torch.where(outputs > 0.5, one, zero)
    equal = torch.eq(y_score, labels)
    accuracy = torch.mean(equal.float())
    return accuracy


def get_loss_curve(epochs, lists):
    plt.plot(range(0, epochs), lists)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss curve')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()


def get_acc_curve(epochs, lists, test_acc, title='v'):
    if title == 'v':
        title_name = 'Vail acc'
        label_name = 'vail acc'
    else:
        title_name = 'Train acc'
        label_name = 'train acc'
    plt.plot(range(0, epochs), lists, label=label_name)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title(title_name)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(0, epochs), [test_acc for i in range(epochs)], label='test acc')
    plt.legend()
    plt.show()


def get_auc_curve(epochs, lists, test_acc, title='v'):
    if title == 'v':
        title_name = 'Vail auc'
        label_name = 'vail auc'
    else:
        title_name = 'Train auc'
        label_name = 'train auc'
    plt.plot(range(0, epochs), lists, label=label_name)
    plt.xlabel('epoch')
    plt.ylabel('auc')
    plt.title(title_name)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(0, epochs), [test_acc for i in range(epochs)], label='test auc')
    plt.legend()
    plt.show()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        # score = -val_loss
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Stopping score ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


