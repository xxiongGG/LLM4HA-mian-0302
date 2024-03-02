import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def read_file(path):
    """
    读取文本文件进行预处理
    :param path: 文件路径
    :return: 分词后的数组
    """
    if 'training_label' in path:
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
            x = [line[2:] for line in lines]
            y = [line[0] for line in lines]
        return x, y
    elif 'training_nolabel' in path:
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x
    else:
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            x = ["".join(line[1:].strip('\n').split(",")) for line in lines[1:]]
            x = [item.split(' ') for item in x]
        return x

def get_acc(outputs, labels):
    corrects_num = get_corrects_num(outputs, labels)
    return corrects_num / len(outputs)

def get_corrects_num(outputs, labels):

    outputs = get_corrects(outputs)
    corrects_num = torch.sum(torch.eq(torch.tensor(outputs), torch.tensor(labels))).item()

    return corrects_num

def get_corrects(outputs):
    modified_list = []
    for num in outputs:
        if num > 0.5:
            modified_list.append(1)
        else:
            modified_list.append(0)
    return modified_list


def draw_fig(list, name, epoch):
    x1 = range(0, epoch)
    y1 = list
    if name == "loss":
        plt.cla()
        plt.title('Train loss vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train loss', fontsize=20)
        plt.grid()
        # plt.savefig("./lossAndacc/Train_loss.png")
        plt.show()
    elif name == "acc":
        plt.cla()
        plt.title('Train accuracy vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train accuracy', fontsize=20)
        plt.grid()
        # plt.savefig("./lossAndacc/Train _accuracy.png")
        plt.show()
