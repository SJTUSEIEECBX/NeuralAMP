import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


class UnbalanceBCELoss(nn.Module):
    def __init__(self, p=1):
        super(UnbalanceBCELoss, self).__init__()
        self.p = p

    def forward(self, predict, target):
        pn = self.p / (self.p + 1)
        loss = -pn * target * torch.log(predict + 1e-5) - (1 - pn) * (1 - target) * torch.log(1 - predict + 1e-5)
        loss = loss.mean()
        return loss

    def dec_p(self, rate=1):
        if self.p > rate:
            self.p -= rate
        else:
            print('cannot decrease p any more')

    def inc_p(self, rate):
        self.p += rate


def criterion(record):
    recall = record['tp'] / (record['tp'] + record['fn'])
    if record['tp'] + record['fp']:
        precision = record['tp'] / (record['tp'] + record['fp'])
    else:
        precision = 0
    accuracy = (record['tp'] + record['tn']) / (record['tp'] + record['tn'] + record['fp'] + record['fn'])
    sparsity = (record['fn'] + record['tn']) / (record['tp'] + record['tn'] + record['fp'] + record['fn'])
    return {'recall': recall, 'precision': precision, 'accuracy': accuracy, 'sparsity': sparsity}


class Records:
    def __init__(self, epochs):
        self.loss_ch = np.zeros(epochs)
        self.loss_act = np.zeros(epochs)
        self.accuracy = np.zeros(epochs)
        self.precision = np.zeros(epochs)
        self.recall = np.zeros(epochs)
        self.sparsity = np.zeros(epochs)

    def record(self, loss_act, loss_ch, criterion, epoch):
        self.loss_ch[epoch] = loss_ch
        self.loss_act[epoch] = loss_act
        self.accuracy[epoch] = criterion['accuracy']
        self.recall[epoch] = criterion['recall']
        self.precision[epoch] = criterion['precision']
        self.sparsity[epoch] = criterion['sparsity']

    def plot(self, filename):
        plt.figure(figsize=[15, 5])
        ax1 = plt.subplot(1, 2, 1)
        ax1.semilogy(self.loss_ch)
        ax1.semilogy(self.loss_act)
        ax1.grid(True, linestyle='-.')
        ax1.minorticks_on()
        ax1.grid(b=True, which='minor', color='#999999', linestyle='-.', alpha=0.2)
        ax1.legend(['channel loss', 'activity loss'])
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(self.accuracy)
        ax2.plot(self.recall)
        ax2.plot(self.precision)
        ax2.plot(self.sparsity)
        ax2.legend(['accuracy', 'recall', 'precision', 'sparsity'])
        plt.savefig('plots/{}.pdf'.format(filename))
