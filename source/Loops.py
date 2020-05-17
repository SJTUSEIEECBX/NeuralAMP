import torch
import numpy as np
import torch.nn as nn


def train_loop(model, optimizer, dataloader, loss_func_a, loss_func_c, nvar, threshold=0.5):
    model.train()
    avg_loss_act = 0
    avg_loss_ch = 0
    tp = 0  # True Positive
    fp = 0  # False Positive
    tn = 0  # True Negative
    fn = 0  # False Negative
    for i, (ch_real, ch_imag, act_train) in enumerate(dataloader):
        ch_real = ch_real.cuda()
        ch_imag = ch_imag.cuda()
        act_train = act_train.cuda()
        ch_hat_real, ch_hat_imag, act = model(ch_real, ch_imag, nvar)
        channel_loss = loss_func_c(ch_real, ch_hat_real) + loss_func_c(ch_imag, ch_hat_imag)
        if torch.isnan(channel_loss):
            print('nan in channel')
        act_loss = loss_func_a(act, act_train)
        if torch.isnan(act_loss):
            print('nan in act')
        loss = act_loss + channel_loss
        avg_loss_act += ch_real.shape[0] * act_loss.cpu().item()
        avg_loss_ch += ch_real.shape[0] * channel_loss.cpu().item()
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            if torch.isnan(param.grad).sum():
                param.grad[torch.isnan(param.grad)] = 0
        nn.utils.clip_grad_value_(model.parameters(), 100)
        optimizer.step()
        act = act > threshold
        tp += ((act == 1) & (act_train == 1)).float().sum().detach().cpu().item()
        fp += ((act == 1) & (act_train == 0)).float().sum().detach().cpu().item()
        tn += ((act == 0) & (act_train == 0)).float().sum().detach().cpu().item()
        fn += ((act == 0) & (act_train == 1)).float().sum().detach().cpu().item()
    avg_loss_act = avg_loss_act / len(dataloader.dataset)
    avg_loss_ch = avg_loss_ch / len(dataloader.dataset)
    return avg_loss_ch, avg_loss_act, {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def test_loop(model, dataloader, loss_func_a, loss_func_c, nvar, threshold=0.5):
    model.eval()
    avg_loss_act = 0
    avg_loss_ch = 0
    tp = 0  # True Positive
    fp = 0  # False Positive
    tn = 0  # True Negative
    fn = 0  # False Negative
    with torch.no_grad():
        for i, (ch_real, ch_imag, act_test) in enumerate(dataloader):
            ch_real = ch_real.cuda()
            ch_imag = ch_imag.cuda()
            act_test = act_test.cuda()
            ch_hat_real, ch_hat_imag, act = model(ch_real, ch_imag, nvar)
            channel_loss = loss_func_c(ch_real, ch_hat_real) + loss_func_c(ch_imag, ch_hat_imag)
            if torch.isnan(channel_loss):
                print('nan in channel')
            act_loss = loss_func_a(act, act_test)
            if torch.isnan(act_loss):
                print('nan in act')
            avg_loss_act += ch_real.shape[0] * act_loss.cpu().item()
            avg_loss_ch += ch_real.shape[0] * channel_loss.cpu().item()
            act = act > threshold
            tp += ((act == 1) & (act_test == 1)).float().sum().detach().cpu().item()
            fp += ((act == 1) & (act_test == 0)).float().sum().detach().cpu().item()
            tn += ((act == 0) & (act_test == 0)).float().sum().detach().cpu().item()
            fn += ((act == 0) & (act_test == 1)).float().sum().detach().cpu().item()
    avg_loss_act = avg_loss_act / len(dataloader.dataset)
    avg_loss_ch = avg_loss_ch / len(dataloader.dataset)
    return avg_loss_ch, avg_loss_act, {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
