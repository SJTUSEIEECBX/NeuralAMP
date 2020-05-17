import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
from PrepareData import generate_batched_active_channel
from Models import NeuralAMP_AutoEncoder
from init import params, str_params, data_params
import utils
from Loops import train_loop, test_loop
import datetime


starttime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
print(str_params)

try:
    dataset = np.load('data/{}.npz'.format(data_params))
    print('Loaded pre-generated data.')
    H_f_train = dataset['H_f_train']
    H_a_train = dataset['H_a_train']
    activity_train = dataset['activity_train']
    H_f_test = dataset['H_f_test']
    H_a_test = dataset['H_a_test']
    activity_test = dataset['activity_test']
except FileNotFoundError:
    print('Start to generate data at {}.'.format(starttime))
    H_f_train, H_a_train, activity_train = generate_batched_active_channel(simulations=params['train_batchsize'],
                                                                           K=params['K'],
                                                                           N_bs=params['N_bs'],
                                                                           P=params['P'],
                                                                           Ka=params['Ka'],
                                                                           P_set=params['P_set'],
                                                                           Lp_min=params['Lp_min'],
                                                                           Lp_max=params['Lp_max'],
                                                                           N_ms=params['N_ms'],
                                                                           fc=params['fc'],
                                                                           sigma2alpha=params['sigma2alpha'],
                                                                           fs=params['fs'],
                                                                           random_active=params['random_active'])

    H_f_test, H_a_test, activity_test = generate_batched_active_channel(simulations=params['test_batchsize'],
                                                                        K=params['K'],
                                                                        N_bs=params['N_bs'],
                                                                        P=params['P'],
                                                                        Ka=params['Ka'],
                                                                        P_set=params['P_set'],
                                                                        Lp_min=params['Lp_min'],
                                                                        Lp_max=params['Lp_max'],
                                                                        N_ms=params['N_ms'],
                                                                        fc=params['fc'],
                                                                        sigma2alpha=params['sigma2alpha'],
                                                                        fs=params['fs'],
                                                                        random_active=params['random_active'])

    np.savez('data/{}.npz'.format(data_params), H_f_train=H_f_train, H_a_train=H_a_train, activity_train=activity_train,
             H_f_test=H_f_test, H_a_test=H_a_test, activity_test=activity_test)

    print('Data generated at {}.'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

channel_real_train = torch.from_numpy(np.real(H_f_train)).squeeze().float().permute(0, 2, 1)
channel_imag_train = torch.from_numpy(np.imag(H_f_train)).squeeze().float().permute(0, 2, 1)
activity_train = torch.from_numpy(activity_train).float()

channel_real_test = torch.from_numpy(np.real(H_f_test)).squeeze().float().permute(0, 2, 1)
channel_imag_test = torch.from_numpy(np.imag(H_f_test)).squeeze().float().permute(0, 2, 1)
activity_test = torch.from_numpy(activity_test).float()

train_set = data.TensorDataset(channel_real_train, channel_imag_train, activity_train)
trainloader = data.DataLoader(train_set, batch_size=params['minibatch'], shuffle=True)

test_set = data.TensorDataset(channel_real_test, channel_imag_test, activity_test)
testloader = data.DataLoader(test_set, batch_size=params['minibatch'], shuffle=False)

model = NeuralAMP_AutoEncoder(params['N_bs'], params['K'], params['T'], device='cuda').cuda()
loss_func_c = nn.MSELoss()
loss_func_a = utils.UnbalanceBCELoss(p=params['unblc'])
optimizer = torch.optim.Adam(model.parameters(), lr=params['LR'])

Records_train = utils.Records(params['EPOCH'])
Records_test = utils.Records(params['EPOCH'])

print('training started at {}.'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

for epoch in range(params['EPOCH']):
    loss_act_train, loss_ch_train, record_train = train_loop(model, optimizer, trainloader, loss_func_a, loss_func_c,
                                                             params['nvar'])
    criterion_train = utils.criterion(record_train)
    loss_act_test, loss_ch_test, record_test = test_loop(model, testloader, loss_func_a, loss_func_c, params['nvar'])
    criterion_test = utils.criterion(record_test)

    Records_train.record(loss_act_train, loss_ch_train, criterion_train, epoch)
    Records_test.record(loss_act_test, loss_ch_test, criterion_test, epoch)

    if epoch % 50 == 0:
        print('Epoch {} at {}'.format(epoch, datetime.datetime.now()))

torch.save(model.state_dict(), 'trained_models/{}.pkl'.format(starttime))
print('Model saved as {}.pkl'.format(starttime))

Records_train.plot('{}-train'.format(starttime))
Records_test.plot('{}-test'.format(starttime))


