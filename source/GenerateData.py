from PrepareData import generate_batched_active_channel
import numpy as np
from init import params, data_params


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
