import argparse
import numpy as np


parser = argparse.ArgumentParser(description="Train and test neural AMP")
parser.add_argument('K', type=int, help='Total number of users.')
parser.add_argument('Ka', type=int, help='Number of active users, or the expectation if -r is set to truth.')
parser.add_argument('N', type=int, help='Number of receiver antennas')
parser.add_argument('T', type=int, help='Length of pilot sequence')
parser.add_argument('--ra', type=bool, default=False, help='Whether to use random number of active users.')
parser.add_argument('--snr', type=float, default=20, help='Signal to noise ratio')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--tb', type=int, default=1e5, help='Training batch size')
parser.add_argument('--vb', type=int, default=1e4, help='Validation batch size')
parser.add_argument('--mb', type=int, default=2e3, help='Mini-batch size')
parser.add_argument('--ep', type=int, default=1.5e3, help='Number of training epochs')
parser.add_argument('--ly', type=int, default=4, help='Number of layers')
parser.add_argument('--up', type=float, default=3, help='Rate of unbalanced BCE loss')

args = parser.parse_args()

params = {
    'K': args.K,  # number of potential users
    'Ka': args.Ka,  # number of active users
    'random_active': args.ra,  # whether to use random number of active users
    'N_bs': args.N,  # number of receiver antennas (base station)
    'T': args.T,  # time overhead (length of pilot sequence)
    'fc': 2e9,  # frequency of carrier
    'sigma2alpha': 1, # average power of path
    'Bw': 1e7,  # bandwidth
    'fs': 1e7,  # frequency of sampling
    'N': 2048,  # number of carriers
    'P': 1,  # number of subcarrier used for pilot (usually set to 1 while not utilizing OFDM)
    'Lp_min': 8,  # min number of pathes
    'Lp_max': 14,  # max number of pathes
    'snr': args.snr,  # SNR in dB
    'train_batchsize': args.tb,  # total amount of training data
    'test_batchsize': args.vb,  # total amount of test data
    'minibatch': args.mb,  # size of minibatch used in Stochastic Gradient Descend
    'LR': args.lr,  # learning rate
    'EPOCH': args.ep,  # training epochs
    'layers': args.ly,  # AMP layers
    'unblc': args.up,  # Rate of unbalanced loss
}
delta_p = np.floor(params['N'] / params['P']).astype(int)
P_set = np.linspace(delta_p, params['P'] * delta_p, params['P'])
params.update({'P_set':  P_set})
nvar = 10 ** (-params['snr'] / 10)
params.update({'nvar': nvar})

str_params = 'K={}, Ka={}, N={}, T={}, snr={}, random active={}'.format(params['K'], params['Ka'], params['N_bs'],
                                                                        params['T'], params['snr'],
                                                                        params['random_active'])
