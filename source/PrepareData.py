import numpy as np
import torch


def generate_channel_single_user(N_bs, N_ms, fc, Lp, sigma2alpha, fs, K, P_set):
    n_bs = np.expand_dims(np.arange(N_bs), axis=1)
    idx_bs = np.expand_dims(np.arange(Lp), axis=0) + np.random.randint(N_bs) + 1
    idx_bs[idx_bs > N_bs] -= N_bs
    theta_bs = idx_bs / N_bs
    A_ms = np.ones((1, Lp))
    A_bs = np.exp(-2j * np.pi * n_bs.dot(theta_bs))
    L_cp = K / 4
    tau_max = L_cp / fs
    tau = np.sort(tau_max * np.random.rand(1, Lp)) * (-2 * np.pi * fs / K)
    alpha = np.sort(np.sqrt(sigma2alpha / 2) * (np.random.randn(1, Lp) + 1j * np.random.randn(1, Lp)))
    A_r = np.fft.fft(np.eye(N_bs)) / np.sqrt(N_bs)
    A_t = np.fft.fft(np.eye(N_ms)) / np.sqrt(N_ms)
    H_frq = np.zeros((N_bs, N_ms, len(P_set)), dtype=complex)
    H_ang = np.zeros((N_bs, N_ms, len(P_set)), dtype=complex)
    for i in range(len(P_set)):
        D = np.diag((alpha * np.exp(1j * P_set[i] * tau)).squeeze())
        H_frq[:, :, i] = A_bs.dot(D).dot(A_ms.T)
        H_ang[:, :, i] = A_r.T.dot(H_frq[:, :, i]).dot(A_t)
    return H_frq, H_ang, theta_bs


def generate_active_user(K, Ka, random_active):
    if not random_active:
        activity = np.zeros((K, 1))
        idx = np.random.permutation(K)
        activity[idx[:Ka]] = 1
    else:
        activity = np.random.rand(K, 1)
        activity = activity < (1 - Ka / K)
    return activity


def generate_active_channel(K, Ka, N_bs, P_set, Lp_min, Lp_max, N_ms, fc, sigma2alpha, fs, random_active):
    P = len(P_set)
    H_frq = np.zeros((K, N_bs, P), dtype=complex)
    H_ang = np.zeros((K, N_bs, P), dtype=complex)
    activity = generate_active_user(K, Ka, random_active)
    for user in range(K):
        if activity[user]:
            Lp = np.random.randint(Lp_min, Lp_max)
            H_f, H_a, _ = generate_channel_single_user(N_bs, N_ms, fc, Lp, sigma2alpha, fs, K, P_set)
            H_frq[user] = np.transpose(H_f, (1, 0, 2))
            H_ang[user] = np.transpose(H_a, (1, 0, 2))
    return H_frq, H_ang, activity


def generate_batched_active_channel(simulations, K, N_bs, P, Ka, P_set, Lp_min, Lp_max, N_ms, fc, sigma2alpha, fs,
                                    random_active=False):
    H_f = np.zeros((simulations, K, N_bs, P), dtype=complex)
    H_a = np.zeros((simulations, K, N_bs, P), dtype=complex)
    act = np.zeros((simulations, K))
    for i in range(simulations):
        H_frq, H_ang, activity = generate_active_channel(K,
                                                         Ka,
                                                         N_bs,
                                                         P_set,
                                                         Lp_min,
                                                         Lp_max,
                                                         N_ms,
                                                         fc,
                                                         sigma2alpha,
                                                         fs,
                                                         random_active,)
        H_f[i] = H_frq
        H_a[i] = H_ang
        act[i] = activity.squeeze()
    return H_f, H_a, act
