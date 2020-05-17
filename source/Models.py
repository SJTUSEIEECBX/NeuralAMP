import torch.nn as nn
import torch
import numpy as np


def cpxmm(a_real, a_imag, b_real, b_imag):
    res_real = a_real.matmul(b_real) - a_imag.matmul(b_imag)
    res_imag = a_real.matmul(b_imag) + a_imag.matmul(b_real)
    return res_real, res_imag


class NonlinearEstimator(nn.Module):
    def __init__(self, N, K, T, activate=nn.Softshrink):
        super(NonlinearEstimator, self).__init__()
        self.left_weight_real = nn.Parameter(torch.FloatTensor(N, N))
        self.left_weight_imag = nn.Parameter(torch.FloatTensor(N, N))
        # self.right_weight_real = nn.Parameter(torch.FloatTensor(K, K))
        # self.right_weight_imag = nn.Parameter(torch.FloatTensor(K, K))
        self.bias_real = nn.Parameter(torch.FloatTensor(N, K))
        self.bias_imag = nn.Parameter(torch.FloatTensor(N, K))
        self.activate_real = activate()
        self.activate_imag = activate()
        nn.init.eye_(self.left_weight_real)
        nn.init.eye_(self.left_weight_imag)
        # nn.init.eye_(self.right_weight_real)
        # nn.init.eye_(self.right_weight_imag)
        nn.init.zeros_(self.bias_real)
        nn.init.zeros_(self.bias_imag)

    def forward(self, X_real, X_imag):
        Y_real, Y_imag = cpxmm(self.left_weight_real, self.left_weight_imag, X_real, X_imag)
        # Y_real, Y_imag = cpxmm(X_real, X_imag, self.right_weight_real, self.right_weight_imag)
        Y_real = Y_real + self.bias_real
        Y_imag = Y_imag + self.bias_imag
        if self.activate_real != None:
            Y_real = self.activate_real(Y_real)
            Y_imag = self.activate_imag(Y_imag)
        return Y_real, Y_imag


class LinearSteper(nn.Module):
    def __init__(self, N, K, T, activate=nn.Tanh):
        super(LinearSteper, self).__init__()
        self.left_weight_real = nn.Parameter(torch.FloatTensor(T, N))
        self.left_weight_imag = nn.Parameter(torch.FloatTensor(T, N))
        self.right_weight_real = nn.Parameter(torch.FloatTensor(K, T))
        self.right_weight_imag = nn.Parameter(torch.FloatTensor(K, T))
        self.bias_real = nn.Parameter(torch.FloatTensor(T, T))
        self.bias_imag = nn.Parameter(torch.FloatTensor(T, T))
        self.activate_real = activate()
        self.activate_imag = activate()
        nn.init.eye_(self.left_weight_real)
        nn.init.eye_(self.left_weight_imag)
        nn.init.eye_(self.right_weight_real)
        nn.init.eye_(self.right_weight_imag)
        nn.init.zeros_(self.bias_real)
        nn.init.zeros_(self.bias_imag)

    def forward(self, X_real, X_imag):
        Y_real, Y_imag = cpxmm(self.left_weight_real, self.left_weight_imag, X_real, X_imag)
        Y_real, Y_imag = cpxmm(Y_real, Y_imag, self.right_weight_real, self.right_weight_imag)
        Y_real = Y_real + self.bias_real
        Y_imag = Y_imag + self.bias_imag
        if self.activate_real != None:
            Y_real = self.activate_real(Y_real)
            Y_imag = self.activate_imag(Y_imag)
        return Y_real, Y_imag


class AMPLayer(nn.Module):
    def __init__(self, N, K, T):
        super(AMPLayer, self).__init__()
        self.linear = LinearSteper(N, K, T)
        self.nonlinear = NonlinearEstimator(N, K, T, activate=nn.Tanhshrink)

    def forward(self, Y_real, Y_imag, H_real, H_imag, P_real, P_imag, H_pre_real, H_pre_imag, R_pre_real, R_pre_imag):
        Y_pre_real, Y_pre_imag = cpxmm(H_real, H_imag, P_real, P_imag)
        linear_step_real, linear_step_imag = self.linear(H_pre_real, H_pre_imag)
        # print(torch.isnan(linear_step_real + linear_step_imag).sum().item(), 'nan in LE', sep='\t')
        if torch.isnan(linear_step_real + linear_step_imag).sum().item():
            print('nan in LE')
        linear_step_real, linear_step_imag = cpxmm(R_pre_real, R_pre_imag, linear_step_real, linear_step_imag)
        R_real = Y_real - Y_pre_real + linear_step_real
        R_imag = Y_imag - Y_pre_imag + linear_step_imag
        H_tmp_real, H_tmp_imag = cpxmm(R_real, R_imag, P_real.permute(1, 0), -P_imag.permute(1, 0))
        H_tmp_real = H_tmp_real + H_real
        H_tmp_imag = H_tmp_imag + H_imag
        H_new_real, H_new_imag = self.nonlinear(H_tmp_real, H_tmp_imag)
        # print(torch.isnan(H_new_real + H_new_imag).sum().item(), 'nan in NLE', sep='\t')
        if torch.isnan(H_new_real + H_new_imag).sum().item():
            print('nan in NLE')
        return H_new_real, H_new_imag, H_tmp_real, H_tmp_imag, R_real, R_imag


class NeuralAMP(nn.Module):
    def __init__(self, N, K, T, num_layers=3, device='cuda'):
        super(NeuralAMP, self).__init__()
        self.Layers = nn.ModuleList([AMPLayer(N, K, T) for i in range(num_layers)])
        self.N = N
        self.K = K
        self.num_layers = num_layers
        self.device = device

    def forward(self, Y_real, Y_imag, P_real, P_imag):
        batch = Y_real.shape[0]
        H_real = torch.zeros(batch, self.N, self.K, requires_grad=True, device=self.device, dtype=torch.float32)
        H_imag = torch.zeros(batch, self.N, self.K, requires_grad=True, device=self.device, dtype=torch.float32)
        H_pre_real = torch.zeros(batch, self.N, self.K, requires_grad=True, device=self.device, dtype=torch.float32)
        H_pre_imag = torch.zeros(batch, self.N, self.K, requires_grad=True, device=self.device, dtype=torch.float32)
        R_pre_real = Y_real.clone()
        R_pre_imag = Y_imag.clone()
        for layer in self.Layers:
            H_real, H_imag, H_pre_real, H_pre_imag, R_pre_real, R_pre_imag = layer(Y_real, Y_imag, H_real, H_imag,
                                                                                   P_real, P_imag, H_pre_real,
                                                                                   H_pre_imag, R_pre_real, R_pre_imag)
        return H_real, H_imag


class Channel2Act(nn.Module):
    def __init__(self, N, K):
        super(Channel2Act, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, N))
        self.bias = nn.Parameter(torch.FloatTensor(1, K))
        nn.init.constant_(self.weight, 1 / N)
        nn.init.zeros_(self.bias)
        self.activate = nn.Sigmoid()

    def forward(self, H_real, H_imag):
        H_abs = (H_real ** 2 + H_imag ** 2).sqrt()
        act = self.weight.matmul(H_abs) + self.bias
        act = self.activate(act)
        return act.squeeze()


class NeuralAMP_AutoEncoder(nn.Module):
    def __init__(self, N, K, T, num_AMP_layers=4, device='cuda'):
        super(NeuralAMP_AutoEncoder, self).__init__()
        self.N = N
        self.T = T
        self.P_real = nn.Parameter(torch.FloatTensor(K, T))
        self.P_imag = nn.Parameter(torch.FloatTensor(K, T))
        # self.threshold = nn.Parameter(torch.Tensor([4]))
        nn.init.orthogonal_(self.P_real)
        nn.init.orthogonal_(self.P_imag)
        self.decoder = NeuralAMP(N, K, T, num_AMP_layers, device)
        # self.activedetect = Channel2Act(N, K)
        # self.output = nn.Softmax()
        self.device = device

    def forward(self, H_real, H_imag, nvar):
        N_real = np.sqrt(nvar / 2) * torch.randn(self.N, self.T, device=self.device)
        N_imag = np.sqrt(nvar / 2) * torch.randn(self.N, self.T, device=self.device)
        Y_real, Y_imag = cpxmm(H_real, H_imag, self.P_real, self.P_imag)
        Y_real = Y_real + N_real
        Y_imag = Y_imag + N_imag
        # print(torch.isnan(Y_real + Y_imag).sum().item(), 'nan in Y', sep='\t')
        H_hat_real, H_hat_imag = self.decoder(Y_real, Y_imag, self.P_real.detach(), self.P_imag.detach())
        if torch.isnan(H_hat_real + H_hat_imag).sum().item():
            print('nan in H')
        # print(torch.isnan(H_hat_real + H_hat_imag).sum().item(), 'nan in H', sep='\t')
        H_abs = (H_hat_real ** 2 + H_hat_imag ** 2).sqrt()
        act = H_abs.mean(dim=1)
        # act = (act + 1).log()
        act = act / act.max(dim=1, keepdim=True)[0]
        # act = self.output(act)
        # act = self.activedetect(H_hat_real, H_hat_imag)
        return H_hat_real, H_hat_imag, act