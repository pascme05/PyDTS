#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         models
# Date:         03.11.2023
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.2
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Deep learning models of PyDTS (PyTorch). Every model takes a batch of windows X with the shape (B, W, F) — B windows,
W samples, F input channels — and returns (B, out). The layer structure follows the former Keras models:
    DNN     3 dense layers (32) on the flattened window
    CNN     5 convolutions (30-30-40-50-50 filters), max pooling (5), 3 dense layers (32)
    LSTM    2 LSTM layers (128), 3 dense layers (32)
    TRAN    dense embedding (32), one transformer encoder block (2 heads), average pooling over time
    INF     informer-style encoder block (as TRAN, full attention)
    DAE     convolutional denoising autoencoder
    TFS     time (convolution + neural ODE), frequency (random Fourier features) and statistical paths, fused
Search spaces of the hyperparameter optimisation (Optuna) are at the end of the file.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import sys

import numpy as np
import torch
import torch.nn as nn


#######################################################################################################################
# Helper
#######################################################################################################################
def outAct(activation):
    """Output activation: 'linear' (regression) or 'sigmoid' (classification)."""
    return nn.Sigmoid() if activation == 'sigmoid' else nn.Identity()


def flatSize(module, shape):
    """Number of features after `module` for one input window of the given shape (W, F)."""
    with torch.no_grad():
        return int(np.prod(module(torch.zeros((1,) + tuple(shape))).shape[1:]))


class Transpose(nn.Module):
    """(B, W, F) <-> (B, F, W) between time-major data and channel-first convolutions."""
    def forward(self, x):
        return x.transpose(1, 2)


def dense(n_in, units=(32, 32, 32)):
    layers = []
    for u in units:
        layers += [nn.Linear(n_in, u), nn.ReLU()]
        n_in = u

    return layers, n_in


#######################################################################################################################
# DNN
#######################################################################################################################
class DNN(nn.Module):
    def __init__(self, shape, out, activation):
        super().__init__()
        layers, n = dense(int(np.prod(shape)))
        self.net = nn.Sequential(nn.Flatten(), *layers, nn.Linear(n, out), outAct(activation))

    def forward(self, x):
        return self.net(x)


#######################################################################################################################
# CNN
#######################################################################################################################
class CNN(nn.Module):
    def __init__(self, shape, out, activation):
        super().__init__()
        W, F = shape
        conv, c_in = [Transpose()], F
        for c, k in ((30, 10), (30, 8), (40, 6), (50, 5), (50, 5)):
            conv += [nn.Conv1d(c_in, c, k, padding='same'), nn.ReLU()]
            c_in = c
        p = min(5, W)
        conv += [nn.MaxPool1d(p, p, ceil_mode=True), nn.Flatten()]
        self.feat = nn.Sequential(*conv)
        layers, n = dense(flatSize(self.feat, shape))
        self.head = nn.Sequential(*layers, nn.Linear(n, out), outAct(activation))

    def forward(self, x):
        return self.head(self.feat(x))


#######################################################################################################################
# LSTM
#######################################################################################################################
class LSTM(nn.Module):
    def __init__(self, shape, out, activation):
        super().__init__()
        self.lstm1 = nn.LSTM(shape[1], 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        layers, n = dense(128)
        self.head = nn.Sequential(*layers, nn.Linear(n, out), outAct(activation))

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.head(x[:, -1, :])                                                                                   # last time step


#######################################################################################################################
# Transformer and Informer
#######################################################################################################################
class EncoderBlock(nn.Module):
    """Self-attention and feed-forward network, each with residual connection, dropout and layer normalisation."""
    def __init__(self, dim=32, heads=2, ff=32, rate=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ff), nn.ReLU(), nn.Linear(ff, dim))
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.drop1, self.drop2 = nn.Dropout(rate), nn.Dropout(rate)

    def forward(self, x):
        a, _ = self.att(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop1(a))
        return self.norm2(x + self.drop2(self.ffn(x)))


class TRAN(nn.Module):
    def __init__(self, shape, out, activation):
        super().__init__()
        self.embed = nn.Linear(shape[1], 32)
        self.block = EncoderBlock(32, 2, 32)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(32, out), outAct(activation))

    def forward(self, x):
        return self.head(self.block(self.embed(x)).mean(dim=1))                                                         # average over time


class INF(TRAN):
    """Informer-style encoder (full self-attention, as the former Keras model)."""


#######################################################################################################################
# Denoising Autoencoder
#######################################################################################################################
class DAE(nn.Module):
    def __init__(self, shape, out, activation):
        super().__init__()
        W, F = shape
        self.out = out
        self.enc = nn.Sequential(Transpose(), nn.Conv1d(F, 32, 3, padding='same'), nn.ReLU(),
                                 nn.Conv1d(32, 32, 3, padding='same'), nn.ReLU(), nn.Flatten(),
                                 nn.Linear(32 * W, 32), nn.ReLU())
        self.up = nn.Sequential(nn.Linear(32, 32 * out), nn.ReLU())
        self.dec = nn.Sequential(nn.Conv1d(32, 32, 3, padding='same'), nn.ReLU(), nn.Flatten(),
                                 nn.Linear(32 * out, out), outAct(activation))

    def forward(self, x):
        z = self.up(self.enc(x)).view(-1, self.out, 32).transpose(1, 2)
        return self.dec(z)


#######################################################################################################################
# Time/Frequency/Statistical (TFS) Model
#######################################################################################################################
class ODEBlock(nn.Module):
    """dz/dt = f(z) from t = 0 to 1 with 4 Runge-Kutta steps (the Keras model used Dormand-Prince)."""
    def __init__(self, n_in, units=64, steps=4):
        super().__init__()
        self.proj = nn.Linear(n_in, units)
        self.f = nn.Sequential(nn.Linear(units, units), nn.Tanh(), nn.Linear(units, units))
        self.steps = steps

    def forward(self, x):
        z, h = self.proj(x), 1.0 / self.steps
        for _ in range(self.steps):
            k1 = self.f(z)
            k2 = self.f(z + h / 2 * k1)
            k3 = self.f(z + h / 2 * k2)
            k4 = self.f(z + h * k3)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return z


class FourierFeatures(nn.Module):
    """Fixed random projection (σ = 0.1) followed by sin and cos."""
    def __init__(self, n_in, dim=64):
        super().__init__()
        self.register_buffer('omega', torch.randn(n_in, dim // 2, generator=torch.Generator().manual_seed(42)) * 0.1)

    def forward(self, x):
        p = x @ self.omega
        return torch.cat([torch.sin(p), torch.cos(p)], dim=-1)


def stats(x):
    """Mean, min, max, std, median, range and IQR over time: (B, W, F) -> (B, 7, F)."""
    q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], device=x.device), dim=1)
    mn, mx = x.min(dim=1).values, x.max(dim=1).values
    return torch.stack([x.mean(dim=1), mn, mx, x.std(dim=1, unbiased=False), q[1], mx - mn, q[2] - q[0]], dim=1)


class TFS(nn.Module):
    def __init__(self, shape, out, activation):
        super().__init__()
        W, F = shape
        self.time = nn.Sequential(Transpose(), nn.Conv1d(F, 32, 6, padding='same'), nn.ReLU(), nn.BatchNorm1d(32),
                                  Transpose(), ODEBlock(32, 64), nn.Flatten())
        self.freq = nn.Sequential(FourierFeatures(F, 64), nn.LayerNorm(64), nn.Linear(64, 64), nn.ReLU(),
                                  nn.Flatten())
        self.stat = nn.Sequential(nn.Linear(F, 64), nn.ReLU(), nn.Flatten())
        n = [W * 64, W * 64, 7 * 64]
        m = min(n)
        self.adj = nn.ModuleList([nn.Linear(k, m) if k != m else nn.Identity() for k in n])
        self.head = nn.Sequential(nn.Linear(3 * m, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, out),
                                  outAct(activation))

    def forward(self, x):
        paths = [self.time(x), self.freq(x), self.stat(stats(x))]
        return self.head(torch.cat([a(p) for a, p in zip(self.adj, paths)], dim=1))


#######################################################################################################################
# Factory
#######################################################################################################################
MODELS = {'DNN': DNN, 'CNN': CNN, 'LSTM': LSTM, 'TRAN': TRAN, 'INF': INF, 'DAE': DAE, 'TFS': TFS}


def buildMdl(name, shape, out, activation):
    """Model `name` for windows of shape (W, F) with `out` outputs."""
    if name not in MODELS:
        sys.exit("ERROR: Model %s not available for the DL/CL solver, use one of %s" % (name, ', '.join(MODELS)))

    return MODELS[name](tuple(int(s) for s in shape), int(out), activation)


#######################################################################################################################
# Hyperparameter Optimisation (Optuna search spaces)
#######################################################################################################################
class Searched(nn.Module):
    """Model assembled from a feature extractor and a dense head chosen by the hyperparameter search."""
    def __init__(self, feat, shape, units, out, activation, flat=True):
        super().__init__()
        self.feat = feat
        layers, n_in = [], flatSize(feat, shape)
        for u in units:
            layers += [nn.Linear(n_in, u), nn.ReLU()]
            n_in = u
        self.head = nn.Sequential(*layers, nn.Linear(n_in, out), outAct(activation))

    def forward(self, x):
        return self.head(self.feat(x))


class LastStep(nn.Module):
    def __init__(self, sizes, n_in):
        super().__init__()
        self.lstm = nn.ModuleList()
        for s in sizes:
            self.lstm.append(nn.LSTM(n_in, s, batch_first=True))
            n_in = s

    def forward(self, x):
        for l in self.lstm:
            x, _ = l(x)
        return x[:, -1, :]


def optiMdl(trial, name, shape, out, activation):
    """
    Model of one trial. Regression: CNN (default), LSTM or DNN search space depending on the configured model;
    classification (sigmoid output): DNN search space. Returns the model and its learning rate.
    """
    W, F = shape
    if activation == 'sigmoid' or name == 'DNN':
        u = trial.suggest_int('units', 64, 512, step=64)
        feat = nn.Flatten()
        units = [u] * trial.suggest_int('dnn_layers', 2, 6)
        lr = trial.suggest_categorical('learning_rate', [1e-1, 5e-2, 1e-2] if activation == 'sigmoid'
                                       else [1e-2, 1e-3, 1e-4])
    elif name == 'LSTM':
        sizes = [trial.suggest_int('nodes_%d' % i, 16, 128, step=16) for i in range(trial.suggest_int('lstm_layers', 0, 3))]
        sizes.append(trial.suggest_int('nodes2', 16, 128, step=16))
        feat = LastStep(sizes, F)
        units = [trial.suggest_int('units_%d' % i, 32, 256, step=32) for i in range(trial.suggest_int('dnn_layers', 1, 4))]
        lr = trial.suggest_categorical('learning_rate', [1e-2, 1e-3, 1e-4])
    else:
        layers, c_in = [Transpose()], F
        drop = trial.suggest_categorical('dropout_opt', [False, True])
        rate = trial.suggest_float('dropout', 0.1, 0.5, step=0.1) if drop else 0
        bn = trial.suggest_categorical('batch_opt', [False, True])
        for i in range(trial.suggest_int('cnn_layers', 1, 5)):
            c = trial.suggest_int('filters_%d' % i, 8, 64, step=8)
            layers += [nn.Conv1d(c_in, c, trial.suggest_int('kernel_size_%d' % i, 2, 10, step=2), padding='same'),
                       nn.ReLU()]
            if drop:
                layers.append(nn.Dropout(rate))
            if bn:
                layers.append(nn.BatchNorm1d(c))
            c_in = c
        if trial.suggest_categorical('pooling_opt', [False, True]):
            p = min(trial.suggest_int('pool_size', 2, 10, step=2), W)
            layers.append(nn.MaxPool1d(p, min(trial.suggest_int('strides', 2, 10, step=2), W), ceil_mode=True))
        feat = nn.Sequential(*layers, nn.Flatten())
        units = [trial.suggest_int('units_%d' % i, 64, 512, step=64) for i in range(trial.suggest_int('dnn_layers', 1, 4))]
        lr = trial.suggest_categorical('learning_rate', [1e-2, 1e-3, 1e-4])

    return Searched(feat, shape, units, out, activation), lr
