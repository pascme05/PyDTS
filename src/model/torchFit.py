#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         torchFit
# Date:         25.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.2
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Training, storing, loading and prediction of the PyTorch models of PyDTS (solvers DL and CL).
    fit()        mini-batch training with validation, early stopping (patience), learning-rate reduction (factor 0.5
                 after patience/2 epochs without improvement) and the epoch with the lowest validation loss as result;
                 every epoch is appended to training_log.csv (epoch, loss, <metric>, lr, val_loss, val_<metric>)
    save/load()  model.pt in mdl/mdl_<model>_<name>/ with the weights and the model description (meta)
    predict()    predictions in batches
The device is the GPU (CUDA) if setupExp['gpu'] = 1 and one is available, otherwise the CPU.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn


#######################################################################################################################
# Device
#######################################################################################################################
def device(gpu):
    """CUDA device if gpu == 1 and available, otherwise the CPU."""
    if gpu == 1:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return torch.device('cuda')
        print("WARN: No CUDA GPU found, using the CPU")

    return torch.device('cpu')


#######################################################################################################################
# Loss functions (per sample, so that sample weights can be applied)
#######################################################################################################################
def _flat(p, y):
    return p.reshape(p.shape[0], -1), y.reshape(y.shape[0], -1)


def lossFn(name):
    """Per-sample loss of the Keras names used in PyDTS (mae, mse, huber, msle, mape, BinaryCrossentropy, ...)."""
    key = str(name).lower().replace('_', '').replace(' ', '')
    eps = 1e-7
    fns = {
        'mae': lambda p, y: (p - y).abs().mean(1),
        'meanabsoluteerror': lambda p, y: (p - y).abs().mean(1),
        'mse': lambda p, y: ((p - y) ** 2).mean(1),
        'meansquarederror': lambda p, y: ((p - y) ** 2).mean(1),
        'huber': lambda p, y: nn.functional.smooth_l1_loss(p, y, reduction='none').mean(1),
        'msle': lambda p, y: ((torch.log1p(p.clamp(min=0)) - torch.log1p(y.clamp(min=0))) ** 2).mean(1),
        'mape': lambda p, y: (100 * (p - y).abs() / y.abs().clamp(min=eps)).mean(1),
        'binarycrossentropy': lambda p, y: nn.functional.binary_cross_entropy(p.clamp(eps, 1 - eps), y,
                                                                              reduction='none').mean(1),
        'kldivergence': lambda p, y: (y.clamp(eps, 1) * torch.log(y.clamp(eps, 1) / p.clamp(eps, 1))).sum(1),
    }
    if key not in fns:
        sys.exit("ERROR: Loss function %s not available, use mae, mse, huber, msle, mape, BinaryCrossentropy or "
                 "KLDivergence" % name)
    fn = fns[key]

    return lambda p, y: fn(*_flat(p, y))


#######################################################################################################################
# Optimiser
#######################################################################################################################
def optimiser(params, setupMdl, lr=None):
    lr = setupMdl['lr'] if lr is None else lr
    if setupMdl['opt'] == 'RMSprop':
        return torch.optim.RMSprop(params, lr=lr, alpha=setupMdl['rho'], momentum=setupMdl['mom'],
                                   eps=setupMdl['eps'])
    if setupMdl['opt'] in ('SGD', 'SDG'):
        return torch.optim.SGD(params, lr=lr, momentum=setupMdl['mom'])

    return torch.optim.Adam(params, lr=lr, betas=(setupMdl['beta1'], setupMdl['beta2']), eps=setupMdl['eps'])


#######################################################################################################################
# Storing and loading
#######################################################################################################################
def save(mdl, path, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'state': {k: v.detach().cpu() for k, v in mdl.state_dict().items()}, 'meta': meta}, path)


def load(mdl, path, meta=None):
    """Loads the weights of `path` into `mdl`; stops if the stored model was built for other settings."""
    try:
        ckpt = torch.load(path, map_location='cpu')
    except Exception as e:
        sys.exit("ERROR: Model %s could not be loaded, train it first (%s)" % (path, e))
    old = ckpt.get('meta', {})
    if meta is not None:
        diff = {k: (old.get(k), v) for k, v in meta.items() if k in old and old.get(k) != v}
        if diff:
            sys.exit("ERROR: Stored model %s was built for other settings %s, delete it or change the name"
                     % (path, diff))
    try:
        mdl.load_state_dict(ckpt['state'])
    except RuntimeError as e:
        sys.exit("ERROR: Stored model %s does not fit the current settings, delete it or change the name (%s)"
                 % (path, str(e)[:300]))

    return old


#######################################################################################################################
# Prediction
#######################################################################################################################
def predict(mdl, X, batch, dev):
    mdl.to(dev).eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[a:a + batch], dtype=np.float32)).to(dev, non_blocking=True)
            out.append(mdl(xb).float().cpu().numpy())

    return np.concatenate(out, axis=0).astype(np.float64)


def summary(mdl, shape):
    """Layer table (name, type, output shape, weights) for one window of the given shape."""
    rows, hooks = [], []

    def hook(m, i, o):
        o = o[0] if isinstance(o, (tuple, list)) else o
        rows.append({'layer': m._name, 'type': type(m).__name__,
                     'output shape': str(tuple(['B'] + list(o.shape[1:]))),
                     'weights': int(sum(p.numel() for p in m.parameters(recurse=False)))})

    skip = (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Identity, nn.Dropout)                                                     # no shape change
    for n, m in mdl.named_modules():
        if n and not list(m.children()) and not isinstance(m, skip) and type(m).__name__ != 'Transpose':
            m._name = n
            hooks.append(m.register_forward_hook(hook))
        elif n and isinstance(m, (nn.LSTM, nn.MultiheadAttention)):
            m._name = n
            hooks.append(m.register_forward_hook(hook))
    mdl.eval()
    with torch.no_grad():
        mdl(torch.zeros((2,) + tuple(shape)))
    for h in hooks:
        h.remove()

    return rows, int(sum(p.numel() for p in mdl.parameters()))


#######################################################################################################################
# Training
#######################################################################################################################
def fit(mdl, T, V, setupMdl, path, meta, dev, weights=None, shuffle=True, verbose=1, lr=None, log=True, trial=None):
    """
    Trains `mdl` on T = (X, y) with validation on V = (X, y). The weights of the epoch with the lowest validation loss
    are kept, stored in `path` (model.pt) and returned in `mdl`. Returns the history (one dict per epoch).
    `trial`: Optuna trial for pruning during the hyperparameter optimisation.
    """
    # ==============================================================================
    # Init
    # ==============================================================================
    XT, yT = T
    XV, yV = V
    nT, nV = len(XT), len(XV)
    if nV == 0:
        sys.exit("ERROR: No validation data, check the data split (rV, idV, val)")
    batch = int(max(1, min(setupMdl['batch'], nT)))
    vbatch = int(max(1, min(setupMdl['batch'], nV)))
    vsteps = int(max(1, min(setupMdl['valsteps'], int(np.ceil(nV / vbatch)))))
    patience = int(setupMdl['patience'])
    loss_fn, metric_fn = lossFn(setupMdl['loss']), lossFn(setupMdl['metric'])
    metric = str(setupMdl['metric'])
    w = None if weights is None else np.asarray(weights, dtype=np.float32)

    mdl.to(dev)
    opt = optimiser(mdl.parameters(), setupMdl, lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=max(1, patience // 2),
                                                       min_lr=1e-9)
    rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)                                                       # shuffle order follows the seed
    best, best_state, wait, hist = np.inf, None, 0, []
    log = log and bool(path)
    log_file = (os.path.join(os.path.dirname(path), os.path.basename(path).replace('model', 'training_log')
                             .replace('.pt', '.csv')) if log else None)                                              # one log per member
    ep0 = 0
    if log and os.path.exists(log_file):
        with open(log_file) as f:
            ep0 = max(0, sum(1 for _ in f) - 1)

    def batches(n, size, order):
        for a in range(0, n, size):
            yield order[a:a + size]

    def tensor(a, idx):
        return torch.from_numpy(np.ascontiguousarray(a[idx], dtype=np.float32)).to(dev, non_blocking=True)

    # ==============================================================================
    # Epochs
    # ==============================================================================
    for ep in range(int(setupMdl['epoch'])):
        start = time.time()

        # ------------------------------------------
        # Training
        # ------------------------------------------
        mdl.train()
        order = rng.permutation(nT) if shuffle else np.arange(nT)
        tl, tm, tn = 0.0, 0.0, 0
        for idx in batches(nT, batch, order):
            if len(idx) < 2 and nT > 1:                                                                              # batch norm needs 2 samples
                continue
            xb, yb = tensor(XT, idx), tensor(yT, idx)
            p = mdl(xb)
            ls = loss_fn(p, yb)
            if w is not None:
                ls = ls * torch.from_numpy(w[idx]).to(dev)
            l = ls.mean()
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            with torch.no_grad():
                tl += float(l) * len(idx)
                tm += float(metric_fn(p, yb).sum())
                tn += len(idx)

        # ------------------------------------------
        # Validation
        # ------------------------------------------
        mdl.eval()
        vl, vm, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for k, idx in enumerate(batches(nV, vbatch, np.arange(nV))):
                if k >= vsteps:
                    break
                xb, yb = tensor(XV, idx), tensor(yV, idx)
                p = mdl(xb)
                vl += float(loss_fn(p, yb).sum())
                vm += float(metric_fn(p, yb).sum())
                vn += len(idx)
        row = {'epoch': ep0 + ep, 'loss': tl / max(tn, 1), metric: tm / max(tn, 1),
               'lr': opt.param_groups[0]['lr'], 'val_loss': vl / max(vn, 1), 'val_' + metric: vm / max(vn, 1)}
        hist.append(row)

        # ------------------------------------------
        # Best epoch, learning rate, early stopping
        # ------------------------------------------
        improved = np.isfinite(row['val_loss']) and row['val_loss'] < best
        if improved:
            best, wait = row['val_loss'], 0
            best_state = {k: v.detach().clone() for k, v in mdl.state_dict().items()}
            if path:
                save(mdl, path, meta)
        else:
            wait += 1
        sched.step(row['val_loss'])
        if log:
            new = not os.path.exists(log_file)
            with open(log_file, 'a', newline='') as f:
                wr = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    wr.writeheader()
                wr.writerow(row)
        if verbose:
            print("Epoch %d/%d - %.1fs - loss: %.4g - val_loss: %.4g - lr: %.2g%s"
                  % (ep + 1, setupMdl['epoch'], time.time() - start, row['loss'], row['val_loss'], row['lr'],
                     ' *' if improved else ''), flush=True)
        if trial is not None:
            trial.report(row['val_loss'], ep)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if wait >= patience:
            print("INFO: Early stopping after %d epochs without improvement" % patience)
            break

    # ==============================================================================
    # Result: weights of the best epoch
    # ==============================================================================
    if best_state is not None:
        mdl.load_state_dict(best_state)
    elif path:
        print("WARN: No validation loss available, the weights of the last epoch are stored")
        save(mdl, path, meta)

    return hist


#######################################################################################################################
# Model description
#######################################################################################################################
def outDim(y, setupDat, setupPar):
    """Number of model outputs: sequence length (outseq), number of output channels or 1."""
    if len(setupDat['out']) == 1:
        return y.shape[1] if setupPar['outseq'] >= 1 else 1

    return len(setupDat['out'])


def mdlPath(setupPar, setupExp):
    """model.pt, or model_<k>.pt for the further members k = 2 .. ensemble of a seed ensemble."""
    k = int(setupExp.get('member', 0))
    return os.path.join('mdl', 'mdl_' + setupPar['model'] + '_' + setupExp['name'],
                        'model.pt' if k == 0 else 'model_%d.pt' % (k + 1))


def meta(X, out, setupPar):
    """Description stored with the weights; loading stops if it differs from the current settings."""
    return {'model': setupPar['model'], 'shape': [int(s) for s in X.shape[1:]], 'out': int(out),
            'method': int(setupPar['method'])}


def mdlSize(mdl):
    """Size of the weights in kB (float32)."""
    return sum(p.numel() for p in mdl.parameters()) * 4 / 1024
