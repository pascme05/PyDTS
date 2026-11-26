#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Black-Box Modeling
# File:         theory.py
# Date:         25.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Theory view of the operating modes: what PyDTS computes, step by step, with explorers that start from the current
configuration and call the functions of PyDTS where possible.
Tabs:   Overview          the chain data → normalisation → windows → model → back-scaling → errors
        Data & windows    window length in time vs. the time constant of the output (estimated from the data)
        Network           the PyTorch network of src/model/models.py for the current window (layers, shapes, weights)
        Training          loss, Adam, early stopping, learning-rate reduction; the stored training log of the model
        Errors            the metrics of src/general/performance.py on an adjustable example
        References        literature
THEORY = {mode: fn(cfg)} adds the view 'Theory' to the mode page.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import contextlib
import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui import adapter as A
from gui.core import data as D
from gui.core import literature as lit
from gui.core.present import panel_tabs, interpretation, tiles, hint
from gui.core.style import show, line, table, fmt, downsample, SERIES
from gui.core.theory_kit import symbols, intro

KERAS = {'DNN': 'DNN', 'CNN': 'CNN', 'LSTM': 'LSTM', 'TRAN': 'TRAN', 'DAE': 'DAE', 'INF': 'INF', 'TFS': 'TFS'}
TABS = ["Overview", "Data & windows", "Network", "Training", "Errors", "References"]
TITLES = {'Overview': "What does PyDTS compute?", 'Data & windows': "Does the window see enough of the past?",
          'Network': "How is the network built and how many weights does it learn?",
          'Training': "How are the weights found?", 'Errors': "How are the errors measured?",
          'References': "Where to read more?"}
SUBS = {'Overview': "The chain of a run from the datasets to the errors, as in src/main.py, src/train.py and "
                    "src/test.py.",
        'Data & windows': "Sliding windows of the inputs (src/general/framing.py) and the thermal memory of the "
                          "output, estimated from the first training dataset.",
        'Network': "Built with the model function of src/model/models.py for the current window and channels.",
        'Training': "src/model/torchFit.py: PyTorch training with early stopping, learning-rate reduction and "
                    "the best epoch as stored model.",
        'Errors': "src/general/performance.py evaluated on a measured-like temperature and an adjustable "
                  "prediction.",
        'References': "Papers and books behind the methods of PyDTS."}


#######################################################################################################################
# Helper
#######################################################################################################################
def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _i(x, d=0):
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return d


def _channels(cfg):
    """(inputs, outputs, units of the outputs, dataset) of the first training dataset (None if not readable)."""
    typ = str(cfg['Dat'].get('type', 'xlsx'))
    train = [str(n) for n in A.to_list(cfg['Dat'].get('train'))]
    if not train:
        return None
    try:
        d = D.load(train[0], typ)
    except Exception:
        d = None
    if d is None:
        return None
    X, y, ux, uy = d
    inp = [str(c) for c in A.to_list(cfg['Dat'].get('inp'))] or [c for c in X.columns if c not in ('time', 'id')]
    out = [str(c) for c in A.to_list(cfg['Dat'].get('out'))] or [c for c in y.columns if c not in ('time', 'id')]
    return [c for c in inp if c in X], [c for c in out if c in y], uy, (X, y, train[0])


def _tau(y, fs):
    """
    Time constant of a signal from its autocorrelation: the lag at which the autocorrelation of the detrended signal
    drops below 1/e (an RC element has ρ(τ) = e^(−τ/τ_th)). Returns seconds (nan if it never drops).
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    s = max(1, len(y) // 20000)
    z = y[::s] - np.mean(y[::s])
    if len(z) < 10 or not np.any(z):
        return np.nan
    f = np.fft.rfft(z, 2 * len(z))
    r = np.fft.irfft(f * np.conj(f))[:len(z)]
    r = r / r[0]
    k = np.argmax(r < np.exp(-1))
    return k * s / fs if k > 0 else np.nan


@st.cache_data(show_spinner="Building the network …", max_entries=16)
def keras_summary(model, W, F, M, method, nDim):
    """Layers (name, type, output shape, weights) of the PyDTS network (PyTorch) for an input of W samples × F channels."""
    from src.model.models import buildMdl
    from src.model.torchFit import summary
    mdl = buildMdl(model, (W, F), M, 'linear' if method == 0 else 'sigmoid')
    return summary(mdl, (W, F))


def _arch_dot(rows, W, F, M, inp, out):
    fill = {'Conv1D': '#e8f4ea', 'LSTM': '#f3e8f7', 'Dense': '#eef3fb', 'MaxPooling1D': '#fffbe6',
            'Flatten': '#fffbe6', 'MultiHeadAttention': '#fff4e5', 'TransformerBlock': '#fff4e5',
            'Conv1d': '#e8f4ea', 'Linear': '#eef3fb', 'MaxPool1d': '#fffbe6', 'MultiheadAttention': '#fff4e5'}
    L = ['digraph G {', 'rankdir=TB; nodesep=0.25; ranksep=0.2; bgcolor="transparent";',
         'node [shape=box, style="rounded,filled", fillcolor="#eef3fb", color="#2a78d6", fontname="Helvetica", '
         'fontsize=11];', 'edge [color="#666666"];',
         f'x [label="input window\\n{", ".join(inp[:6])}{" …" if len(inp) > 6 else ""}\\nB × {W} × {F}", '
         f'fillcolor="#fff4e5", color="#e07b00"];']
    prev = 'x'
    for i, r in enumerate(rows):
        if r['type'] == 'InputLayer':
            continue
        lab = f"{r['type']}\\nout: {r['output shape']}" + (f"\\n{r['weights']:,} weights" if r['weights'] else '')
        L.append(f'n{i} [label="{lab}", fillcolor="{fill.get(r["type"], "#eef3fb")}"];')
        L.append(f'{prev} -> n{i};')
        prev = f'n{i}'
    L.append(f'y [label="prediction ŷ\\n{", ".join(out[:6])}{" …" if len(out) > 6 else ""}  (B × {M})", '
             f'fillcolor="#fde8e8", color="#d03b3b"];')
    L.append(f'{prev} -> y;')
    L.append('}')
    return '\n'.join(L)


#######################################################################################################################
# Tabs
#######################################################################################################################
def _overview(cfg):
    solver, model = cfg['Par'].get('solver'), cfg['Par'].get('model')
    W, ov = _i(cfg['Par'].get('window'), 1), _i(cfg['Par'].get('overlap'))
    st.markdown(f"PyDTS learns a **black-box model** from measured data: no physics is put in, the relation between "
                f"inputs X (e.g. currents, voltages, speed, coolant temperature) and outputs y (e.g. winding and "
                f"magnet temperature) is learned from examples. The current configuration uses the solver "
                f"**{solver}** with the model **{model}**.")
    st.latex(r"\hat{y}(k) = f_\theta\big(\,\tilde{x}(k-W+1),\;\dots,\;\tilde{x}(k)\,\big),\qquad "
             r"\tilde{x} = \frac{x-\mu_x}{\sigma_x}")
    st.latex(r"\theta^\ast = \arg\min_\theta \frac{1}{N}\sum_{i=1}^{N} \ell\big(f_\theta(X_i),\,y_i\big)")
    from gui.core.workflow import Flow
    f = Flow(rankdir='LR')
    f.node('d', 'Datasets', [', '.join(str(n) for n in A.to_list(cfg['Dat'].get('train')))[:30]], kind='inp')
    f.node('n', 'Normalise', [f"inputs: {cfg['Dat'].get('inpNorm')}", f"outputs: {cfg['Dat'].get('outNorm')}"])
    f.node('w', 'Windows', [f"W = {W}", f"overlap {ov}"], active=cfg['Par'].get('frame') == 1)
    f.node('m', 'Model', [f"{solver} · {model}"], kind='opt')
    f.node('b', 'Scale back', ['clip to [min, max]'])
    f.node('e', 'Errors', ['MAE, RMSE, R², MAX'], kind='out')
    f.edge('d', 'n').edge('n', 'w').edge('w', 'm').edge('m', 'b').edge('b', 'e')
    st.graphviz_chart(f.dot(), width='content')
    interpretation(f"Every prediction ŷ(k) is made from the last W = {W} samples of the normalised inputs. The model "
                   f"never sees the output itself during testing (unless *Initial values* is on), so it must infer "
                   f"the thermal state from the input history — which is why the window length, or rolled features "
                   f"that summarise a long history, decide how well temperatures can be predicted.")
    symbols([('x(k)', 'input vector at sample k (Data › Inputs)', 'unit of the channel', 'currents, voltages, T_cool'),
             ('y(k), ŷ(k)', 'measured and predicted output', 'unit of the channel', 'temperatures in °C'),
             ('μ_x, σ_x', 'mean and standard deviation of the training inputs (Data › Input normalisation = 3)',
              'unit of x', '–'),
             ('W', 'window length (Parameters › Window length)', 'samples', '10 … 1000'),
             ('f_θ', 'model with weights θ (Parameters › Model)', '–', 'DNN, CNN, LSTM'),
             ('ℓ', 'loss per window (Model › Loss function)', '–', 'mae, mse'),
             ('N', 'number of training windows', '–', '10⁴ … 10⁶')])


def _windows(cfg):
    W, ov = max(_i(cfg['Par'].get('window'), 1), 1), _i(cfg['Par'].get('overlap'))
    fs = _f(cfg['Dat'].get('fs'), 1.0) or 1.0
    ch = _channels(cfg)
    st.latex(r"X_i = [\,\tilde x(s_i),\dots,\tilde x(s_i+W-1)\,]\in\mathbb{R}^{W\times F},\qquad "
             r"s_{i+1}-s_i = W-o,\qquad T_W = W/f_s")
    if ch is None:
        interpretation("The first training dataset could not be read — check Data › Training datasets and File type. "
                       f"With the current settings one window spans T_W = {fmt(W / fs)} s.")
        return
    inp, out, uy, (X, y, name) = ch
    n = len(y)
    c1, c2 = st.columns([1, 3])
    with c1:
        o = st.selectbox("Output", out, key='thW|o') if out else None
        Ws = st.number_input("Window W (samples)", 1, max(n - 1, 2), min(W, max(n - 1, 2)), key='thW|W',
                             help="Starts at Parameters › Window length; changing it here does not change the "
                                  "configuration.")
    tau = _tau(y[o].to_numpy(dtype=float), fs) if o else np.nan
    TW = Ws / fs
    step = max(Ws - ov, 1) if Ws == W else 1
    nwin = max((n - Ws) // step + 1, 0)
    with c2:
        tiles([("Window span T_W", f"{fmt(TW)} s", "W / fs: the history one prediction can see."),
               ("Memory of the output τ", f"{fmt(tau)} s" if np.isfinite(tau) else "> data",
                "Lag at which the autocorrelation of the output falls below 1/e (like the time constant of an RC "
                "element)."),
               ("T_W / τ", fmt(TW / tau) if np.isfinite(tau) and tau > 0 else '–',
                "≥ 1: the window covers the main thermal memory; ≪ 1: the model must guess the state."),
               ("Windows", f"{nwin:,}", f"(n − W) / (W − overlap) + 1 with n = {n:,} samples of {name}.")])
    if o:
        s = downsample(n)
        t = np.arange(0, n, s) / fs
        k0 = int(n * 0.4)
        fig = go.Figure([line(t, y[o].iloc[::s], f"{o}", 0)])
        fig.add_vrect(x0=k0 / fs, x1=(k0 + Ws) / fs, fillcolor=SERIES[1], opacity=0.2, line_width=0,
                      annotation_text=f"one window ({fmt(TW)} s)", annotation_position='top left')
        if np.isfinite(tau):
            fig.add_vrect(x0=(k0 + Ws) / fs - tau, x1=(k0 + Ws) / fs, fillcolor=SERIES[2], opacity=0.12,
                          line_width=0, annotation_text=f"τ = {fmt(tau)} s", annotation_position='bottom left')
        fig.update_layout(xaxis_title='t (s)', yaxis_title=f"{o} ({D.unit(uy.get(o))})")
        show(fig, key='thW|fig')
    if np.isfinite(tau) and tau > 0:
        r = TW / tau
        interpretation(f"The output {o} changes with a memory of about τ = {fmt(tau)} s, one window spans "
                       f"{fmt(TW)} s (T_W/τ = {fmt(r)}). "
                       + ("The window covers the thermal memory: the model can infer the state from the inputs."
                          if r >= 1 else
                          f"The window sees only {fmt(100 * r)} % of the memory: the model cannot know how hot the "
                          f"machine already is. Increase W (cost: more weights and memory), down-sample the data, or "
                          f"add rolled EWMA features (Parameters › Features = 2, Model › EWMA spans ≈ "
                          f"{fmt(2 * tau * fs)} samples), which carry the long history at a short window."))
    else:
        interpretation("The autocorrelation of the output does not drop below 1/e within the data: the memory is "
                       "longer than the dataset, or the output follows a slow trend. Rolled EWMA features with long "
                       "spans are then essential.")
    symbols([('X_i', 'i-th input window', '–', 'W × F'), ('s_i', 'start sample of window i', 'samples', '–'),
             ('o', 'overlap of consecutive windows (Parameters › Overlap)', 'samples', 'W − 1'),
             ('F', 'number of input channels (after features)', '–', str(len(inp))),
             ('f_s', 'sampling frequency (Data › fs)', 'Hz', fmt(fs)), ('T_W', 'time span of one window', 's', '–'),
             ('τ', 'memory (time constant) of the output, autocorrelation ρ(τ) = 1/e', 's', '10 … 10⁴ s')])


def _network(cfg):
    solver, model = str(cfg['Par'].get('solver')), str(cfg['Par'].get('model'))
    W = max(_i(cfg['Par'].get('window'), 1), 1)
    ch = _channels(cfg)
    F = len(ch[0]) if ch else max(len(A.to_list(cfg['Dat'].get('inp'))), 1)
    out = ch[1] if ch else [str(o) for o in A.to_list(cfg['Dat'].get('out'))]
    oseq = _i(cfg['Par'].get('outseq'))
    M = (oseq if (len(out) == 1 and oseq >= 1) else max(len(out), 1))
    if solver not in ('DL', 'CL') or model not in KERAS:
        st.markdown(f"The solver **{solver}** with model **{model}** is not a PyTorch network of PyDTS. "
                    + {'SP': "SP identifies a linear state-space model x(k+1) = A x(k) + B u(k), y = C x + D u "
                             "(subspace method, order Model › SS order) or a transfer function — the classic model of a "
                             "thermal RC network.",
                       'ML': "ML fits a random forest, KNN or SVM on the flattened windows or on features.",
                       'SF': "SF uses the forecasting networks of NeuralForecast (LSTM, N-HiTS, N-BEATS) on the "
                             "history of the output."}.get(solver, ""))
        interpretation("Switch the solver to DL to see the network diagram; the Theory & Tools page *Neural "
                       "networks* explains DNN, CNN, LSTM and attention on thermal examples.")
        return
    if _i(cfg['Par'].get('feat')) in (1, 3):
        hint("Frame features are on (Parameters › Features): the real input is the feature vector of each window; "
             "the diagram shows the raw window.")
    try:
        rows, n = keras_summary(model, W, F, M, _i(cfg['Par'].get('method')), _i(cfg['Par'].get('nDim'), 3))
    except Exception as e:
        interpretation(f"The network could not be built for W = {W}, F = {F} ({type(e).__name__}: {e}).")
        return
    nwin = None
    if ch:
        nT = len(ch[3][1]) * (_f(cfg['Dat'].get('rT'), 0.9) if cfg['Exp'].get('method') == 0 else 1)
        nwin = int(max((nT - W) // max(W - _i(cfg['Par'].get('overlap')), 1) + 1, 0))
    tiles([("Weights", f"{n:,}", "Trainable parameters of the network (PyTorch parameters)."),
           ("Layers", f"{len([r for r in rows if r['type'] != 'InputLayer'])}", "Layers with weights or shape changes in src/model/models.py."),
           ("Input", f"{W} × {F}", "Window length × input channels."),
           ("Training windows", f"{nwin:,}" if nwin is not None else '–', "Approximate number of training windows."),
           ("Weights per window", fmt(n / nwin) if nwin else '–',
            "Rule of thumb: well below 1 for a robust model.")])
    c1, c2 = st.columns([3, 2])
    with c1:
        st.graphviz_chart(_arch_dot(rows, W, F, M, ch[0] if ch else [f"x{i + 1}" for i in range(F)], out),
                          width='content')
    with c2:
        table(pd.DataFrame(rows), hide_index=True)
    big = max(rows, key=lambda r: r['weights'])
    interpretation(f"The {model} has {n:,} weights; the largest layer is {big['type']} ({big['weights']:,}, "
                   f"{fmt(100 * big['weights'] / max(n, 1))} %)."
                   + (" In the DNN the first dense layer connects every sample of the window to every neuron, so its "
                      "weights grow with W·F — long windows make the DNN large and prone to overfitting; a CNN or "
                      "LSTM shares its weights over time." if model == 'DNN' else
                      " The convolutions share their filters over time (few weights); the Flatten + Dense head after "
                      "them grows with the window length." if model == 'CNN' else
                      " The LSTM carries a state through the window, its weights do not depend on W." if model == 'LSTM'
                      else "")
                   + (f" With {fmt(n / nwin)} weights per training window the model can memorise the data — compare "
                      f"training and validation loss after training." if nwin and n > nwin else ""))


def _training(cfg):
    st.latex(r"\ell_{\mathrm{mae}} = \frac1B\sum_{i}|\hat y_i-y_i|,\qquad \ell_{\mathrm{mse}} = \frac1B\sum_i(\hat y_i-y_i)^2")
    st.latex(r"m_t = \beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,\quad "
             r"\theta_t=\theta_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}")
    st.markdown("Training stops when the validation loss has not improved for *Patience* epochs and the best weights "
                "are restored (EarlyStopping); the learning rate η is halved after *Patience*/2 epochs without "
                "improvement (ReduceLROnPlateau); every epoch is appended to `training_log.csv` of the model folder.")
    f = os.path.join(A.ROOT, A.model_dir(cfg), 'training_log.csv')
    if os.path.isfile(f):
        try:
            df = pd.read_csv(f)
            ep = np.arange(1, len(df) + 1)
            fig = go.Figure([line(ep, df[c], c, i, mode='lines+markers') for i, c in enumerate(['loss', 'val_loss'])
                             if c in df])
            if 'lr' in df:
                fig.add_trace(line(ep, df['lr'], 'learning rate', 3, dash='dot', yaxis='y2'))
                fig.update_layout(yaxis2=dict(title='η', overlaying='y', side='right', type='log'))
            fig.update_layout(xaxis_title='epoch', yaxis_title='loss', yaxis_type='log')
            show(fig, key='thT|log')
            b = int(df['val_loss'].idxmin()) if 'val_loss' in df else len(df) - 1
            interpretation(f"Stored log of {A.model_dir(cfg)}: {len(df)} epochs, best validation loss at epoch "
                           f"{b + 1}. Steps per epoch = training windows / batch size ({cfg['Mdl'].get('batch')}).")
        except Exception as e:
            interpretation(f"The training log could not be read ({e}).")
    else:
        interpretation(f"No training log for {A.model_dir(cfg)} yet — it is written when the model is trained "
                       f"(Training = 1). With lr = {cfg['Mdl'].get('lr')}, {cfg['Mdl'].get('epoch')} epochs and "
                       f"patience {cfg['Mdl'].get('patience')} the training stops at the latest after "
                       f"{cfg['Mdl'].get('epoch')} epochs.")
    symbols([('B', 'batch size (Model › Batch size)', 'windows', '64 … 1000'),
             ('g_t', 'gradient of the loss w.r.t. the weights in step t', '–', '–'),
             ('η', 'learning rate (Model › Learning rate)', '–', '10⁻³'),
             ('β₁, β₂', 'decay of the first and second moment (Adam)', '–', '0.9, 0.999'),
             ('ε', 'numerical constant', '–', '10⁻⁸')])


def _errors(cfg):
    from src.general.performance import performance
    st.latex(r"\mathrm{MAE}=\frac1N\sum|\hat y-y|,\quad \mathrm{RMSE}=\sqrt{\tfrac1N\sum(\hat y-y)^2},\quad "
             r"R^2 = 1-\frac{\sum(\hat y-y)^2}{\sum(y-\bar y)^2},\quad \mathrm{MAX}=\max|\hat y-y|")
    c1, c2 = st.columns([1, 3])
    with c1:
        bias = st.slider("Bias (K)", -10.0, 10.0, 0.0, 0.5, key='thE|b')
        lag = st.slider("Lag (s)", 0, 300, 0, 10, key='thE|l', help="Prediction delayed against the measurement.")
        noise = st.slider("Noise (K)", 0.0, 5.0, 1.0, 0.25, key='thE|n')
        gain = st.slider("Gain error (%)", -30, 30, 0, 5, key='thE|g', help="ŷ − 40 °C scaled by (1 + gain).")
    t = np.arange(3600.0)
    rng = np.random.default_rng(1)
    P = np.repeat(rng.uniform(0, 1, 36), 100)
    y = 40 + 60 * np.convolve(P, np.exp(-np.arange(1500) / 400.0) / 400.0)[:len(t)]
    yp = 40 + (1 + gain / 100) * (np.r_[np.full(lag, y[0]), y[:len(y) - lag]] - 40) + bias + \
        noise * rng.standard_normal(len(t))
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = performance(yp[:, None], y[:, None], (yp[:, None] > 60) * 1, (y[:, None] > 60) * 1,
                             {'out': ['T']})
    r = dict(zip(['ACC', 'F1', 'R2', 'TECA', 'RMSE', 'MAE', 'MAX'], res[0][:7]))
    with c2:
        tiles([("MAE", f"{fmt(r['MAE'])} K"), ("RMSE", f"{fmt(r['RMSE'])} K"), ("R²", fmt(r['R2'])),
               ("MAX", f"{fmt(r['MAX'])} K"), ("TECA", fmt(r['TECA']))])
        fig = go.Figure([line(t, y, 'measured T', 0, c='#52514e', width=2.5), line(t, yp, 'predicted T̂', 1)])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='T (°C)')
        show(fig, height=330, key='thE|fig')
    worst = max((('bias', abs(bias)), ('lag', lag / 60), ('noise', noise), ('gain', abs(gain) / 5)),
                key=lambda x: x[1])[0]
    interpretation(f"Computed with performance() of PyDTS: MAE {fmt(r['MAE'])} K, RMSE {fmt(r['RMSE'])} K, MAX "
                   f"{fmt(r['MAX'])} K, R² {fmt(r['R2'])}. A constant bias shifts MAE and RMSE equally; noise raises "
                   f"the RMSE more than the MAE; a lag or gain error hits the MAX at the load steps and the hot end "
                   f"— the relevant error for protecting a machine. Dominant effect now: {worst}. R² depends on the "
                   f"range of the data: the same 2 K error gives a high R² on a cycle from 40 to 100 °C and a poor one "
                   f"on an almost constant temperature.")
    symbols([('y, ŷ', 'measured and predicted output', 'K / °C', '–'), ('N', 'test samples', '–', '–'),
             ('ȳ', 'mean of the measured output', '°C', '–'),
             ('TECA', '1 − Σ|ŷ − y| / (2 Σ|y|), energy assigned correctly', '–', '0.9 … 1')])


def _refs(cfg):
    keys = ['Schirmer2024', 'Kirchgaessner2021', 'Kirchgaessner2021b', 'Kirchgaessner2023', 'Goodfellow2016',
            'Hochreiter1997', 'Bai2018', 'Vaswani2017', 'Kingma2015', 'Ljung1999', 'Hyndman2021', 'Raissi2019']
    for k in keys:
        if k in lit.REFS:
            r = lit.REFS[k]
            st.markdown(f"- [{r['label']}]({lit.url(k)}) — {r['cite']}  \n  *{r.get('topic', '')}*")


#######################################################################################################################
# View
#######################################################################################################################
def theory(cfg):
    intro("Theory of the current configuration.")
    T = panel_tabs(TABS, subs=SUBS, titles=TITLES, skip=('Overview', 'References'))
    for tab, fn in zip(T, (_overview, _windows, _network, _training, _errors, _refs)):
        with tab:
            try:
                fn(cfg)
            except Exception as e:                                                   # never let a tab kill the page
                interpretation(f"This part could not be shown for the current configuration ({type(e).__name__}: "
                               f"{e}).")


THEORY = {0: theory, 1: theory, 2: theory}
