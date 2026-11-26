#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Theory & Tools
# File:         nn.py
# Date:         08.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Neural networks: the networks of PyDTS (Parameters › Model) explained on thermal examples; the PINN tab shows the physics-informed extension (TherMOS PI mode) for comparison. One tab per network (dense network
DNN, convolutional network CNN, long short-term memory LSTM, transformer TRA, physics-informed network PINN) and one
each on the data set (initial state, coverage) and on training. Each tab has a sketch, a small live demo that runs in
NumPy (closed-form or least squares, no training loop, so that it reacts instantly), the equations with a symbol table
and an interpretation. The demos show the thermal meaning of each building block: a CNN filter is a truncated impulse
response, the LSTM forget gate is exp(-h/τ), attention weights can form an exponential memory, the physics loss of a
PINN fixes the extrapolation.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core.style import table, SERIES, show, line, fmt
from gui.views.tools import _common as cm
from gui.views.tools.sketch import _t, _l, _r, _arrow, _dot, _svg, show as show_svg

#######################################################################################################################
# Settings
#######################################################################################################################
K = f"{cm.KEY}|nn"
H = 1.0                                                                        # sample time of the demos (s)
NETS = pd.DataFrame([
    ('DNN', 'Dense (fully connected) network', 'every input is connected to every neuron of the next layer',
     'the window flattened to one long vector, without time order', 'simple, fast, universal approximator',
     'weights grow with W·F, no notion of time', 'Model = DNN; dense head of CNN/LSTM'),
    ('CNN', 'Convolutional network', 'one short filter of k samples slides along time (shared weights)',
     'the last R samples (receptive field)', 'few weights, parallel, sharp transients',
     'forgets everything older than R samples', 'Model = CNN'),
    ('LSTM', 'Long short-term memory', 'a memory cell is updated sample by sample; gates decide what is kept',
     'the whole history through its state', 'long thermal memory, variable time constants',
     'sequential (slow on a GPU), harder to train', 'Model = LSTM'),
    ('TRA', 'Transformer (self-attention)', 'every sample weighs every other sample by a learned similarity',
     'the whole window at once', 'long range and parallel', 'cost grows with W², needs much data',
     'Model = TRAN'),
    ('PINN', 'Physics-informed neural network', 'any of the above, trained with an equation residual in the loss',
     'data and the heat balance', 'fewer data, physical extrapolation', 'needs an (approximate) model and loss weights',
     'not in PyDTS (TherMOS PI mode); closest: solver CL'),
], columns=['abbr.', 'network', 'idea', 'what it sees', 'strength', 'weakness', 'in PyDTS'])
NET_HELP = {'abbr.': "Abbreviation used in this page and in the mode names.",
            'idea': "How the network processes a window of inputs.",
            'what it sees': "Which part of the input history can influence the predicted temperature.",
            'in PyDTS': "Where to select it (Parameters of the Simulation mode)."}


#######################################################################################################################
# Thermal reference used by all demos
#######################################################################################################################
def _profile(n=900, seed=3):
    """Loss profile P (W): piecewise constant segments of 10–60 s with 0–80 W, sample time H."""
    rng = np.random.default_rng(seed)
    P, k = np.zeros(n), 0
    while k < n:
        L = int(rng.uniform(10, 60))
        P[k:k + L] = rng.uniform(0, 80)
        k += L
    return P


def _foster(P, R, tau):
    """Exact discrete response of a Foster network (zero-order hold of P): θ = Σ θ_i, θ_i,k = a θ_i,k-1 + R(1-a)P_k."""
    out = np.zeros(len(P))
    for Ri, ti in zip(R, tau):
        a, x = np.exp(-H / ti), 0.0
        for k, p in enumerate(P):
            x = a * x + Ri * (1 - a) * p
            out[k] += x
    return out


def _impulse(R, tau, n):
    j = np.arange(n)
    return sum(Ri * (1 - np.exp(-H / ti)) * np.exp(-H / ti) ** j for Ri, ti in zip(R, tau))


def _fir(P, h):
    return np.convolve(P, h)[0:len(P)]


#######################################################################################################################
# Sketches
#######################################################################################################################
def _c(x, y, r, fill='none', col='currentColor', sw=1.2):
    return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{fill}' stroke='{col}' stroke-width='{sw}'/>"


def _sk_dnn():
    L = [3, 5, 5, 1]
    xs = [70, 230, 390, 550]
    b, pos = [], []
    for i, n in enumerate(L):
        pos.append([(xs[i], 110 + (k - (n - 1) / 2) * 40) for k in range(n)])
    for i in range(len(L) - 1):
        for p in pos[i]:
            for q in pos[i + 1]:
                b.append(_l(p[0] + 12, p[1], q[0] - 12, q[1], op=0.35))
    names = ['P(t−W+1)', '⋮', 'P(t)']
    for i, ps in enumerate(pos):
        for k, (x, y) in enumerate(ps):
            b.append(_c(x, y, 12, fill=[SERIES[0], SERIES[2], SERIES[2], SERIES[1]][i], col='none'))
            if i == 0:
                b.append(_t(x - 18, y + 4, names[k], 'end', 10))
    b.append(_t(570, 114, 'θ̂(t)', 'start', 11))
    for x, s in zip(xs, ['inputs x', 'hidden layer 1', 'hidden layer 2', 'output']):
        b.append(_t(x, 18, s, 'middle', 10))
    b.append(_r(615, 70, 140, 80, col=SERIES[2], dash='4 3'))
    b += [_t(685, 92, 'one neuron', 'middle', 10), _t(685, 114, 'z = Σ wᵢxᵢ + b', 'middle', 10),
          _t(685, 136, 'y = φ(z)', 'middle', 10)]
    return _svg(''.join(b), 770, 215)


def _sk_cnn(k, L, dil):
    """Receptive field of the last output of L stacked convolutions (kernel k, dilation 2^l or 1)."""
    d = [2 ** l if dil else 1 for l in range(L)]
    R = 1 + (k - 1) * sum(d)
    n = min(R + 2, 40)
    dx = 720 / n
    ys = [190 - l * 150 / max(L, 1) for l in range(L + 1)]
    b, act = [], {L: {n - 1}}
    for l in range(L, 0, -1):
        act[l - 1] = set()
        for j in act[l]:
            for m in range(k):
                i = j - m * d[l - 1]
                if i >= 0:
                    act[l - 1].add(i)
                    b.append(_l(30 + j * dx, ys[l] + 5, 30 + i * dx, ys[l - 1] - 5, col=SERIES[0], op=0.7))
    for l in range(L + 1):
        for j in range(n):
            on = j in act.get(l, ())
            b.append(_c(30 + j * dx, ys[l], 4.5, fill=(SERIES[1] if l == L else SERIES[0]) if on else 'none',
                        col='currentColor' if not on else 'none', sw=0.6))
        b.append(_t(8, ys[l] + 4, 'P' if l == 0 else f'{l}', 'start', 10))
    b.append(_t(30 + (n - 1) * dx, ys[L] - 12, 'θ̂(t)', 'middle', 10))
    b.append(_t(380, 214, f"receptive field R = {R} samples" + (" (drawing cut at 40)" if R + 2 > 40 else ''),
                'middle', 11))
    return _svg(''.join(b), 770, 222)


def _sk_lstm():
    b = [_r(140, 30, 470, 160, col='currentColor', sw=1.2), _t(375, 22, 'LSTM cell at time step t', 'middle', 11)]
    b += [_arrow(40, 70, 140, 70), _t(40, 62, 'c(t−1)', 'start', 10), _l(140, 70, 610, 70, w=2.2),
          _arrow(610, 70, 720, 70), _t(720, 62, 'c(t) (memory)', 'end', 10)]
    for x, s in ((230, '×'), (360, '+')):
        b += [_c(x, 70, 13, fill='none', col=SERIES[1], sw=2), _t(x, 75, s, 'middle', 14, SERIES[1])]
    gates = [(230, 'f', 'forget', SERIES[0]), (330, 'i', 'input', SERIES[2]), (400, 'g', 'candidate', SERIES[3]),
             (520, 'o', 'output', SERIES[4])]
    for x, s, n, col in gates:
        b += [_r(x - 22, 130, 44, 28, fill=col, col='none', op=0.85), _t(x, 149, s, 'middle', 12, '#fff'),
              _t(x, 176, n, 'middle', 9)]
    b += [_arrow(230, 130, 230, 84), _l(330, 130, 360, 84), _l(400, 130, 360, 84), _arrow(360, 90, 360, 84)]
    b += [_arrow(40, 214, 140, 214), _t(40, 206, 'x(t) = P(t), h(t−1)', 'start', 10), _l(140, 214, 520, 214),
          _l(230, 214, 230, 158), _l(330, 214, 330, 158), _l(400, 214, 400, 158), _l(520, 214, 520, 158)]
    b += [_c(560, 110, 13, col=SERIES[4], sw=2), _t(560, 114, '×', 'middle', 13, SERIES[4]), _l(560, 70, 560, 97),
          _l(520, 130, 552, 120), _arrow(573, 110, 720, 110), _t(720, 102, 'h(t) → θ̂(t)', 'end', 10)]
    return _svg(''.join(b), 770, 230)


def _sk_tra(alpha, n=20):
    a = alpha[-n:]
    a = a / max(a.max(), 1e-12)
    dx = 700 / n
    b = []
    for j in range(n):
        x = 40 + j * dx
        b.append(_r(x - 12, 150, 24, 24, fill=SERIES[0], col='none', op=0.25 + 0.75 * a[j]))
        if j < n - 1:
            xq = 40 + (n - 1) * dx
            b.append(f"<path d='M{xq:.1f},146 Q{(x + xq) / 2:.1f},{146 - 30 - 90 * (n - 1 - j) / n:.1f} "
                     f"{x:.1f},146' fill='none' stroke='{SERIES[1]}' stroke-width='{0.4 + 3.5 * a[j]:.2f}' "
                     f"stroke-opacity='{0.15 + 0.85 * a[j]:.2f}'/>")
    b.append(_r(40 + (n - 1) * dx - 12, 150, 24, 24, fill='none', col=SERIES[1], sw=2.5))
    b += [_t(40, 196, f'P(t−{n - 1})', 'middle', 10), _t(40 + (n - 1) * dx, 196, 'P(t) = query', 'middle', 10),
          _t(380, 20, 'attention of the last sample on the window (line width and box shade = weight αⱼ)',
             'middle', 11)]
    return _svg(''.join(b), 770, 206)


def _sk_pinn():
    b = [_r(20, 85, 110, 50, fill=SERIES[0], col='none', op=0.2), _t(75, 106, 'inputs', 'middle', 11),
         _t(75, 122, 't, P(t)', 'middle', 10), _arrow(130, 110, 180, 110),
         _r(180, 80, 120, 60, fill=SERIES[2], col='none', op=0.3), _t(240, 106, 'network', 'middle', 11),
         _t(240, 124, 'DNN/CNN/LSTM', 'middle', 9), _arrow(300, 110, 350, 110), _t(372, 114, 'θ̂', 'middle', 13),
         _arrow(390, 100, 450, 50), _arrow(390, 120, 450, 170),
         _r(450, 25, 200, 50, fill=SERIES[1], col='none', op=0.25), _t(550, 46, 'data loss', 'middle', 11),
         _t(550, 64, 'mean (θ̂ − θ_meas)²', 'middle', 10),
         _r(450, 145, 200, 50, fill=SERIES[3], col='none', op=0.25), _t(550, 166, 'physics loss', 'middle', 11),
         _t(550, 184, 'mean (C dθ̂/dt + θ̂/R − P)²', 'middle', 10),
         _arrow(650, 50, 700, 100), _arrow(650, 170, 700, 120), _c(715, 110, 15, col='currentColor', sw=1.5),
         _t(715, 115, 'Σ', 'middle', 13), _t(690, 152, '× α', 'end', 10),
         f"<path d='M715,126 Q715,230 240,215 L240,142' fill='none' stroke='{SERIES[4]}' stroke-width='1.4' "
         f"stroke-dasharray='5 3'/>", _t(470, 228, 'gradient of the total loss updates the weights', 'middle', 10)]
    return _svg(''.join(b), 770, 240)


#######################################################################################################################
# Tabs
#######################################################################################################################
def _tab_dnn():
    st.markdown("A **neuron** forms a weighted sum of its inputs plus a bias and passes it through a non-linear "
                "**activation** φ. A **dense network** (DNN, also multilayer perceptron) stacks layers of such "
                "neurons; every neuron is connected to all outputs of the previous layer. With enough neurons it "
                "can approximate any smooth function (universal approximation). For a time series the window of "
                "W samples × F features is simply flattened into one input vector — the network does not know "
                "which sample is older.")
    show_svg(_sk_dnn(), "Dense network with a window of losses as inputs, two hidden layers and one output; the box "
                        "shows what one neuron computes.")
    c = st.columns(3)
    N = int(c[0].slider("Hidden ReLU neurons N", 2, 40, 6, key=f"{K}|dn"))
    act = c[1].selectbox("Activation shown", ['ReLU', 'tanh', 'sigmoid', 'GELU'], key=f"{K}|act")
    W = int(c[2].slider("Window W for the size estimate", 10, 2000, 300, step=10, key=f"{K}|dW"))

    # Z_th of a 3-term Foster network over log10(t); one hidden ReLU layer with kinks at evenly spaced positions,
    # output layer by least squares (the hidden layer is fixed so that the demo needs no training loop)
    R, tau = np.array([0.05, 0.15, 0.3]), np.array([0.01, 0.5, 20.0])
    x = np.linspace(-3, 2, 400)
    Z = (R[None, :] * (1 - np.exp(-10 ** x[:, None] / tau[None, :]))).sum(1)
    kinks = np.linspace(x[0], x[-1], N + 1)[0:-1]
    Phi = np.column_stack([np.ones_like(x)] + [np.maximum(x - b, 0) for b in kinks])
    w = np.linalg.lstsq(Phi, Z, rcond=None)[0]
    Zh = Phi @ w
    err = np.abs(Zh - Z).max()
    p_dnn = 3 * N + 1
    p_win = W * 64 + 64 + 65
    cm.tiles([("Neurons N", f"{N}", "Hidden neurons with ReLU activation."),
              ("Weights", f"{p_dnn}", "Input weight + bias per neuron, output weight per neuron, output bias."),
              ("Max. error", f"{fmt(err * 1e3)} mK/W", f"{100 * err / R.sum():.1f} % of R_th = {R.sum():.2f} K/W"),
              ("Dense layer on a window", f"{p_win:,} weights", f"W = {W} samples, 1 feature, 64 neurons: "
                                                                 "W·64 + 64 + 65.")])
    t1, t2 = st.tabs(["Can a small network learn a Z_th curve?", "What does the activation do?"])
    with t1:
        fig = go.Figure([line(10 ** x, Z, 'Z_th (3-term Foster)', 0, width=3),
                         line(10 ** x, Zh, f'DNN, {N} ReLU neurons', 1, dash='dot')])
        for bb in kinks:
            fig.add_vline(x=10 ** bb, line=dict(color='gray', width=0.6, dash='dot'))
        fig.update_layout(xaxis=dict(type='log', title='t (s)'), yaxis_title='Z_th (K/W)')
        show(fig, key=f"{K}|dnnfit", info="Thick line: thermal impedance Z_th(t) of a 3-term Foster network "
             "(time constants 10 ms, 0.5 s, 20 s). Dotted: the network output. Grey lines: kink positions of the "
             "ReLU neurons. Input of the network: log10(t).")
        st.caption(f"With N = {N} neurons the network is a line with {N} kinks; the largest deviation is "
                   f"{fmt(err * 1e3)} mK/W ({100 * err / R.sum():.1f} % of the total {R.sum():.2f} K/W). "
                   "More neurons → more kinks → smaller error, but also more weights to learn from data.")
    with t2:
        z = np.linspace(-4, 4, 200)
        from math import erf
        f = {'ReLU': np.maximum(z, 0), 'tanh': np.tanh(z), 'sigmoid': 1 / (1 + np.exp(-z)),
             'GELU': z * 0.5 * (1 + np.array([erf(v / np.sqrt(2)) for v in z]))}
        fig = go.Figure([line(z, f[n], n, i, width=3 if n == act else 1.2) for i, n in enumerate(f)])
        fig.update_layout(xaxis_title='z (weighted sum)', yaxis_title='φ(z)')
        show(fig, key=f"{K}|actfig", info="Activation functions: output of a neuron over its weighted input sum z. "
             "The selected one is drawn thick.")
        st.caption({'ReLU': "ReLU = max(0, z): cheap, no saturation for z > 0; used in all hidden layers of the PyDTS networks.",
                    'tanh': "tanh: bounded to ±1, used inside the LSTM for the candidate and the output.",
                    'sigmoid': "sigmoid: 0…1, used for the gates of the LSTM (0 = closed, 1 = open).",
                    'GELU': "GELU: smooth ReLU, standard in transformers."}[act] +
                   " Without a non-linear φ, any number of layers collapses to one linear map.")
    cm.equations([
        ("Neuron", r"y = \varphi\!\left(\sum_{i=1}^{n} w_i x_i + b\right)", "Weighted sum and activation."),
        ("Layer", r"\mathbf{h}^{(l)} = \varphi\!\left(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right),"
                  r"\quad \mathbf{h}^{(0)} = \mathbf{x}", "All neurons of layer l at once."),
        ("Weights of a dense layer", r"n_\mathrm{par} = n_\mathrm{in}\,n_\mathrm{out} + n_\mathrm{out}",
         "For a window: n_in = W·F, so the input layer grows linearly with the window length."),
    ], sym=[('x, xᵢ', 'inputs (window of losses, currents, coolant temperature, …)', 'scaled, –'),
            ('wᵢ, W⁽ˡ⁾', 'weights of one neuron / weight matrix of layer l (learned)', '–'),
            ('b, b⁽ˡ⁾', 'bias (learned offset)', '–'), ('φ', 'activation function', '–'),
            ('h⁽ˡ⁾', 'outputs of layer l (hidden features)', '–'), ('l', 'layer index 1 … number of layers', '–'),
            ('W, F', 'window length (samples) and number of input features', '–'),
            ('θ̂', 'predicted temperature (rise)', 'K')])
    cm.interpret([
        f"A {N}-neuron network reproduces a three-time-constant Z_th curve within {100 * err / R.sum():.1f} %. "
        "Dense networks are therefore good **heads**: they map extracted features to temperatures.",
        f"As the first layer on a raw window they are wasteful: W = {W} samples already need {p_win:,} weights in one "
        "layer of 64 neurons, and the network must learn separately that a loss 10 s ago acts like one 11 s ago. "
        "CNN, LSTM and transformer build that time structure in and need far fewer data."])


def _tab_cnn(P, Rf, tf):
    st.markdown("A **1D convolution** slides a short filter of k weights along time and computes at every step a "
                "weighted sum of the last k samples — the same weights everywhere (weight sharing). Stacking L "
                "layers widens the view; **dilation** (gaps of 1, 2, 4, … samples between the taps) widens it "
                "exponentially. The number of past samples that can influence one output is the **receptive "
                "field** R. Thermally, a linear filter of length R is a **truncated impulse response**: "
                "θ(t) = Σ h(j)·P(t−j). The network can only be exact if R covers the slowest time constant.")
    c = st.columns(3)
    k = int(c[0].slider("Kernel k (samples)", 2, 9, 3, key=f"{K}|ck"))
    L = int(c[1].slider("Layers L", 1, 8, 4, key=f"{K}|cL"))
    dil = c[2].toggle("Dilation 1, 2, 4, …", True, key=f"{K}|cd")
    Rf_ = 1 + (k - 1) * sum(2 ** l if dil else 1 for l in range(L))
    show_svg(_sk_cnn(k, L, dil), "Blue: samples and hidden values that feed the last output (orange). Each line is "
                                 "one filter tap.")
    h_full = _impulse(Rf, tf, len(P))
    th = _fir(P, h_full)
    thc = _fir(P, h_full[0:Rf_])
    err = np.abs(thc - th).max()
    gain = h_full[0:Rf_].sum() / h_full.sum()
    cm.tiles([("Receptive field R", f"{Rf_} samples", f"= {Rf_ * H:g} s at h = {H:g} s"),
              ("Static gain captured", f"{100 * gain:.1f} %", "Share of R_th = Σh(j) that lies inside R."),
              ("Max. error", f"{fmt(err)} K", f"{100 * err / th.max():.1f} % of the peak {fmt(th.max())} K"),
              ("Slowest τ", f"{tf.max():g} s", "≈ 3–5 τ of receptive field are needed.")])
    t1, t2 = st.tabs(["How much of the thermal memory does the CNN see?", "What does the truncation cost?"])
    with t1:
        j = np.arange(min(len(P), int(6 * tf.max())))
        fig = go.Figure([line(j * H, h_full[j], 'impulse response h(j)', 0, width=2)])
        fig.add_vrect(x0=0, x1=Rf_ * H, fillcolor=SERIES[1], opacity=0.15, line_width=0,
                      annotation_text=f"receptive field {Rf_} s", annotation_position='top right')
        fig.update_layout(xaxis_title='age of the loss sample j·h (s)', yaxis_title='h(j) (K/W per sample)')
        show(fig, key=f"{K}|cimp", info="Blue: weight with which a loss sample of age j·h acts on the temperature "
             "now (impulse response of the Foster network). Orange area: what the CNN can see.")
        st.caption(f"The receptive field of {Rf_} s holds {100 * gain:.1f} % of the area under h(j), i.e. of the "
                   f"thermal resistance {Rf.sum():.2f} K/W; the rest (slow heating) is invisible to the CNN.")
    with t2:
        t = np.arange(len(P)) * H
        fig = go.Figure([line(t, th, 'exact θ', 0, width=3), line(t, thc, f'best CNN with R = {Rf_}', 1, dash='dot')])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='θ (K)')
        show(fig, key=f"{K}|cres", info="Thick: exact temperature rise for the random loss profile. Dotted: the "
             "best linear CNN with the current receptive field (the true impulse response cut at R).")
        st.caption(f"Maximum deviation {fmt(err)} K: the CNN follows the fast pulses but "
                   f"{'misses' if gain < 0.95 else 'also captures'} the slow drift of the τ = {tf.max():g} s term.")
    cm.equations([
        ("Convolution layer", r"y_c(t) = \varphi\!\left(\sum_{f=1}^{F}\sum_{m=0}^{k-1} w_{c,f,m}\,"
                              r"x_f(t - m\,d) + b_c\right)", "Filter c, dilation d, causal (only past samples)."),
        ("Receptive field", r"R = 1 + (k-1)\sum_{l=1}^{L} d_l,\quad d_l = 2^{\,l-1}\;\Rightarrow\; "
                            r"R = 1 + (k-1)(2^L-1)", "Dilation doubles the view with every layer."),
        ("Thermal meaning", r"\theta(t) = \sum_{j=0}^{\infty} h(j)\,P(t-j) \approx \sum_{j=0}^{R-1} h(j)\,P(t-j)",
         "A linear CNN is a finite impulse response; the missing tail is the error."),
    ], sym=[('x_f, y_c', 'input feature f, output channel c (filter c)', '–'),
            ('w_c,f,m, b_c', 'filter weights and bias (learned, shared over time)', '–'),
            ('k', 'kernel size (taps per filter)', 'samples'), ('d, d_l', 'dilation of layer l', 'samples'),
            ('L', 'number of convolution layers', '–'), ('R', 'receptive field', 'samples'),
            ('h(j)', 'impulse response: temperature per watt of a loss sample j steps ago', 'K/W'),
            ('m, j', 'tap index 0 … k−1, age of a sample', '–')])
    cm.interpret([
        f"With k = {k}, L = {L} and {'dilation' if dil else 'no dilation'} the CNN sees {Rf_} s and captures "
        f"{100 * gain:.1f} % of the thermal resistance; the error on the profile is {fmt(err)} K.",
        "Choose the receptive field (and the window W) ≥ 3 τ of the slowest time constant that matters; dilation "
        "achieves this with few layers. Anything slower (heat sink, coolant) must come in as an input (e.g. a "
        "measured case or coolant temperature) or be handled by an LSTM — that is the idea of combining CNN and LSTM layers."])


def _tab_lstm(P, Rf, tf):
    st.markdown("A **recurrent** network reads the window sample by sample and carries a state. The **LSTM** has "
                "a memory cell c and three **gates** (values 0…1): the forget gate f keeps a share of the old "
                "memory, the input gate i lets new information in, the output gate o decides what is passed on. "
                "Thermally the cell update c(t) = f·c(t−1) + i·g is the discrete RC equation: with a constant "
                "f = exp(−h/τ) one cell is exactly one Foster term. Because f depends on the inputs, an LSTM can "
                "also change its time constant (e.g. with coolant flow).")
    show_svg(_sk_lstm(), "Memory c runs along the top; the gates (σ: sigmoid, 0…1) scale it (×) and add (+) new "
                         "candidates g. The hidden output h(t) feeds the dense head that predicts θ̂.")
    tau_true = float(tf.max())
    c = st.columns(2)
    f = c[0].slider("Forget gate f (constant)", 0.80, 0.999, float(np.round(np.exp(-H / tau_true), 3)),
                    step=0.001, format='%.3f', key=f"{K}|lf")
    tau_c = -H / np.log(f)
    c[1].metric("Equivalent time constant τ = −h / ln f", f"{fmt(tau_c)} s", help=f"Exact value for the slow term: "
                f"f = exp(−h/τ) = {np.exp(-H / tau_true):.4f}.")
    th = _foster(P, Rf[-1:], tf[-1:])
    cc, out = 0.0, np.zeros(len(P))
    for kk, p in enumerate(P):
        cc = f * cc + Rf[-1] * (1 - f) * p                                     # i·g = R(1 − f)·P (correct gain)
        out[kk] = cc
    err = np.abs(out - th).max()
    n = np.arange(0, int(5 * max(tau_c, tau_true)))
    half = np.log(0.5) / np.log(f)
    cm.tiles([("Cell time constant", f"{fmt(tau_c)} s", "τ = −h / ln f"),
              ("Memory half-life", f"{fmt(half * H)} s", "Steps after which f^n = 0.5."),
              ("Max. error vs. RC", f"{fmt(err)} K", f"Slow Foster term R = {Rf[-1]} K/W, τ = {tau_true:g} s.")])
    t1, t2 = st.tabs(["Does one LSTM cell reproduce an RC term?", "How long does the cell remember?"])
    with t1:
        t = np.arange(len(P)) * H
        fig = go.Figure([line(t, th, f'RC term, τ = {tau_true:g} s', 0, width=3),
                         line(t, out, f'LSTM cell, f = {f:.3f}', 1, dash='dot')])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='θ (K)')
        show(fig, key=f"{K}|lres", info="Thick: slow Foster term driven by the loss profile. Dotted: one LSTM "
             "memory cell with a constant forget gate f and the input scaled for the correct static gain.")
        st.caption(f"With f = {f:.3f} the cell behaves like τ = {fmt(tau_c)} s instead of {tau_true:g} s; the "
                   f"maximum deviation is {fmt(err)} K. Training finds f (through its weights) from data.")
    with t2:
        fig = go.Figure([line(n * H, f ** n, f'f^n, f = {f:.3f}', 1),
                         line(n * H, np.exp(-n * H / tau_true), f'exp(−t/τ), τ = {tau_true:g} s', 0, dash='dot')])
        fig.update_layout(xaxis_title='age (s)', yaxis_title='share of the memory kept')
        show(fig, key=f"{K}|lmem", info="Share of an old input that is still in the memory cell after n steps.")
        st.caption(f"After {fmt(half * H)} s half of an input is forgotten. A gate value only 0.01 below 1 "
                   "already shortens the memory considerably — which is why LSTMs are initialised with a forget "
                   "bias that keeps f close to 1.")
    cm.equations([
        ("Gates", [r"f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f),\quad i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)",
                   r"o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o),\quad g_t = \tanh(W_g x_t + U_g h_{t-1} + b_g)"],
         "All gates see the current input and the previous output."),
        ("Memory and output", r"c_t = f_t \odot c_{t-1} + i_t \odot g_t,\qquad h_t = o_t \odot \tanh(c_t)",
         "⊙: element-wise product."),
        ("Thermal analogue", r"\theta_t = e^{-h/\tau}\,\theta_{t-1} + R\,(1-e^{-h/\tau})\,P_t \;\Leftrightarrow\; "
                             r"f = e^{-h/\tau},\; i\,g = R(1-f)P", "Discrete first-order RC (zero-order hold)."),
        ("Weights", r"n_\mathrm{par} = 4\,(H_u(F+H_u) + H_u)", "Four gate blocks; independent of the window W."),
    ], sym=[('x_t', 'input vector at step t (losses, currents, …)', '–'), ('h_t', 'hidden output (H_u values)', '–'),
            ('c_t', 'memory cell state', '–'), ('f, i, o', 'forget, input, output gate (0 … 1)', '–'),
            ('g_t', 'candidate value (−1 … 1)', '–'), ('W, U, b', 'input weights, recurrent weights, biases', '–'),
            ('σ', 'sigmoid function', '–'), ('H_u, F', 'hidden units, input features', '–'),
            ('h, τ, R', 'sample time, time constant, thermal resistance of the RC term', 's, s, K/W'),
            ('t', 'time step index', '–')])
    cm.interpret([
        f"One LSTM cell with f = {f:.3f} acts like an RC term with τ = {fmt(tau_c)} s (error {fmt(err)} K for the "
        f"{tau_true:g} s term). An LSTM with H_u units can thus represent H_u time constants and, through "
        "input-dependent gates, time constants that change with the operating point.",
        "The memory does not depend on the window length, so LSTMs suit slow thermal paths; the price is sequential "
        "computation (slower training on a GPU) and vanishing gradients over very long windows. GRU (layer type "
        "gru) merges the gates and has 25 % fewer weights."])


def _tab_tra(P, Rf, tf):
    st.markdown("A **transformer** layer lets every sample of the window look at every other one. Each sample is "
                "mapped to a **query** q, a **key** k and a **value** v. The similarity q·k, scaled and passed "
                "through a softmax, gives **attention weights** α (positive, sum 1); the output is the weighted sum "
                "of the values. A **position code** is added so the network knows the age of each sample. Several "
                "**heads** attend in parallel to different patterns. If the score falls linearly with the age, "
                "the weights decay exponentially — the attention then forms a thermal memory.")
    c = st.columns(2)
    tau_a = c[0].slider("Score slope → memory τ_a (s)", 2.0, 200.0, float(tf.max()), step=1.0, key=f"{K}|ta")
    W = int(c[1].slider("Window W (samples)", 20, 600, 200, step=10, key=f"{K}|tW"))
    age = np.arange(W)[::-1] * H                                               # age of the window samples
    s = -age / tau_a                                                           # score of the last-sample query
    alpha = np.exp(s - s.max())
    alpha /= alpha.sum()
    show_svg(_sk_tra(alpha), f"Last 20 of the W = {W} samples; the query (orange box) is the newest sample.")
    th = _foster(P, Rf[-1:], tf[-1:])
    Pp = np.concatenate([np.zeros(W - 1), P])
    out = np.array([Rf[-1] * (alpha @ Pp[kk:kk + W]) for kk in range(len(P))])
    err = np.abs(out - th).max()
    cov = 1 - np.exp(-W * H / tau_a)
    cm.tiles([("Weight of the newest sample", f"{100 * alpha[-1]:.2f} %", None),
              ("Weight older than τ_a", f"{100 * alpha[age > tau_a].sum():.1f} %", None),
              ("Max. error vs. RC", f"{fmt(err)} K", f"Slow Foster term τ = {tf.max():g} s."),
              ("Window / τ_a", f"{W * H / tau_a:.1f}", f"The exponential is cut at W: {100 * cov:.1f} % covered.")])
    t1, t2 = st.tabs(["Can attention form a thermal memory?", "What does attention cost compared to an LSTM?"])
    with t1:
        t = np.arange(len(P)) * H
        fig = go.Figure([line(t, th, f'RC term, τ = {tf.max():g} s', 0, width=3),
                         line(t, out, f'attention, τ_a = {tau_a:g} s, W = {W}', 1, dash='dot')])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='θ (K)')
        show(fig, key=f"{K}|tres", info="Thick: slow Foster term. Dotted: one attention head whose scores fall "
             "linearly with the age of the sample (hand-set, not trained), output scaled by R.")
        st.caption(f"Softmax weights renormalise over the window: with W / τ_a = {W * H / tau_a:.1f} the output "
                   f"{'matches the RC term closely' if err < 0.05 * th.max() else 'deviates'} (max. {fmt(err)} K). "
                   "A trained transformer learns such patterns, plus content-dependent ones, from data.")
    with t2:
        Ws = np.arange(50, 4001, 50)
        Fi, d = 4, 64
        lstm = Ws * 4 * d * (Fi + d)
        tra = Ws * (Fi * d + 8 * d * d) + 2 * Ws ** 2 * d
        fig = go.Figure([line(Ws, lstm / 1e6, 'LSTM, 64 units', 0), line(Ws, tra / 1e6, 'transformer, d = 64', 1)])
        fig.add_vline(x=W, line=dict(color='gray', dash='dot'))
        fig.update_layout(xaxis_title='window W (samples)', yaxis_title='multiply-adds per window (millions)')
        show(fig, key=f"{K}|tcost", info="Approximate arithmetic per window for one layer with 64 units and 4 input "
             "features. The LSTM grows linearly with W but must run sequentially; attention grows with W² but "
             "runs in parallel.")
        i = int(np.argmin(np.abs(Ws - W)))
        st.caption(f"At W = {W}: LSTM {lstm[i] / 1e6:.1f} M, transformer {tra[i] / 1e6:.1f} M multiply-adds. For "
                   "long thermal windows, sub-sample the input or put a CNN with stride in front of the transformer.")
    cm.equations([
        ("Query, key, value", r"Q = X W_Q,\quad K = X W_K,\quad V = X W_V", "Linear maps of the window X (W × d)."),
        ("Attention", r"\mathrm{Att}(Q,K,V) = \mathrm{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V,\qquad "
                      r"\alpha_{tj} = \frac{e^{s_{tj}}}{\sum_m e^{s_{tm}}}", "Each row of weights sums to 1."),
        ("Thermal analogue", r"s_{tj} = -\frac{(t-j)h}{\tau_a} \;\Rightarrow\; \alpha_{tj} \propto "
                             r"e^{-(t-j)h/\tau_a}", "A score linear in the age gives an exponential memory."),
        ("Encoder layer", r"X' = X + \mathrm{MHA}(\mathrm{LN}(X)),\quad X'' = X' + \mathrm{FFN}(\mathrm{LN}(X'))",
         "Pre-norm with residual connections; PyDTS TRAN: embedding 32, 2 heads, feed-forward 32."),
    ], sym=[('X', 'window of embedded inputs plus position code', '–'), ('W_Q, W_K, W_V', 'learned projections', '–'),
            ('d, d_k', 'model width (units), width per head', '–'), ('s_tj, α_tj', 'score and attention weight of '
             'sample j for query t', '–'), ('τ_a', 'memory time constant formed by the attention', 's'),
            ('MHA, LN, FFN', 'multi-head attention, layer normalisation, feed-forward network', '–'),
            ('W', 'window length', 'samples')])
    cm.interpret([
        f"With a score slope of 1/{tau_a:g} s the attention reproduces an RC memory (error {fmt(err)} K for W = {W}); "
        f"{100 * alpha[age > tau_a].sum():.1f} % of the weight lies on samples older than τ_a.",
        "Transformers are strongest when the relevant history is long and irregular (e.g. drive cycles with "
        "repeated events) and plenty of data is available. For smooth thermal dynamics an LSTM or CNN is usually as "
        "accurate with fewer weights; keep W moderate because the cost grows with W²."])


def _tab_pinn():
    st.markdown("A **physics-informed neural network** (PINN) is trained on two losses: the usual **data loss** "
                "(prediction vs. measurement) and a **physics loss**, the residual of the governing equation "
                "evaluated at **collocation points** (any time, no measurement needed). Here the equation is the "
                "RC heat balance C dθ/dt + θ/R − P = 0. The weight α balances both. The physics fills the gaps "
                "between and beyond the measurements and keeps the model plausible when it extrapolates.")
    show_svg(_sk_pinn(), "Both losses use the same prediction θ̂; dθ̂/dt comes from automatic differentiation.")
    c = st.columns(3)
    a_log = c[0].slider("Physics weight log₁₀ α", -4.0, 3.0, 0.0, step=0.5, key=f"{K}|pa")
    nd = int(c[1].slider("Measurements (first 150 s)", 3, 30, 8, key=f"{K}|pn"))
    sig = c[2].slider("Noise σ (K)", 0.0, 2.0, 0.5, step=0.1, key=f"{K}|ps")
    alpha = 10 ** a_log
    R, C = 0.5, 120.0                                                          # τ = 60 s
    t = np.linspace(0, 600, 601)
    P = np.where(t < 200, 40.0, np.where(t < 400, 10.0, 60.0))
    th = _foster(P, [R], [R * C])
    rng = np.random.default_rng(7)
    td = np.sort(rng.uniform(0, 150, nd))
    yd = np.interp(td, t, th) + sig * rng.standard_normal(nd)
    cen, wd = np.linspace(-40, 640, 35), 30.0                                 # Gaussian basis (stands for the net)

    def basis(x, der=False):
        g = np.exp(-0.5 * ((x[:, None] - cen[None, :]) / wd) ** 2)
        return -g * (x[:, None] - cen[None, :]) / wd ** 2 if der else g

    def fit(al):
        sT, sP = th.max(), P.max()
        A = [basis(td) / sT / np.sqrt(nd)]
        b = [yd / sT / np.sqrt(nd)]
        if al > 0:
            A.append(np.sqrt(al / len(t)) * (C * basis(t, True) + basis(t) / R) / sP)
            b.append(np.sqrt(al / len(t)) * P / sP)
        A.append(1e-4 * np.eye(len(cen)))
        b.append(np.zeros(len(cen)))
        return basis(t) @ np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0]

    y0, ya = fit(0.0), fit(alpha)
    m = t <= 150
    e = lambda y, s: np.sqrt(np.mean((y[s] - th[s]) ** 2))
    cm.tiles([("RMSE in the data range", f"{fmt(e(ya, m))} K", f"0 … 150 s; data only: {fmt(e(y0, m))} K"),
              ("RMSE extrapolated", f"{fmt(e(ya, ~m))} K", f"150 … 600 s; data only: {fmt(e(y0, ~m))} K"),
              ("Physics weight α", f"{alpha:g}", None), ("Measurements", f"{nd}", f"noise σ = {sig:g} K")])
    fig = go.Figure([line(t, th, 'true θ (RC, τ = 60 s)', 0, width=3), line(t, y0, 'data loss only', 7, dash='dot'),
                     line(t, ya, f'data + physics, α = {alpha:g}', 1, dash='dash'),
                     go.Scatter(x=td, y=yd, mode='markers', name='measurements',
                                marker=dict(color=SERIES[2], size=8))])
    fig.add_vrect(x0=0, x1=150, fillcolor=SERIES[2], opacity=0.08, line_width=0, annotation_text='data range')
    fig.update_layout(xaxis_title='t (s)', yaxis_title='θ (K)', yaxis_range=[-5, max(40, 1.3 * th.max())])
    show(fig, key=f"{K}|pfit", info="Thick: true temperature rise for losses 40 W → 10 W (200 s) → 60 W (400 s). "
         "Dots: noisy measurements, only in the first 150 s. Red dotted: fit with the data loss only. Orange "
         "dashed: fit with data and physics loss. A smooth basis-function model stands in for the network so that "
         "the fit is solved instantly; the loss is the same as in a PINN.")
    st.caption(f"Beyond the data the pure data fit has an RMSE of {fmt(e(y0, ~m))} K — it knows nothing about the "
               f"loss steps. With physics (α = {alpha:g}) the RMSE is {fmt(e(ya, ~m))} K: the heat balance "
               "carries the solution through the 10 W and 60 W segments.")
    cm.equations([
        ("Total loss", r"\mathcal{L} = \underbrace{\frac{1}{N_d}\sum_{i=1}^{N_d}\big(\hat\theta(t_i)-\theta_i\big)^2}"
                       r"_{\text{data}} + \alpha\,\underbrace{\frac{1}{N_c}\sum_{j=1}^{N_c} r(t_j)^2}_{\text{physics}}",
         "Both terms are normalised so that α is dimensionless."),
        ("Residual", r"r(t) = C\,\frac{d\hat\theta}{dt} + \frac{\hat\theta}{R} - P(t)",
         "Zero when the prediction obeys the RC heat balance; in TherMOS any Foster/Cauer network or own equation."),
        ("In TherMOS", r"\mathcal{L} = \alpha_\mathrm{T}\,\mathcal{L}_\text{data} + \beta_\mathrm{T}\,"
                       r"\mathcal{L}_\text{phys} + \gamma_\mathrm{T}\,\mathcal{L}_\text{init}",
         "Setup alpha, beta, gamma of the PI mode (data, physics, initial state); the demo's α corresponds to "
         "β_T / α_T."),
    ], sym=[('θ̂, θᵢ', 'predicted and measured temperature rise', 'K'), ('tᵢ, N_d', 'measurement times, number', 's, –'),
            ('t_j, N_c', 'collocation points (no measurement needed), number', 's, –'),
            ('r', 'residual of the heat balance', 'W'), ('C, R', 'thermal capacitance and resistance', 'J/K, K/W'),
            ('P', 'losses', 'W'), ('α', 'physics weight of the demo', '–'),
            ('α_T, β_T, γ_T', 'loss weights of the PI mode (data, physics, initial state)', '–')])
    cm.interpret([
        f"With {nd} measurements in the first 150 s the physics loss reduces the extrapolation error from "
        f"{fmt(e(y0, ~m))} K to {fmt(e(ya, ~m))} K. A too small α ignores the physics; a too large α trusts the "
        "physics model more than the data — if R or C are wrong, the fit then follows the wrong model.",
        "Use a PINN when measurements are few or do not cover all operating points and an approximate thermal "
        "network exists. Start with α ≈ 1 (normalised losses) and compare the validation error for α = 0.1, 1, 10."])


def _sk_ic():
    """Heat-up at constant load: two windows with identical inputs but different target temperatures."""
    x0, x1, tau_px, W = 80, 520, 85, 70
    th = lambda x: 185 - 105 * (1 - np.exp(-max(x - x0, 0) / tau_px))
    xs = np.arange(40, x1 + 1, 4)
    b = [_t(20, 44, 'P', 'middle', 11), _t(20, 150, 'θ', 'middle', 11),
         _l(40, 60, x0, 60, col=SERIES[3], w=2), _l(x0, 60, x0, 35, col=SERIES[3], w=2),
         _l(x0, 35, x1, 35, col=SERIES[3], w=2), _t(x1 + 4, 39, '10 W', 'start', 10),
         _l(40, 190, x1, 190, op=0.5), _t(x0, 204, 'load on', 'middle', 9)]
    b.append("<polyline fill='none' stroke='%s' stroke-width='2' points='%s'/>"
             % (SERIES[0], ' '.join(f"{x:.0f},{th(x):.1f}" for x in xs)))
    for k, xt in enumerate((230, 450)):
        b += [_r(xt - W, 26, W, 18, fill=SERIES[2], col=SERIES[2], op=0.25),
              _l(xt, 44, xt, th(xt) - 5, dash='3 3', op=0.6), _dot(xt, th(xt), col=SERIES[1], r=4),
              _t(xt - W / 2, 20, f'window at t{"₁₂"[k]}', 'middle', 9),
              _t(xt + 8, th(xt) + 16, f'θ(t{"₁₂"[k]})', 'start', 10)]
    b += [_t(560, 70, 'both windows contain', 'start', 11), _t(560, 88, 'only "10 W, 10 W, …"', 'start', 11),
          _t(560, 118, '→ same input,', 'start', 11), _t(560, 136, '   different targets', 'start', 11),
          _t(560, 166, '→ a network can only', 'start', 11), _t(560, 184, '   predict their mean', 'start', 11)]
    return _svg(''.join(b), 770, 212)


def _bayes(theta, keys):
    """Best possible prediction of any model that sees only `keys`: the mean of θ over samples with equal keys."""
    _, inv = np.unique(np.round(keys, 6), axis=0, return_inverse=True)
    inv = inv.ravel()
    return (np.bincount(inv, theta) / np.bincount(inv))[inv]


def _runs(P0, R, tau, n0, T):
    """Heat-up runs at constant load P0 from n0 different initial temperatures θ0 = 0 … 0.8·R·P0."""
    a = np.exp(-H / tau)
    th0 = np.linspace(0, 0.8 * R * P0, n0) if n0 > 1 else np.zeros(1)
    out = np.zeros((n0, T))
    for i, x in enumerate(th0):
        for k in range(T):
            x = a * x + R * (1 - a) * P0
            out[i, k] = x
    return th0, out


def _keys(th0, th, Wn, use_t, use_0, use_p):
    """Features a network sees for every sample of every run (constant load: the window is set by min(k+1, W))."""
    n0, T = th.shape
    k = np.tile(np.arange(T), n0)
    cols = [np.minimum(k + 1, Wn)]                                             # loaded samples in the window
    if use_t:
        cols.append(k)
    if use_0:
        cols.append(np.repeat(th0, T))
    if use_p:
        cols.append(np.concatenate([np.r_[x0, r[:-1]] for x0, r in zip(th0, th)]))
    return np.column_stack(cols)


def _train_profile(kind, pmax, dmax, n, seed=11):
    """Training profiles: one constant heat-up, a staircase up and down, or random steps (APRBS)."""
    if kind == 0:
        return np.full(n, pmax)
    if kind == 1:
        lv = np.r_[np.linspace(0, pmax, 5), np.linspace(pmax, 0, 5)[1:]]
        return np.resize(np.repeat(lv, int(dmax)), n)
    rng, P, k = np.random.default_rng(seed), np.zeros(n), 0
    while k < n:
        L = int(rng.uniform(2, dmax))
        P[k:k + L] = rng.uniform(0, pmax)
        k += L
    return P


DATA_RULES = pd.DataFrame([
    ('Start state known', 'the same loss window can belong to a cold or a warm device', 'every profile starts at a '
     'known temperature (cold soak) or the start temperature is an input', 'Parameters › Initial values = 1, or start each profile cold'),
    ('Window long enough', 'the network must see the whole relevant heat-up', 'W ≥ 3–5 τ of the slowest path that '
     'matters; otherwise use a recurrent net or the start state', 'Parameters › Window length; LSTM or EWMA features for long τ'),
    ('Operating range covered', 'networks interpolate, they do not extrapolate', 'training (P, θ) points cover the '
     'test and field range, incl. peak load and peak temperature', 'tab "Does the training data cover…"'),
    ('All time scales excited', 'only excited time constants can be learned', 'segment durations from ≈ τ_min to '
     '≥ 3 τ_max, heat-ups and cool-downs, random steps (APRBS) or drive cycles', 'data/: several datasets (transfer split)'),
    ('Sample time fine enough', 'fast transients are lost between samples', 'h ≤ τ_min / 5; down-sample slow '
     'channels instead of shortening the window', 'Data › fs; resample the datasets'),
    ('Split by profile', 'overlapping windows of one profile leak into validation', 'training, validation and test '
     'are different profiles; the test profile is never used for tuning', 'Exp › Data split = 2 (transfer) or 3 (ID based)'),
    ('Same inputs as in use', 'a feature missing in operation cannot be used', 'only signals that are measured in '
     'the application (losses, currents, coolant temperature, flow)', 'Data › Inputs'),
    ('Enough data', 'weights ≫ samples → memorisation', 'samples ≫ weights; else smaller net, more profiles, PINN',
     'Theory › Network (weights per window)'),
], columns=['requirement', 'why', 'how', 'in PyDTS'])
DATA_HELP = {'requirement': "What the training data set must fulfil.",
             'why': "What goes wrong without it.", 'how': "How to design or check the data.",
             'in PyDTS': "Where the setting or the data lives."}


def _tab_data(P, Rf, tf):
    st.markdown("A network learns a **map from what it sees to the temperature**. If two samples have the same "
                "input but different temperatures, no network can learn both — it can at best predict their mean. "
                "This happens in every transient problem with a constant load: a window of *10 W, 10 W, …* looks "
                "the same one second after switch-on and ten minutes later, and the same for a cold and an already "
                "warm device. The thermal **state** (the stored heat) is not in the losses; the network must get it "
                "from a long enough window, from a known start, or as an extra input. The data set decides which of "
                "these works.")
    show_svg(_sk_ic(), "Constant load: the input windows at t₁ and t₂ are identical, the temperatures are not.")
    tau = float(tf.max())
    R = float(Rf.sum())
    c = st.columns(3)
    P0 = c[0].slider("Constant load P₀ (W)", 1.0, 80.0, 10.0, step=1.0, key=f"{K}|ds_P")
    Wr = c[1].slider("Window W / τ", 0.1, 6.0, 1.0, step=0.1, key=f"{K}|ds_win",
                     help=f"Window length in multiples of the time constant τ = {tau:g} s of the demo.")
    n0 = int(c[2].slider("Different start temperatures θ₀ in the data", 1, 5, 3, key=f"{K}|ds_n",
                         help="1: every run starts cold. More: runs start already warm (0 … 80 % of the steady "
                              "state) — e.g. a profile recorded after earlier operation."))
    c = st.columns(3)
    use_t = c[0].toggle("+ time since start t", key=f"{K}|ds_u_t", help="A clock that is reset at switch-on.")
    use_0 = c[1].toggle("+ start temperature θ₀", key=f"{K}|ds_u_0", help="Measured once at the start of the run "
                        "(PyDTS: Parameters › Initial values = 1).")
    use_p = c[2].toggle("+ last temperature θ(t−h)", key=f"{K}|ds_u_p", help="The temperature of the previous step "
                        "(measured, or the network's own prediction in closed loop).")
    T = int(5 * tau / H)
    Wn = max(1, int(round(Wr * tau / H)))
    th0, th = _runs(P0, R, tau, n0, T)
    best = _bayes(th.ravel(), _keys(th0, th, Wn, use_t, use_0, use_p)).reshape(th.shape)
    err = best - th
    rmse, emax = np.sqrt(np.mean(err ** 2)), np.abs(err).max()
    amb = np.mean(np.abs(err) > 0.01 * R * P0)
    feats = "window" + (" + t" if use_t else '') + (" + θ₀" if use_0 else '') + (" + θ(t−h)" if use_p else '')
    cm.tiles([("Smallest possible RMSE", f"{fmt(rmse)} K", "Error of the best possible model with these inputs "
               "(mean of all temperatures that share the same input); no network can be better."),
              ("Largest error", f"{fmt(emax)} K", f"Steady state R·P₀ = {fmt(R * P0)} K."),
              ("Ambiguous samples", f"{100 * amb:.0f} %", "Share of samples whose input also occurs with a "
               "temperature more than 1 % of R·P₀ away."),
              ("Window", f"{Wn * H:g} s", f"{Wr:g} τ, τ = {tau:g} s, R = {R:g} K/W")])
    t1, t2, t3, t4 = st.tabs(["Why can't the network tell where in the heat-up it is?",
                              "How long must the window be?", "Does the training data cover the test?",
                              "Checklist for a training data set"])
    with t1:
        tt = np.arange(T) * H
        fig = go.Figure()
        for i in range(n0):
            fig.add_trace(line(tt, th[i], f'true, θ₀ = {th0[i]:.1f} K', i % len(SERIES), width=3))
            fig.add_trace(line(tt, best[i], f'best model, θ₀ = {th0[i]:.1f} K', i % len(SERIES), dash='dot'))
        fig.add_vline(x=Wn * H, line=dict(color='gray', dash='dash'), annotation_text='t = W')
        fig.update_layout(xaxis_title='time since switch-on (s)', yaxis_title='θ (K)')
        show(fig, key=f"{K}|ds_fig", info="Thick: true temperature rise of every run (constant load P₀, start "
             "temperature θ₀). Dotted, same colour: the best prediction any model can make from the selected "
             "inputs — the mean of all true temperatures that share the same input. The window before switch-on "
             "is filled with zeros (load off).")
        if not (use_t or use_0 or use_p):
            msg = (f"With the loss window only, every sample after t = W = {Wn * H:g} s has the same input. The "
                   f"best model predicts one constant {fmt(best[0, -1])} K there, although the true value rises "
                   f"from {fmt(th[0, min(Wn, T - 1)])} K to {fmt(th[0, -1])} K. ")
        else:
            msg = f"With the inputs {feats} the remaining RMSE is {fmt(rmse)} K. "
        if n0 > 1 and not (use_0 or use_p):
            msg += (f"The {n0} runs with different θ₀ have identical inputs from the start, so they collapse onto "
                    "one dotted curve: the network cannot know that a run started warm.")
        elif rmse < 1e-3 * R * P0:
            msg += "The input fixes the state uniquely: the temperature is learnable without systematic error."
        st.caption(msg)
    with t2:
        Wg = np.linspace(0.1, 6.0, 40)
        rm = [np.sqrt(np.mean((_bayes(th.ravel(), _keys(th0, th, max(1, int(round(w * tau / H))), use_t, use_0,
                                                          use_p)) - th.ravel()) ** 2)) for w in Wg]
        th0c, thc = _runs(P0, R, tau, 1, T)
        rc = [np.sqrt(np.mean((_bayes(thc.ravel(), _keys(th0c, thc, max(1, int(round(w * tau / H))), False, False,
                                                            False)) - thc.ravel()) ** 2)) for w in Wg]
        fig = go.Figure([line(Wg, rc, 'window only, all runs cold', 0, dash='dot'),
                         line(Wg, rm, f'selected: {feats}, {n0} start temperature(s)', 1)])
        fig.add_vline(x=Wr, line=dict(color='gray', dash='dash'))
        fig.update_layout(xaxis_title='window W / τ', yaxis_title='smallest possible RMSE (K)')
        show(fig, key=f"{K}|ds_Wfig", info="Smallest possible error over the window length. Blue dotted: only cold "
             "starts and only the loss window — the error vanishes once the window reaches back to switch-on. "
             "Orange: the selected inputs and number of start temperatures. Runs last 5 τ.")
        i3 = int(np.argmin(np.abs(Wg - 3.0)))
        st.caption(f"Cold starts only: RMSE {fmt(rc[0])} K at W = 0.1 τ, {fmt(rc[i3])} K at 3 τ and "
                   f"{fmt(rc[-1])} K at 6 τ (the window then always contains the switch-on). With the selected "
                   f"set-up the curve ends at {fmt(rm[-1])} K"
                   + (": a longer window cannot recover an unknown warm start." if rm[-1] > 0.01 * R * P0
                      else "."))
    with t3:
        c = st.columns(3)
        kinds = ["one heat-up at constant load", "staircase up and down", "random steps (APRBS)"]
        kind = kinds.index(c[0].selectbox("Training profile", kinds, index=2, key=f"{K}|ds_k"))
        pmax = c[1].slider("Max. training load (W)", 10.0, 100.0, 80.0, step=5.0, key=f"{K}|ds_pm")
        dmax = c[2].slider("Segment length (s)", 5.0, 300.0, 60.0, step=5.0, key=f"{K}|ds_d",
                           help="Staircase: hold time per level. APRBS: longest random segment (shortest 2 s).")
        Ptr = _train_profile(kind, pmax, dmax, len(P))
        ttr, tte = _foster(Ptr, Rf, tf), _foster(P, Rf, tf)
        pb, tb = 10.0, max(tte.max(), ttr.max()) / 15
        cell = lambda p, x: set(zip((p // pb).astype(int), (x // tb).astype(int)))
        have = cell(Ptr, ttr)
        inside = np.array([(int(p // pb), int(x // tb)) in have for p, x in zip(P, tte)])
        cov = 100 * inside.mean()
        nst = int(np.sum(np.abs(np.diff(Ptr)) > 1e-9))
        cm.tiles([("Test samples covered", f"{cov:.0f} %", f"Share of test samples whose (P, θ) cell of {pb:g} W × "
                   f"{tb:.1f} K also occurs in the training data."),
                  ("Peak θ training / test", f"{fmt(ttr.max())} / {fmt(tte.max())} K", None),
                  ("Load steps in training", f"{nst}", "Each step excites all time constants once.")])
        fig = go.Figure([go.Scatter(x=Ptr, y=ttr, mode='markers', name='training', marker=dict(color='gray',
                                    size=4, opacity=0.4)),
                         go.Scatter(x=P[inside], y=tte[inside], mode='markers', name='test, covered',
                                    marker=dict(color=SERIES[0], size=5)),
                         go.Scatter(x=P[~inside], y=tte[~inside], mode='markers', name='test, not covered',
                                    marker=dict(color=SERIES[7], size=6, symbol='x'))])
        fig.update_layout(xaxis_title='loss P (W)', yaxis_title='temperature rise θ (K)')
        show(fig, key=f"{K}|ds_cov", info="Operating points (load, temperature) of the training profile (grey) and "
             "of the test profile from section 2 (blue: a training point lies in the same cell; red crosses: "
             "none does — the network has to extrapolate there). Both use the Foster network of section 2.")
        st.caption(f"The training profile '{kinds[kind]}' covers {cov:.0f} % of the test samples with {nst} load "
                   "steps. " + ("A single constant heat-up shows one load level and one rising curve; the network "
                                "never sees cooling or other loads. " if kind == 0 else '')
                   + ("Short segments keep the device near its mean temperature: high temperatures are missing. "
                      if kind == 2 and dmax < tau else '')
                   + ("Long segments give few steps: few transients to learn from. " if dmax > 3 * tau and kind
                      else '') + "Aim for > 95 % coverage and peak values beyond those of the application.")
        with st.expander("Training profile over time", icon=':material/show_chart:'):
            tt = np.arange(len(Ptr)) * H
            fig = go.Figure([line(tt, Ptr, 'P (W)', 3), line(tt, ttr, 'θ (K)', 0, yaxis='y2')])
            fig.update_layout(xaxis_title='t (s)', yaxis_title='P (W)',
                              yaxis2=dict(title='θ (K)', overlaying='y', side='right'))
            show(fig, key=f"{K}|ds_tr", info="Yellow: training losses. Blue: resulting temperature rise (right axis).")
    with t4:
        table(DATA_RULES, hide_index=True,
              column_config={c: st.column_config.TextColumn(help=h) for c, h in DATA_HELP.items()})
        st.caption("Read row by row as a checklist before training. The first two rows decide whether the "
                   "temperature is learnable at all (tabs 1 and 2); the third whether the model is valid where it "
                   "is used (tab 3).")
    cm.equations([
        ("Best possible model", r"f^*(\mathbf{x}) = \mathbb{E}[\theta \mid \mathbf{x}]",
         "With the MSE loss the ideal network returns the mean of all temperatures seen with input x."),
        ("Irreducible error", r"\mathrm{RMSE}_\mathrm{min} = \sqrt{\mathbb{E}\big[\mathrm{Var}(\theta\mid"
                              r"\mathbf{x})\big]}", "Zero only if the input fixes θ uniquely; size and training "
                                                    "time of the network do not change it."),
        ("Heat-up at constant load", r"\theta(t) = \theta_\infty + (\theta_0 - \theta_\infty)\,e^{-t/\tau},\quad "
                                     r"\theta_\infty = R\,P_0", "Depends on t and θ₀ — neither is in the losses."),
        ("State as input", r"\theta_k = e^{-h/\tau}\,\theta_{k-1} + R\,(1-e^{-h/\tau})\,P_k",
         "With θ(t−h) as input the next value is fixed exactly: the state carries the whole history."),
    ], sym=[('x', 'everything the network sees (loss window, extra inputs)', '–'),
            ('θ, θ₀, θ_∞', 'temperature rise, at switch-on, in steady state', 'K'),
            ('P₀', 'constant load', 'W'), ('R, τ', 'thermal resistance and time constant of the demo', 'K/W, s'),
            ('t, k, h', 'time since switch-on, step index, sample time', 's, –, s'),
            ('W', 'window length', 's'), ('E[·|x], Var(·|x)', 'mean and variance over all samples with input x', '–')])
    cm.interpret([
        f"With the inputs {feats}, a window of {Wn * H:g} s ({Wr:g} τ) and {n0} start temperature(s), the best "
        f"possible model still has an RMSE of {fmt(rmse)} K (max. {fmt(emax)} K, steady state {fmt(R * P0)} K); "
        f"{100 * amb:.0f} % of the samples are ambiguous. "
        + ("This error is a property of the data, not of the network: a larger net or longer training will not "
           "remove it." if rmse > 0.01 * R * P0 else "The temperature is learnable from these inputs."),
        "Ways out, in order of robustness: (1) give the start state — record from a known cold soak or pass θ₀ as an "
        "input (PyDTS: Initial values = 1; testing then needs only the first measured temperature); (2) use a recurrent model "
        "that runs over the whole profile from the start (the state is carried; PyDTS solver CL feeds the prediction back); "
        "(3) make the window ≥ 3–5 τ of the slowest relevant path. The last temperature θ(t−h) solves it exactly "
        "but in closed loop the network feeds back its own errors, which can drift; train with noise or "
        "multi-step losses then. A clock t alone fails as soon as a run starts warm.",
        "The training set must then cover the application: several profiles with random load steps from short to "
        "≥ 3 τ_max, heat-ups and cool-downs, loads and temperatures up to the application peak, and separate "
        "profiles for validation and test (checklist in tab 4)."])


def _tab_train():
    st.markdown("All networks are trained the same way: the inputs are **normalised**, the data are split into "
                "**training, validation and test** sets, and an optimiser (Adam) changes the weights step by step "
                "against the gradient of the loss. One pass over the training data is an **epoch**; it is split "
                "into **mini-batches**. The **learning rate** sets the step size. Training stops when the "
                "validation loss no longer improves (**early stopping**), which limits **overfitting**.")
    c = st.columns(2)
    lr = 10 ** c[0].slider("Learning rate log₁₀ η", -3.0, 0.5, -1.0, step=0.1, key=f"{K}|lr")
    it = int(c[1].slider("Iterations", 10, 300, 100, key=f"{K}|it"))
    # quadratic loss of a linear model with two weights (unscaled feature → ill-conditioned)
    Hm = np.array([[1.0, 0.0], [0.0, 0.1]]) * 2
    w0, ws = np.array([2.0, 2.0]), np.zeros(2)
    lam = np.linalg.eigvalsh(Hm)
    curves = []
    for fac, lab in ((1.0, f'η = {lr:.3g}'), (0.1, f'η/10'), (np.nan, 'gradient descent, η = 1/λ_max')):
        eta = 1 / lam.max() if np.isnan(fac) else lr * fac
        w, L = w0.copy(), []
        for _ in range(it):
            L.append(0.5 * (w - ws) @ Hm @ (w - ws))
            w = w - eta * Hm @ (w - ws)
        curves.append((lab, np.array(L)))
    div = lr > 2 / lam.max()
    cm.tiles([("Stability limit", f"η < {2 / lam.max():.2f}", "2 / λ_max of the loss curvature"),
              ("Loss after training", 'diverged' if div else f"{curves[0][1][-1]:.2e}",
               f"start {curves[0][1][0]:.2f}"),
              ("Condition number", f"{lam.max() / lam.min():.0f}", "λ_max / λ_min; 1 for normalised inputs.")])
    fig = go.Figure([line(np.arange(it), np.clip(L, 1e-12, 1e6), lab, i) for i, (lab, L) in enumerate(curves)])
    fig.update_layout(xaxis_title='iteration', yaxis=dict(type='log', title='training loss'))
    show(fig, key=f"{K}|lrfig", info="Loss of gradient descent on a two-weight model whose inputs are not normalised "
         "(curvatures 2 and 0.2). Blue: chosen learning rate, orange: ten times smaller, green: the largest "
         "stable step for the steep direction.")
    st.caption(("The chosen step is larger than 2 / λ_max: the loss **diverges**. " if div else
                f"After {it} iterations the loss is {curves[0][1][-1]:.2e}. ") +
               "Too small a step needs many epochs; too large a step oscillates or diverges. Normalising the inputs "
               "makes all directions equally steep, so one learning rate fits all weights.")
    table(pd.DataFrame([
        ('Mdl › epoch', 'Epochs', 'maximum passes over the training data', '50–500'),
        ('Mdl › batch', 'Mini-batch', 'windows per weight update; small = noisy but often better generalisation', '32–512'),
        ('Mdl › lr', 'Learning rate η', 'initial step of Adam', '1e-4 … 1e-2'),
        ('(automatic)', 'Learning-rate schedule', 'PyDTS halves η after patience/2 epochs without improvement',
         'factor 0.5'),
        ('Mdl › patience', 'Patience', 'epochs without validation improvement before stopping', '5–20'),
                ('Par › window, outseq', 'Window, output', 'input length; seq2point (one θ per window) or seq2seq', '≥ 3 τ_relevant'),
        ('Mdl › loss', 'Loss function', 'MSE (smooth), MAE (robust to outliers)', 'mae'),
    ], columns=['setup', 'name', 'meaning', 'typical']), hide_index=True,
        column_config={'setup': st.column_config.TextColumn(help="Parameter of PyDTS (section › variable)."),
                       'typical': st.column_config.TextColumn(help="Typical range for thermal data sets.")})
    cm.equations([
        ("Loss", r"\mathcal{L}(\mathbf{w}) = \frac{1}{N}\sum_{n=1}^{N}\big(\hat\theta_n(\mathbf{w}) - \theta_n\big)^2",
         "Mean squared error over the samples of one mini-batch."),
        ("Gradient step", r"\mathbf{w} \leftarrow \mathbf{w} - \eta\,\nabla_{\mathbf{w}}\mathcal{L}",
         "The gradient is computed by back-propagation (chain rule through all layers)."),
        ("Adam", r"m \leftarrow \beta_1 m + (1-\beta_1)g,\quad v \leftarrow \beta_2 v + (1-\beta_2)g^2,\quad "
                 r"\mathbf{w} \leftarrow \mathbf{w} - \eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon}",
         "Per-weight step size from running means of the gradient and its square."),
        ("Normalisation", r"\tilde x = \frac{x - \mu_x}{\sigma_x}", "Mean and standard deviation of the training set."),
    ], sym=[('w', 'all weights and biases', '–'), ('η', 'learning rate', '–'), ('g', 'gradient ∇L', '–'),
            ('m, v', 'running mean of gradient and squared gradient (m̂, v̂ bias-corrected)', '–'),
            ('β₁, β₂, ε', 'Adam constants (0.9, 0.999, 1e-8)', '–'), ('N', 'samples in a mini-batch', '–'),
            ('μ_x, σ_x', 'mean and standard deviation of an input', 'unit of x'),
            ('λ_max, λ_min', 'largest/smallest curvature (eigenvalue of the Hessian) of the loss', '–')])
    cm.interpret([
        f"Here the learning rate η = {lr:.3g} is {'above' if div else 'below'} the stability limit "
        f"{2 / lam.max():.2f}. Adam adapts the step per weight and is therefore much less sensitive, but the same "
        "rule applies: if the training loss jumps or becomes NaN, reduce the learning rate (Model › Learning rate).",
        "Judge a model on the **test** profile, never on the training data: a training error far below the test "
        "error means overfitting (more data, smaller network, dropout, earlier stop); both high means underfitting "
        "(larger network, longer window W, more informative inputs)."])


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    cm.page_head("Neural networks", "Dense, convolutional, recurrent, attention-based and physics-informed networks "
                                    "on thermal examples — the models of PyDTS for temperature prediction.")
    cm.question("What does each network of PyDTS do with a window of losses, and which one fits which thermal "
                "task?")
    cm.section(1, "The networks at a glance", "All of them map a window of W samples × F input features (losses, "
                                              "currents, coolant temperature, …) to one temperature (seq2point) or "
                                              "a sequence (seq2seq).", divider=False)
    table(NETS, hide_index=True, column_config={c: st.column_config.TextColumn(help=h) for c, h in NET_HELP.items()})

    cm.section(2, "Thermal reference of the demos", "The CNN, LSTM and transformer demos use the same random loss "
                                                    "profile (sample time 1 s) and a two-term Foster network as "
                                                    "the true thermal model.")
    c = st.columns(2)
    tau1 = cm.num(c[0], "Fast time constant τ₁ (s)", 5.0, 'nn|t1', min_value=1.0, max_value=50.0)
    tau2 = cm.num(c[1], "Slow time constant τ₂ (s)", 60.0, 'nn|t2', min_value=10.0, max_value=300.0)
    Rf, tf = np.array([0.3, 0.5]), np.array([tau1, tau2])
    P = _profile()
    with st.expander("Loss profile and true temperature", icon=':material/show_chart:'):
        tt = np.arange(len(P)) * H
        th = _foster(P, Rf, tf)
        fig = go.Figure([line(tt, P, 'P (W)', 3), line(tt, th, 'θ (K)', 0, yaxis='y2')])
        fig.update_layout(xaxis_title='t (s)', yaxis_title='P (W)', yaxis2=dict(title='θ (K)', overlaying='y',
                                                                               side='right'))
        show(fig, key=f"{K}|ref", info="Yellow: random loss profile (segments of 10–60 s, 0–80 W). Blue: exact "
             "temperature rise of the Foster network R = 0.3 / 0.5 K/W with τ₁ and τ₂ (right axis).")
        st.caption(f"Peak rise {fmt(th.max())} K; the slow term (τ₂ = {tau2:g} s, 0.5 K/W) carries "
                   f"{100 * 0.5 / 0.8:.0f} % of the static resistance of 0.8 K/W.")

    cm.section(3, "Networks one by one")
    tabs = st.tabs(["DNN · neuron and layer", "CNN · convolution", "LSTM · memory", "TRA · attention",
                    "PINN · physics loss", "Data & initial state", "Training"])
    with tabs[0]:
        _tab_dnn()
    with tabs[1]:
        _tab_cnn(P, Rf, tf)
    with tabs[2]:
        _tab_lstm(P, Rf, tf)
    with tabs[3]:
        _tab_tra(P, Rf, tf)
    with tabs[4]:
        _tab_pinn()
    with tabs[5]:
        _tab_data(P, Rf, tf)
    with tabs[6]:
        _tab_train()
    cm.refs(['Goodfellow2016', 'Hochreiter1997', 'Bai2018', 'Vaswani2017', 'Raissi2019', 'Kirchgaessner2023'])
