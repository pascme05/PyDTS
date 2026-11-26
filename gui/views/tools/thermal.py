#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Theory & Tools
# File:         thermal.py
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
Temperature prediction with data-driven models: why a temperature needs the history of the losses, how long the
window must be, and how rolled EWMA features (PyDTS featuresRoll) carry that history at a short window.
Tabs:   1) Design calculator   thermal time constants and sampling → window length, EWMA spans, model size;
                               the values can be written into the configuration
        2) Window vs features  least-squares models on a two-term Foster network: static, FIR window of W samples
                               and EWMA features, test error over W
        3) Physics of the task the thermal network as a convolution of the losses, and what follows for the model
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core import forms
from gui.core.style import show, line, table, fmt, SERIES, status_badge
from gui.views.tools import _common as cm

K = f"{cm.KEY}|th"


#######################################################################################################################
# Thermal reference
#######################################################################################################################
def _profile(n, seed=5, dmin=5, dmax=400):
    """Loss profile: random steps of dmin … dmax samples and 0 … 1 (normalised loss)."""
    rng = np.random.default_rng(seed)
    P, k = np.zeros(n), 0
    while k < n:
        L = int(rng.uniform(dmin, dmax))
        P[k:k + L] = rng.uniform(0, 1)
        k += L
    return P


def _foster(P, R, tau, h=1.0):
    """Temperature rise of a Foster network (exact discretisation for piecewise constant P)."""
    th = np.zeros(len(P))
    for r, t in zip(R, tau):
        a = np.exp(-h / t)
        x = 0.0
        for k, p in enumerate(P):
            x = a * x + (1 - a) * r * p
            th[k] += x
    return th


def _ewma(P, span):
    """EWMA as pandas ewm(span) — the rolled feature of PyDTS (src/general/featuresRoll.py)."""
    return pd.Series(P).ewm(span=span, adjust=False).mean().to_numpy()


def _lstsq(Xtr, ytr, Xte):
    A = np.c_[Xtr, np.ones(len(Xtr))]
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.c_[Xte, np.ones(len(Xte))] @ w


def _fir(P, W):
    """Matrix of the last W samples of P (zero padded at the start)."""
    Z = np.r_[np.zeros(W - 1), P]
    return np.lib.stride_tricks.sliding_window_view(Z, W)[:, ::-1]


#######################################################################################################################
# Tab 1: design calculator
#######################################################################################################################
def _design():
    cfg = forms.cfg()
    cm.section(1, "Thermal and data inputs", "Time constants from a step response, a datasheet Z_th curve or the "
                                             "Theory › Data & windows tab of the mode (memory τ of the output).",
               divider=False)
    c = cm.grid([1, 1])
    with c[0]:
        with cm.box("Thermal system", "Fastest and slowest time constant that matter for the predicted temperature.",
                    icon='thermostat', foot="Winding: seconds to minutes; stator yoke, magnet, coolant: tens of "
                                            "minutes to hours."):
            g = cm.grid(2, 'bottom')
            tmin = cm.num(g[0], "τ_min (s)", 20.0, 'th|tmin', min_value=0.01, help="Fastest relevant time constant.")
            tmax = cm.num(g[1], "τ_max (s)", 1800.0, 'th|tmax', min_value=0.1, help="Slowest relevant time constant.")
    with c[1]:
        with cm.box("Data and model", "Sampling of the data and number of inputs (from the configuration).",
                    icon='dataset', foot="fs and the inputs start at the configuration (Data › fs, Data › Inputs)."):
            g = cm.grid(2, 'bottom')
            fs = cm.num(g[0], "Sampling frequency fs (Hz)", float(cfg['Dat'].get('fs') or 1.0), 'th|fs',
                        min_value=1e-6)
            from gui import adapter as A
            F = int(cm.num(g[1], "Input channels F", max(len(A.to_list(cfg['Dat'].get('inp'))), 4), 'th|F',
                           min_value=1, format='%d'))
    if tmax < tmin:
        tmin, tmax = tmax, tmin
    h = 1 / fs
    W_full = int(np.ceil(3 * tmax * fs))
    W_short = int(np.ceil(3 * tmin * fs))
    nsp = max(2, int(np.ceil(np.log10(max(tmax / tmin, 1.0001)) / np.log10(4))) + 1)
    spans = np.unique(np.round(np.geomspace(2 * tmin * fs, 2 * tmax * fs, nsp)).astype(int))
    spans = spans[spans >= 1]
    w_dnn = W_full * F * 32 + 2 * 32 * 33 + 33
    w_short = W_short * F * (1 + len(spans)) * 32 + 2 * 32 * 33 + 33

    cm.section(None, "Result")
    cm.tiles([("Window for 3 τ_max", f"{W_full:,} samples", f"3·τ_max·fs = {fmt(3 * tmax)} s of history."),
              ("Window with EWMA", f"{max(W_short, 1):,} samples", "3·τ_min·fs: the short window sees the fast "
                                                                   "transients, the EWMA features the slow history."),
              ("EWMA spans", ', '.join(str(s) for s in spans), "Geometric from 2·τ_min·fs to 2·τ_max·fs (an EWMA of "
                                                               "span s has τ ≈ s/2 samples)."),
              ("DNN weights, full window", f"{w_dnn:,}", "First dense layer W·F·32 + PyDTS DNN head (3 × 32)."),
              ("DNN weights, EWMA", f"{w_short:,}", "Short window with F·(1 + spans) channels.")])
    c = cm.grid([3, 2])
    with c[0]:
        tt = np.geomspace(max(h, tmin / 20), tmax * 5, 300)
        fig = go.Figure([line(tt, 1 - np.exp(-tt / tmin), f"RC τ_min = {fmt(tmin)} s", 1),
                         line(tt, 1 - np.exp(-tt / tmax), f"RC τ_max = {fmt(tmax)} s", 0)])
        for s, col in ((W_full * h, SERIES[2]), (max(W_short, 1) * h, SERIES[3])):
            fig.add_vline(x=s, line_dash='dash', line_color=col)
        for i, s in enumerate(spans):
            fig.add_trace(line(tt, 1 - np.exp(-tt / (s / 2 * h)), f"EWMA span {s}", 4 + i, dash='dot', width=1.3))
        fig.update_layout(xaxis_type='log', xaxis_title='time after a load step (s)', yaxis_title='step response',
                          title="Step responses, window lengths and EWMA filters")
        show(fig, height=380, key=f"{K}|step", info="Solid: step responses of the slowest and fastest RC element. "
             "Dashed: window for 3 τ_max (green) and for 3 τ_min (yellow). Dotted: step responses of the EWMA "
             "features — they should span the solid curves.")
    with c[1]:
        with cm.box("Checks", "Whether the design fits the data.", icon='fact_check'):
            ok_fs = h <= tmin / 5
            status_badge(f"Sampling: h = {fmt(h)} s ≤ τ_min/5 = {fmt(tmin / 5)} s" if ok_fs else
                         f"Sampling too coarse: h = {fmt(h)} s > τ_min/5", 'good' if ok_fs else 'warning')
            big = W_full > 2000
            status_badge(f"Full window {W_full:,} samples" + (" — too long for a DNN/CNN, use EWMA features or an "
                                                              "LSTM" if big else " — feasible"),
                         'warning' if big else 'good')
            status_badge(f"{len(spans)} EWMA spans × {F} inputs = {len(spans) * F} extra channels", 'good')
        with cm.box("Apply", "Writes the design into the configuration (Parameters of the Simulation mode).",
                    icon='input'):
            mode = st.radio("Design", ["Short window + EWMA", "Full window"], key=f"{K}|apply|m", horizontal=True)
            if st.button("Apply to configuration", icon=':material/check:', key=f"{K}|apply", type='primary'):
                W = max(W_short, 2) if mode.startswith('Short') else max(W_full, 2)
                cfg['Par']['window'], cfg['Par']['overlap'] = W, W - 1
                cfg['Par']['yFocus'] = W - 1
                if mode.startswith('Short'):
                    cfg['Par']['feat'] = 2
                    cfg['Mdl']['feat_roll.EWMA'] = ', '.join(str(s) for s in spans)
                for k in [k for k in st.session_state if isinstance(k, str) and k.startswith(('p|Par|', 'p|Mdl|'))]:
                    del st.session_state[k]
                st.toast(f"Window {W}, overlap {W - 1}" + (f", EWMA {', '.join(map(str, spans))}" if
                                                          mode.startswith('Short') else ''), icon=':material/check:')
    cm.equations([
        ("Window", r"W \ge 3\,\tau_{max}\,f_s", "After 3 τ an RC element has reached 95 % of its step response: the "
                                                "window then sees the history that matters."),
        ("EWMA feature", [r"\bar P_s(k) = \alpha P(k) + (1-\alpha)\,\bar P_s(k-1),\qquad \alpha = \frac{2}{s+1}",
                          r"\tau_s \approx \frac{s}{2}\,h"], "pandas ewm(span = s) as in PyDTS featuresRoll — a "
                                                             "first-order low pass, i.e. one RC element."),
        ("Spans", r"s_j \in [\,2\tau_{min} f_s,\; 2\tau_{max} f_s\,],\ \text{geometric}",
         "A linear combination of first-order filters with these time constants is a Foster network."),
    ], sym=[('τ_min, τ_max', 'fastest and slowest relevant thermal time constant', 's'),
            ('f_s, h', 'sampling frequency and sample time 1/f_s', 'Hz, s'), ('W', 'window length', 'samples'),
            ('s', 'EWMA span (Model › EWMA spans)', 'samples'), ('α', 'EWMA smoothing factor', '–'),
            ('F', 'number of input channels', '–')])
    cm.interpret([
        f"Covering τ_max = {fmt(tmax)} s with the raw window needs W = {W_full:,} samples and a DNN with about "
        f"{w_dnn:,} weights. With {len(spans)} EWMA spans ({', '.join(map(str, spans))} samples) a window of "
        f"{max(W_short, 1):,} samples is enough and the DNN has {w_short:,} weights"
        + (f" — {fmt(w_dnn / max(w_short, 1))} × fewer." if w_dnn > w_short else "."),
        "Rolled EWMA features are the classic choice for motor temperatures (Kirchgässner et al.): they turn the "
        "dynamic problem into an almost static one that even small models solve. Use the full window when the "
        "shape of the transients matters (CNN) or an LSTM, whose memory does not depend on the window length."])
    cm.refs(['Kirchgaessner2021b', 'Kirchgaessner2021', 'Schirmer2024'])


#######################################################################################################################
# Tab 2: window vs features
#######################################################################################################################
def _window():
    cm.section(1, "Thermal reference", "A two-term Foster network driven by random load steps (1 s samples); the "
                                       "first half trains a linear least-squares model, the second half tests it.",
               divider=False)
    c = cm.grid(3, 'bottom')
    t1 = cm.num(c[0], "τ₁ (s)", 10.0, 'th|t1', min_value=1.0, max_value=200.0)
    t2 = cm.num(c[1], "τ₂ (s)", 300.0, 'th|t2', min_value=5.0, max_value=3000.0)
    noise = cm.num(c[2], "Measurement noise (K)", 0.05, 'th|ns', min_value=0.0, max_value=2.0)
    R = np.array([0.4, 0.6])
    n = 6000
    P = _profile(n)
    th = _foster(P, R, [t1, t2]) * 100
    thm = th + noise * np.random.default_rng(2).standard_normal(n)
    a = n // 2
    Ws = np.unique(np.r_[1, np.geomspace(2, 1500, 14).astype(int)])
    e_fir = []
    for W in Ws:
        X = _fir(P, W)
        e_fir.append(np.mean(np.abs(_lstsq(X[:a], thm[:a], X[a:]) - th[a:])))
    spans = [int(max(2, 2 * t1)), int(max(2, 2 * t2))]
    Xe = np.c_[P, np.column_stack([_ewma(P, s) for s in spans])]
    pe = _lstsq(Xe[:a], thm[:a], Xe[a:])
    e_ew = np.mean(np.abs(pe - th[a:]))
    p0 = _lstsq(P[:a, None], thm[:a], P[a:, None])
    e0 = np.mean(np.abs(p0 - th[a:]))
    W95 = int(3 * t2)

    cm.section(None, "Result")
    cm.tiles([("Static model (P only)", f"{fmt(e0)} K", "MAE of T = a·P(k) + b on the test half."),
              ("Best FIR window", f"{fmt(min(e_fir))} K at W = {Ws[int(np.argmin(e_fir))]}",
               "MAE of a linear model on the last W samples of P."),
              ("EWMA features", f"{fmt(e_ew)} K", f"MAE with P and EWMA spans {spans} (3 inputs)."),
              ("Peak rise", f"{fmt(th.max())} K", "Largest temperature rise of the reference.")])
    c = cm.grid([1, 1])
    with c[0]:
        fig = go.Figure([line(Ws, e_fir, 'FIR window (W weights)', 0, mode='lines+markers')])
        fig.add_hline(y=e_ew, line_dash='dash', line_color=SERIES[2], annotation_text='EWMA (3 inputs)')
        fig.add_hline(y=e0, line_dash='dot', line_color=SERIES[1], annotation_text='static')
        fig.add_vline(x=W95, line_dash='dot', line_color='#898781', annotation_text='3 τ₂')
        fig.update_layout(xaxis_type='log', yaxis_type='log', xaxis_title='window W (samples)',
                          yaxis_title='test MAE (K)', title="Test error over the window length")
        show(fig, height=360, key=f"{K}|w", info="Linear least-squares models on the test half. Blue: the window "
             "of the last W losses; it improves until W covers about 3 τ₂. Green dashed: three EWMA features reach "
             "the same accuracy with 3 instead of W weights.")
    with c[1]:
        tt = np.arange(a, n)
        Wb = int(Ws[int(np.argmin(np.abs(Ws - min(W95, Ws[-1]) // 10)))])
        X = _fir(P, Wb)
        pw = _lstsq(X[:a], thm[:a], X[a:])
        fig = go.Figure([line(tt, th[a:], 'true T', 0, c='#52514e', width=2.5),
                         line(tt, pw, f"FIR W = {Wb}", 1), line(tt, pe, 'EWMA features', 2),
                         line(tt, p0, 'static', 3, dash='dot')])
        fig.update_layout(xaxis_title='sample (test half)', yaxis_title='temperature rise (K)',
                          title="Predictions on the test half")
        show(fig, height=360, key=f"{K}|p", info="A short window misses the slow heating (offset), the static model "
             "only follows the load steps; the EWMA model follows both.")
    cm.equations([
        ("Thermal network (Foster)", r"\theta(k) = \sum_i x_i(k),\quad x_i(k) = a_i x_i(k-1) + (1-a_i) R_i P(k),\quad "
                                     r"a_i = e^{-h/\tau_i}", "Exact for piecewise constant losses."),
        ("FIR model", r"\hat\theta(k) = \sum_{j=0}^{W-1} w_j P(k-j) + b", "Truncated impulse response: what a "
                                                                          "linear DNN/CNN on a window can learn."),
        ("EWMA model", r"\hat\theta(k) = w_0 P(k) + \sum_s w_s \bar P_s(k) + b", "If the spans match the τ_i, the "
                                                                                "model is the Foster network "
                                                                                "itself."),
    ], sym=[('θ', 'temperature rise', 'K'), ('P', 'loss, normalised (peak 1)', '–'),
            ('R_i, τ_i', 'thermal resistance and time constant of term i', 'K/W, s'),
            ('W', 'window length', 'samples'), ('w, b', 'weights and bias of the regression', '–')])
    cm.interpret([
        f"The static model has an MAE of {fmt(e0)} K: the temperature cannot be predicted from the current loss "
        f"alone. A window must reach about 3 τ₂ = {W95} samples before the FIR error stops falling "
        f"(best {fmt(min(e_fir))} K); three EWMA features reach {fmt(e_ew)} K with three weights.",
        "This carries over to neural networks: a DNN or CNN on a window is a non-linear FIR model and needs the same "
        "window length; with EWMA inputs (PyDTS Features = 2) a much smaller network suffices. Real machines add "
        "non-linearities (losses ∝ I², temperature-dependent resistance, speed-dependent cooling) — that is where "
        "the network earns its place over the linear model."])


#######################################################################################################################
# Tab 3: physics of the task
#######################################################################################################################
def _physics():
    cm.section(1, "Why a temperature needs history", divider=False)
    st.markdown("The temperature of a component follows its losses through **heat storage**: the thermal network is "
                "a low-pass filter. Mathematically the temperature is the **convolution** of the losses with the "
                "thermal impulse response — the prediction at time *k* depends on all past losses, weighted by "
                "how much of their heat is still stored.")
    cm.equations([
        ("Heat balance of node i", r"C_i\,\frac{d\theta_i}{dt} = P_i - \sum_j G_{ij}(\theta_i-\theta_j)",
         "Capacitances C store heat, conductances G conduct it to neighbours and the coolant."),
        ("Convolution", r"\theta(t) = \int_0^{t} h(t-t')\,P(t')\,dt',\qquad "
                        r"h(t) = \sum_i \frac{R_i}{\tau_i} e^{-t/\tau_i}",
         "The impulse response h of a Foster network; Z_th(t) = ∫h = Σ R_i (1 − e^(−t/τ_i))."),
        ("What the network learns", r"\hat\theta(k) = f_\theta\big(P(k-W+1..k),\; T_{cool}(k), n(k), \dots\big)",
         "A data-driven model must recover h (and its non-linear dependencies) from examples."),
    ], sym=[('C_i', 'heat capacitance of node i', 'J/K'), ('G_ij', 'thermal conductance between i and j', 'W/K'),
            ('θ_i', 'temperature of node i', '°C'), ('P_i', 'losses injected at node i', 'W'),
            ('h(t)', 'thermal impulse response', 'K/(W·s)'), ('R_i, τ_i', 'Foster resistance and time constant',
                                                            'K/W, s')], expanded=True)
    table(pd.DataFrame([
        ('Losses ∝ I², f, B²', 'inputs are currents, voltages, speed — not the losses', 'the network must learn the '
         'squares; add I², U·I or speed features if the model struggles'),
        ('Resistance rises with T', 'copper losses grow ~0.4 %/K', 'non-linear feedback: a static linear model fails '
         'at the hot end'),
        ('Cooling depends on flow, speed', 'G changes with the operating point', 'include coolant temperature and '
         'speed as inputs'),
        ('Unknown initial state', 'the same input can belong to a cold or warm machine', 'long window, EWMA features, '
         'LSTM or initial values as input'),
        ('Unmeasured paths (rotor)', 'magnet temperature is far from the measured quantities', 'larger errors — '
         'see Tutorial 3 (PM error ≫ stator errors)'),
    ], columns=['effect', 'consequence', 'what to do in PyDTS']), hide_index=True)
    cm.interpret([
        "Temperature prediction is a *dynamic* regression: choose the window or the rolled features from the thermal "
        "time constants (tab Design calculator), give the model the quantities that drive the losses and the "
        "cooling, and test on a separate drive cycle (Data split = transfer).",
        "Physics-informed alternatives (thermal neural networks, PINNs; see the *Neural networks* page and TherMOS) "
        "embed the heat balance and usually extrapolate better; PyDTS gives the pure data-driven baseline."])


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    cm.page_head("Temperature prediction", "How to set up a data-driven temperature model: window length, rolled "
                                           "features and the thermal background.")
    cm.question("How long must the model look into the past to predict a temperature, and how can it do so with "
                "few weights?")
    tabs = st.tabs(["Design calculator", "Window vs features", "Physics of the task"])
    with tabs[0]:
        _design()
    with tabs[1]:
        _window()
    with tabs[2]:
        _physics()
