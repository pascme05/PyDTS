#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Black-Box Modeling
# File:         viz_inputs.py
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
This function shows the input data of PyDTS. dataset_view() shows one dataset of data/ in four panels: overview
(channels, units, ranges), signals over time, the split into training / validation / test data of the current
configuration, and the correlation between inputs and outputs. It is the Inputs view of every mode (INPUTS) and the
Input data page. workbook()/sheet() are the generic viewers of the scaffold.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui import adapter as A
from gui.core import data as D
from gui.core.present import panel_tabs, interpretation, tiles
from gui.core.style import show, line, downsample, table, fmt, SERIES


#######################################################################################################################
# Generic viewer
#######################################################################################################################
def arrow_safe(df):
    """Mixed text/number columns as text, so the table renders without warnings."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object and not all(isinstance(v, str) for v in df[c].dropna()):
            df[c] = df[c].map(lambda v: '' if v is None or (isinstance(v, float) and v != v) else str(v))
    return df


#######################################################################################################################
# Helper
#######################################################################################################################
def dur(sec):
    """Readable duration."""
    for f, u in ((86400, 'd'), (3600, 'h'), (60, 'min')):
        if sec >= 2 * f:
            return f"{sec / f:.3g} {u}"
    return f"{sec:.3g} s"


def tscale(sec):
    """Divisor and unit of a time axis that spans sec seconds."""
    for f, u in ((86400, 'd'), (3600, 'h'), (60, 'min')):
        if sec >= 3 * f:
            return f, u
    return 1, 's'


def _cols(cfg, df, key):
    """Channels of the configuration (Dat › inp / out) that exist in the dataset; all if none is configured."""
    want = [str(c) for c in A.to_list(cfg['Dat'].get(key))]
    have = [c for c in df.columns if c not in ('time', 'id')]
    sel = [c for c in want if c in have]
    return sel or have


def tcol(df, fs):
    """Time of the rows in s: the column time if present, else index / fs."""
    if 'time' in df:
        t = pd.to_numeric(df['time'], errors='coerce').to_numpy(dtype=float)
        if np.isfinite(t).all() and len(t) > 1:
            return t - t[0]
    return np.arange(len(df)) / fs


def aligned(X, y, cols, fs):
    """Inputs interpolated onto the time of the outputs (inputs and outputs may have different sample rates)."""
    if len(X) == len(y):
        return X[cols].to_numpy(dtype=float)
    tx, ty = tcol(X, fs), tcol(y, fs)
    return np.column_stack([np.interp(ty, tx, X[c].to_numpy(dtype=float)) for c in cols])


def roles(cfg):
    """{dataset: role} of the configuration."""
    role = {str(n): 'training' for n in A.to_list(cfg['Dat'].get('train'))}
    if cfg['Exp'].get('method') == 2:
        role[str(cfg['Dat'].get('val'))] = 'validation'
        role[str(cfg['Dat'].get('test'))] = 'test'
    return role


#######################################################################################################################
# Dataset view
#######################################################################################################################
def dataset_view(cfg, key='ds', choose=True):
    typ = str(cfg['Dat'].get('type', 'xlsx'))
    names = D.datasets(typ)
    role = roles(cfg)
    if not names:
        st.info(f"No .{typ} datasets in data/ (Data › File type).")
        return
    opts = [n for n in role if n in names] + [n for n in names if n not in role]
    name = opts[0]
    if choose:
        name = st.selectbox("Dataset", opts, key=f"{key}|name",
                            format_func=lambda n: f"{n} · {role[n]}" if n in role else f"{n} · not used",
                            help=f"Datasets in data/ of type .{typ}; the ones of the configuration come first "
                                 f"(Data › Training / Test / Validation datasets).")
    missing = [n for n in role if n not in names]
    if missing:
        st.warning(f"Configured dataset(s) not found as .{typ} in data/: {', '.join(missing)}.",
                   icon=':material/warning:')
    try:
        X, y, ux, uy = D.load(name, typ)
    except Exception as e:
        st.error(f"The dataset could not be read: {e}")
        return
    fs = float(cfg['Dat'].get('fs', 1) or 1)
    n = len(y)
    inp, out = _cols(cfg, X, 'inp'), _cols(cfg, y, 'out')
    ts, tu = tscale(max(tcol(y, fs)[-1], 1e-9))
    T = panel_tabs(["Overview", "Signals", "Split", "Correlation"],
                   subs={'Overview': "Channels of the dataset with unit and range; 'used' marks the inputs and "
                                     "outputs of the configuration.",
                         'Signals': "Inputs and outputs over time (thinned to 4000 points for the browser).",
                         'Split': "Which samples train, validate and test the model with the current data split.",
                         'Correlation': "Linear correlation (Pearson ρ) between every input and output."},
                   titles={'Overview': "What is in the dataset?", 'Signals': "How do the signals evolve?",
                           'Split': "Which data train and which test the model?",
                           'Correlation': "Which inputs carry information about the outputs?"})

    # Overview
    with T[0]:
        tiles([("Samples", f"{n:,}", "Rows of the dataset."),
               ("Duration", dur(n / fs), f"Samples / sampling frequency fs = {fmt(fs)} Hz (Data › fs)."),
               ("Inputs used", f"{len(inp)} of {len([c for c in X.columns if c not in ('time', 'id')])}",
                "Data › Inputs (empty: all)."),
               ("Outputs", f"{len(out)}", ', '.join(out)),
               ("IDs", f"{X['id'].nunique() if 'id' in X else 1}",
                "Distinct values of the column id (blocks, e.g. measurement sessions).")])
        tab = D.summary(X, y, ux, uy, fs)
        tab.insert(1, 'used', [c in inp if r == 'input' else c in out for c, r in zip(tab['channel'], tab['role'])])
        table(tab, hide_index=True, height=min(460, 38 + 35 * len(tab)))
        nan = int(tab['NaN'].sum())
        interpretation(f"{n:,} samples at {fmt(fs)} Hz ({dur(n / fs)}). The model maps {len(inp)} inputs to "
                       f"{len(out)} outputs ({', '.join(out[:6])}{' …' if len(out) > 6 else ''})."
                       + (f" {nan} missing values will be interpolated by PyDTS." if nan else " No missing values."))

    # Signals
    with T[1]:
        c1, c2 = st.columns([1, 3])
        with c1:
            pi = st.multiselect("Inputs", [c for c in X.columns if c not in ('time', 'id')], default=inp[:3],
                                key=f"{key}|pi")
            po = st.multiselect("Outputs", [c for c in y.columns if c not in ('time', 'id')], default=out[:3],
                                key=f"{key}|po")
            rng = st.slider("Range (%)", 0, 100, (0, 100), key=f"{key}|rng")
        tx, ty = tcol(X, fs), tcol(y, fs)
        t_end = max(tx[-1], ty[-1])
        lo, hi = t_end * rng[0] / 100, max(t_end * rng[1] / 100, t_end * rng[0] / 100 + 1e-9)
        ix = np.flatnonzero((tx >= lo) & (tx <= hi))
        iy = np.flatnonzero((ty >= lo) & (ty <= hi))
        ix, iy = ix[::downsample(len(ix))], iy[::downsample(len(iy))]
        with c2:
            from plotly.subplots import make_subplots
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
            for i, c in enumerate(pi):
                fig.add_trace(line(tx[ix] / ts, X[c].to_numpy()[ix], f"{c} [{D.unit(ux.get(c))}]", i), row=1, col=1)
            for i, c in enumerate(po):
                fig.add_trace(line(ty[iy] / ts, y[c].to_numpy()[iy], f"{c} [{D.unit(uy.get(c))}]", i + len(pi)),
                              row=2, col=1)
            fig.update_yaxes(title_text='inputs', row=1, col=1)
            fig.update_yaxes(title_text='outputs', row=2, col=1)
            fig.update_xaxes(title_text=f"t ({tu})", row=2, col=1)
            show(fig, height=520, key=f"{key}|sig")
        if po:
            yo = y[po[0]].to_numpy(dtype=float)
            interpretation(f"{po[0]} spans {fmt(np.nanmin(yo))} … {fmt(np.nanmax(yo))} {D.unit(uy.get(po[0]))}. "
                           f"A model can only interpolate: operating points outside the training range (higher "
                           f"losses, other ambient or coolant temperatures) are extrapolated and less reliable.")
        else:
            interpretation("Select an output to see its range.")

    # Split
    with T[2]:
        sp = D.split(cfg, n)
        m = cfg['Exp'].get('method')
        if sp is None and m == 2:
            st.markdown(f"**Transfer split:** training on {', '.join(r for r, v in role.items() if v == 'training')}, "
                        f"validation on {cfg['Dat'].get('val')}, test on {cfg['Dat'].get('test')} — every dataset "
                        f"is used completely.")
            interpretation("The test runs on a separate measurement: the errors show how well the model generalises "
                           "to a new operating cycle — the most honest test of a temperature model.")
        elif sp is None:
            st.markdown(f"**ID based split:** test IDs {cfg['Dat'].get('idT')}, validation IDs "
                        f"{cfg['Dat'].get('idV')}; all other IDs train the model.")
            if 'id' in y:
                ids = y['id'].value_counts().sort_index()
                fig = go.Figure(go.Bar(x=[str(i) for i in ids.index], y=ids.values, marker_color=SERIES[0]))
                fig.update_layout(xaxis_title='id', yaxis_title='samples')
                show(fig, key=f"{key}|ids")
            interpretation("Whole blocks (IDs) are held out, so the test data are not neighbours of training samples "
                           "in time.")
        else:
            o = out[0] if out else None
            s = downsample(n)
            t = np.arange(0, n, s) / fs / ts
            fig = go.Figure()
            if o:
                fig.add_trace(line(t, y[o].iloc[::s], f"{o}", 0, c='#52514e'))
            cols = {'training': SERIES[0], 'validation': SERIES[3], 'test': SERIES[1]}
            for i, (lab, a, b) in enumerate(sp):
                fig.add_vrect(x0=a / fs / ts, x1=b / fs / ts, fillcolor=cols.get(lab, SERIES[i % len(SERIES)]),
                              opacity=0.15, line_width=0, annotation_text=lab if len(sp) <= 3 else str(i + 1),
                              annotation_position='top left')
            fig.update_layout(xaxis_title=f"t ({tu})", yaxis_title=o or '')
            show(fig, key=f"{key}|split")
            if m == 0:
                rT, rV = float(cfg['Dat']['rT']), float(cfg['Dat']['rV'])
                interpretation(f"1-fold: the first {fmt(100 * rT)} % train the model (the last {fmt(100 * rV)} % of "
                               f"them validate it), the last {fmt(100 * (1 - rT))} % "
                               f"({dur((sp[2][2] - sp[2][1]) / fs)}) test it. The test period is later in time — "
                               f"check that it contains the operating points you care about (e.g. the hottest "
                               f"phase)." + (" Shuffling is on: training and test windows interleave and the errors "
                                             "will be optimistic." if str(cfg['Dat'].get('Shuffle')) == 'True' else ''))
            else:
                interpretation(f"k-fold: {len(sp)} folds of {dur(n / len(sp) / fs)} each; every fold is the test "
                               f"data once and the errors are averaged over the folds.")

    # Correlation
    with T[3]:
        num = [c for c in inp if c in X][:30]
        if not out or not num:
            interpretation("No inputs or outputs to correlate.")
        else:
            s = downsample(n, 20000)
            with np.errstate(all='ignore'):
                M = np.corrcoef(np.c_[aligned(X, y, num, fs)[::s], y[out].to_numpy(dtype=float)[::s]].T)
            C = M[:len(num), len(num):]
            fig = go.Figure(go.Heatmap(z=C, x=out, y=num, zmin=-1, zmax=1, colorscale='RdBu_r',
                                       text=np.round(C, 2), texttemplate='%{text}', colorbar=dict(title='ρ')))
            fig.update_layout(xaxis_title='output', yaxis_title='input')
            show(fig, height=max(320, 26 * len(num) + 120), key=f"{key}|corr")
            Ca = np.nan_to_num(np.abs(C), nan=-1)
            k = np.unravel_index(np.argmax(Ca), C.shape)
            interpretation(f"Strongest linear relation: {num[k[0]]} → {out[k[1]]} (ρ = {fmt(C[k])}). A low "
                           f"correlation does not make an input useless — temperatures follow the *history* of the "
                           f"losses (∝ I²), not their current value; the network learns this from the window, rolled "
                           f"EWMA features make it explicit.")


#######################################################################################################################
# Inputs per mode: mode -> function(cfg)
#######################################################################################################################
INPUTS = {0: dataset_view, 1: dataset_view, 2: dataset_view}
