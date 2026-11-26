#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Black-Box Modeling
# File:         viz_results.py
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
Results of a PyDTS run. The adapter records the test predictions (wrapper around src.test.performance), the feature
ranking and the training log. A result answers in the OPTELIA order: summary panel (answer, key figures, what it
means), then one tab per question (accuracy per output, time series, error, scatter, training, features), each in a
panel with its interpretation. The metrics are the ones of PyDTS (src/general/performance.py); the matplotlib figures
of PyDTS stay available under 'Figures & files'.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core import files
from gui.core.present import result_base, answer, tiles, qtabs, interpretation, hint, details
from gui.core.style import show, line, table, fmt, downsample, SERIES

METRICS = ['ACC', 'F1', 'R2', 'TECA', 'RMSE', 'MAE', 'MAX', 'EST', 'TRUE']
MET_HELP = {'ACC': "Accuracy of the on/off states (output > Data › threshold).", 'F1': "F1 score of the on/off states.",
            'R2': "Coefficient of determination: share of the variance explained (1 = perfect).",
            'TECA': "Total energy correctly assigned: 1 − Σ|ŷ − y| / (2 Σ|y|).",
            'RMSE': "Root mean square error √mean((ŷ − y)²).", 'MAE': "Mean absolute error mean|ŷ − y|.",
            'MAX': "Largest absolute error max|ŷ − y|.", 'EST': "Share of the predicted sum Σ|ŷ| in the total.",
            'TRUE': "Share of the measured sum Σ|y| in the total."}


#######################################################################################################################
# Helper
#######################################################################################################################
def _unit(u):
    u = str(u or '').strip()
    return {'degC': '°C', 'deC': '°C', 'sec': 's', '-': '', 'nan': ''}.get(u, u)


def _eunit(u):
    """Unit of an error (differences of °C are K)."""
    u = _unit(u)
    return 'K' if u == '°C' else u


def _tscale(sec):
    for f, u in ((86400, 'd'), (3600, 'h'), (60, 'min')):
        if sec >= 3 * f:
            return f, u
    return 1, 's'


def _level(r2):
    return 'good' if r2 >= 0.95 else 'warning' if r2 >= 0.8 else 'critical'


#######################################################################################################################
# Mode renderer
#######################################################################################################################
def mode0(data, setup, key):
    tests = data.get('tests') or []
    if not tests:
        st.info("This run has no test results (Testing = 0, or the run stopped before testing). The trained model is "
                "stored in mdl/; switch on *Testing* to evaluate it.")
        _training(data, key, standalone=True)
        return
    out = list(data.get('out') or [])
    units = list(data.get('units') or [])
    if len(tests) > 1:
        k = st.selectbox("Fold", list(range(len(tests))), format_func=lambda i: f"fold {i + 1}", key=f"{key}|fold",
                         help="k-fold split: one test per fold; the key figures are averaged over all folds.")
    else:
        k = 0
    t0 = tests[k]
    res = np.mean([t['result'] for t in tests], axis=0)                       # averaged over folds (as printResults)
    out = out if len(out) == res.shape[0] else [f"y{i + 1}" for i in range(res.shape[0])]
    units = units if len(units) == len(out) else [''] * len(out)
    M = pd.DataFrame(res, columns=METRICS[:res.shape[1]], index=out)
    mae, rmse, r2, mx = M['MAE'], M['RMSE'], M['R2'], M['MAX']
    worst = mae.idxmax()
    eu = _eunit(units[out.index(worst)])
    yT, yP = t0['yTrue'], t0['yPred']
    fs = float(data.get('fs') or 1) / float(t0.get('stride', 1))

    # Summary
    result_base()
    reg = setup is None or (setup.get('Par') or {}).get('method', 0) == 0
    if reg:
        answer(f"The {data.get('model', '')} model predicts {len(out)} output{'s' if len(out) > 1 else ''} with a mean "
               f"absolute error of {fmt(mae.mean())} {_eunit(units[0])} and R² = {fmt(r2.mean())}; the largest error "
               f"is {fmt(mx.max())} {_eunit(units[int(np.argmax(mx.values))])} ({mx.idxmax()}).", _level(r2.mean()))
    else:
        answer(f"The {data.get('model', '')} model classifies with an accuracy of {fmt(100 * M['ACC'].mean())} % "
               f"(F1 = {fmt(M['F1'].mean())}).", 'good' if M['ACC'].mean() > 0.9 else 'warning')
    tiles([("Mean absolute error", f"{fmt(mae.mean())} {_eunit(units[0])}", MET_HELP['MAE'] + " Mean of all outputs."),
           ("RMSE", f"{fmt(rmse.mean())} {_eunit(units[0])}", MET_HELP['RMSE']),
           ("R²", fmt(r2.mean()), MET_HELP['R2']),
           ("Largest error", f"{fmt(mx.max())} {_eunit(units[0])}", MET_HELP['MAX']),
           ("Test samples", f"{len(yT) * int(t0.get('stride', 1)):,}", "Predicted samples (windows) of the test data"
                                                                       + (f", averaged over {len(tests)} folds"
                                                                          if len(tests) > 1 else '') + ".")])
    st.markdown(f"**What it means:** the hardest output is **{worst}** (MAE {fmt(mae[worst])} {eu}, R² "
                f"{fmt(r2[worst])}). " + ("RMSE is much larger than the MAE: a few large errors dominate — look at "
                                          "the *Error* tab for where they occur." if rmse.mean() > 1.5 * mae.mean()
                                          else "RMSE and MAE are close: the errors are evenly spread, without large "
                                               "outliers."))

    T = qtabs(['Accuracy', 'Time series', 'Error', 'Scatter', 'Training', 'Features'])

    # Accuracy per output
    with T[0]:
        cols = ['MAE', 'RMSE', 'MAX', 'R2'] + (['ACC', 'F1'] if not reg else ['TECA'])
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure()
            for i, m in enumerate(['MAE', 'RMSE', 'MAX']):
                fig.add_trace(go.Bar(x=out, y=M[m], name=m, marker_color=SERIES[i]))
            fig.update_layout(barmode='group', yaxis_title=f"error ({eu})", xaxis_title='output')
            show(fig, key=f"{key}|acc")
        with c2:
            tab = M[cols].copy()
            tab.insert(0, 'unit', [_unit(u) for u in units])
            table(tab, column_config={c: st.column_config.NumberColumn(c, help=MET_HELP.get(c)) for c in cols})
        best = mae.idxmin()
        interpretation(f"Best output {best} (MAE {fmt(mae[best])}), worst {worst} (MAE {fmt(mae[worst])}). Outputs "
                       f"that are far from the inputs in the heat path (e.g. rotor magnet vs. stator winding) are "
                       f"usually harder to predict because their temperature depends on longer history and "
                       f"unmeasured losses — a longer window or rolled EWMA features help.")

    # Time series
    with T[1]:
        c1, c2 = st.columns([1, 4])
        with c1:
            o = st.selectbox("Output", out, index=out.index(worst), key=f"{key}|ts|o")
            rng = st.slider("Range (%)", 0, 100, (0, 100), key=f"{key}|ts|r")
        j = out.index(o)
        n = len(yT)
        a, b = int(n * rng[0] / 100), max(int(n * rng[1] / 100), int(n * rng[0] / 100) + 2)
        s = downsample(b - a)
        ts, tu = _tscale(n / fs)
        t = np.arange(a, b, s) / fs / ts
        fig = go.Figure([line(t, yT[a:b:s, j], f"measured {o}", 0, c='#52514e', width=2.5),
                         line(t, yP[a:b:s, j], f"predicted {o}", 1)])
        fig.update_layout(xaxis_title=f"t ({tu}, test data)", yaxis_title=f"{o} ({_unit(units[j])})")
        with c2:
            show(fig, height=440, key=f"{key}|ts")
        e = yP[:, j] - yT[:, j]
        kmax = int(np.nanargmax(np.abs(e)))
        interpretation(f"Mean error (bias) {fmt(np.nanmean(e))} {_eunit(units[j])}: the model "
                       f"{'over' if np.nanmean(e) > 0 else 'under'}estimates {o} on average. The largest deviation "
                       f"({fmt(e[kmax])} {_eunit(units[j])}) occurs at t = {fmt(kmax / fs / ts)} {tu} — zoom there "
                       f"with the range slider: errors at fast load steps point to a too short window or missing "
                       f"inputs, slow drifts to a missing thermal memory or boundary temperature.")

    # Error
    with T[2]:
        o = st.selectbox("Output", out, index=out.index(worst), key=f"{key}|er|o")
        j = out.index(o)
        e = yP[:, j] - yT[:, j]
        n = len(e)
        s = downsample(n)
        ts, tu = _tscale(n / fs)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure([line(np.arange(0, n, s) / fs / ts, e[::s], 'ŷ − y', 1)])
            fig.add_hline(y=0, line_color='#898781')
            fig.update_layout(xaxis_title=f"t ({tu})", yaxis_title=f"error ({_eunit(units[j])})",
                              title="Error over time")
            show(fig, key=f"{key}|er|t")
        with c2:
            fig = go.Figure(go.Histogram(x=e, nbinsx=60, marker_color=SERIES[0]))
            fig.update_layout(xaxis_title=f"error ({_eunit(units[j])})", yaxis_title='samples',
                              title="Distribution of the error")
            show(fig, key=f"{key}|er|h")
        p95 = np.nanpercentile(np.abs(e), 95)
        interpretation(f"95 % of the test samples of {o} are within ±{fmt(p95)} {_eunit(units[j])}; the worst "
                       f"{fmt(np.nanmax(np.abs(e)))} {_eunit(units[j])}. A narrow distribution centred on zero is a "
                       f"good model; a shifted one is a bias (e.g. a different coolant or ambient temperature in the "
                       f"test data), long tails are rare situations the training did not contain.")

    # Scatter
    with T[3]:
        o = st.selectbox("Output", out, index=out.index(worst), key=f"{key}|sc|o")
        j = out.index(o)
        s = downsample(len(yT), 6000)
        x, yy = yT[::s, j], yP[::s, j]
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        fig = go.Figure([go.Scattergl(x=x, y=yy, mode='markers', name='test samples',
                                      marker=dict(size=4, color=SERIES[0], opacity=0.45)),
                         go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='ŷ = y',
                                    line=dict(color='#52514e', dash='dash'))])
        fig.update_layout(xaxis_title=f"measured {o} ({_unit(units[j])})",
                          yaxis_title=f"predicted {o} ({_unit(units[j])})")
        show(fig, height=440, key=f"{key}|sc")
        ok = np.isfinite(yT[:, j]) & np.isfinite(yP[:, j])
        slope = np.polyfit(yT[ok, j], yP[ok, j], 1)[0] if ok.sum() > 2 else np.nan
        top = yT[:, j] >= np.nanpercentile(yT[:, j], 90)
        etop = np.nanmean(yP[top, j] - yT[top, j])
        interpretation(f"Slope of ŷ over y: {fmt(slope)} (1 = unbiased over the range). In the top 10 % of {o} the "
                       f"mean error is {fmt(etop)} {_eunit(units[j])}"
                       + (" — the model underestimates the highest values; for temperatures this is the critical "
                          "side (hot spots are predicted too cold)." if etop < -0.5 * mae[o] else "."))

    # Training
    with T[4]:
        _training(data, key)

    # Features
    with T[5]:
        rk = data.get('rank')
        sc = rk[0] if isinstance(rk, (list, tuple)) and len(rk) else None
        try:
            sc = pd.Series(sc).astype(float) if sc is not None else None
        except (TypeError, ValueError):
            sc = None
        if sc is None or sc.empty or not np.any(np.abs(sc.values) > 0):
            interpretation("No feature ranking in this run — set Parameters › Feature ranking (Par › rank) to 1 to "
                           "rank the inputs by their importance.")
        else:
            sc = sc.sort_values()
            fig = go.Figure(go.Bar(x=sc.values, y=[str(i) for i in sc.index], orientation='h',
                                   marker_color=SERIES[0]))
            fig.update_layout(xaxis_title='importance', yaxis_title='input')
            show(fig, height=max(300, 24 * len(sc) + 100), key=f"{key}|rk")
            interpretation(f"Most important input: {sc.index[-1]}, least important: {sc.index[0]}. Inputs with "
                           f"near-zero importance can be removed (Data › Inputs) to reduce the model size.")


def _training(data, key, standalone=False):
    h = data.get('hist')
    df = None
    if isinstance(h, dict):
        df = h.get('new') if h.get('new') is not None and len(h.get('new')) else h.get('all')
    if df is None or not len(df):
        msg = ("No training in this run (Training = 0: the stored model was loaded). Train with *Training* = 1 on "
               "the Setup page to see the learning curves.")
        st.info(msg) if standalone else interpretation(msg)
        return
    ep = np.arange(1, len(df) + 1)
    fig = go.Figure()
    for i, c in enumerate([c for c in ('loss', 'val_loss') if c in df]):
        fig.add_trace(line(ep, df[c], {'loss': 'training loss', 'val_loss': 'validation loss'}[c], i, mode='lines+markers'))
    fig.update_layout(xaxis_title='epoch', yaxis_title='loss (normalised)', yaxis_type='log')
    show(fig, key=f"{key}|hist")
    if 'val_loss' in df:
        b = int(df['val_loss'].idxmin())
        gap = df['val_loss'].iloc[b] / max(df['loss'].iloc[b], 1e-12)
        interpretation(f"Best validation loss {fmt(df['val_loss'].iloc[b])} at epoch {b + 1} of {len(df)} "
                       f"(these weights are kept). Validation / training loss = {fmt(gap)}: "
                       + ("close to 1 — no overfitting." if gap < 1.5 else
                          "the network fits the training data much better than unseen data (overfitting) — use "
                          "fewer units, more data, or stop earlier (patience).")
                       + (" Training ended at the epoch limit while the loss still fell — more epochs may help."
                          if b >= len(df) - 2 else ""))
    else:
        interpretation(f"{len(df)} epochs; final training loss {fmt(df['loss'].iloc[-1])}.")


#######################################################################################################################
# Figures and files
#######################################################################################################################
def gallery(run, key='gal'):
    figs = [(t, b) for t, b in run.get('figs') or []]
    disk, reports, datas = [], [], []
    for f in run.get('folders') or []:
        rf = files.run_files(f)
        disk += rf['figures']
        reports += rf['report']
        datas += rf['data']
    items = figs + [(os.path.splitext(os.path.basename(f))[0].replace('_', ' '), f) for f in disk]
    if not items and not reports and not datas:
        st.info("No figures or files for this run (enable *Plots of PyDTS* on the Setup page).")
        return
    if items:
        st.markdown(f"**Figures of PyDTS** ({len(items)})")
        mode = st.segmented_control("View", ['Grid', 'Single'], default='Grid', key=f"{key}|view",
                                    label_visibility='collapsed')
        if mode == 'Single':
            i = st.selectbox("Figure", range(len(items)), format_func=lambda i: items[i][0], key=f"{key}|one")
            st.image(items[i][1], caption=items[i][0], width='stretch')
        else:
            cols = st.columns(2)
            for i, (n, img) in enumerate(items):
                with cols[i % 2]:
                    st.image(img, caption=n, width='stretch')
    if reports or datas:
        st.markdown("**Files**")
        cols = st.columns(3)
        for i, f in enumerate(reports + datas):
            with cols[i % 3]:
                with open(f, 'rb') as fh:
                    st.download_button(os.path.basename(f), fh.read(), file_name=os.path.basename(f),
                                       icon=':material/download:', key=f"{key}|dl|{f}", width='stretch')


#######################################################################################################################
# Interactive results per mode: mode -> function(data, setup, key)
#######################################################################################################################
RENDER = {0: mode0, 1: mode0, 2: mode0}


#######################################################################################################################
# Run
#######################################################################################################################
def run_view(run, key='res'):
    """Interactive results, figures and files, and the log of one run."""
    if run is None:
        st.info("No results yet — start a simulation in the **Run** view or open a results folder.")
        return
    if run.get('error'):
        st.error(f"The simulation stopped with an error:\n\n```\n{run['error'][-3000:]}\n```")
    st.caption(f"Run {run.get('time', '')} · {fmt(run.get('duration', 0))} s")
    view = st.segmented_control("Show", ['Interactive', 'Figures & files', 'Console log'], default='Interactive',
                                key=f"{key}|v", label_visibility='collapsed')
    if view == 'Interactive':
        if run.get('data') is not None:
            try:
                RENDER.get(run['mode'], mode0)(run['data'], run.get('setup'), key=f"{key}|{run['mode']}")
            except Exception as e:
                st.warning(f"Interactive charts not available for this run ({type(e).__name__}: {e}). The figures "
                           f"of PyDTS are shown under 'Figures & files'.")
        else:
            st.info("No interactive data stored for this run — see 'Figures & files'.")
    elif view == 'Figures & files':
        gallery(run, key=f"{key}|g")
    elif view == 'Console log':
        st.code(run.get('log') or '(log not stored for this run)', language=None, height=500)
