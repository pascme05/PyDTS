#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Black-Box Modeling
# File:         data.py
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
Datasets of PyDTS for the GUI: list of the files in data/, reading of one dataset in the same way as
src/data/loadData.py (xlsx/csv: sheets input, output, units; mat: input, output, labelInp, labelOut, unitInp, unitOut),
cached by file and modification time, and the split of the samples into training, validation and test data for the
current configuration (as loadData does it, without loading the data twice).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import os

import numpy as np
import pandas as pd
import streamlit as st

from gui import adapter as A

DATA_DIR = os.path.join(A.ROOT, 'data')


#######################################################################################################################
# Files
#######################################################################################################################
def datasets(typ=None):
    """Names of the datasets in data/ (without extension), optionally of one file type."""
    if not os.path.isdir(DATA_DIR):
        return []
    out = []
    for f in sorted(os.listdir(DATA_DIR)):
        n, e = os.path.splitext(f)
        if e.lower() in ('.xlsx', '.csv', '.mat') and (typ is None or e.lower() == '.' + str(typ).lower()):
            out.append(n)
    return out


def path(name, typ):
    return os.path.join(DATA_DIR, f"{name}.{typ}")


@st.cache_data(show_spinner="Loading dataset …", max_entries=6)
def _read(filename, _mtime):
    """(X, y, units of X, units of y) of a dataset file, like src/data/loadData.py."""
    if filename.lower().endswith('.mat'):
        import scipy.io
        raw = scipy.io.loadmat(filename)
        li = [str(s).strip() for s in raw['labelInp']]
        lo = [str(s).strip() for s in raw['labelOut']]
        X = pd.DataFrame(raw['input'], columns=li)
        y = pd.DataFrame(raw['output'], columns=lo)
        ui = [str(s).strip() for s in raw['unitInp']]
        uo = [str(s).strip() for s in raw['unitOut']]
    else:
        X = pd.read_excel(filename, sheet_name='input')
        y = pd.read_excel(filename, sheet_name='output')
        u = pd.read_excel(filename, sheet_name='units')
        ui = [str(v).strip() for v in u['Input'].dropna().values]
        uo = [str(v).strip() for v in u['Output'].dropna().values]
    return X, y, dict(zip(X.columns, ui)), dict(zip(y.columns, uo))


def load(name, typ):
    """Dataset (X, y, units X, units y) or None if the file does not exist."""
    f = path(name, typ)
    if not os.path.isfile(f):
        return None
    return _read(f, os.path.getmtime(f))


def unit(u):
    """Readable unit ('degC' -> '°C')."""
    u = str(u or '').strip()
    return {'degC': '°C', 'deC': '°C', 'sec': 's', '-': ''}.get(u, u)


#######################################################################################################################
# Split of the samples
#######################################################################################################################
def split(cfg, n):
    """
    Share of the n samples of the first training dataset that is used for training, validation and testing with the
    current data split (Exp › method), as in loadData: [(label, start, end)] in samples; None for the transfer split.
    """
    m = cfg['Exp'].get('method')
    rT, rV = float(cfg['Dat'].get('rT', 0.9)), float(cfg['Dat'].get('rV', 0.2))
    if m == 0:
        nT = int(round(n * rT))
        nTr = int(round(nT * (1 - rV)))
        return [('training', 0, nTr), ('validation', nTr, nT), ('test', nT, n)]
    if m == 1:
        k = int(cfg['Exp'].get('kfold', 10) or 10)
        return [(f"fold {i + 1}", int(i * n / k), int((i + 1) * n / k)) for i in range(k)]
    return None


def summary(X, y, ux, uy, fs):
    """Table of the channels: role, unit, min, mean, max, std."""
    rows = []
    for role, df, u in (('input', X, ux), ('output', y, uy)):
        for c in df.columns:
            if c in ('time', 'id'):
                continue
            v = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float)
            rows.append({'channel': c, 'role': role, 'unit': unit(u.get(c)), 'min': np.nanmin(v),
                         'mean': np.nanmean(v), 'max': np.nanmax(v), 'std': np.nanstd(v),
                         'NaN': int(np.isnan(v).sum())})
    return pd.DataFrame(rows)
