#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Reliability
# File:         theory_kit.py
# Date:         15.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Building blocks of the theory tabs of the operating modes (gui/core/theory.py and the per-mode modules
theory_m1 … theory_m5): number inputs, numbered sections, symbol tables and the findings box. Every theory tab follows
the same order: idea → equations with a symbol table → inputs → key figures → chart with its reading → findings.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import numpy as np
import pandas as pd
import streamlit as st

from gui.core.style import table

KB = 8.617333262145e-5                                                                        # Boltzmann constant eV/K


#######################################################################################################################
# Helpers
#######################################################################################################################
def f_(x, default=0.0):
    """Float of x, default for NaN / empty / text."""
    try:
        x = float(x)
        return default if np.isnan(x) else x
    except (TypeError, ValueError):
        return default


def num(label, value, key, help=None, **kw):
    """Number input with compact formatting."""
    return st.number_input(label, value=float(value), key=key, format='%g', help=help, **kw)


def intro(text):
    st.caption(text + "  The values start from the current configuration, and changes made here do not modify it.")


def gap(h=1.0):
    st.markdown(f"<div style='height:{h}rem'></div>", unsafe_allow_html=True)


def section(n, title, text=None):
    """Numbered section heading with a one-line explanation."""
    gap(1.2)
    st.markdown(f"#### {n} · {title}")
    if text:
        st.caption(text)


def insights(items, title=None):
    """Derived findings in plain words as the Interpretation of the panel (thin rule, bulb, bullets)."""
    from gui.core.present import interpretation
    interpretation(list(items))


def symbols(rows, label=None, expanded=False, now=False):
    """
    Symbol table: rows = [(symbol, meaning, unit[, typical value])], a collapsed box. Inside a panel tab it is shown at
    the end of the panel, below the Interpretation (now=True shows it in place).
    """
    if not now and st.session_state.get('_ptab_depth', 0) > 0:
        st.session_state.setdefault('_sym_pending', []).append((rows, label, expanded))
        return
    cols = ['Symbol', 'Meaning', 'Unit', 'Typical'][:len(rows[0])]
    label = label or ("Variables of the equations explained (unit, typical value)" if len(cols) == 4 else
                      "Variables of the equations explained (unit)")
    with st.expander(f":material/function: {label}", expanded=expanded):
        table(pd.DataFrame(rows, columns=cols), hide_index=True)


def tiles(items):
    """Key figures [(label, value, help)] as tiles in one row."""
    from gui.core.present import tiles as _t
    _t(items)
