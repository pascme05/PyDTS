#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Black-Box Modeling
# File:         workflow.py
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
This function draws the workflow of every operating mode (and sub-mode) as a flow chart (Graphviz, rendered in the
browser) and explains the steps. The nodes show the settings of the current configuration that control the step;
steps that are skipped with the current settings are drawn dashed. Add one function per mode to WORKFLOWS.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core import forms

import streamlit as st

#######################################################################################################################
# Style
#######################################################################################################################
COL = {'inp': ('#e8f1fc', '#2a78d6'), 'step': ('#ffffff', '#52514e'), 'out': ('#e7f6ef', '#1baf7a'),
       'opt': ('#fdf1e8', '#eb6834'), 'off': ('#f6f6f3', '#c3c2b7')}


def opt(sec, cat, var):
    """'value) label' of a parameter with options."""
    return f"{forms.value(sec, cat, var)}) {forms.option_label(sec, cat, var).split(' (')[0]}"


def val(sec, cat, var, unit=''):
    v = forms.value(sec, cat, var)
    if isinstance(v, float) and forms.find(sec, cat, var)['percent']:
        return f"{100 * v:g} %"
    return f"{v:g}{(' ' + unit) if unit else ''}" if isinstance(v, (int, float)) else str(v)


class Flow:
    """Small builder of a Graphviz flow chart."""

    def __init__(self, rankdir='TB'):
        self.nodes, self.edges, self.ranks = [], [], []
        self.rankdir = rankdir

    def node(self, nid, title, lines=(), kind='step', active=True):
        fill, line = COL[kind if active else 'off']
        txt = f"<<B>{_esc(title)}</B>" + ''.join(f"<BR/><FONT POINT-SIZE='10' COLOR='#52514e'>{_esc(l)}</FONT>"
                                                  for l in lines if l) + ">"
        shape = {'inp': 'note', 'out': 'box', 'opt': 'box', 'step': 'box'}[kind]
        style = 'filled,rounded' + (',dashed' if not active else '') + (',bold' if kind == 'out' else '')
        self.nodes.append(f'"{nid}" [label={txt}, shape={shape}, style="{style}", fillcolor="{fill}", '
                          f'color="{line}", fontcolor="{"#0b0b0b" if active else "#898781"}"];')
        return self

    def edge(self, a, b, label='', active=True):
        lab = f', label="{label}", fontsize=9, fontcolor="#52514e"' if label else ''
        self.edges.append(f'"{a}" -> "{b}" [color="{"#52514e" if active else "#c3c2b7"}"'
                          f'{", style=dashed" if not active else ""}{lab}];')
        return self

    def same(self, *ids):
        self.ranks.append('{rank=same; ' + ' '.join(f'"{i}";' for i in ids) + '}')
        return self

    def dot(self):
        return ('digraph G {\n'
                f'  rankdir={self.rankdir}; bgcolor="transparent"; nodesep=0.35; ranksep=0.4;\n'
                '  node [fontname="Helvetica", fontsize=12, margin="0.18,0.08"];\n'
                '  edge [arrowsize=0.7, penwidth=1.2];\n  ' + '\n  '.join(self.nodes + self.edges + self.ranks) +
                '\n}')


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _legend():
    st.caption(":blue[■] inputs (workbooks, config) · ■ processing step · :orange[■] option / loop · "
               ":green[■] results · dashed: skipped with the current settings")


#######################################################################################################################
# Workflows: mode -> function(cfg, sub) returning (Flow, [(step title, explanation), ...])
#######################################################################################################################
def _example(cfg, sub):
    f = Flow()
    f.node('cfg', 'Configuration', [A.NAME], kind='inp')
    f.node('sim', 'Simulation', ['main(setup, path)'])
    f.node('res', 'Results', ['figures, data'], kind='out')
    f.edge('cfg', 'sim').edge('sim', 'res')
    return f, [('Configuration', "parameters of the GUI"), ('Simulation', "toolkit"), ('Results', "outputs")]


def _pydts(cfg, sub, mode=0):
    E, Dd, P, Md = cfg['Exp'], cfg['Dat'], cfg['Par'], cfg['Mdl']
    split = {0: f"1-fold, {100 * float(Dd.get('rT', 0.9)):g} % train", 1: f"k-fold, k = {E.get('kfold')}",
             2: "transfer: separate test set", 3: "ID based"}.get(E.get('method'), '')
    train, test = E.get('train') == 1, E.get('test') == 1
    frame, feat = P.get('frame') == 1, P.get('feat')
    f = Flow()
    f.node('dat', 'Datasets', [', '.join(str(x) for x in A.to_list(Dd.get('train')))[:40], f".{Dd.get('type')}"],
           kind='inp')
    f.node('split', 'Load and split', [split])
    f.node('pre', 'Pre-processing', [f"normalise X: {Dd.get('inpNorm')}, y: {Dd.get('outNorm')}",
                                     'filter / noise' if Dd.get('inpFil') or Dd.get('inpNoise') else ''])
    f.node('roll', 'Rolled features', [f"EWMA {Md.get('feat_roll.EWMA')}"], kind='opt', active=feat in (2, 3))
    f.node('frm', 'Windows', [f"W = {P.get('window')}, overlap {P.get('overlap')}",
                              'seq2point' if not P.get('outseq') else f"seq2seq {P.get('outseq')}"], active=frame)
    f.node('feat', 'Frame features', ['mean, std, RMS, …'], kind='opt', active=feat in (1, 3))
    if mode == 1:
        f.node('mdl', 'Hyperparameter search', ['Optuna', f"{P.get('model')}"], kind='opt')
    elif mode == 2:
        f.node('mdl', 'Grid search', ['window × overlap', f"{P.get('model')}"], kind='opt')
    else:
        f.node('mdl', 'Train model' if train else 'Load model', [f"{P.get('solver')} · {P.get('model')}",
                                                                 f"{Md.get('epoch')} epochs, lr {Md.get('lr')}"
                                                                 if train else A.model_dir(cfg)])
    f.node('test', 'Test', ['predict, scale back', f"clip [{P.get('outMin'):g}, {P.get('outMax'):g}]"
                            if isinstance(P.get('outMin'), (int, float)) else ''], active=test)
    f.node('res', 'Errors and plots', ['MAE, RMSE, R², MAX', 'feature ranking' if P.get('rank') else ''],
           kind='out', active=test)
    f.edge('dat', 'split').edge('split', 'pre').edge('pre', 'roll', active=feat in (2, 3))         .edge('roll', 'frm', active=feat in (2, 3)).edge('pre', 'frm').edge('frm', 'feat', active=feat in (1, 3))         .edge('feat', 'mdl', active=feat in (1, 3)).edge('frm', 'mdl').edge('mdl', 'test', active=test)         .edge('test', 'res', active=test)
    steps = [('Datasets', "the measured time series of data/ (sheets/variables input and output)."),
             ('Load and split', "src/data/loadData.py separates training, validation and test samples."),
             ('Pre-processing', "src/preprocess.py normalises with the training statistics, filters and adds noise."),
             ('Rolled features', "src/general/featuresRoll.py adds EWMA/EWMS low-pass features (thermal memory)."),
             ('Windows', "src/general/framing.py cuts sliding windows of W samples."),
             ('Frame features', "src/general/features.py replaces each window by statistics."),
             ('Model', "src/train.py trains the model (or loads mdl/…); optiHyp / optiGrid search settings."),
             ('Test', "src/test.py predicts the test windows and scales them back to physical units."),
             ('Errors and plots', "src/general/performance.py computes the errors shown under Results.")]
    return f, steps


WORKFLOWS = {0: _pydts, 1: lambda c, s: _pydts(c, s, 1), 2: lambda c, s: _pydts(c, s, 2)}


#######################################################################################################################
# View
#######################################################################################################################
def view(m):
    cfg = forms.cfg()
    sub = None
    spec = A.MODES[m].get('sub')
    if spec:
        p = forms.find(*spec)
        cur = forms.value(*spec)
        if p['options']:
            names = p['options']
            sub = st.segmented_control("Sub-mode", list(names), default=cur, key=f"wf|{m}",
                                       format_func=lambda k: f"{names[k]}" + (' (selected)' if k == cur else ''))
            sub = cur if sub is None else sub
            if sub != cur:
                st.caption(f"Showing sub-mode {sub}; the configuration runs sub-mode {cur} — change it in "
                           f"*Parameters*.")
    f, steps = WORKFLOWS[m](cfg, sub)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.graphviz_chart(f.dot(), width='content')
        _legend()
    with c2:
        st.markdown("##### What happens")
        for i, (t, txt) in enumerate(steps, start=1):
            st.markdown(f"**{i}. {t}** — {txt}")
        st.caption("The boxes show the settings of the current configuration; change them in *Parameters*, the "
                   "diagram follows.")
