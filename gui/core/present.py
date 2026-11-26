#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Reliability
# File:         present.py
# Date:         14.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function provides the building blocks of the result views in plain language. Every result answers its question
in the same order: the answer (one sentence), the key figures (tiles with plain labels; symbol, equation and
abbreviation on hover), what it means (findings) and one tab per question with one main chart or table and a one-line
reading. In the focus mode (sidebar, on by default) the further tables, equations, charts and long notes are folded
into 'Details' sections and long notes become hover texts; with the focus mode off everything is shown at once.

Abbreviations are explained once in GLOSSARY and shown as underlined terms with a hover text (term), in the column help
of tables (col_help) and in the tile help.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import contextlib
import html
import re

import streamlit as st
from streamlit.delta_generator_singletons import context_dg_stack

#######################################################################################################################
# Glossary: abbreviation -> plain explanation (hover text)
#######################################################################################################################
GLOSSARY = {
    # errors and metrics (src/general/performance.py)
    'MAE': "Mean absolute error: average size of the prediction error, in the unit of the output (e.g. K).",
    'RMSE': "Root mean square error: like the MAE but large errors count more; RMSE ≥ MAE, a big gap means a few "
            "large errors.",
    'R2': "Coefficient of determination: share of the variance of the output that the model explains (1 = perfect, "
          "0 = no better than the mean).",
    'TECA': "Total energy correctly assigned: 1 − Σ|ŷ − y| / (2·Σ|y|); 1 = perfect, common in energy "
            "disaggregation.",
    'SAE': "Signal aggregate error: relative error of the sum (energy) of the prediction, |Σŷ − Σy| / Σy.",
    'ACC': "Accuracy of the on/off states (output above the threshold): share of correctly classified samples.",
    'F1': "F1 score of the on/off states: harmonic mean of precision and recall (1 = perfect).",
    'MAX': "Largest absolute error of all test samples (worst case, e.g. the hot-spot error).",
    # networks
    'DNN': "Dense (fully connected) neural network: the window is flattened and every input connects to every neuron.",
    'CNN': "Convolutional neural network: short filters slide along time and detect local patterns.",
    'LSTM': "Long short-term memory: a recurrent network with a memory cell that carries the state through time.",
    'TRAN': "Transformer: self-attention weighs every sample of the window against every other one.",
    'DAE': "Denoising autoencoder: compresses the window and reconstructs the clean signal.",
    'seq2point': "The network sees a window and predicts one sample of the output (the focus point).",
    'seq2seq': "The network sees a window and predicts a whole sequence of outputs.",
    'epoch': "One pass over all training windows.",
    'EWMA': "Exponentially weighted moving average: a first-order low pass of an input, like one thermal RC element.",
}

_PAT = None


#######################################################################################################################
# Helpers
#######################################################################################################################
def focus():
    """Focus mode of the results (sidebar toggle, on by default)."""
    return bool(st.session_state.get('focus', True))


def strict():
    """Focus mode on a page of an operating mode (or the results browser): footnotes and extra blocks are hidden."""
    return focus() and bool(st.session_state.get('_mode_page', False))


def explain(abbr):
    """Plain explanation of an abbreviation ('' if unknown)."""
    return GLOSSARY.get(abbr, '')


def term(abbr, label=None):
    """Abbreviation as an underlined term with its explanation on hover (HTML for st.markdown(unsafe_allow_html))."""
    txt = explain(abbr)
    lab = html.escape(label or abbr)
    return f'<abbr class="term" title="{html.escape(txt)}">{lab}</abbr>' if txt else lab


def col_help(text, *abbrs):
    """Column / tile help: the text plus the explanation of the abbreviations it uses."""
    extra = [f"{a}: {explain(a)}" for a in abbrs if explain(a)]
    return '\n\n'.join([text] + extra) if extra else text


def terms(text):
    """Markdown text with the known abbreviations (whole words) turned into hover terms."""
    global _PAT
    if _PAT is None:
        keys = sorted((k for k in GLOSSARY if len(k) > 1), key=len, reverse=True)
        _PAT = re.compile(r'(?<![\w/>.\-="\[${`])(' + '|'.join(re.escape(k) for k in keys) + r')(?![\w<\-"\]$}`])')
    return _PAT.sub(lambda m: term(m.group(1)), text)


#######################################################################################################################
# Blocks
#######################################################################################################################
def answer(sentence, level='good'):
    """The answer of the result in one plain sentence, coloured by its status (icon + text, not colour alone)."""
    summary_open()
    box = {'good': st.success, 'warning': st.warning, 'serious': st.warning, 'critical': st.error}.get(level, st.info)
    icon = {'good': ':material/check_circle:', 'warning': ':material/warning:', 'serious': ':material/error:',
            'critical': ':material/cancel:'}.get(level, ':material/info:')
    box(sentence, icon=icon)


def tiles(items, cols=None):
    """
    Key figures as bordered tiles in one row: items = [(plain label, value, hover help[, delta[, delta_color]])].
    The label is plain language, the symbol / abbreviation and the equation belong into the help.
    """
    items = [x for x in items if x]
    if not items:
        return
    cc = st.columns(cols or len(items), gap='small')
    for i, it in enumerate(items):
        lab, val, hlp = it[0], it[1], it[2] if len(it) > 2 else None
        kw = {}
        if len(it) > 3 and it[3] is not None:
            kw['delta'] = it[3]
            kw['delta_color'] = it[4] if len(it) > 4 else 'normal'
        cc[i % len(cc)].metric(lab, val, help=hlp, **kw)   # tile frame: style.inject_css


def note(text, icon=None, **kw):
    """
    Explanation of a result in readable size. In the focus mode a long note (more than ~160 characters) is not shown
    on the mode pages and folded into a hover elsewhere; a short one is shown as it is.
    """
    text = str(text or '')
    if not text.strip():
        return
    if focus() and len(re.sub(r'[*_`]', '', text)) > 160:
        if strict():                                                       # mode pages: no long footnotes
            return
        first = re.split(r'(?<=[.:;])\s', text.strip(), maxsplit=1)[0]
        st.caption(f":material/info: {first if len(first) <= 160 else 'More about this'}", help=text)
        return
    if kw:
        st.markdown(text, **kw)
    else:
        st.markdown(terms(text), unsafe_allow_html=True)                   # abbreviations explained on hover


def reading(text):
    """Reading of the main chart or table = the interpretation line of the panel (thin rule above it)."""
    interpretation(text)


def interpretation(text):
    """
    Closing line of a result panel: what the result means in plain words (shown in both views). A list of findings
    is shown as bullets below the label.
    """
    if isinstance(text, (list, tuple)):
        items = [s for s in text if s]
        if not items:
            return
        body = "\n" + "\n".join(f"- {terms(s)}" for s in items)
    elif text:
        body = " " + terms(text)
    else:
        return
    st.session_state['_interp_n'] = st.session_state.get('_interp_n', 0) + 1
    st.markdown('<div class="panel-read"></div>', unsafe_allow_html=True)           # thin rule above the line
    st.markdown(f":material/lightbulb: **Interpretation:**{body}", unsafe_allow_html=True)


def hint(text, icon=None):
    """Footnote-like extra information: only shown with the focus mode off."""
    if not focus():
        note(text, icon=icon)


@contextlib.contextmanager
def details(label="Details", expanded=False):
    """
    Further tables, equations and charts: on the mode pages not shown in the focus mode (the content is still built,
    in a hidden container, so that the code after it does not change); elsewhere a collapsed section in the focus
    mode; shown directly otherwise.
    """
    if strict():
        n = st.session_state['_hide_n'] = st.session_state.get('_hide_n', 0) + 1
        with st.container(key=f"hide-{n}"):                                  # CSS: display none (style.inject_css)
            yield
    elif focus():
        with st.expander(f"Details: {label}", expanded=expanded, icon=':material/unfold_more:'):
            yield
    else:
        st.markdown(f"**{label}**")
        yield


def essential(full_value, focus_value):
    """Value for the current view: focus_value in the focus mode, full_value otherwise (e.g. column lists)."""
    return focus_value if focus() else full_value


#######################################################################################################################
# Tabs named by the question they answer
#######################################################################################################################
QUESTION = {
    'Accuracy': "How accurate is the prediction?", 'Time series': "How does the prediction follow the measurement?",
    'Error': "Where and how large are the errors?", 'Scatter': "Is the model biased over the range?",
    'Training': "Did the training converge?", 'Features': "Which inputs matter?",
}

# Short tab labels for names that are questions themselves (the question becomes the panel title)
LABEL = {}

# One line below the panel title: what the panel shows
EXPLAIN = {
    'Accuracy': "Errors per output as computed by PyDTS (src/general/performance.py) on the test data.",
    'Time series': "Measured and predicted output over the test data; zoom with the range slider.",
    'Error': "Prediction error ŷ − y over time and its distribution.",
    'Scatter': "Predicted over measured value; on the diagonal the prediction is exact.",
    'Training': "Training and validation loss per epoch from the training log (mdl/…/training_log.csv).",
    'Features': "Feature ranking of PyDTS (Par › rank): importance of every input.",
}


def _depth():
    return len(context_dg_stack.get())


def result_base():
    """
    Start of a result page: marks its nesting level and opens the summary box, which takes everything up to the first
    tab row (answer, key figures, 'What the result means').
    """
    st.session_state['_res_base'] = _depth()
    st.session_state['_sum_open'] = None
    summary_open()


def summary_open(title="Result in brief", sub="The answer, the key figures and what they mean for the design."):
    """Opens the first box of a result (answer, key figures, what it means); closed by the next tab row."""
    ss = st.session_state
    if ss.get('_sum_open') is not None or ss.get('_res_base') is None or _depth() != ss['_res_base']:
        return
    from gui.core.style import panel
    box = panel(title, sub)
    box.__enter__()
    ss['_sum_open'] = _depth()


def summary_close():
    """Closes the summary box if it is open at this level."""
    ss = st.session_state
    d = ss.get('_sum_open')
    if d is not None and _depth() == d:
        context_dg_stack.set(context_dg_stack.get()[:-1])
    ss['_sum_open'] = None


class _PanelTab:
    """
    A tab whose content sits in a round panel (title = the question, sub = what the tab shows). Tabs inside a panel
    stay plain. A panel without an Interpretation line is recorded in session_state['_no_interp'] (smoke tests).
    """

    def __init__(self, tab, label, title, sub, track=True):
        self.tab, self.label, self.title, self.sub, self.box, self.n0 = tab, label, title, sub, None, 0
        self.track = track

    def __enter__(self):
        from gui.core.style import panel
        self.tab.__enter__()
        d = st.session_state.get('_ptab_depth', 0)
        if d == 0:
            self.box = panel(self.title, self.sub)
            self.box.__enter__()
            self.n0 = st.session_state.get('_interp_n', 0)
        st.session_state['_ptab_depth'] = d + 1
        return self

    def __exit__(self, typ, *a):
        st.session_state['_ptab_depth'] = max(0, st.session_state.get('_ptab_depth', 1) - 1)
        if self.box is not None:
            pend = st.session_state.pop('_sym_pending', [])
            if typ is None and pend:                                    # variables boxes below the Interpretation
                from gui.core.theory_kit import symbols
                for rows, label, exp in pend:
                    symbols(rows, label, exp, now=True)
            if self.track and typ is None and st.session_state.get('_interp_n', 0) == self.n0:
                st.session_state.setdefault('_no_interp', []).append(self.label)
            self.box.__exit__(typ, *a)
        self.tab.__exit__(typ, *a)
        return False


def panel_tabs(names, subs=None, titles=None, skip=()):
    """
    st.tabs whose content is wrapped in round panels (only the top level; tabs inside a panel stay plain).
    names: tab labels (a leading '1 · ' is dropped in the panel title); subs / titles: label -> explanation / title;
    skip: labels that need no Interpretation line (overview, references, inputs) or True for all.
    """
    summary_close()
    labels = [str(n) for n in names]
    tabs = st.tabs(labels)
    clean = [re.sub(r'^\d+\s*·?\s*', '', n) for n in labels]
    return [_PanelTab(t, n, (titles or {}).get(n, c), (subs or {}).get(n), track=not (skip is True or n in skip))
            for t, c, n in zip(tabs, clean, labels)]


def _base(n):
    """Tab name without a count in brackets ('Not applicable (3)' -> 'Not applicable')."""
    return re.sub(r'\s*\(\d+\)$', '', n)


def qtabs(names):
    """
    Result tabs: the tab is labelled with the topic (LABEL or the name), its panel is titled with the question it
    answers (QUESTION) and explained in one line (EXPLAIN).
    """
    summary_close()
    tabs = st.tabs([LABEL.get(_base(n), n) for n in names])
    return [_PanelTab(t, n, QUESTION.get(_base(n), n), EXPLAIN.get(_base(n))) for t, n in zip(tabs, names)]
