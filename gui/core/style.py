#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Reliability
# File:         style.py
# Date:         21.10.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function defines the common style of the GUI charts (plotly template, categorical/sequential/diverging colours)
and small chart helpers. Categorical colours are assigned in fixed order and follow the entity (e.g. a component keeps
its colour on every chart).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import re
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

#######################################################################################################################
# Colours
#######################################################################################################################
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SEQ = [[0.0, '#cde2fb'], [0.25, '#86b6ef'], [0.5, '#3987e5'], [0.75, '#1c5cab'], [1.0, '#0d366b']]
DIV = [[0.0, '#1c5cab'], [0.25, '#86b6ef'], [0.5, '#f0efec'], [0.75, '#ec8a85'], [1.0, '#b32d2d']]
STATUS = {'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a', 'critical': '#d03b3b'}
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
SYS_COLOR = INK2                                                       # the system total is drawn neutral


def color(i):
    """Categorical colour of slot i (folded onto the eight slots, callers keep series counts small)."""
    return SERIES[i % len(SERIES)]


def entity_colors(names):
    """Fixed colour per entity name; 'Sys' is drawn neutral."""
    out, k = {}, 0
    for n in names:
        if str(n) == 'Sys':
            out[n] = SYS_COLOR
        else:
            out[n] = color(k)
            k += 1
    return out


#######################################################################################################################
# Template
#######################################################################################################################
_axis = dict(gridcolor=GRID, gridwidth=1, linecolor=AXIS, zerolinecolor=AXIS, ticks='outside', tickcolor=AXIS,
             exponentformat='power', separatethousands=True,              # 10⁻⁴ instead of SI suffixes like 100µ
             tickfont=dict(color=INK2, size=12), title=dict(font=dict(color=INK2, size=13)), automargin=True)
pio.templates['gui'] = go.layout.Template(layout=dict(
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK, size=13),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=SURFACE, colorway=SERIES,
    xaxis=_axis, yaxis=_axis, margin=dict(l=10, r=10, t=84, b=10),
    # title in the first row of the top margin, legend in the second row (they never overlap)
    legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0, font=dict(color=INK2)),
    hoverlabel=dict(bgcolor='white', bordercolor=AXIS, font=dict(color=INK)),
    title=dict(font=dict(size=15, color=INK), x=0, xanchor='left', xref='paper', y=1, yref='container',
               yanchor='top', pad=dict(t=10)),
    bargap=0.25, colorscale=dict(sequential=SEQ, diverging=DIV)))
pio.templates.default = 'gui'


#######################################################################################################################
# Helpers
#######################################################################################################################
def show(fig, height=380, key=None, legend_below=False, info=None):
    """
    Renders a figure with the GUI template and a consistent height (legend below for narrow columns). An
    explanation of the chart (argument info or gui/core/chart_help.py for the calling view) is shown on hover over
    the info line above the chart.
    """
    if info is None:
        from gui.core.chart_help import lookup
        caller = sys._getframe(1)
        scope = caller.f_code.co_name
        if caller.f_globals.get('__name__', '').endswith('theory') and not scope.startswith('_theory'):
            scope = f"theory.{scope}"                   # theory.mode0 must not get the help of viz_results.mode0
        info = lookup(scope, key)
    tools = st.session_state.get('_tools_page', False)                          # Theory & Tools: help as footer
    if tools:
        from gui.views.tools import _common as cm
        title = re.sub(r'<[^>]+>', ' ', str(fig.layout.title.text or '')).strip()
        if not cm.in_box():                                                    # chart alone: own box, title on top
            fig.update_layout(title=None, margin=dict(t=30))
            with cm.box(title or "Chart", icon='show_chart'):
                return show(fig, height, key, legend_below, info)
        if title and title == st.session_state.get('_tbox_title'):          # same as the box header: once only
            fig.update_layout(title=None, margin=dict(t=30))
    if info and not tools and not (st.session_state.get('focus', True) and st.session_state.get('_mode_page')):
        st.caption(":material/info: How to read this chart", help=info)
    fix_log_labels(fig)
    fig.update_layout(height=height)
    if legend_below or any(a.yanchor == 'bottom' and (a.y or 0) >= 0.9 for a in fig.layout.annotations or []):
        fig.update_layout(legend=dict(y=-0.3, yanchor='top'), margin=dict(b=100))      # subplot titles on top
    st.plotly_chart(fig, theme=None, key=key, config={'displaylogo': False, 'toImageButtonOptions': {'scale': 2}})
    if info and tools:
        import html as _html
        st.markdown(f"<div class='tbox-foot'>{_html.escape(info)}</div>", unsafe_allow_html=True)


def _axis_type(fig, ref):
    """Type of the data axis behind a reference such as 'x', 'x2', 'y3 domain' (None for paper/domain refs)."""
    ref = str(ref or '')
    if not ref or ref == 'paper' or ' ' in ref:
        return None
    ax = fig.layout[f"{ref[0]}axis{ref[1:]}"]
    return getattr(ax, 'type', None)


def fix_log_labels(fig):
    """
    Moves the labels of add_hline/add_vline lines on log axes to log10 of the value. Plotly draws the line itself in
    data coordinates, but expects annotation positions on a log axis as log10(value); plotly.py copies the raw value,
    so a label at 'target 100 FIT' sat at 10^100 and the autorange of the axis grew up to there.

    The labels are found by their line: a line shape with x0 == x1 (vline) or y0 == y1 (hline) that spans the domain
    of the other axis, and an annotation on the same axis reference at the same value.
    """
    shapes, anns = fig.layout.shapes or (), fig.layout.annotations or ()
    if not shapes or not anns:
        return fig
    vl = {(s.xref, s.x0) for s in shapes if s.type == 'line' and s.x0 == s.x1 and 'domain' in str(s.yref)}
    hl = {(s.yref, s.y0) for s in shapes if s.type == 'line' and s.y0 == s.y1 and 'domain' in str(s.xref)}
    for a in anns:
        if a.name == '_lg':                                         # already converted (figure shown twice)
            continue
        done = False
        if 'domain' in str(a.yref) and (a.xref, a.x) in vl and _axis_type(fig, a.xref) == 'log':
            a.x = np.log10(a.x) if a.x and a.x > 0 else None
            done = True
        if 'domain' in str(a.xref) and (a.yref, a.y) in hl and _axis_type(fig, a.yref) == 'log':
            a.y = np.log10(a.y) if a.y and a.y > 0 else None
            done = True
        if done:
            a.name = '_lg'
            if a.x is None or a.y is None:
                a.visible = False
    return fig


def line(x, y, name, i=0, dash=None, width=2, **kw):
    return go.Scatter(x=x, y=y, name=name, mode=kw.pop('mode', 'lines'),
                      line=dict(color=kw.pop('c', color(i)), width=width, dash=dash), **kw)


def band(x, lo, hi, name, c, opacity=0.18, showlegend=True):
    """Filled confidence band between lo and hi (two traces)."""
    rgba = _rgba(c, opacity)
    return [go.Scatter(x=x, y=hi, mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False),
            go.Scatter(x=x, y=lo, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=rgba, name=name,
                       hoverinfo='skip', showlegend=showlegend)]


def _rgba(hex_color, a):
    h = hex_color.lstrip('#')
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def downsample(n, max_points=4000):
    """Stride that keeps long time series responsive in the browser."""
    return max(1, int(n // max_points))


SCI_HI, SCI_LO = 1e6, 1e-3                                  # outside this range numbers are shown scientifically


def fmt(x, unit='', digits=3):
    """
    Number for tiles, captions and labels: `digits` significant digits, thousands separators, no exponent between
    1e-3 and 1e6 (1,192 · 0.0959 · 146), compact scientific notation outside (4.36e-8, 2.5e6).
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return '–'
    if not np.isfinite(x):
        return '–' if x != x else ('∞' if x > 0 else '−∞')
    if x == 0:
        s = '0'
    elif abs(x) >= SCI_HI or abs(x) < SCI_LO:
        m, e = f"{x:.{max(digits - 1, 0)}e}".split('e')
        m = m.rstrip('0').rstrip('.') if '.' in m else m
        s = f"{m}e{int(e)}"
    else:
        d = max(0, digits - 1 - int(np.floor(np.log10(abs(x)))))
        s = f"{round(x, d):,.{d}f}"
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
    return f"{s} {unit}".strip()


def fmt_p(p, digits=3, ppm_below=1e-2):
    """
    Probability for tiles and captions: small values (below ppm_below, default 1 %) in ppm (1 ppm = 1e-6, e.g.,
    '420 ppm', '0.35 ppm'), larger values in percent ('3.5 %'), since the failure probabilities of robust designs are
    usually far below 1 %.
    """
    try:
        p = float(p)
    except (TypeError, ValueError):
        return '–'
    if not np.isfinite(p):
        return fmt(p)
    if abs(p) < ppm_below:
        return fmt(1e6 * p, 'ppm', digits)
    return fmt(100 * p, '%', digits)


GENERIC = {None, '%g', '%.2g', '%.3g', '%.4g', '%.5g', '%.6g'}


def num_format(values, digits=4):
    """printf format for a numeric table column: integers, fixed decimals or scientific (see fmt)."""
    v = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    a = np.abs(v)
    amax = a.max()
    nz = a[a > 0]
    amin = nz.min() if nz.size else 0.0
    if np.all(v == np.round(v)) and amax < SCI_HI:                    # exact integers only
        return '%d'
    if amax >= SCI_HI or (amin and amin < SCI_LO) or (amin and amax / amin > 1e5):
        return f"%.{digits - 2}e"
    d = int(np.clip(digits - 1 - np.floor(np.log10(amax)), 0, digits))
    return f"%.{d}f"


def table(df, column_config=None, **kw):
    """st.dataframe with an automatic number format per numeric column (explicit formats are kept)."""
    cfg = dict(column_config or {})
    if isinstance(df, pd.DataFrame):
        for c in df.columns:
            col = df[c]
            if not (pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)):
                continue
            entry = cfg.get(c)
            if isinstance(entry, dict):
                tc = entry.get('type_config') or {}
                if tc.get('type') not in (None, 'number') or tc.get('format') not in GENERIC:
                    continue
                f = num_format(col)
                if f:
                    cfg[c] = {**entry, 'type_config': {**tc, 'type': 'number', 'format': f}}
            elif entry is None:
                f = num_format(col)
                if f:
                    cfg[c] = st.column_config.NumberColumn(format=f)
    return st.dataframe(df, column_config=cfg, **kw)


#######################################################################################################################
# Cards: tinted, numbered groups of parameters and inputs
#######################################################################################################################
# Role -> (light tint, accent). The same role has the same colour on every page: question blue, decision amber,
# sizing violet, models green, risk red, scaling / environment orange, numerics and outputs grey.
CAT_TINT = {'blue': ('#f3f7fd', '#2a78d6'), 'amber': ('#fdf8ec', '#eda100'), 'violet': ('#f5f3fd', '#6250d6'),
            'green': ('#f1faf5', '#1baf7a'), 'red': ('#fdf3f2', '#e34948'), 'orange': ('#fdf5f0', '#eb6834'),
            'grey': ('#f7f7f5', '#898781')}


def inject_css():
    """Shared CSS of the GUI: tinted cards with a left accent, separators between grid columns inside a card,
    a readable minimum size of captions and the hover underline of explained terms."""
    css = []

    def fields(pre=''):
        """Selector of the frames of all editable fields (Streamlit >= 1.66 with react-aria widgets; the older
        baseweb frames are kept for older Streamlit versions)."""
        fr = ['div[data-testid="stTextInputRootElement"]', 'div[data-testid="stNumberInputContainer"]',
              'div[data-testid="stTextAreaRootElement"]', 'div[data-testid="stTextArea"] div:has(> textarea)',
              'div[data-testid="stSelectbox"] .e1oobldh0', 'div[data-testid="stSelectbox"] div:has(> input)',
              'div[data-testid="stMultiSelect"] .ew56mvo0',
              'div[data-testid="stMultiSelect"] div:has(> div[data-testid="stMultiSelectTagsContainer"])',
              'div[data-testid="stDateInputField"]', 'div[data-testid="stTimeInput"] div:has(> input)',
              'div[data-baseweb="input"]', 'div[data-baseweb="select"]>div', 'div[data-baseweb="textarea"]']
        return ','.join(f'{pre} {f}'.strip() for f in fr)

    # fields outside the cards: white with a thin border instead of the grey of the theme
    css.append(f'{fields()}{{background:#ffffff !important;border-color:#c9c7bf !important;}}')
    for role, (bg, acc) in CAT_TINT.items():
        k = f'div[class*="st-key-card-{role}-"]'
        css.append(f'{k}{{background:{bg};border-left:4px solid {acc} !important;}}')
        css.append(f'{k} .card-num{{background:{acc};}}')
        # editable fields: the card colour, less transparent than the card, with a thin border of the accent
        css.append(f'{fields(k)}{{background:{acc}1c !important;border-color:{acc}66 !important;}}')
        css.append(f'{fields(k).replace(",", ":focus-within,")}:focus-within{{border-color:{acc} !important;}}')
    # inner elements transparent, so the colour of the frame shows through (input, text area, step buttons)
    css.append('div[data-testid="stTextInputRootElement"] input,div[data-testid="stNumberInputContainer"] input,'
               'div[data-testid="stNumberInputContainer"] button,div[data-testid="stTextArea"] textarea,'
               'div[data-testid="stSelectbox"] input,div[data-testid="stMultiSelect"] input,'
               'div[data-testid="stDateInputField"] input,div[data-baseweb="input"] input,'
               'div[data-baseweb="base-input"],div[data-baseweb="textarea"] textarea'
               '{background:transparent !important;}')
    # a number field (old Streamlit) has an input frame inside its container: one frame only
    css.append('div[data-testid="stNumberInputContainer"] div[data-baseweb="input"]'
               '{border:none !important;background:transparent !important;}')
    css += [
        'div[class*="st-key-card-"] div[data-testid="stColumn"] + div[data-testid="stColumn"]'
        '{border-left:1px solid #e1e0d9;padding-left:0.9rem;}',
        '.card-head{display:flex;align-items:center;gap:0.6rem;margin:0 0 0.2rem 0;font-weight:650;'
        'font-size:1.05rem;}',
        '.card-num{color:white;border-radius:50%;min-width:1.6rem;height:1.6rem;display:inline-flex;'
        'align-items:center;justify-content:center;font-size:0.85rem;font-weight:700;}',
        '.card-sub{font-size:0.95rem;color:#52514e;margin:0 0 0.4rem 0;}',
        '.card-line{border-bottom:1px solid #d6d4cc;margin:-0.6rem 0 0.2rem 0;}',
        'div[data-testid="stCaptionContainer"]{font-size:0.92rem;}',
        'abbr.term{text-decoration:underline dotted #898781;cursor:help;}',
        'div[data-testid="stMetric"]{border:1px solid #e1e0d9;border-radius:0.5rem;padding:0.5rem 0.8rem;'
        'background:#ffffff;height:100%;}',
        # panels: round white boxes like the cards of the parameters (results, theory, inputs); thin rules
        # between explanation, content and interpretation
        'div[class*="st-key-panel-"]{background:#ffffff;border:1px solid #d6d4cc;border-left:6px solid #0b0b0b;'
        'border-radius:0.75rem;padding:1rem 1.3rem 1.1rem 1.3rem;}',
        'div[class*="st-key-hide-"]{display:none;}',                 # hidden content of the focus mode
        # grid: columns of one row start at the same height, charts of a row have the same width
        'div[class*="st-key-panel-"] div[data-testid="stHorizontalBlock"]{align-items:flex-start;gap:1.5rem;}',
        'div[class*="st-key-panel-"] .card-head{font-size:1.15rem;}',
        'div[class*="st-key-panel-"] .card-line{margin:0.1rem 0 0.6rem 0;}',
        '.panel-read{border-top:1px solid #d6d4cc;margin:0.6rem 0 0.1rem 0;}',
    ]
    st.html(f"<style>{''.join(css)}</style>")


def card_head(num, title, sub=None, badge=''):
    """Header strip of a card: number disc, title, optional badge and one line of explanation."""
    import html as _h
    n = f'<span class="card-num">{num}</span>' if num not in (None, '') else ''
    st.markdown(f'<div class="card-head">{n}<span>{_h.escape(str(title))}</span></div>', unsafe_allow_html=True)
    if badge:
        st.markdown(badge)
    if sub:
        st.markdown(f'<div class="card-sub">{_h.escape(str(sub))}</div>', unsafe_allow_html=True)


_CARD_N = {}


def card(role='grey', num=None, title=None, sub=None, badge='', key=None):
    """Bordered, tinted container with a numbered header; use as a context manager (with card(...):)."""
    if key is None:
        cnt = st.session_state.setdefault('_card_n', 0) + 1
        st.session_state['_card_n'] = cnt
        key = f"auto{cnt}"
    box = st.container(border=True, key=f"card-{role if role in CAT_TINT else 'grey'}-{key}")
    if title:
        with box:
            card_head(num, title, sub, badge)
    return box


def panel_reset():
    """Kept for compatibility: the panel keys restart once per script run (gui/app.py), never in the middle."""
    return None


def panel(title, sub=None, num=None, rule=True):
    """
    Round white box of a view (results, theory, inputs), like the cards of the parameters: the title says what the
    box is about, one line below it what it shows, a thin rule, then the content and (results) the interpretation
    below a second rule. Returns the container, so content (and later the interpretation) can be written into it
    with `with box:`.
    """
    n = st.session_state.get('_panel_n', 0) + 1
    st.session_state['_panel_n'] = n
    if st.session_state.get('_ptab_depth', 0) > 0:                       # inside a panel: no box in a box
        box = st.container()
    else:
        box = st.container(key=f"panel-{n}")
    with box:
        card_head(num, title, sub)
        if rule:
            st.markdown('<div class="card-line"></div>', unsafe_allow_html=True)    # rule under the explanation
    return box


def group_head(title):
    """Small header with a thin line below, separating the sub-groups inside a card."""
    st.markdown(f"**{title}**")                                  # markdown: badges like :gray-badge[...] render
    st.markdown('<div class="card-line"></div>', unsafe_allow_html=True)


def status_badge(text, level):
    """Status shown as icon + label (colour never carries the meaning alone)."""
    icon = {'good': ':material/check_circle:', 'warning': ':material/warning:', 'serious': ':material/error:',
            'critical': ':material/cancel:'}[level]
    col = {'good': 'green', 'warning': 'orange', 'serious': 'orange', 'critical': 'red'}[level]
    st.markdown(f":{col}-badge[{icon} {text}]")
