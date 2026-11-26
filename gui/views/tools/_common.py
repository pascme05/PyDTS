#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Theory & Tools
# File:         _common.py
# Date:         05.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Shared building blocks of the Theory & Tools pages: the question a tool answers, symbol tables, the box 'What the
result means', literature lines, key figures and export buttons. Every tool follows the same order: question →
inputs → equation with symbols → key figures → charts with 'How to read' → what the result means → export.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import html
import io
import json
from contextlib import contextmanager

import numpy as np
import pandas as pd
import streamlit as st

from gui.core.style import table, fmt, SERIES
from gui.core import literature as lit

KEY = 'tl'                                                                     # prefix of all widget keys


#######################################################################################################################
# Functions
#######################################################################################################################
GAP = 'medium'                                                                 # gap of every grid row
CSS = (
    # box: white, thin frame, round corners; at least as high as its grid column (boxes of a row end together) but
    # growing with its content (min-height, not height: nothing overlaps the next row)
    'div[class*="st-key-tbox-"]{background:#ffffff;border:1px solid #d6d4cc;border-radius:0.6rem;'
    'padding:1rem 1.25rem 0.9rem 1.25rem;box-sizing:border-box;flex:1 1 auto;gap:0.75rem;}'
    'div[data-testid="stColumn"]:has(> div > div[class*="st-key-tbox-"]) > div{min-height:100%;}'
    'div[data-testid="stColumn"]:has(> div > div[class*="st-key-tbox-"]) > div > div[class*="st-key-tbox-"]'
    '{min-height:100%;}'
    # header: explanation line and rule below the title; footer: rule + explanation in grey
    '.tbox-sub{font-size:0.9rem;color:#52514e;margin:0.15rem 0 0 0;line-height:1.45;}'
    '.tbox-rule{border-bottom:1px solid #e1e0d9;margin:0.2rem 0 0.35rem 0;}'
    '.tbox-foot{font-size:0.86rem;color:#6b6a66;border-top:1px solid #e1e0d9;padding-top:0.6rem;margin-top:0.35rem;'
    'line-height:1.5;}'
    # rows inside a box: room between wrapped fields, no field touches the next row
    'div[class*="st-key-tbox-"] div[data-testid="stHorizontalBlock"]{row-gap:0.75rem;}'
    # metrics of the key figures: long values wrap instead of running into the neighbour
    'div[class*="st-key-tbox-"] [data-testid="stMetricValue"] > div{white-space:normal;overflow-wrap:anywhere;'
    'line-height:1.2;}'
)


def page_head(title, text):
    st.session_state['_tbox_n'] = 0                                            # box keys restart per run
    st.session_state['_tbox_depth'] = 0
    st.session_state['_tools_page'] = True                                     # chart help as box footer (style.show)
    st.html(f"<style>{CSS}</style>")
    st.title(title)
    st.caption(text + " The tools are self-contained calculators: they do not change the configuration or the "
                      "workbooks. Values start from typical"
                      " settings and can be exported.")


def question(q, text=None, art=None):
    """
    The engineering question the tool answers (first line of every tool); art: name of a task picture
    (gui/views/tools/task_art.py) shown next to it.
    """
    st.session_state['_sec_n'] = 0                                             # every tool numbers its own steps
    pic = ''
    if art:
        from gui.views.tools import task_art
        pic = task_art.svg(art)
    cols = st.columns([3, 2], vertical_alignment='center') if pic else [st.container()]
    with cols[0].container(border=True):
        st.markdown(f":material/help: **Question.** {q}")
        if text:
            st.caption(text)
    if pic:
        cols[1].markdown(pic, unsafe_allow_html=True)


def section(n, title, text=None, divider=True):
    """
    Numbered step heading with a coloured bar (n = None: unnumbered, e.g. 'Result'). Separates the inputs, the result
    and the interpretation of a tool so that a page reads as a sequence of steps.
    """
    if divider:
        st.divider()
    if n is None:                                                              # 'Result' etc.: next step number
        n = st.session_state.get('_sec_n', 0) + 1
    if isinstance(n, int):
        st.session_state['_sec_n'] = n
    lab = f"{n} · {title}"
    col = SERIES[0]
    st.markdown(f"<div style='border-left:4px solid {col};padding:2px 0 2px 10px;margin:0 0 6px 0;"
                f"font-weight:600;font-size:1.08rem'>{lab}</div>", unsafe_allow_html=True)
    if text:
        st.caption(text)


def grid(spec=2, align='top'):
    """
    One row of the page grid: equal columns (spec = n) or relative widths (spec = list), the same gap everywhere.
    align 'top' for boxes (they start and, with the CSS of page_head, end together), 'bottom' for input fields
    (fields line up although their labels wrap differently).
    """
    return st.columns(spec, gap=GAP, vertical_alignment=align)


@contextmanager
def box(title, sub=None, foot=None, icon=None):
    """
    Box of things that belong together: header (icon, title, one line saying what the box is for), thin rule, the
    content, and a grey footer with the explanation of the content (how to read it, where the values come from).
    foot can be a callable that returns the text, for a footer that depends on the content.
    """
    n = st.session_state.get('_tbox_n', 0) + 1
    st.session_state['_tbox_n'] = n
    st.session_state['_tbox_depth'] = st.session_state.get('_tbox_depth', 0) + 1
    outer = st.session_state.get('_tbox_title')
    st.session_state['_tbox_title'] = title                                    # a chart of the same title drops it
    try:
        with st.container(key=f"tbox-{n}"):
            st.markdown(f":material/{icon}: **{title}**" if icon else f"**{title}**")
            if sub:
                st.markdown(f"<div class='tbox-sub'>{html.escape(sub)}</div>", unsafe_allow_html=True)
            st.markdown("<div class='tbox-rule'></div>", unsafe_allow_html=True)
            yield
            txt = foot() if callable(foot) else foot
            if txt:
                st.markdown(f"<div class='tbox-foot'>{html.escape(txt)}</div>", unsafe_allow_html=True)
    finally:
        st.session_state['_tbox_depth'] -= 1
        st.session_state['_tbox_title'] = outer


def in_box():
    """True inside a box (helpers that box themselves then render plainly)."""
    return st.session_state.get('_tbox_depth', 0) > 0


@contextmanager
def _maybe_box(*a, **kw):
    """A box, unless already inside one."""
    if in_box():
        yield
    else:
        with box(*a, **kw):
            yield


def foot(text):
    """Footer line inside a box for content that changes (e.g. per tab): thin rule and grey explanation."""
    st.markdown(f"<div class='tbox-foot'>{html.escape(text)}</div>", unsafe_allow_html=True)


def expert():
    """True when the global sidebar toggle 'Expert mode' is on (advanced inputs are shown)."""
    return bool(st.session_state.get('expert', False))


def hidden(n, what="advanced inputs"):
    """Caption for inputs that are hidden outside the expert mode (they keep their default values)."""
    if n and not expert():
        st.caption(f":material/visibility_off: {n} {what} use their default values — switch on *Expert mode* in "
                   f"the sidebar to change them.")


def equations(steps, sym=None, title="Show the equations and symbols", expanded=False):
    """
    Numbered equation steps [(title, latex, meaning)] in the order input → intermediate → result, followed by the
    symbol table (rows symbol, meaning, unit). Outside a box it opens its own box 'Equations and symbols'.
    """
    with _maybe_box("Equations and symbols", "How the result follows from the inputs, step by step.",
                    foot="Every symbol is defined in the table below the equations.", icon='function'), \
            st.expander(title, icon=':material/function:', expanded=expanded):
        for i, s in enumerate(steps, 1):
            st.markdown(f"**{i} · {s[0]}**")
            for tex in ([s[1]] if isinstance(s[1], str) else s[1]):
                st.latex(tex)
            if len(s) > 2 and s[2]:
                st.caption(s[2])
        if sym:
            st.markdown("**Symbols**")
            table(pd.DataFrame(sym, columns=['symbol', 'meaning', 'unit']), hide_index=True,
                  column_config={'meaning': st.column_config.TextColumn(width='large')})


def interpret(paragraphs, title="Interpretation"):
    """
    Plain-language interpretation at the bottom of a tool: a few paragraphs of prose built from the current results
    (what was computed, the key number and its meaning, which input drives it, what to do).
    """
    with _maybe_box(title, "What the result means and what to do, built from the current results.",
                    icon='lightbulb'):
        for p in paragraphs:
            if p:
                st.markdown(p)


def symbols(rows, title="Symbols of the equation", expanded=False):
    """Symbol table (symbol, meaning, unit) below an equation."""
    with st.expander(title, icon=':material/function:', expanded=expanded):
        table(pd.DataFrame(rows, columns=['symbol', 'meaning', 'unit']), hide_index=True,
              column_config={'meaning': st.column_config.TextColumn(width='large')})


def findings(lines, title="What the result means"):
    """Plain-language interpretation of the result (list of markdown sentences)."""
    with _maybe_box(title, "What the result means, built from the current results.", icon='lightbulb'):
        st.markdown("\n".join(f"- {x}" for x in lines if x))


def refs(keys, prefix="Literature: "):
    """Linked literature line."""
    keys = [k for k in dict.fromkeys(keys) if k in lit.REFS]
    if keys:
        st.caption(prefix + " · ".join(f"[{lit.label(k)}]({lit.url(k)})" for k in keys))


def tiles(items, title="Key figures", sub="The answer for the current inputs.", foot=None):
    """
    Row of key figures [(label, value, help)]; outside a box in its own box 'Key figures'. More than five figures
    wrap into two rows so that long values keep their room.
    """
    with _maybe_box(title, sub, foot=foot or "Hover a figure for its definition.", icon='speed'):
        per = len(items) if len(items) <= 5 else (len(items) + 1) // 2
        for i in range(0, len(items), per):
            for c, it in zip(grid(per), items[i:i + per]):
                c.metric(it[0], it[1], help=it[2] if len(it) > 2 else None)


def num(c, label, value, key, help=None, **kw):
    """Number input with the generic %g format (integer input for format '%d')."""
    f = kw.pop('format', '%g')
    cast = int if f == '%d' else float
    kw = {k: cast(v) if k in ('min_value', 'max_value', 'step') else v for k, v in kw.items()}
    if f == '%d':
        return c.number_input(label, value=int(value), key=f"{KEY}|{key}", help=help, format=f, **kw)
    return c.number_input(label, value=float(value), key=f"{KEY}|{key}", help=help, format=f, **kw)


def export(df, name, label="Download CSV", key=None, note=None):
    """CSV download of a table (with an optional note on where to paste it)."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(label, buf.getvalue(), file_name=f"{name}.csv", mime='text/csv',
                       icon=':material/download:', key=key or f"{KEY}|dl|{name}")
    if note:
        st.caption(note)


def export_json(obj, name, label="Download JSON", key=None):
    st.download_button(label, json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o)), file_name=f"{name}.json",
                       mime='application/json', icon=':material/download:', key=key or f"{KEY}|dj|{name}")


def chain():
    """Hand-over between the tools in the session."""
    return st.session_state.setdefault(f"{KEY}|chain", {})


def f_life(x, unit):
    """Life with its unit and a conversion to years for hours."""
    if unit == 'h' and np.isfinite(x):
        return f"{fmt(x)} h ({fmt(x / 8760)} a)"
    return f"{fmt(x)} {unit}"
