#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Black-Box Modeling
# File:         app.py
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
Browser GUI of the toolkit (runs locally). Start with 'python start_gui.py' or 'streamlit run gui/app.py' from the
root folder of the repository.
Pages:  1) Project:     run settings, input files, shared parameters; input data
        2) Modes:       workflow, parameters, inputs, theory, run and results of every operating mode
        3) Results:     browser of all results folders
        4) Help:        tutorials, documentation (docs/GUI.md) and parameter reference
The configuration is edited in the GUI and saved/loaded as JSON (sidebar); config workbooks can be imported.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from gui import adapter as A
for p in [os.path.normpath(os.path.join(ROOT, s)) for s in A.SRC_PATHS]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

from gui.core import config as conf
from gui.core import forms
from gui.core import tutorials as tut
from gui.core import style
from gui.views import project, inputs, mode, results, docs, tutorials
from gui.views.registry import PAGES

#######################################################################################################################
# Page
#######################################################################################################################
_logo = os.path.join(ROOT, A.LOGO) if A.LOGO else ''
_icon = '_icon'.join(os.path.splitext(_logo)) if _logo else ''
_icon = _icon if os.path.isfile(_icon) else None
st.set_page_config(page_title=A.NAME, page_icon=_icon, layout='wide', initial_sidebar_state='expanded')
if os.path.isfile(_logo):
    st.logo(_logo, size='large', icon_image=_icon)
forms.cfg()
st.session_state.setdefault('runs', {})
st.session_state['_card_n'] = 0                                          # automatic keys restart per run
for _k, _v in (('_mode_page', False), ('_tools_page', False), ('_hide_n', 0), ('_ptab_depth', 0), ('_panel_n', 0),
               ('_sum_open', None), ('_res_base', None), ('_interp_n', 0), ('_no_interp', []), ('_sym_pending', []),
               ('_tbox_n', 0), ('_tbox_depth', 0)):
    st.session_state[_k] = _v
st.session_state.setdefault('focus', True)                               # focus mode on by default
style.inject_css()


#######################################################################################################################
# Sidebar: configuration
#######################################################################################################################
def _import(upload):
    """Imports an uploaded JSON config or config workbook (once per file)."""
    if upload is None or st.session_state.get('imported') == upload.file_id:
        return
    try:
        name = os.path.splitext(upload.name)[0]
        if upload.name.lower().endswith('.json'):
            cfg = conf.from_json(upload.getvalue().decode('utf-8'))
        else:
            cfg = conf.from_xlsx(upload.getvalue(), forms.schema())
        forms.set_config(cfg, name)
        st.session_state.imported = upload.file_id
        st.toast(f"Configuration '{name}' imported", icon=':material/check:')
        st.rerun()
    except Exception as e:
        st.session_state.imported = upload.file_id
        st.sidebar.error(f"Import failed: {e}")


with st.sidebar:
    st.toggle("Focus mode", key='focus', help="On: the results and theory show the answer, the key figures, the main "
              "chart and the interpretation. Off: also footnotes, secondary tables and the chart explanations.")
    st.toggle("Expert mode", key='expert', help="Off: only the main parameters that have an effect for the current "
              "settings are shown. On: all parameters incl. numerics, sub-mode options and inactive ones. Hidden "
              "parameters keep their values and are used by the simulation.")
    st.markdown("### :material/tune: Configuration")
    st.caption(f"Active: **{st.session_state.cfg_name}**")

    saved = conf.list_configs()
    c1, c2 = st.columns([3, 2], vertical_alignment='bottom')
    with c1:
        pick = st.selectbox("Saved configs", saved, index=saved.index(st.session_state.cfg_name)
                            if st.session_state.cfg_name in saved else 0, key='sb|pick')
    with c2:
        if st.button("Load", icon=':material/folder_open:', width='stretch', disabled=not saved):
            forms.set_config(conf.read_config(pick), pick)
            st.toast(f"Configuration '{pick}' loaded", icon=':material/check:')
            st.rerun()

    c1, c2 = st.columns([3, 2], vertical_alignment='bottom')
    with c1:
        name = st.text_input("Save as", value=st.session_state.cfg_name, key='sb|name')
    with c2:
        if st.button("Save", icon=':material/save:', width='stretch', type='primary'):
            f = conf.save_config(st.session_state.cfg, name)
            st.session_state.cfg_name = os.path.splitext(os.path.basename(f))[0]
            st.toast(f"Saved to {A.CONFIG_DIR}/{os.path.basename(f)}", icon=':material/check:')
            st.rerun()

    st.download_button("Download JSON", conf.to_json(st.session_state.cfg),
                       file_name=f"{st.session_state.cfg_name}.json", mime='application/json',
                       icon=':material/download:', width='stretch')
    _import(st.file_uploader("Import config (.json or .xlsx)", type=['json', 'xlsx'], key='sb|up'))
    if st.button("Reset all to defaults", icon=':material/restart_alt:', type='tertiary'):
        forms.set_config(conf.default_config(), 'default')
        st.rerun()

    st.divider()
    runs = st.session_state.runs
    if runs:
        st.markdown("**Runs in this session**")
        for m, r in sorted(runs.items()):
            icon = ':material/error:' if r.get('error') else ':material/check_circle:'
            st.caption(f"{icon} {conf.MODE_TITLES[m]} · {r['time'][11:]} · {r['duration']:.0f} s")


#######################################################################################################################
# Navigation
#######################################################################################################################
multi = len(A.MODES) > 1
PAGES.update({m: st.Page(mode.make_page(m), title=f"{m} · {d['title']}" if multi else d['title'], icon=d['icon'],
                         url_path=f"mode{m}") for m, d in A.MODES.items()})
PAGES['results'] = st.Page(results.page, title='Results browser', icon=':material/folder_data:', url_path='results')
TOOLS = {}
for _url, (_mod, _title, _icon) in A.TOOLS.items():
    try:
        _m = __import__(f"gui.views.tools.{_mod}", fromlist=['page'])
        TOOLS[_url] = st.Page(_m.page, title=_title, icon=_icon, url_path=_url)
        PAGES[_url] = TOOLS[_url]
    except ImportError as _e:                                         # a tool page that is not available is skipped
        st.sidebar.caption(f"{_title}: {_e}")
pages = {
    'Project': [st.Page(project.page, title='Setup', icon=':material/settings:', default=True),
                st.Page(inputs.page, title='Input data', icon=':material/table_chart:', url_path='data')],
    'Operating modes' if multi else 'Simulation': [PAGES[m] for m in A.MODES],
    'Theory & Tools': list(TOOLS.values()),
    'Results': [PAGES['results']],
    'Help': ([st.Page(tutorials.page, title='Tutorials', icon=':material/school:', url_path='tutorials')]
             if tut.TUTORIALS else []) +
            [st.Page(docs.page, title='Documentation', icon=':material/menu_book:', url_path='docs')],
}
st.navigation(pages, expanded=True).run()
st.markdown(f"<div style='margin-top:3rem;padding-top:.6rem;border-top:1px solid rgba(128,128,128,.25);"
            f"font-size:.78rem;opacity:.65;text-align:center'>{A.NAME} — {A.TITLE} · "
            f"{getattr(A, 'COPYRIGHT', '')}</div>", unsafe_allow_html=True)
