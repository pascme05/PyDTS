#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Black-Box Modeling
# File:         inputs.py
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
Input data page: browser of the datasets in data/ (the datasets of the configuration first) with overview, signals,
split and correlation (gui/core/viz_inputs.dataset_view).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import streamlit as st

from gui.core import forms
from gui.core import viz_inputs as vi


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    st.title("Input data")
    st.caption("Datasets in data/: xlsx/csv with the sheets input, output and units, or mat files with input, output, "
               "labelInp, labelOut, unitInp and unitOut (both with the columns time and id). The datasets of the "
               "configuration are listed first; the file type is set under Data › File type.")
    vi.dataset_view(forms.cfg(), key='in')
