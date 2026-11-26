#######################################################################################################################
#######################################################################################################################
# Title:        Python Deep Timeseries Simulation (PyDTS)
# Topic:        Black-Box Modeling
# File:         adapter.py
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
Connection of the GUI to PyDTS. PyDTS is configured in start.py (setupExp, setupDat, setupPar) and mdlPara.py
(setupMdl); gui/build_config.py turns both into the config workbook config/default.xlsx (sheets Exp, Dat, Par, Mdl),
from which the forms are generated. For a run the configuration is cast back into the four setup dictionaries and
main / optiHyp / optiGrid are called exactly as in start.py. The predictions of the test are recorded (wrapper around
src.test.performance) so that the GUI can show them as interactive charts.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import contextlib
import copy
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#######################################################################################################################
# Names
#######################################################################################################################
NAME = 'PyDTS'
TITLE = 'Python Deep Timeseries Simulation'
COPYRIGHT = '© Dr. Pascal A. Schirmer'
TAGLINE = "Black-box modelling of time series with signal processing, machine learning and deep learning: denoising, " \
          "forecasting, non-linear modelling (e.g. temperatures of electric machines), anomaly detection and " \
          "degradation. Set up the project here, then open an operating mode in the sidebar."
SRC_PATHS = ['']                                                      # added to sys.path (relative to ROOT)
LOGO = None
DOC = 'docs/GUI.md'

#######################################################################################################################
# Config workbook (written by gui/build_config.py from start.py and mdlPara.py)
#######################################################################################################################
CONFIG_DIR = 'config'
CONFIG_WORKBOOK = 'config/default.xlsx'
SHEETS = {'Exp': ('Exp', False), 'Dat': ('Dat', False), 'Par': ('Par', False), 'Mdl': ('Mdl', False)}
LISTS = {('Dat', 'train'), ('Dat', 'idT'), ('Dat', 'idV'), ('Dat', 'inp'), ('Dat', 'out'), ('Dat', 'his'),
         ('Dat', 'fut'), ('Mdl', 'feat_roll.EWMA'), ('Mdl', 'feat_roll.EWMS')}


class _ByVar(set):
    """Set of (section, var) that also answers (section, category, var) as used by build_schema (flat sections)."""

    def __contains__(self, k):
        return set.__contains__(self, (k[0], k[-1]))


INTS = _ByVar({('Exp', 'kfold'), ('Dat', 'batch'), ('Dat', 'lim'), ('Dat', 'inpFilLen'), ('Dat', 'outFilLen'),
               ('Dat', 'balance'), ('Par', 'lag'), ('Par', 'ahead'), ('Par', 'window'), ('Par', 'overlap'),
               ('Par', 'outseq'), ('Par', 'yFocus'), ('Mdl', 'batch'), ('Mdl', 'epoch'), ('Mdl', 'patience'),
               ('Mdl', 'valsteps'), ('Mdl', 'SP_SS_block'), ('Mdl', 'SP_SS_order'), ('Mdl', 'SP_TF_poles'),
               ('Mdl', 'SP_TF_zeros'), ('Mdl', 'SK_RF_depth'), ('Mdl', 'SK_RF_state'), ('Mdl', 'SK_RF_estimators'),
               ('Mdl', 'SK_KNN_neighbors')})
STRS = _ByVar(LISTS | {('Exp', 'name'), ('Dat', 'test'), ('Dat', 'val')})
NO_OPTIONS = set()
OPTIONAL = _ByVar({('Dat', 'inp'), ('Dat', 'his'), ('Dat', 'fut'), ('Dat', 'idT'), ('Dat', 'idV')})
OPTIONS = {}
LABELS = {}
PERCENT_AS_FRACTION = False                                           # PyDTS divides the noise in % by 100 itself
EXTRA_CONFIGS = {}

#######################################################################################################################
# Run settings (formerly start.py)
#######################################################################################################################
RUN_SECTION = 'Exp'
CONFIG_VAR = None
MODE_VAR = 'sim'
RUN = {'name': 'Test', 'sim': 0, 'train': 0, 'test': 1, 'plot': 1, 'save': 0, 'gpu': 0, 'warn': 3}
RUN_FIELDS = [
    ('name', 'text', "Simulation name", "Name of the simulation; the trained model is stored as "
                                        "mdl/mdl_<model>_<name> and re-used for testing."),
    ('train', 'select:0|1', "Training", "0: load the stored model mdl/mdl_<model>_<name>, 1: train a new model (or "
                                        "continue training a stored one)."),
    ('test', 'select:0|1', "Testing", "1: predict the test data and compute the errors."),
    ('plot', 'select:0|1', "Plots of PyDTS", "1: the matplotlib figures of PyDTS are shown under Results › Figures."),
    ('save', 'select:0|1', "Save results", "1: PyDTS writes the predictions and errors to results/."),
    ('gpu', 'expert:select:0|1', "Device", "0: CPU, 1: GPU (PyTorch with CUDA)."),
    ('warn', 'expert:select:0|1|2|3', "Library messages", "0: all, 3: none."),
]

#######################################################################################################################
# Operating modes
#######################################################################################################################
MODES = {
    0: dict(title='Simulation', section='Par', extra=['Dat', 'Mdl'], icon=':material/model_training:',
            text="Train and test a model: the data are loaded, normalised, framed into windows, the model is trained "
                 "on the training data and predicts the test data; errors and predictions are shown under Results.",
            sub=None),
    1: dict(title='Hyperparameter optimisation', section='Mdl', extra=['Par', 'Dat'], icon=':material/tune:',
            text="Optuna searches the layers and units of the network (src/optiHyp, models.optiMdl). Long "
                 "runs: start with few epochs.", sub=None),
    2: dict(title='Grid optimisation', section='Par', extra=['Mdl', 'Dat'], icon=':material/grid_on:',
            text="Grid search over framing parameters (window length, overlap) with training and testing for every "
                 "grid point (src/optiGrid).", sub=None),
}

#######################################################################################################################
# Theory & Tools pages: url -> (module in gui/views/tools, title, icon)
#######################################################################################################################
TOOLS = {
    'tools-temp': ('thermal', 'Temperature prediction', ':material/thermostat:'),
    'tools-nn': ('nn', 'Neural networks', ':material/neurology:'),
}
TEST_PAGES = {u: f"from gui.views.tools import {m}; {m}.page()" for u, (m, _, _) in TOOLS.items()}

#######################################################################################################################
# Files
#######################################################################################################################
FOLDERS = {'data': 'data'}
FILE_EXT = ('.xlsx', '.csv', '.mat')
FILE_GROUPS = []
GROUP_SELECTOR = None
RESULTS_DIR = 'results'
RESULT_PROJECTS = False


def results_root(cfg):
    """Folder of the GUI runs: every run gets results/gui/<name>_<time>."""
    return os.path.join(ROOT, RESULTS_DIR, 'gui')


#######################################################################################################################
# Casting between config and setup
#######################################################################################################################
def _num(s):
    try:
        f = float(s)
        return int(f) if f.is_integer() and 'e' not in s.lower() and '.' not in s else f
    except ValueError:
        return s


def to_cfg(sec, var, val):
    """Python value of a setup script -> value of the GUI config (lists as comma separated text)."""
    if (sec, var) in LISTS:
        return ', '.join(str(v) for v in (val if isinstance(val, (list, tuple)) else [val]))
    if isinstance(val, bool):
        return str(val)
    return val


def to_list(txt):
    if isinstance(txt, (list, tuple)):
        return list(txt)
    return [_num(s.strip()) for s in ('' if txt is None else str(txt)).replace(';', ',').split(',') if s.strip()]


def setups(cfg):
    """The four setup dictionaries of PyDTS from the GUI config."""
    out = {}
    for sec in ('Exp', 'Dat', 'Par', 'Mdl'):
        d = {}
        for var, val in (cfg.get(sec) or {}).items():
            if (sec, var) in LISTS:
                val = to_list(val)
            elif sec == 'Dat' and var == 'Shuffle':
                val = {'True': True, 'False': False}.get(str(val), val)
            elif sec == 'Dat' and var in ('test', 'val', 'type'):
                val = str(val)
            if '.' in var:
                a, b = var.split('.', 1)
                d.setdefault(a, {})[b] = val
            else:
                d[var] = val
        out[sec] = d
    out['Mdl'].setdefault('feat', {})
    out['Mdl'].setdefault('feat_roll', {})
    out['Exp']['author'] = 'GUI'
    return out


#######################################################################################################################
# Toolkit
#######################################################################################################################
FEED = 'dict'


def init_setup():
    return {'Exp': {}, 'Dat': {}, 'Par': {}, 'Mdl': {}}


def paths():
    """Path variable of PyDTS (initPath uses the parent of the working directory, which is ROOT during a run)."""
    from src.general.helpFnc import initPath
    return initPath(os.path.basename(ROOT))


def cast(setup):
    return setup


def model_dir(cfg):
    """Folder of the stored model mdl/mdl_<model>_<name> (relative to ROOT)."""
    return os.path.join('mdl', f"mdl_{cfg['Par'].get('model')}_{cfg['Exp'].get('name')}")


def run(setup, path, mode):
    """
    Runs PyDTS like start.py. The test predictions are recorded by a wrapper around src.test.performance (and the
    feature ranking around src.test.featureRank); the training log (training_log.csv) is read afterwards.
    """
    import src.test as T
    S = setups(setup)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(S['Exp'].get('warn', 3))
    folder = os.path.join(results_root(setup), f"{S['Exp']['name']}_{time.strftime('%y%m%d-%H%M%S')}")
    os.makedirs(folder, exist_ok=True)

    rec = {'tests': [], 'rank': None}
    perf, rank = T.performance, T.featureRank

    def performance(yPred, yTrue, yPred_L, yTrue_L, setupDat):
        res = perf(yPred, yTrue, yPred_L, yTrue_L, setupDat)
        rec['tests'].append({'yPred': np.array(yPred, dtype=float), 'yTrue': np.array(yTrue, dtype=float),
                             'result': np.array(res[0], dtype=float), 'avg': np.array(res[1], dtype=float)})
        return res

    def featureRank(*a, **k):
        out = rank(*a, **k)
        rec['rank'] = copy.deepcopy(out)
        return out

    log0 = os.path.join(ROOT, model_dir(setup), 'training_log.csv')
    n0 = _log_rows(log0)
    T.performance, T.featureRank = performance, featureRank
    try:
        if mode == 1:
            from src.optiHyp import optiHyp
            optiHyp(S['Exp'], S['Dat'], S['Par'], S['Mdl'], path)
        elif mode == 2:
            from src.optiGrid import optiGrid
            optiGrid(S['Exp'], S['Dat'], S['Par'], S['Mdl'], path)
        else:
            from src.main import main
            main(S['Exp'], S['Dat'], S['Par'], S['Mdl'], path)
    finally:
        T.performance, T.featureRank = perf, rank

    dat = S['Dat']
    out = {'mode': mode, 'out': [str(o) for o in dat.get('outLabel', dat.get('out', []))],
           'inp': [str(i) for i in dat.get('inpLabel', [])], 'fs': float(dat.get('fs', 1.0)),
           'units': _units(dat.get('outUnits')), 'tests': rec['tests'], 'rank': rec['rank'],
           'hist': _history(log0, n0), 'model': S['Par'].get('model'), 'solver': S['Par'].get('solver'),
           'folder': folder}
    return out


def _log_rows(f):
    try:
        with open(f, 'r', encoding='utf-8') as h:
            return sum(1 for _ in h)
    except OSError:
        return 0


def _history(f, n0):
    """Training log rows written by this run (CSVLogger appends; earlier runs are kept separately)."""
    import pandas as pd
    try:
        df = pd.read_csv(f)
    except Exception:
        return None
    new = df.iloc[max(n0 - 1, 0):] if n0 else df
    return {'new': new.reset_index(drop=True), 'all': df}


def _units(u):
    try:
        return [str(x) for x in u.iloc[0].values]
    except Exception:
        return []


def compact(data, mode):
    """Stored with the run: predictions thinned to 20 000 samples per test."""
    if not isinstance(data, dict):
        return data
    d = dict(data)
    tests = []
    for t in d.get('tests') or []:
        n = len(t['yTrue'])
        s = max(1, n // 20000)
        tests.append({**t, 'yPred': t['yPred'][::s], 'yTrue': t['yTrue'][::s], 'stride': s})
    d['tests'] = tests
    return d


@contextlib.contextmanager
def session():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield
