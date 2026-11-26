#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         test_smoke
# Date:         26.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.1
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
End-to-end checks of the whole chain (loading, split, resampling, windows, training, storing, loading, test, errors)
for the DL models DNN, CNN and LSTM, the closed loop (CL), a seed ensemble and mean resampling. One epoch on a few
thousand samples of data/dataTest_*.xlsx on the CPU, about one minute in total. The models are written to
mdl/mdl_<model>_SmokeTest and deleted afterwards. Run with `python -m pytest tests` or `python tests/test_smoke.py`.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import contextlib
import io
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')


#######################################################################################################################
# Helper
#######################################################################################################################
NAME = 'SmokeTest'


def _run(**over):
    """Trains and tests the start.py example (default config) with small settings; returns the R2 of the test."""
    from gui import adapter as A
    from gui.core import config as C
    os.chdir(ROOT)
    cfg = C.read_config('default')
    cfg['Exp'].update(name=NAME, train=1, test=1, plot=0, save=0, gpu=0, method=0)
    cfg['Dat'].update(lim=3000, train=['dataTest_1'], test='dataTest_1', val='dataTest_1', fs=1 / 3600)
    cfg['Par'].update(window=24, overlap=23)
    cfg['Mdl'].update(epoch=1, batch=256)
    for k, v in over.items():
        sec, var = k.split('__')
        cfg[sec][var] = v
    model = cfg['Par']['model']
    folder = os.path.join(ROOT, 'mdl', 'mdl_%s_%s' % (model, NAME))
    shutil.rmtree(folder, ignore_errors=True)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            r = A.run(A.setups(cfg), A.paths(), 0)
        files = sorted(os.listdir(folder))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    avg = r['tests'][-1]['avg']
    assert np.isfinite(avg[[2, 4, 5]]).all(), log.getvalue()[-2000:]

    return avg[2], files, log.getvalue()


#######################################################################################################################
# Tests
#######################################################################################################################
def test_dl_models():
    """DNN, CNN and LSTM train, are stored as model.pt and reloaded for the test."""
    for model in ('DNN', 'CNN', 'LSTM'):
        _, files, _ = _run(Par__solver='DL', Par__model=model)
        assert 'model.pt' in files, (model, files)


def test_closed_loop():
    _, files, _ = _run(Par__solver='CL', Par__model='DNN', Dat__out=['Solar'])                                         # CL: one output
    assert 'model.pt' in files, files


def test_ensemble_and_mean_resampling():
    """Two seeds give two members whose mean is tested; mean resampling to a lower rate runs through."""
    _, files, log = _run(Par__solver='DL', Par__model='DNN', Exp__ensemble=2, Dat__resample='mean',
                         Dat__fs=1 / 7200)
    assert {'model.pt', 'model_2.pt'} <= set(files), files
    assert 'member 2' in log and 'ensemble:' in log and '(mean)' in log


def test_repeatable():
    """Same seed and settings, same result (deterministic training)."""
    a, _, _ = _run(Par__solver='DL', Par__model='CNN')
    b, _, _ = _run(Par__solver='DL', Par__model='CNN')
    assert a == b, (a, b)


#######################################################################################################################
# Main
#######################################################################################################################
if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print("PASS", name)
            except BaseException as e:                                                                          # also sys.exit
                fails += 1
                print("FAIL", name, repr(e)[:3000])
    sys.exit(1 if fails else 0)
