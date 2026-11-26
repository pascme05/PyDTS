#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         test_core
# Date:         25.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.1
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Fast checks of the data handling of PyDTS (no training): data split, normalisation, time lag, sequences, rolling
features and initial values. Run from the root folder with `python -m pytest tests` or `python tests/test_core.py`.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loadData import splitIdx
from src.data.normData import normData
from src.data.resampleData import resampleData
from src.general.helpFnc import shiftEdge, sampleWeights, normVal, toBool
from src.general.createSeq import createSeq
from src.general.featuresRoll import featuresRoll
from src.general.addInitValues import addInitValues


#######################################################################################################################
# Helper
#######################################################################################################################
DAT = {'rT': 0.8, 'rV': 0.25, 'idT': [3], 'idV': [2]}
EXP = {'kfold': 4}


def _sets(method, n=1000, ids=None, shu=False, fold=2, dat=None):
    ids = np.ones(n) if ids is None else ids
    return {tr: splitIdx(n, ids, EXP, dat or DAT, method, tr, fold, shu) for tr in (1, 2, 3)}


#######################################################################################################################
# Tests
#######################################################################################################################
def test_split_disjoint_and_pool():
    """Training, validation and test rows never overlap; the normalisation pool is training + validation."""
    ids = np.repeat([1, 2, 3, 4], 250)
    for method, kw in ((0, {}), (0, {'shu': True}), (1, {}), (1, {'shu': True}), (3, {'ids': ids})):
        s = _sets(method, **kw)
        T, V, X = (set(s[k][0]) for k in (1, 3, 2))
        assert not (T & V) and not (T & X) and not (V & X), (method, kw)
        assert T | V == set(s[1][1]), (method, kw)
        assert len(V) > 0 and len(T) > 0 and len(X) > 0, (method, kw)


def test_split_repeatable_with_shuffle():
    a, b = _sets(0, shu=True), _sets(0, shu=True)
    assert all(np.array_equal(a[k][0], b[k][0]) for k in a)


def test_split_ids_validation_per_id():
    """ID split: all non-test IDs are trained on; validation is the last share rV of the validation IDs, or of every
    training ID if the validation IDs are test IDs."""
    ids = np.repeat([1, 2, 3], 500)
    s = _sets(3, n=1500, ids=ids, dat={'rT': 0.8, 'rV': 0.2, 'idT': [3], 'idV': [3]})
    T, V, X = s[1][0], s[3][0], s[2][0]
    assert set(ids[X]) == {3} and set(ids[T]) == {1, 2} and set(ids[V]) == {1, 2} and not set(T) & set(V)
    assert len(V) == 200 and set(V) == set(range(400, 500)) | set(range(900, 1000))
    s = _sets(3, n=1500, ids=ids, dat={'rT': 0.8, 'rV': 0.2, 'idT': [3], 'idV': [2]})
    assert set(ids[s[3][0]]) == {2} and len(s[3][0]) == 100 and set(ids[s[1][0]]) == {1, 2}


def test_resample_per_id():
    """IDs with a time gap between them are resampled separately: no held filler samples in the gap, step 1/fs."""
    d = pd.DataFrame({'time': np.r_[np.arange(0, 10, 0.5), np.arange(100, 105, 0.5)], 'id': [1] * 20 + [2] * 10,
                      'v': np.arange(30.)})
    out, t = resampleData(d, 2, {'fs': 1})
    assert len(out) == 15 and np.allclose(np.diff(t[:10]), 1) and t[10] == 100
    assert np.allclose(out['v'].values, np.r_[np.arange(0, 20, 2), np.arange(20, 30, 2)])


def test_framing_per_id():
    """No window spans two IDs; frame IDs match the frames; without ids the old single-series framing is kept."""
    from src.general.framing import framing, frameIds
    d = pd.DataFrame({'a': np.arange(10.), 'b': np.arange(10.) * 10})
    ids = [1] * 6 + [2] * 4
    f = framing(d, 3, 2, ids)
    assert f.shape == (6, 3, 2) and f[4, :, 0].tolist() == [6, 7, 8]
    assert frameIds(ids, 10, 3, 2).tolist() == [1, 1, 1, 1, 2, 2]
    assert framing(d, 3, 2).shape == (8, 3, 2) and framing(d[['a']], 3, 0).shape == (3, 3)


def test_shift_edge_per_id():
    """The lag shift does not move values from one ID into the next."""
    out = shiftEdge(np.arange(6.), 1, [1, 1, 1, 2, 2, 2])
    assert out.tolist() == [0, 0, 1, 3, 3, 4]


def test_shift_edge():
    y = np.arange(6.0)
    assert np.array_equal(shiftEdge(y, 2), [0, 0, 0, 1, 2, 3])
    assert np.array_equal(shiftEdge(y, -2), [2, 3, 4, 5, 5, 5])
    assert np.array_equal(shiftEdge(y, 0), y)


def test_norm_constant_channel_and_inverse():
    X = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'c': [5.0, 5.0, 5.0]})
    y = pd.DataFrame({'o': [0.0, 1.0, 4.0]})
    mx, my, nx, ny, ux, uy, sx, sy = normVal(X, y)
    sd = {'weightNorm': 0, 'normMaxX': mx, 'normMaxY': my, 'normMinX': nx, 'normMinY': ny, 'normAvgX': ux,
          'normAvgY': uy, 'normVarX': sx, 'normVarY': sy}
    for k in (1, 2, 3):
        sd['inpNorm'] = sd['outNorm'] = k
        Xn, yn = normData(X, y, sd, 0)
        assert np.isfinite(Xn.values).all(), k
        Xb, yb = normData(Xn, yn, sd, 1)
        assert np.allclose(Xb.values, X.values) and np.allclose(yb.values, y.values), k


def test_norm_zscore_unit_std():
    """inpNorm/outNorm = 3 is a z-score: mean 0 and standard deviation 1."""
    X = pd.DataFrame({'a': np.random.default_rng(1).normal(50, 7, 1000)})
    y = pd.DataFrame({'o': np.random.default_rng(2).normal(-3, 0.2, 1000)})
    mx, my, nx, ny, ux, uy, sx, sy = normVal(X, y)
    sd = {'weightNorm': 0, 'inpNorm': 3, 'outNorm': 3, 'normMaxX': mx, 'normMaxY': my, 'normMinX': nx,
          'normMinY': ny, 'normAvgX': ux, 'normAvgY': uy, 'normVarX': sx, 'normVarY': sy}
    Xn, yn = normData(X, y, sd, 0)
    assert np.isclose(Xn.values.std(), 1) and np.isclose(yn.values.std(), 1) and np.isclose(Xn.values.mean(), 0)


def test_create_seq_lengths():
    data = np.random.rand(5, 10, 2)
    for W, out in ((10, 3), (10, 4)):
        par = {'window': W, 'outseq': out, 'yFocus': 0}
        assert createSeq(data, par).shape[1] == out
    par = {'window': 10, 'outseq': 0, 'yFocus': 50}                                                                     # focus beyond the window
    assert np.array_equal(createSeq(data, par), data[:, 9, :])


def test_features_roll_restart_per_id():
    X = pd.DataFrame({'p': np.r_[np.zeros(50), np.ones(50)]})
    mdl = {'feat_roll': {'diff': 0, 'EWMA': [20], 'EWMS': [0]}}
    ids = np.r_[np.ones(50), 2 * np.ones(50)]
    out = featuresRoll(X, mdl, ids)
    assert len(out) == 100 and out['p_EWMA_0'].iloc[50] == 1.0                                                          # no carry-over from id 1


def test_init_values_once():
    X = pd.DataFrame({'a': np.arange(6.0)})
    y = pd.DataFrame({'o': np.arange(6.0) * 10})
    ids = pd.Series([1, 1, 1, 2, 2, 2])
    sd = {'normMaxX': np.array([5.0]), 'normMinX': np.array([0.0]), 'normAvgX': np.array([2.5]),
          'normVarX': np.array([1.0]), 'normMaxY': np.array([50.0]), 'normMinY': np.array([0.0]),
          'normAvgY': np.array([25.0]), 'normVarY': np.array([1.0])}
    out = addInitValues(X, y, ids, sd)
    addInitValues(X, y, ids, sd)                                                                                        # second call
    assert out.iloc[:, 1].tolist() == [0, 0, 0, 30, 30, 30] and len(sd['normMaxX']) == 2


def test_sample_weights_and_bool():
    y = np.r_[np.zeros(90), np.ones(10)]
    w = sampleWeights(y, 1)
    assert np.isclose(w[y == 0].sum(), w[y == 1].sum())
    assert sampleWeights(y, 0) is None
    assert toBool('False') is False and toBool('True') is True and toBool(1) is True


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
            except Exception as e:
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
