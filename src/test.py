#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         test
# Date:         03.11.2023
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.1
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.preprocess import preprocess
from src.postprocess import postprocess
from src.general.framing import framing, frameIds
from src.general.features import features
from src.general.createSeq import createSeq
from src.model.testMdlDL import testMdlDL
from src.model.testMdlML import testMdlML
from src.model.testMdlSP import testMdlSP
from src.model.testMdlSF import testMdlSF
from src.model.testMdlCL import testMdlCL
from src.general.performance import performance
from src.general.plotting import plotting
from src.general.printResults import printResults
from src.general.save import save
from src.general.featureRank import featureRank
from src.general.adaptDim import adaptDim
from src.data.summaryData import summaryData
from src.general.performance import performance as _performance                                                       # members (not recorded by the GUI)

# ==============================================================================
# External
# ==============================================================================
import contextlib
import io
import numpy as np


#######################################################################################################################
# Helper
#######################################################################################################################
def checkRange(X, setupDat):
    """
    Warns when test inputs lie outside the range of the training data (training and validation rows, the
    normalisation pool): the network then has to extrapolate and its error can grow strongly. Prints the share of
    samples outside the range per input and per test ID.
    """
    cols = [c for c in X.columns if c not in ('time', 'id')]
    lo, hi = np.asarray(setupDat['normMinX'], float), np.asarray(setupDat['normMaxX'], float)
    if len(cols) != len(lo):
        return
    v = X[cols].values.astype(float)
    tol = 0.02 * np.maximum(hi - lo, 1e-12)                                                                             # 2 % of the range
    out = (v < lo - tol) | (v > hi + tol)
    if not out.any():
        print("INFO: All test inputs within the range of the training data")
        return
    ids = X['id'].values if 'id' in X.columns else np.zeros(len(X))
    for j in np.flatnonzero(out.any(axis=0)):
        per = ', '.join("ID %g: %.0f %%" % (i, 100 * out[ids == i, j].mean()) for i in np.unique(ids)
                        if out[ids == i, j].any())
        print("WARN: Input %s outside the training range [%.4g, %.4g] (test %.4g .. %.4g) - %s of the samples; "
              "the model extrapolates there" % (cols[j], lo[j], hi[j], np.nanmin(v[:, j]), np.nanmax(v[:, j]), per))


def ensemble(fn, dataTest, setupDat, setupPar, setupMdl, setupExp):
    """
    Prediction of a seed ensemble (setupExp['ensemble'] = N): every member (model.pt, model_2.pt, ...) predicts the
    test data and the prediction is the mean of the members. Returns the (reshaped) test data, the mean prediction and
    the member predictions (empty for a single model).
    """
    n = max(1, int(setupExp.get('ensemble', 1)))
    runs = []
    for k in range(n):
        d = {key: dict(v) for key, v in dataTest.items()}                                                               # models replace arrays
        runs.append((d, fn(d, setupDat, setupPar, setupMdl, dict(setupExp, member=k))))
    if n == 1:
        return [runs[0][0], runs[0][1], []]
    print("INFO: Prediction is the mean of %d ensemble members" % n)
    members = [np.array(p['T']['y'], copy=True) for _, p in runs]
    pred = runs[0][1]
    pred['T']['y'] = np.mean(members, axis=0)

    return [runs[0][0], pred, members]


def score(dataT, yPred, ndim, setupPar, setupDat):
    """Average errors (as performance) of one ensemble member; the inputs are not needed, one window is passed."""
    X = np.asarray(dataT['X'])[:1].reshape((1,) + tuple(ndim[1:]))
    with contextlib.redirect_stdout(io.StringIO()):
        [t, p] = postprocess({'X': X.copy(), 'y': np.array(dataT['y'], copy=True)},
                             {'X': X.copy(), 'y': np.array(yPred, copy=True)}, setupPar, setupDat)
        return _performance(p['y'], t['y'], p['L'], t['L'], setupDat)[1]


def printMembers(scores, resultAvg):
    """Errors of every ensemble member, their mean and spread, and the errors of the ensemble mean."""
    if not scores:
        return
    s = np.array(scores, dtype=float)
    print("INFO: Ensemble members (average over the outputs)")
    for k, r in enumerate(s):
        print("  member %d:  R2 %.4f  RMSE %.4g  MAE %.4g  MAX %.4g" % (k + 1, r[2], r[4], r[5], r[6]))
    print("  members:   R2 %.4f +- %.4f  RMSE %.4g +- %.2g  MAE %.4g +- %.2g (mean +- standard deviation)"
          % (s[:, 2].mean(), s[:, 2].std(), s[:, 4].mean(), s[:, 4].std(), s[:, 5].mean(), s[:, 5].std()))
    print("  ensemble:  R2 %.4f  RMSE %.4g  MAE %.4g  MAX %.4g"
          % (resultAvg[2], resultAvg[4], resultAvg[5], resultAvg[6]))


#######################################################################################################################
# Function
#######################################################################################################################
def test(data, setupExp, setupDat, setupPar, setupMdl, setupPath):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Running PyDTS (Testing)")
    print("\n")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    dataTest = {'T': {}, 'V': {}}
    dataPred = {'T': {'X': [], 'y': []}}
    members = []

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Msg
    # ==============================================================================
    print("------------------------------------------")
    print("START: Preprocessing")
    print("------------------------------------------")

    # ==============================================================================
    # Data Summary
    # ==============================================================================
    summaryData(data['T']['X'], data['T']['y'], setupDat)
    checkRange(data['T']['X'], setupDat)

    # ==============================================================================
    # Testing Data
    # ==============================================================================
    [data['T'], t, idsT] = preprocess(data['T'], setupDat, setupPar)

    # ==============================================================================
    # Msg
    # ==============================================================================
    print("\n")

    ###################################################################################################################
    # Framing
    ###################################################################################################################
    # ==============================================================================
    # Msg
    # ==============================================================================
    print("------------------------------------------")
    print("START: Framing and Edge Detection")
    print("------------------------------------------")

    # ==============================================================================
    # Testing Data
    # ==============================================================================
    if setupPar['frame'] == 1:
        if setupPar['overlap'] == -1:
            ov = 0
        elif setupPar['outseq'] == 0:
            ov = setupPar['window'] - 1
        else:
            ov = setupPar['window'] - setupPar['outseq']
        dataTest['T']['X'] = framing(data['T']['X'], setupPar['window'], ov, idsT)                                     # per ID
        dataTest['T']['y'] = framing(data['T']['y'], setupPar['window'], ov, idsT)
        dataTest['T']['id'] = frameIds(idsT, len(data['T']['y']), setupPar['window'], ov)                               # ID per frame (CL)
    else:
        dataTest['T']['X'] = data['T']['X'].values
        dataTest['T']['y'] = data['T']['y'].values
        print("INFO: No frames calculated using raw data")

    # ==============================================================================
    # Msg
    # ==============================================================================
    print("\n")

    ###################################################################################################################
    # Features
    ###################################################################################################################
    # ==============================================================================
    # Msg
    # ==============================================================================
    print("------------------------------------------")
    print("START: Feature Extraction")
    print("------------------------------------------")

    # ==============================================================================
    # Testing Data
    # ==============================================================================
    if setupPar['feat'] == 1 or setupPar['feat'] == 3:
        dataTest['T']['X'] = features(dataTest['T']['X'], setupMdl['feat'])
    else:
        print("INFO: No features calculated using raw data")

    # ==============================================================================
    # Msg
    # ==============================================================================
    print("\n")

    ###################################################################################################################
    # Sequence
    ###################################################################################################################
    # ==============================================================================
    # Msg
    # ==============================================================================
    print("------------------------------------------")
    print("START: Creating Input Sequence")
    print("------------------------------------------")

    # ==============================================================================
    # Calc
    # ==============================================================================
    if setupPar['frame'] == 1:
        dataTest['T']['y'] = createSeq(dataTest['T']['y'], setupPar)

    # ==============================================================================
    # Adapt Dimension
    # ==============================================================================
    [dataTest['T']['X'], dataTest['T']['y'], ndim] = adaptDim(dataTest['T']['X'], dataTest['T']['y'], setupPar)

    # ==============================================================================
    # Msg
    # ==============================================================================
    print("\n")

    ###################################################################################################################
    # Training
    ###################################################################################################################
    # ==============================================================================
    # Msg
    # ==============================================================================
    print("------------------------------------------")
    print("START: Testing")
    print("------------------------------------------")

    # ==============================================================================
    # Model
    # ==============================================================================
    # ------------------------------------------
    # Classical Learning
    # ------------------------------------------
    if setupPar['solver'] == 'SP':
        dataPred = testMdlSP(dataTest, t, setupPar, setupMdl, setupExp)

    # ------------------------------------------
    # Machine Learning
    # ------------------------------------------
    elif setupPar['solver'] == 'ML':
        dataPred = testMdlML(dataTest, setupPar, setupExp)

    # ------------------------------------------
    # Deep Learning
    # ------------------------------------------
    elif setupPar['solver'] == 'DL':
        [dataTest, dataPred, members] = ensemble(testMdlDL, dataTest, setupDat, setupPar, setupMdl, setupExp)

    # ------------------------------------------
    # Short-Time-Forecasting
    # ------------------------------------------
    elif setupPar['solver'] == 'SF':
        dataPred = testMdlSF(dataTest, setupDat, setupPar, setupMdl, setupExp)

    # ------------------------------------------
    # Closed Loop Modeling
    # ------------------------------------------
    elif setupPar['solver'] == 'CL':
        [dataTest, dataPred, members] = ensemble(testMdlCL, dataTest, setupDat, setupPar, setupMdl, setupExp)

    # ==============================================================================
    # Msg
    # ==============================================================================
    print("\n")

    ####################################################################################################################
    # Post-Processing
    ####################################################################################################################
    # ==============================================================================
    # Msg
    # ==============================================================================
    print("------------------------------------------")
    print("START: Postprocessing")
    print("------------------------------------------")

    # ==============================================================================
    # Ensemble members (spread over the seeds)
    # ==============================================================================
    scores = [score(dataTest['T'], m, ndim, setupPar, setupDat) for m in members]

    # ==============================================================================
    # Unfold
    # ==============================================================================
    if len(ndim) == 2:
        dataTest['T']['X'] = dataTest['T']['X'].reshape((dataTest['T']['X'].shape[0], ndim[1]))
        dataPred['T']['X'] = dataPred['T']['X'].reshape((dataPred['T']['X'].shape[0], ndim[1]))
    else:
        dataTest['T']['X'] = dataTest['T']['X'].reshape((dataTest['T']['X'].shape[0], ndim[1], ndim[2]))
        dataPred['T']['X'] = dataPred['T']['X'].reshape((dataPred['T']['X'].shape[0], ndim[1], ndim[2]))

    # ==============================================================================
    # Calc
    # ==============================================================================
    [dataTest['T'], dataPred['T']] = postprocess(dataTest['T'], dataPred['T'], setupPar, setupDat)

    # ==============================================================================
    # Msg
    # ==============================================================================
    print("\n")

    ####################################################################################################################
    # Output
    ####################################################################################################################
    # ==============================================================================
    # Msg
    # ==============================================================================
    print("------------------------------------------")
    print("START: Output")
    print("------------------------------------------")

    # ==============================================================================
    # Performance
    # ==============================================================================
    # ------------------------------------------
    # Error
    # ------------------------------------------
    [result, resultAvg] = performance(dataPred['T']['y'], dataTest['T']['y'], dataPred['T']['L'], dataTest['T']['L'], setupDat)

    # ------------------------------------------
    # Ranking
    # ------------------------------------------
    [feaScore, feaError] = featureRank(data['T']['X'], data['T']['y'], setupDat, setupMdl, setupPar)

    # ==============================================================================
    # Console
    # ==============================================================================
    printResults(result, resultAvg, setupDat)
    printMembers(scores, resultAvg)

    # ==============================================================================
    # Plotting
    # ==============================================================================
    if setupExp['plot'] != 0:
        plotting(data, dataTest['T'], dataPred['T'], resultAvg, feaScore, feaError, setupDat, setupPar, setupExp)

    # ==============================================================================
    # Saving
    # ==============================================================================
    if setupExp['save'] != 0:
        save(dataTest['T']['y'], dataPred['T']['y'], result, resultAvg, setupDat, setupPar, setupExp, setupPath)

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Running PyDTS (Testing)")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [result, resultAvg]
