#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         testMdlCL
# Date:         03.11.2023
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.2
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
Closed-loop (CL) testing: the feedback channel holds the model's own prediction lagW windows earlier. All windows of a
block of lagW windows only depend on earlier predictions, so they are predicted together.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.general.helpFnc import reshapeMdlData
from src.model.models import buildMdl
from src.model.trainMdlCL import lagWindows, addFeedback
from src.model import torchFit as tfit

# ==============================================================================
# External
# ==============================================================================
import copy
import time

import numpy as np
import torch


#######################################################################################################################
# Function
#######################################################################################################################
def testMdlCL(data, setupDat, setupPar, setupMdl, setupExp):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Test Model (CL)")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    dev = tfit.device(setupExp['gpu'])
    mdlName = tfit.mdlPath(setupPar, setupExp)
    dataTest = copy.deepcopy(data)

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    lagW = lagWindows(setupPar, False)
    ids = dataTest['T'].get('id')
    dataTest['T']['X'] = addFeedback(dataTest['T']['X'], dataTest['T']['y'], lagW, ids)
    [dataTest['T']['X'], dataTest['T']['y']] = reshapeMdlData(dataTest['T']['X'], dataTest['T']['y'], setupDat,
                                                              setupPar, 0)
    X = np.ascontiguousarray(dataTest['T']['X'], dtype=np.float32)
    N = X.shape[0]
    if ids is None or len(ids) != N:
        ids = np.zeros(N)
    ids = np.asarray(ids)
    first = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]                                                        # first window of each ID
    first = first[np.searchsorted(first, np.arange(N), side='right') - 1]                                              # ... for every window

    ###################################################################################################################
    # Model
    ###################################################################################################################
    out = tfit.outDim(dataTest['T']['y'], setupDat, setupPar)
    activation = 'linear' if setupPar['method'] == 0 else 'sigmoid'
    mdl = buildMdl(setupPar['model'], X.shape[1:], out, activation)
    tfit.load(mdl, mdlName, tfit.meta(X, out, setupPar))
    mdl.to(dev).eval()
    print("INFO: Model loaded")

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    start = time.time()
    pred = np.zeros(N)
    blk = max(1, lagW)
    with torch.no_grad():
        for a in range(0, N, blk):
            b = min(a + blk, N)
            xb = X[a:b].copy()
            src = np.arange(a, b) - lagW
            use = src >= first[a:b] if lagW > 0 else np.ones(b - a, dtype=bool)                          # same ID only
            xb[use, :, -1] = pred[src[use]][:, None]                                                                    # own predictions
            pred[a:b] = mdl(torch.from_numpy(xb).to(dev)).float().cpu().numpy().reshape(b - a, -1)[:, 0]
    testTime = time.time() - start

    ###################################################################################################################
    # Output
    ###################################################################################################################
    print("INFO: Total inference time (ms): %.2f" % (testTime * 1000))
    print("INFO: Inference time per sample (us): %.2f" % (testTime / N * 1000 * 1000))
    print("INFO: Model size (kB): %.2f" % tfit.mdlSize(mdl))

    return {'T': {'X': dataTest['T']['X'][:, :, :-1], 'y': pred}}
