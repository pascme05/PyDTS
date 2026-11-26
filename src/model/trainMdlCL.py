#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         trainMdlCL
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
Closed-loop (CL) training: the output |lag| samples in the past is appended as an additional input channel (teacher
forcing with the measured output). During testing the model's own predictions are fed back instead (testMdlCL).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.general.helpFnc import shiftEdge, sampleWeights, toBool
from src.general.helpFnc import reshapeMdlData
from src.model.models import buildMdl
from src.model import torchFit as tfit

# ==============================================================================
# External
# ==============================================================================
import numpy as np
import os
import sys
import time


#######################################################################################################################
# Helper
#######################################################################################################################
def lagWindows(setupPar, train):
    """Lag in windows: |lag| divided by the window step of the framing (training and testing framing differ)."""
    if setupPar['frame'] != 1:
        step = 1
    elif train:
        step = setupPar['window'] if setupPar['overlap'] == -1 else setupPar['window'] - setupPar['overlap']
    else:
        step = setupPar['window'] if setupPar['overlap'] == -1 else 1
    lagW = int(round(abs(setupPar['lag']) / step))
    if train and abs(setupPar['lag']) % step != 0:
        print("WARN: Lag %d is not a multiple of the window step %d, feedback uses %d samples"
              % (abs(setupPar['lag']), step, lagW * step))

    return lagW


def addFeedback(X, y, lagW, ids=None):
    """
    Appends the output lagW windows earlier as last input channel: (N, W, F) -> (N, W, F + 1). With ids (one per
    window) the feedback never reaches into another ID: the first lagW windows of an ID hold its first output value.
    """
    if np.ndim(np.squeeze(y)) > 1:
        sys.exit("ERROR: Closed loop (CL) supports one output only")
    fb = shiftEdge(np.squeeze(y).reshape(-1), lagW, ids)

    return np.concatenate((X, np.repeat(fb[:, None, None], X.shape[1], axis=1)), axis=2)


#######################################################################################################################
# Function
#######################################################################################################################
def trainMdlCL(data, setupDat, setupPar, setupMdl, setupExp):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Training Model (CL)")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    dev = tfit.device(setupExp['gpu'])
    mdlName = tfit.mdlPath(setupPar, setupExp)
    print("INFO: Device %s" % dev)

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    weights = sampleWeights(data['T']['y'], setupDat['balance'])
    lagW = lagWindows(setupPar, True)
    for k in ('T', 'V'):
        data[k]['X'] = addFeedback(data[k]['X'], data[k]['y'], lagW, data[k].get('id'))
        [data[k]['X'], data[k]['y']] = reshapeMdlData(data[k]['X'], data[k]['y'], setupDat, setupPar, 0)

    ###################################################################################################################
    # Model
    ###################################################################################################################
    out = tfit.outDim(data['T']['y'], setupDat, setupPar)
    activation = 'linear' if setupPar['method'] == 0 else 'sigmoid'
    mdl = buildMdl(setupPar['model'], data['T']['X'].shape[1:], out, activation)
    info = tfit.meta(data['T']['X'], out, setupPar)
    print(mdl)

    if os.path.exists(mdlName):
        tfit.load(mdl, mdlName, info)
        print("INFO: Model will be retrained")
    else:
        print("INFO: Model will be created")

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    start = time.time()
    tfit.fit(mdl, (data['T']['X'], data['T']['y']), (data['V']['X'], data['V']['y']), setupMdl, mdlName, info, dev,
             weights=weights, shuffle=toBool(setupMdl['shuffle']), verbose=setupMdl['verbose'])
    trainTime = time.time() - start

    ###################################################################################################################
    # Output
    ###################################################################################################################
    print("INFO: Total training time (sec): %.2f" % trainTime)
    print("INFO: Training time per sample (ms): %.2f" % (trainTime / data['T']['X'].shape[0] * 1000))
    print("INFO: Model size (kB): %.2f" % tfit.mdlSize(mdl))
