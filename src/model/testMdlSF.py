#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         testMdlSF
# Date:         21.04.2024
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

# ==============================================================================
# External
# ==============================================================================
from src.general.helpFnc import setDevice, mdlSize
import os
import time
import sys
from sys import getsizeof
from neuralforecast.core import NeuralForecast
import copy
import pandas as pd
import numpy as np


#######################################################################################################################
# Function
#######################################################################################################################
def testMdlSF(data, setupDat, setupPar, _, setupExp):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Test Model (SF)")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # CPU/GPU
    # ==============================================================================
    setDevice(setupExp['gpu'])

    # ==============================================================================
    # Parameters
    # ==============================================================================
    sampling_times = 1 / setupDat['fs']

    # ==============================================================================
    # Variables
    # ==============================================================================
    mdl = []
    dataPred = {'T': {}}

    # ==============================================================================
    # Name
    # ==============================================================================
    mdlName = 'mdl/mdl_' + setupPar['model'] + '_' + setupExp['name']

    ###################################################################################################################
    # Loading
    ###################################################################################################################
    try:
        mdl = NeuralForecast.load(path=mdlName)
        print("INFO: Model loaded")
    except Exception as e:
        sys.exit("ERROR: Model %s could not be loaded, train it first (%s)" % (mdlName, e))
    start_time = mdl.last_dates[0]
    datetime_index = pd.date_range(start=start_time, periods=(len(data['T']['X'])+2), freq=f'{sampling_times}S')
    datetime_index = datetime_index[1:-1]

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Reshape Data
    # ==============================================================================
    dataTest = copy.deepcopy(data['T']['X'])
    dataTest = pd.DataFrame(data=dataTest, columns=setupDat['inpLabel'])
    dataTest.drop(setupDat['out'], axis=1, inplace=True)
    dataTest.drop(setupDat['his'], axis=1, inplace=True)
    dataTest.insert(0, 'ds', datetime_index)
    dataTest.insert(0, 'unique_id', 1.0)

    # ==============================================================================
    # Init Output
    # ==============================================================================
    dataPred['T']['y'] = np.zeros((dataTest.shape[0], 1))
    K = dataPred['T']['y'].shape[0] // setupPar['ahead']

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Start timer
    # ==============================================================================
    start = time.time()

    # ==============================================================================
    # Test
    # ==============================================================================
    try:
        if K > 1:
            for i in range(0, K):
                start = i * setupPar['ahead']
                ende = (i + 1) * setupPar['ahead']
                dataTest = copy.deepcopy(data['T']['X'][start:ende])
                dataTest = pd.DataFrame(data=dataTest, columns=setupDat['inpLabel'])
                dataTest.drop(setupDat['out'], axis=1, inplace=True)
                dataTest.drop(setupDat['his'], axis=1, inplace=True)
                datetime_index = pd.date_range(start=start_time, periods=(setupPar['ahead'] + 2),
                                               freq=f'{sampling_times}S')
                datetime_index = datetime_index[1:-1]
                dataTest.insert(0, 'ds', datetime_index)
                dataTest.insert(0, 'unique_id', 1.0)
                dataPred['T']['y'][start:ende] = mdl.predict(futr_df=dataTest).to_numpy('float')[:, 1].reshape(-1, 1)
        else:
            dataPred['T']['y'] = mdl.predict(futr_df=dataTest).to_numpy('float')[:, 1].reshape(-1, 1)
    except Exception:
        if K > 1:
            for i in range(0, K):
                start = i * setupPar['ahead']
                ende = (i + 1) * setupPar['ahead']
                dataTest = copy.deepcopy(data['T']['X'][start:ende])
                dataTest = pd.DataFrame(data=dataTest, columns=setupDat['inpLabel'])
                dataTest.drop(setupDat['out'], axis=1, inplace=True)
                dataTest.drop(setupDat['his'], axis=1, inplace=True)
                datetime_index = pd.date_range(start=start_time, periods=(setupPar['ahead'] + 2),
                                               freq=f'{sampling_times}S')
                datetime_index = datetime_index[1:-1]
                dataTest.insert(0, 'ds', datetime_index)
                dataTest.insert(0, 'unique_id', 1.0)
                dataPred['T']['y'][start:ende] = mdl.predict().to_numpy('float')[:, 1].reshape(-1, 1)
        else:
            dataPred['T']['y'] = mdl.predict().to_numpy('float')[:, 1].reshape(-1, 1)
        print("WARN: Using future values failed, predicting without additional inputs")
    dataPred['T']['X'] = data['T']['X']

    # ==============================================================================
    # End timer
    # ==============================================================================
    ende = time.time()
    testTime = (ende - start)

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    print("INFO: Total inference time (ms): %.2f" % (testTime * 1000))
    print("INFO: Inference time per sample (us): %.2f" % (testTime / data['T']['X'].shape[0] * 1000 * 1000))
    print("INFO: Model size (kB): %.2f" % mdlSize(mdl))

    ###################################################################################################################
    # References
    ###################################################################################################################
    return dataPred
