#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         postprocess
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
from src.data.normData import normData

# ==============================================================================
# External
# ==============================================================================
import copy


#######################################################################################################################
# Function
#######################################################################################################################
def postprocess(data, dataPred, setupPar, setupDat):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Start Postprocessing Prediction")

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Reshape
    # ==============================================================================
    # ------------------------------------------
    # Seq2Seq
    # ------------------------------------------
    if setupPar['outseq'] >= 1:
        # One Output
        if len(setupDat['out']) == 1:
            data['y'] = data['y'].reshape((data['y'].shape[0] * data['y'].shape[1], 1))
            dataPred['y'] = dataPred['y'].reshape((dataPred['y'].shape[0] * dataPred['y'].shape[1], 1))

        # Multiple Outputs
        else:
            dataPred['y'] = dataPred['y'].reshape((dataPred['y'].shape[0] * dataPred['y'].shape[1], dataPred['y'].shape[2]))

    # ------------------------------------------
    # Seq2Point
    # ------------------------------------------
    else:
        # One Output
        if len(setupDat['out']) == 1:
            data['y'] = data['y'].reshape((data['y'].shape[0], 1))
            dataPred['y'] = dataPred['y'].reshape((dataPred['y'].shape[0], 1))

    # ==============================================================================
    # Inverse Normalisation
    # ==============================================================================
    [data['X'], data['y']] = normData(data['X'], data['y'], setupDat, 1)
    [dataPred['X'], dataPred['y']] = normData(dataPred['X'], dataPred['y'], setupDat, 1)

    # ==============================================================================
    # Limiting
    # ==============================================================================
    dataPred['y'][dataPred['y'] < setupPar['outMin']] = setupPar['outMin']
    dataPred['y'][dataPred['y'] > setupPar['outMax']] = setupPar['outMax']

    # ==============================================================================
    # Labelling
    # ==============================================================================
    # ------------------------------------------
    # Init
    # ------------------------------------------
    data['L'] = copy.deepcopy(data['y'])
    dataPred['L'] = copy.deepcopy(dataPred['y'])

    # ------------------------------------------
    # Calc
    # ------------------------------------------
    if setupPar['method'] == 0:
        data['L'][data['L'] <= setupDat['threshold']] = 0
        data['L'][data['L'] > setupDat['threshold']] = 1
        dataPred['L'][dataPred['L'] <= setupDat['threshold']] = 0
        dataPred['L'][dataPred['L'] > setupDat['threshold']] = 1
    else:
        data['L'][data['L'] <= 0.5] = 0
        data['L'][data['L'] > 0.5] = 1
        dataPred['L'][dataPred['L'] <= 0.5] = 0
        dataPred['L'][dataPred['L'] > 0.5] = 1

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [data, dataPred]