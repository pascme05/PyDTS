#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         addInitValues
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

# ==============================================================================
# External
# ==============================================================================
import numpy as np
import copy


#######################################################################################################################
# Function
#######################################################################################################################
def addInitValues(X, y, idT, setupDat):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Adding initial values")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    T = X.shape[0]

    # ==============================================================================
    # Variables
    # ==============================================================================
    yout = copy.deepcopy(y)

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Changes
    # ==============================================================================
    yout.values[0, :] = y.values[0, :]
    for i in range(1, T):
        if idT.T.values[i-1] != idT.T.values[i]:
            yout.values[i, :] = y.values[i, :]
        else:
            yout.values[i, :] = yout.values[i-1, :]

    # ==============================================================================
    # Appending
    # ==============================================================================
    Xout = X.join(yout)

    ###################################################################################################################
    # Postprocessing
    ###################################################################################################################
    setupDat['normMaxX'] = np.append(setupDat['normMaxX'], setupDat['normMaxY'])
    setupDat['normMinX'] = np.append(setupDat['normMinX'], setupDat['normMinY'])
    setupDat['normAvgX'] = np.append(setupDat['normAvgX'], setupDat['normAvgY'])
    setupDat['normVarX'] = np.append(setupDat['normVarX'], setupDat['normVarY'])

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return Xout