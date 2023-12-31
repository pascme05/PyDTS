#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         featuresRoll
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
import pandas as pd
import numpy as np
import copy


#######################################################################################################################
# Function
#######################################################################################################################
def featuresRoll(X, setupMdl):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Calculate Rolling Features")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Variables
    # ==============================================================================
    Xout = copy.deepcopy(X)

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Derivative
    # ==============================================================================
    if setupMdl['feat_roll']['diff'] != 0:
        for i in range(0, int(setupMdl['feat_roll']['diff'])):
            temp = X.diff().fillna(0)
            temp.columns = X.columns + "_Diff_" + str(i)
            Xout = pd.concat([Xout, temp], axis=1)

    # ==============================================================================
    # EWMA
    # ==============================================================================
    if np.sum(setupMdl['feat_roll']['EWMA']) != 0:
        for i in range(0, len(setupMdl['feat_roll']['EWMA'])):
            temp = X.ewm(span=setupMdl['feat_roll']['EWMA'][i]).mean()
            temp.columns = X.columns + "_EWMA_" + str(i)
            Xout = pd.concat([Xout, temp], axis=1)

    # ==============================================================================
    # EWMS
    # ==============================================================================
    if np.sum(setupMdl['feat_roll']['EWMS']) != 0:
        for i in range(0, len(setupMdl['feat_roll']['EWMS'])):
            temp = X.ewm(span=setupMdl['feat_roll']['EWMS'][i]).std()
            temp.columns = X.columns + "_EWMS_" + str(i)
            Xout = pd.concat([Xout, temp], axis=1)

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return Xout
