#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         start
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import pandas as pd


#######################################################################################################################
# Function
#######################################################################################################################
def featureRank(X, y, setupDat, setupMdl):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Ranking Input Features")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    inpLabel = X.columns

    # ==============================================================================
    # Variables
    # ==============================================================================
    scores = np.zeros((X.shape[1], len(setupDat['out'])))
    err = np.zeros((X.shape[1], len(setupDat['out'])))

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Variables
    # ==============================================================================
    mdl = RandomForestRegressor(max_depth=setupMdl['SK_RF_depth'], random_state=setupMdl['SK_RF_state'],
                                n_estimators=setupMdl['SK_RF_estimators'])

    # ==============================================================================
    # Variables
    # ==============================================================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Calc
    # ==============================================================================
    for i in range(0, len(setupDat['out'])):
        mdl = mdl.fit(X_train.values, np.squeeze(y_train.values[:, i]))
        result = permutation_importance(mdl, X_test.values, np.squeeze(y_test.values[:, i]), n_repeats=10, random_state=42, n_jobs=2)
        scores[:, i] = result.importances_mean
        err[:, i] = result.importances_std

    # ==============================================================================
    # Rank
    # ==============================================================================
    score = pd.Series(np.mean(scores, axis=1), index=inpLabel)
    error = pd.Series(np.mean(err, axis=1), index=inpLabel)

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [score, error]