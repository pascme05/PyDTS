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
import matplotlib.pyplot as plt
import seaborn as sns


#######################################################################################################################
# Function
#######################################################################################################################
def featureRank(X, y, setupDat, setupMdl, setupPar):
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
    outLabel = y.columns

    # ==============================================================================
    # Variables
    # ==============================================================================
    if setupPar['rank'] == 3:
        scores = np.zeros((X.shape[1], X.shape[1]))
        err = np.zeros((X.shape[1], X.shape[1]))
    else:
        scores = np.zeros((X.shape[1], len(setupDat['out'])))
        err = np.zeros((X.shape[1], len(setupDat['out'])))
    heatInp = []

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
    if setupPar['rank'] == 1:
        print("INFO: Running Feature Ranking")
        for i in range(0, len(setupDat['out'])):
            mdl = mdl.fit(X_train.values, np.squeeze(y_train.values[:, i]))
            result = permutation_importance(mdl, X_test.values, np.squeeze(y_test.values[:, i]), n_repeats=10, random_state=42, n_jobs=-2)
            scores[:, i] = result.importances_mean
            err[:, i] = result.importances_std
    elif setupPar['rank'] == 2:
        print("INFO: Running Correlation Analysis Inp-Out")
        heatInp = pd.concat([X_train, y_train], axis=1)
        for i in range(0, len(setupDat['out'])):
            mdl = mdl.fit(X_train.values, np.squeeze(y_train.values[:, i]))
            result = permutation_importance(mdl, X_test.values, np.squeeze(y_test.values[:, i]), n_repeats=10, random_state=42, n_jobs=-2)
            scores[:, i] = result.importances_mean
            err[:, i] = result.importances_std
    elif setupPar['rank'] == 3:
        print("INFO: Running Correlation Analysis Inp-Inp")
        heatInp = X_train
        for i in range(0, X_train.shape[1]):
            mdl = mdl.fit(X_train.values, np.squeeze(X_train.values[:, i]))
            result = permutation_importance(mdl, X_test.values, np.squeeze(X_test.values[:, i]), n_repeats=10, random_state=42, n_jobs=-2)
            scores[:, i] = result.importances_mean
            err[:, i] = result.importances_std
    else:
        print("INFO: Feature ranking aborted")

    # ==============================================================================
    # Rank
    # ==============================================================================
    score = pd.Series(np.mean(scores, axis=1), index=inpLabel)
    error = pd.Series(np.mean(err, axis=1), index=inpLabel)

    ###################################################################################################################
    # Plotting
    ###################################################################################################################
    if setupPar['rank'] >= 2:
        # ==============================================================================
        # Heatmap
        # ==============================================================================
        plt.figure()
        sns.heatmap(heatInp.corr(), annot=True, cmap="Blues")
        plt.title("Heatmap of Input and Output Features")

        # ==============================================================================
        # Feature Ranking
        # ==============================================================================
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rank = scores/np.max(np.max(scores))
        pos = ax.matshow(rank, cmap='Blues', interpolation='none')
        fig.colorbar(pos)
        if setupPar['rank'] == 2:
            ax.set_xticklabels(outLabel)
            ax.set_yticklabels(inpLabel)
        else:
            ax.set_xticklabels(inpLabel)
            ax.set_yticklabels(inpLabel)
        plt.title("Feature Ranking")

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [score, error]
