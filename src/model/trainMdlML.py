#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         trainMdlML
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
from sklearn import neighbors
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
import joblib
import numpy as np


#######################################################################################################################
# Function
#######################################################################################################################
def trainMdlML(data, setupPar, setupMdl, setupExp):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Training Model (ML)")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    mdlName = 'mdl/mdl_' + setupPar['model'] + '_' + setupExp['name'] + '.joblib'

    # ==============================================================================
    # Variables
    # ==============================================================================
    mdl = []

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Reshape data
    # ==============================================================================
    if np.size(data['T']['X'].shape) == 3:
        data['T']['X'] = data['T']['X'].reshape((data['T']['X'].shape[0], data['T']['X'].shape[1] * data['T']['X'].shape[2]))

    # ==============================================================================
    # Build Model
    # ==============================================================================
    # ------------------------------------------
    # Single Output
    # ------------------------------------------
    if data['T']['y'].ndim == 1:
        # KNN
        if setupPar['model'] == "KNN":
            for ii, weights in enumerate(['uniform', 'distance']):
                if setupPar['method'] == 0:
                    mdl = neighbors.KNeighborsRegressor(n_neighbors=setupMdl['SK_KNN_neighbors'], weights=weights)
                else:
                    mdl = neighbors.KNeighborsClassifier(n_neighbors=setupMdl['SK_KNN_neighbors'], weights=weights)

        # RF
        if setupPar['model'] == "RF":
            if setupPar['method'] == 0:
                mdl = RandomForestRegressor(max_depth=setupMdl['SK_RF_depth'], random_state=setupMdl['SK_RF_state'],
                                            n_estimators=setupMdl['SK_RF_estimators'])
            else:
                mdl = RandomForestClassifier(max_depth=setupMdl['SK_RF_depth'], random_state=setupMdl['SK_RF_state'],
                                             n_estimators=setupMdl['SK_RF_estimators'])

        # SVM
        if setupPar['model'] == "SVM":
            if setupPar['method'] == 0:
                mdl = SVR(kernel=setupMdl['SK_SVM_kernel'], C=setupMdl['SK_SVM_C'], gamma=setupMdl['SK_SVM_gamma'],
                          epsilon=setupMdl['SK_SVM_epsilon'])
            else:
                mdl = SVC(kernel=setupMdl['SK_SVM_kernel'], C=setupMdl['SK_SVM_C'], gamma=setupMdl['SK_SVM_gamma'])

    # ------------------------------------------
    # Multi Output
    # ------------------------------------------
    else:
        # KNN
        if setupPar['model'] == "KNN":
            for ii, weights in enumerate(['uniform', 'distance']):
                if setupPar['method'] == 0:
                    mdl = MultiOutputRegressor(neighbors.KNeighborsRegressor(n_neighbors=setupMdl['SK_KNN_neighbors'],
                                                                             weights=weights))
                else:
                    mdl = MultiOutputClassifier(neighbors.KNeighborsClassifier(n_neighbors=setupMdl['SK_KNN_neighbors'],
                                                                               weights=weights))

        # RF
        if setupPar['model'] == "RF":
            if setupPar['method'] == 0:
                mdl = MultiOutputRegressor(RandomForestRegressor(max_depth=setupMdl['SK_RF_depth'],
                                                                 random_state=setupMdl['SK_RF_state'],
                                                                 n_estimators=setupMdl['SK_RF_estimators']))
            else:
                mdl = MultiOutputClassifier(RandomForestClassifier(max_depth=setupMdl['SK_RF_depth'],
                                                                   random_state=setupMdl['SK_RF_state'],
                                                                   n_estimators=setupMdl['SK_RF_estimators']))

        # SVM
        if setupPar['model'] == "SVM":
            if setupPar['method'] == 0:
                mdl = MultiOutputRegressor(SVR(kernel=setupMdl['SK_SVM_kernel'], C=setupMdl['SK_SVM_C'],
                                               gamma=setupMdl['SK_SVM_gamma'], epsilon=setupMdl['SK_SVM_epsilon']))
            else:
                mdl = MultiOutputClassifier(SVC(kernel=setupMdl['SK_SVM_kernel'], C=setupMdl['SK_SVM_C'],
                                                gamma=setupMdl['SK_SVM_gamma']))

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Load Model
    # ==============================================================================
    try:
        mdl = joblib.load(mdlName)
        print("INFO: Model exist and will be retrained!")
    except:
        joblib.dump(mdl, mdlName)
        print("INFO: Model does not exist and will be created!")

    # ==============================================================================
    # Train
    # ==============================================================================
    mdl.fit(data['T']['X'], data['T']['y'])

    # ==============================================================================
    # Save model
    # ==============================================================================
    joblib.dump(mdl, mdlName)
