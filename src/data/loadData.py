#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         loadData
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
from src.general.helpFnc import normVal
from src.general.featuresRoll import featuresRoll

# ==============================================================================
# External
# ==============================================================================
import pandas as pd
import numpy as np
from os.path import join as pjoin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import copy
import scipy.io


#######################################################################################################################
# Function
#######################################################################################################################
def loadData(setupExp, setupDat, setupPar, setupMdl, setupPath, name, method, train, fold):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Loading Dataset")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    shu = setupDat['Shuffle']

    # ==============================================================================
    # Variables
    # ==============================================================================
    data = {}
    units = {}

    # ==============================================================================
    # Path
    # ==============================================================================
    name = name + '.' + setupDat['type']
    path = setupPath['datPath']
    filename = pjoin(path, name)

    ###################################################################################################################
    # Loading
    ###################################################################################################################
    # ==============================================================================
    # Excel
    # ==============================================================================
    if setupDat['type'] == 'xlsx' or setupDat['type'] == 'csv':
        try:
            data['X'] = pd.read_excel(filename, sheet_name='input')
            data['y'] = pd.read_excel(filename, sheet_name='output')
            unitsRaw = pd.read_excel(filename, sheet_name='units')
            units['Input'] = pd.DataFrame(columns=data['X'].columns)
            units['Input'].loc[0] = unitsRaw['Input'].dropna().values
            units['Output'] = pd.DataFrame(columns=data['y'].columns)
            units['Output'].loc[0] = unitsRaw['Output'].dropna().values
            print("INFO: Data file loaded")
        except:
            print("ERROR: Data file could not be loaded")

    # ==============================================================================
    # Mat-file
    # ==============================================================================
    elif setupDat['type'] == 'mat':
        try:
            raw = scipy.io.loadmat(filename)
            for i in range(0, len(raw['labelInp'])):
                raw['labelInp'][i] = raw['labelInp'][i].rstrip()
            for i in range(0, len(raw['labelOut'])):
                raw['labelOut'][i] = raw['labelOut'][i].rstrip()
            data['X'] = pd.DataFrame(data=raw['input'], columns=raw['labelInp'])
            data['y'] = pd.DataFrame(data=raw['output'], columns=raw['labelOut'])
            units['Input'] = pd.DataFrame(columns=raw['labelInp'])
            units['Input'].loc[0] = raw['unitInp']
            units['Output'] = pd.DataFrame(columns=raw['labelOut'])
            units['Output'].loc[0] = raw['unitOut']
            print("INFO: Data file loaded")
        except:
            print("ERROR: Data file could not be loaded")

    # ==============================================================================
    # Default
    # ==============================================================================
    else:
        print("ERROR: Data format not available")

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Selecting Input and Output
    # ==============================================================================
    # ------------------------------------------
    # Input
    # ------------------------------------------
    if len(setupDat['inp']) != 0:
        inp = copy.deepcopy(setupDat['inp'])
        inp.append('time')
        inp.append('id')
        data['X'] = data['X'][inp]
        units['Input'] = units['Input'][inp]

    # ------------------------------------------
    # Output
    # ------------------------------------------
    if len(setupDat['out']) != 0:
        out = copy.deepcopy(setupDat['out'])
        out.append('time')
        out.append('id')
        data['y'] = data['y'][out]
        units['Output'] = units['Output'][out]

    # ==============================================================================
    # Limiting
    # ==============================================================================
    if setupDat['lim'] != 0:
        data['X'] = data['X'].head(setupDat['lim'])
        data['y'] = data['y'].head(setupDat['lim'])
        print("INFO: Data limited to ", setupDat['lim'], " samples")
    else:
        print("INFO: Data samples not limited")

    ###################################################################################################################
    # Calculating
    ###################################################################################################################
    # ==============================================================================
    # 1-Fold
    # ==============================================================================
    if method == 0:
        # ------------------------------------------
        # Training
        # ------------------------------------------
        if train == 1:
            data['X'], _ = train_test_split(data['X'], test_size=(1 - setupDat['rT']), random_state=None, shuffle=shu)
            data['y'], _ = train_test_split(data['y'], test_size=(1 - setupDat['rT']), random_state=None, shuffle=shu)

        # ------------------------------------------
        # Testing
        # ------------------------------------------
        elif train == 2:
            _, data['X'] = train_test_split(data['X'], test_size=(1 - setupDat['rT']), random_state=None, shuffle=shu)
            _, data['y'] = train_test_split(data['y'], test_size=(1 - setupDat['rT']), random_state=None, shuffle=shu)

        # ------------------------------------------
        # Validation
        # ------------------------------------------
        else:
            # Split
            X, _ = train_test_split(data['X'], test_size=(1 - setupDat['rT']), random_state=None, shuffle=False)
            y, _ = train_test_split(data['y'], test_size=(1 - setupDat['rT']), random_state=None, shuffle=False)

            # Extract
            data['X'], _ = train_test_split(X, test_size=(1 - setupDat['rV']), random_state=None, shuffle=shu)
            data['y'], _ = train_test_split(y, test_size=(1 - setupDat['rV']), random_state=None, shuffle=shu)

    # ==============================================================================
    # k-Fold
    # ==============================================================================
    elif method == 1:
        # ------------------------------------------
        # Init
        # ------------------------------------------
        kfX = KFold(n_splits=setupExp['kfold'])
        kfX.get_n_splits(data['X'])
        kfy = KFold(n_splits=setupExp['kfold'])
        kfy.get_n_splits(data['y'])

        # ------------------------------------------
        # Training
        # ------------------------------------------
        if train == 1:
            # X
            iter = 0
            for idx, _ in kfX.split(data['X']):
                iter = iter + 1
                if iter == fold:
                    data['X'] = data['X'].iloc[idx, :]
                    break

            # y
            iter = 0
            for idx, _ in kfy.split(data['y']):
                iter = iter + 1
                if iter == fold:
                    data['y'] = data['y'].iloc[idx, :]
                    break

        # ------------------------------------------
        # Testing
        # ------------------------------------------
        elif train == 2:
            # X
            iter = 0
            for idx1, idx2 in kfX.split(data['X']):
                iter = iter + 1
                if iter == fold:
                    data['X'] = data['X'].iloc[idx2, :]
                    break

            # y
            iter = 0
            for idx1, idx2 in kfy.split(data['y']):
                iter = iter + 1
                if iter == fold:
                    data['y'] = data['y'].iloc[idx2, :]
                    break

        # ------------------------------------------
        # Validation
        # ------------------------------------------
        else:
            # X
            iter = 0
            for idx, _ in kfX.split(data['X']):
                iter = iter + 1
                if iter == fold:
                    data['X'] = data['X'].iloc[idx, :]
                    break

            # y
            iter = 0
            for idx, _ in kfy.split(data['y']):
                iter = iter + 1
                if iter == fold:
                    data['y'] = data['y'].iloc[idx, :]
                    break

            # Extract
            data['X'], _ = train_test_split(data['X'], test_size=(1 - setupDat['rV']), random_state=None, shuffle=shu)
            data['y'], _ = train_test_split(data['y'], test_size=(1 - setupDat['rV']), random_state=None, shuffle=shu)

    # ==============================================================================
    # Transfer
    # ==============================================================================
    elif method == 2:
        data['X'] = data['X']
        data['y'] = data['y']

    # ==============================================================================
    # IDs
    # ==============================================================================
    else:
        # ------------------------------------------
        # Init
        # ------------------------------------------
        idTest = setupDat['idT']
        idVal = setupDat['idV']

        # ------------------------------------------
        # Training
        # ------------------------------------------
        if train == 1:
            # Split
            for sel in idTest:
                data['X'].drop(data['X'][data['X']['id'] == sel].index, inplace=True)
                data['y'].drop(data['y'][data['y']['id'] == sel].index, inplace=True)

        # ------------------------------------------
        # Testing
        # ------------------------------------------
        elif train == 2:
            # Init
            idTrain = [sel for sel in data['X']['id'].unique() if sel not in idTest]

            # Split
            for sel in idTrain:
                data['X'].drop(data['X'][data['X']['id'] == sel].index, inplace=True)
                data['y'].drop(data['y'][data['y']['id'] == sel].index, inplace=True)

        # ------------------------------------------
        # Validation
        # ------------------------------------------
        else:
            # Init
            idTrain = [sel for sel in data['X']['id'].unique() if sel not in idVal]

            # Split
            for sel in idTrain:
                data['X'].drop(data['X'][data['X']['id'] == sel].index, inplace=True)
                data['y'].drop(data['y'][data['y']['id'] == sel].index, inplace=True)

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    # ==============================================================================
    # Rolling input features
    # ==============================================================================
    if setupPar['feat'] >= 2:
        tempTime = copy.deepcopy(data['X']['time'])
        tempID = copy.deepcopy(data['X']['id'])
        data['X'] = featuresRoll(data['X'].drop(['time', 'id'], axis=1), setupMdl)
        data['X']['time'] = tempTime
        data['X']['id'] = tempID

    # ==============================================================================
    # Norm
    # ==============================================================================
    [maxX, maxY, minX, minY, uX, uY, sX, sY] = normVal(data['X'].drop(['time', 'id'], axis=1),
                                                       data['y'].drop(['time', 'id'], axis=1))

    # ==============================================================================
    # Labels
    # ==============================================================================
    setupDat['inpLabel'] = data['X'].columns
    setupDat['inpLabel'] = setupDat['inpLabel'].drop(['time', 'id'])
    setupDat['outLabel'] = data['y'].columns
    setupDat['outLabel'] = setupDat['outLabel'].drop(['time', 'id'])
    setupDat['inpUnits'] = units['Input'].drop(['time', 'id'], axis=1)
    setupDat['outUnits'] = units['Output'].drop(['time', 'id'], axis=1)

    # ==============================================================================
    # Sampling Time
    # ==============================================================================
    setupDat['Ts_raw_X'] = data['X']['time'].iloc[1] - data['X']['time'].iloc[0]
    setupDat['fs_raw_X'] = 1 / setupDat['Ts_raw_X']
    setupDat['Ts_raw_y'] = data['y']['time'].iloc[1] - data['y']['time'].iloc[0]
    setupDat['fs_raw_y'] = 1 / setupDat['Ts_raw_y']

    # ==============================================================================
    # Interpolation
    # ==============================================================================
    # ------------------------------------------
    # Input
    # ------------------------------------------
    for names in data['X']:
        if pd.isna(data['X'][names]).any():
            data['X'][names] = data['X'][names].interpolate(limit_direction='both')
            print("WARN: NaN in data column %s detected using interpolation", names)

    # ------------------------------------------
    # Output
    # ------------------------------------------
    for names in data['y']:
        if pd.isna(data['y'][names]).any():
            data['y'][names] = data['y'][names].interpolate(limit_direction='both')
            print("WARN: NaN in data column %s detected using interpolation", names)

    # ==============================================================================
    # Removing NaN/Inf
    # ==============================================================================
    # ------------------------------------------
    # Input
    # ------------------------------------------
    for names in data['X']:
        data['X'][names] = data['X'][names].fillna(0)
        data['X'][names].replace([np.inf, -np.inf], 0, inplace=True)

    # ------------------------------------------
    # Output
    # ------------------------------------------
    for names in data['y']:
        data['y'][names] = data['y'][names].fillna(0)
        data['y'][names].replace([np.inf, -np.inf], 0, inplace=True)

    # ==============================================================================
    # Normalisation Values
    # ==============================================================================
    setupDat['normMaxX'] = maxX
    setupDat['normMaxY'] = maxY
    setupDat['normMinX'] = minX
    setupDat['normMinY'] = minY
    setupDat['normAvgX'] = uX
    setupDat['normAvgY'] = uY
    setupDat['normVarX'] = sX
    setupDat['normVarY'] = sY

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [data, setupDat]
