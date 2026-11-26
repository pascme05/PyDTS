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
from src.general.helpFnc import normVal, shuffle_subarrays
from src.general.featuresRoll import featuresRoll
from src.general.featuresPhy import featuresPhy

# ==============================================================================
# External
# ==============================================================================
import pandas as pd
import numpy as np
from os.path import join as pjoin
from sklearn.model_selection import KFold
import copy
import math
import sys
import scipy.io


#######################################################################################################################
# Helper
#######################################################################################################################
def splitIdx(n, ids, setupExp, setupDat, method, train, fold, shu):
    """
    Positional rows of one data frame (n rows, id column ids) for training (train=1), testing (train=2) or validation
    (train=3), and the pool of rows the normalisation is computed from (training + validation rows, never test rows).
    Validation rows are always removed from the training rows. ID split: every non-test ID is trained on; validation is
    the last share rV of each validation ID (idV) that is a training ID, or of every training ID if there is none.
    The split uses one fixed random permutation (seed 42)
    when shuffling, so that inputs and outputs of equal length are split identically and repeated calls (training,
    testing, normalisation) return disjoint sets.
    """
    # ==============================================================================
    # Init
    # ==============================================================================
    idx = np.arange(n)
    if shu is True:
        idx = np.random.default_rng(42).permutation(n)

    # ==============================================================================
    # Pool (training + validation) and test rows
    # ==============================================================================
    # ------------------------------------------
    # 1-Fold
    # ------------------------------------------
    if method == 0:
        nTest = math.ceil((1 - setupDat['rT']) * n)
        pool, test = idx[:n - nTest], idx[n - nTest:]

    # ------------------------------------------
    # k-Fold
    # ------------------------------------------
    elif method == 1:
        pool, test = list(KFold(n_splits=setupExp['kfold']).split(idx))[fold - 1]
        pool, test = idx[pool], idx[test]

    # ------------------------------------------
    # Transfer (separate files for training, validation and testing)
    # ------------------------------------------
    elif method == 2:
        return idx, idx

    # ------------------------------------------
    # IDs
    # ------------------------------------------
    else:
        ids = np.asarray(ids)
        isTest = np.isin(ids[idx], setupDat['idT'])
        pool, test = idx[~isTest], idx[isTest]
        if train == 2:
            return np.sort(test), pool
        trainIds = list(pd.unique(ids[pool]))
        valIds = [i for i in trainIds if i in list(np.atleast_1d(setupDat['idV']))]
        if not valIds:
            valIds = trainIds
            if len(np.atleast_1d(setupDat['idV'])) and str(setupDat['idV']) not in ('', '[]', "['']"):
                print("WARN: Validation IDs %s are no training IDs, the last %.0f %% of every training ID is used for "
                      "validation" % (setupDat['idV'], 100 * setupDat['rV']))
        val = []
        for v in valIds:                                                                                                # last share rV of each ID
            rows = pool[ids[pool] == v]
            val.append(rows[len(rows) - int(np.floor(setupDat['rV'] * len(rows))):])
        val = np.concatenate(val) if val else np.array([], dtype=int)
        sel = val if train == 3 else pool[~np.isin(pool, val)]

        return np.sort(sel), pool

    # ==============================================================================
    # Selection
    # ==============================================================================
    if train == 2:
        return np.sort(test), pool
    nVal = int(np.floor(setupDat['rV'] * len(pool)))
    sel = pool[len(pool) - nVal:] if train == 3 else pool[:len(pool) - nVal]

    return np.sort(sel), pool


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
    if setupDat['Shuffle'] == 'ID':
        shu = False
    else:
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
    if setupDat['type'] == 'xlsx':
        try:
            data['X'] = pd.read_excel(filename, sheet_name='input')
            data['y'] = pd.read_excel(filename, sheet_name='output')
            unitsRaw = pd.read_excel(filename, sheet_name='units')
            units['Input'] = pd.DataFrame(columns=data['X'].columns)
            units['Input'].loc[0] = unitsRaw['Input'].dropna().values
            units['Output'] = pd.DataFrame(columns=data['y'].columns)
            units['Output'].loc[0] = unitsRaw['Output'].dropna().values
            print("INFO: Data file loaded")
        except Exception as e:
            sys.exit("ERROR: Data file %s could not be loaded (%s)" % (filename, e))

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
        except Exception as e:
            sys.exit("ERROR: Data file %s could not be loaded (%s)" % (filename, e))

    # ==============================================================================
    # Default
    # ==============================================================================
    else:
        sys.exit("ERROR: Data format %s not available, use xlsx (sheets input, output, units) or mat"
                 % setupDat['type'])

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
    # Shuffle IDs
    # ==============================================================================
    if setupDat['Shuffle'] == 'ID' and method in (0, 1):
        [data['X'], data['y']] = shuffle_subarrays(data['X'], data['y'], data['y']['id'], seed=42)
    if shu is True and len(data['X']) != len(data['y']):
        print("WARN: Inputs and outputs have different lengths, data are split without shuffling")
        shu = False

    # ==============================================================================
    # Split (rows of the set and of the normalisation pool)
    # ==============================================================================
    pool = {}
    for key in ('X', 'y'):
        [sel, rows] = splitIdx(len(data[key]), data[key]['id'].values, setupExp, setupDat, method, train, fold, shu)
        pool[key] = data[key].iloc[np.sort(rows)]
        data[key] = data[key].iloc[sel]

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    # ==============================================================================
    # Rolling input features
    # ==============================================================================
    if setupPar['feat'] >= 2:
        for frame in (data, pool):
            tempTime = copy.deepcopy(frame['X']['time'])
            tempID = copy.deepcopy(frame['X']['id'])
            frame['X'] = featuresRoll(frame['X'].drop(['time', 'id'], axis=1), setupMdl, tempID.values)
            frame['X']['time'] = tempTime
            frame['X']['id'] = tempID

    # ==============================================================================
    # Physics input features
    # ==============================================================================
    # [data['X'], units['Input']] = featuresPhy(data['X'])

    # ==============================================================================
    # Norm
    # ==============================================================================
    [maxX, maxY, minX, minY, uX, uY, sX, sY] = normVal(pool['X'].drop(['time', 'id'], axis=1),
                                                       pool['y'].drop(['time', 'id'], axis=1))

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
    setupDat['Ts_raw_X'] = float(np.median(np.diff(data['X']['time'].values[:1000])))                      # robust to gaps
    setupDat['fs_raw_X'] = 1 / setupDat['Ts_raw_X']
    setupDat['Ts_raw_y'] = float(np.median(np.diff(data['y']['time'].values[:1000])))
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
            print("WARN: NaN in data column %s detected using interpolation" % names)

    # ------------------------------------------
    # Output
    # ------------------------------------------
    for names in data['y']:
        if pd.isna(data['y'][names]).any():
            data['y'][names] = data['y'][names].interpolate(limit_direction='both')
            print("WARN: NaN in data column %s detected using interpolation" % names)

    # ==============================================================================
    # Removing NaN/Inf
    # ==============================================================================
    # ------------------------------------------
    # Input
    # ------------------------------------------
    for names in data['X']:
        data['X'][names] = data['X'][names].replace([np.inf, -np.inf], 0).fillna(0)

    # ------------------------------------------
    # Output
    # ------------------------------------------
    for names in data['y']:
        data['y'][names] = data['y'][names].replace([np.inf, -np.inf], 0).fillna(0)

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
