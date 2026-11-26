#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         train
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
from src.preprocess import preprocess
from src.general.framing import framing, frameIds
from src.general.features import features
from src.general.createSeq import createSeq
from src.model.trainMdlDL import trainMdlDL
from src.model.trainMdlML import trainMdlML
from src.model.trainMdlSP import trainMdlSP
from src.model.trainMdlSF import trainMdlSF
from src.model.trainMdlCL import trainMdlCL
from src.general.adaptDim import adaptDim
from src.data.summaryData import summaryData
from src.general.helpFnc import balance_window, setSeed

# ==============================================================================
# External
# ==============================================================================
import copy


#######################################################################################################################
# Helper
#######################################################################################################################
def members(setupExp):
    """
    Settings of the members of a seed ensemble (setupExp['ensemble'] = N networks, DL and CL): member k is trained with
    the seed seed + k and stored as model.pt (k = 0) or model_<k+1>.pt. The test averages the predictions of all members.
    """
    n, seed = max(1, int(setupExp.get('ensemble', 1))), int(setupExp.get('seed', 42))
    for k in range(n):
        if n > 1:
            print("INFO: Ensemble member %d/%d (seed %d)" % (k + 1, n, seed + k))
        setSeed(seed + k)
        yield dict(setupExp, member=k)


def shallow(data):
    """Copy of the set dictionaries, the models replace their arrays (reshaping) without changing the original."""
    return {k: dict(v) for k, v in data.items()}


#######################################################################################################################
# Function
#######################################################################################################################
def train(data, setupExp, setupDat, setupPar, setupMdl):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Running PyDTS (Training)")
    print("\n")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    dataTrain = {'T': {}, 'V': {}}

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Msg
    # ==============================================================================
    print("------------------------------------------")
    print("START: Preprocessing")
    print("------------------------------------------")

    # ==============================================================================
    # Data Summary
    # ==============================================================================
    summaryData(data['T']['X'], data['T']['y'], setupDat)

    # ==============================================================================
    # Calc
    # ==============================================================================
    [data['T'], t, idsT] = preprocess(data['T'], copy.deepcopy(setupDat), setupPar)
    [data['V'], _, idsV] = preprocess(data['V'], setupDat, setupPar)

    # ==============================================================================
    # Balancing
    # ==============================================================================
    if setupPar['frame'] == 1 and setupDat['balance'] == -1:
        [data['T']['X'], data['T']['y']] = balance_window(data['T']['X'], data['T']['y'], window_size=setupMdl['batch'],
                                                          step=setupMdl['batch'], strategy='avg', n_bins=3, n_per_bin=10)
        [data['V']['X'], data['V']['y']] = balance_window(data['V']['X'], data['V']['y'], window_size=setupMdl['batch'],
                                                          step=setupMdl['batch'], strategy='avg', n_bins=3, n_per_bin=10)
        print("INFO: Data equally balanced")
        [idsT, idsV] = [None, None]                                                                                     # rows re-ordered
    else:
        print("INFO: Data is not balanced")

    # ==============================================================================
    # Msg
    # ==============================================================================
    print("\n")

    ###################################################################################################################
    # Framing
    ###################################################################################################################
    # ==============================================================================
    # Msg
    # ==============================================================================
    print("------------------------------------------")
    print("START: Framing and Edge Detection")
    print("------------------------------------------")

    # ==============================================================================
    # Calc
    # ==============================================================================
    if setupPar['frame'] == 1:
        if setupPar['overlap'] == -1:
            dataTrain['T']['X'] = framing(data['T']['X'], setupPar['window'], 0, idsT)
            dataTrain['T']['y'] = framing(data['T']['y'], setupPar['window'], 0, idsT)
            dataTrain['V']['X'] = framing(data['V']['X'], setupPar['window'], 0, idsV)
            dataTrain['V']['y'] = framing(data['V']['y'], setupPar['window'], 0, idsV)
        else:
            dataTrain['T']['X'] = framing(data['T']['X'], setupPar['window'], setupPar['overlap'], idsT)
            dataTrain['T']['y'] = framing(data['T']['y'], setupPar['window'], setupPar['overlap'], idsT)
            dataTrain['V']['X'] = framing(data['V']['X'], setupPar['window'], setupPar['overlap'], idsV)
            dataTrain['V']['y'] = framing(data['V']['y'], setupPar['window'], setupPar['overlap'], idsV)
        ov = 0 if setupPar['overlap'] == -1 else setupPar['overlap']
        dataTrain['T']['id'] = frameIds(idsT, len(data['T']['y']), setupPar['window'], ov)                              # ID per frame (CL)
        dataTrain['V']['id'] = frameIds(idsV, len(data['V']['y']), setupPar['window'], ov)
    else:
        dataTrain['T']['X'] = data['T']['X'].values
        dataTrain['T']['y'] = data['T']['y'].values
        dataTrain['V']['X'] = data['V']['X'].values
        dataTrain['V']['y'] = data['V']['y'].values
        print("INFO: No frames calculated using raw data")

    # ==============================================================================
    # Msg
    # ==============================================================================
    print("\n")

    ###################################################################################################################
    # Features
    ###################################################################################################################
    # ==============================================================================
    # Msg
    # ==============================================================================
    print("------------------------------------------")
    print("START: Feature Extraction")
    print("------------------------------------------")

    # ==============================================================================
    # Calc
    # ==============================================================================
    if setupPar['feat'] == 1 or setupPar['feat'] == 3:
        dataTrain['T']['X'] = features(dataTrain['T']['X'], setupMdl['feat'])
        dataTrain['V']['X'] = features(dataTrain['V']['X'], setupMdl['feat'])
    else:
        print("INFO: No features calculated using raw data")

    # ==============================================================================
    # Msg
    # ==============================================================================
    print("\n")

    ###################################################################################################################
    # Sequence
    ###################################################################################################################
    # ==============================================================================
    # Msg
    # ==============================================================================
    print("------------------------------------------")
    print("START: Creating Input Sequence")
    print("------------------------------------------")

    # ==============================================================================
    # Calc
    # ==============================================================================
    if setupPar['frame'] == 1:
        dataTrain['T']['y'] = createSeq(dataTrain['T']['y'], setupPar)
        dataTrain['V']['y'] = createSeq(dataTrain['V']['y'], setupPar)

    # ==============================================================================
    # Adapt Dimension
    # ==============================================================================
    [dataTrain['T']['X'], dataTrain['T']['y'], _] = adaptDim(dataTrain['T']['X'], dataTrain['T']['y'], setupPar)
    [dataTrain['V']['X'], dataTrain['V']['y'], _] = adaptDim(dataTrain['V']['X'], dataTrain['V']['y'], setupPar)

    # ==============================================================================
    # Msg
    # ==============================================================================
    print("\n")

    ###################################################################################################################
    # Training
    ###################################################################################################################
    # ==============================================================================
    # Msg
    # ==============================================================================
    print("------------------------------------------")
    print("START: Training")
    print("------------------------------------------")

    # ==============================================================================
    # Model
    # ==============================================================================
    # ------------------------------------------
    # Classical Learning
    # ------------------------------------------
    if setupPar['solver'] == 'SP':
        trainMdlSP(dataTrain, t, setupPar, setupMdl, setupExp)

    # ------------------------------------------
    # Machine Learning
    # ------------------------------------------
    elif setupPar['solver'] == 'ML':
        trainMdlML(dataTrain, setupPar, setupMdl, setupExp)

    # ------------------------------------------
    # Deep Learning
    # ------------------------------------------
    elif setupPar['solver'] == 'DL':
        for e in members(setupExp):
            trainMdlDL(shallow(dataTrain), setupDat, setupPar, setupMdl, e)

    # ------------------------------------------
    # Short-Time-Forecasting
    # ------------------------------------------
    elif setupPar['solver'] == 'SF':
        trainMdlSF(dataTrain, setupDat, setupPar, setupMdl, setupExp)

    # ------------------------------------------
    # Closed Loop Modeling
    # ------------------------------------------
    elif setupPar['solver'] == 'CL':
        for e in members(setupExp):
            trainMdlCL(shallow(dataTrain), setupDat, setupPar, setupMdl, e)

    ###################################################################################################################
    # MSG Out
    ###################################################################################################################
    print("DONE: Running PyDTS (Training)")
    print('----------------------')
    print("\n\n")
