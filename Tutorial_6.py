#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling (Cross-Domain Modelling)
# File:         Tutorial_6
# Date:         03.11.2023
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.1
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Import external libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.main import main
from src.optiHyp import optiHyp
from src.optiGrid import optiGrid
from src.general.helpFnc import initPath, initSetup
from mdlPara import mdlPara

# ==============================================================================
# External
# ==============================================================================
import warnings
import os

#######################################################################################################################
# Format
#######################################################################################################################
warnings.filterwarnings("ignore")

#######################################################################################################################
# Paths
#######################################################################################################################
setupPath = initPath('PyDTS')

#######################################################################################################################
# Init
#######################################################################################################################
[setupExp, setupDat, setupPar, setupMdl] = initSetup()

#######################################################################################################################
# Setup and Configuration
#######################################################################################################################
# ==============================================================================
# Experimental Parameters
# ==============================================================================
# ------------------------------------------
# Names
# ------------------------------------------
setupExp['name'] = 'Tutorial_6'                                                                                         # Name of the simulation
setupExp['author'] = 'Pascal Schirmer'                                                                                  # Name of the author

# ------------------------------------------
# General
# ------------------------------------------
setupExp['sim'] = 0                                                                                                     # 0) simulation, 1) optimisation hyperparameters, 2) optimising grid
setupExp['gpu'] = 1                                                                                                     # 0) cpu, 1) gpu
setupExp['warn'] = 3                                                                                                    # 0) all msg are logged, 1) INFO not logged, 2) INFO and WARN not logged, 3) disabled

# ------------------------------------------
# Training/Testing
# ------------------------------------------
setupExp['method'] = 3                                                                                                  # 0) 1-fold with data split, 1) k-fold with cross validation, 2) transfer learning with different datasets, 3) id based
setupExp['trainBatch'] = 0                                                                                              # 0) all no batching, 1) fixed batch size (see data batch parameter), 2) id based
setupExp['kfold'] = 10                                                                                                  # number of folds for method 1)
setupExp['train'] = 0                                                                                                   # 0) no training (trying to load model), 1) training new model (or retraining)
setupExp['test'] = 1                                                                                                    # 0) no testing, 1) testing

# ------------------------------------------
# Output Control
# ------------------------------------------
setupExp['save'] = 0                                                                                                    # 0) results are not saved, 1) results are saved
setupExp['log'] = 0                                                                                                     # 0) no data logging, 1) logging input data
setupExp['plot'] = 1                                                                                                    # 0) no plotting, 1) plotting

# ==============================================================================
# Data Parameters
# ==============================================================================
# ------------------------------------------
# General
# ------------------------------------------
setupDat['type'] = 'mat'                                                                                                # data input type: 1) 'xlsx', 2) 'csv', 3) 'mat'
setupDat['batch'] = 100000                                                                                              # number of samples fed at once to training
setupDat['Shuffle'] = False                                                                                             # False: no shuffling, True: shuffling data when splitting
setupDat['rT'] = 0.9                                                                                                    # training proportion (0, 1)
setupDat['rV'] = 0.2                                                                                                    # validation proportion (0, 1) as percentage from training proportion
setupDat['idT'] = [36, 37, 39]                                                                                          # list of testing ids for method 3)
setupDat['idV'] = [33, 34, 35]                                                                                          # list of validation ids for method 3)

# ------------------------------------------
# Datasets
# ------------------------------------------
setupDat['train'] = ['Tutorial_6']                                                                                      # name of training datasets (multiple)
setupDat['test'] = 'Tutorial_6'                                                                                         # name of testing datasets (one)
setupDat['val'] = 'Tutorial_6'                                                                                          # name of validation dataset (one)

# ------------------------------------------
# Input/ Output Mapping
# ------------------------------------------
setupDat['inp'] = ['v', 'T', 'V_DC', 'T_a2']                                                                            # names of the input variables (X) if empty all
setupDat['out'] = ['I_DC']                                                                                              # names of the output variables (y)

# ------------------------------------------
# Sampling
# ------------------------------------------
setupDat['fs'] = 10                                                                                                     # sampling frequency (Hz)
setupDat['lim'] = 0                                                                                                     # 0) data is not limited, x) limited to x samples

# ------------------------------------------
# Pre-processing
# ------------------------------------------
setupDat['weightNorm'] = 0                                                                                              # 0) separate normalisation per input/output channel, 1) weighted normalisation
setupDat['inpNorm'] = 3                                                                                                 # normalising input values (X): 0) None, 1) -1/+1, 2) 0/1, 3) avg/sig
setupDat['outNorm'] = 2                                                                                                 # normalising output values (y): 0) None, 1) -1/+1, 2) 0/1, 3) avg/sig
setupDat['inpNoise'] = 0                                                                                                # adding gaussian noise (dB) to input
setupDat['outNoise'] = 0                                                                                                # adding gaussian noise (dB) to output
setupDat['inpFil'] = 0                                                                                                  # filtering input data (X): 0) None, 1) Median
setupDat['outFil'] = 0                                                                                                  # filtering output data (y): 0) None, 1) Median
setupDat['inpFilLen'] = 61                                                                                              # filter length input data (samples)
setupDat['outFilLen'] = 61                                                                                              # filter length output data (samples)
setupDat['threshold'] = 0.1                                                                                             # 0) no threshold x) threshold to transform regressio into classification data
setupDat['balance'] = 0                                                                                                 # 0) no balancing 1) balancing based classes, x) balancing based on x bins

# ==============================================================================
# General Parameters
# ==============================================================================
# ------------------------------------------
# Solver
# ------------------------------------------
setupPar['method'] = 0                                                                                                  # 0) regression, 1) classification
setupPar['solver'] = 'DL'                                                                                               # solver 1) 'SP': Signal Processing, 2) 'ML': Machine Learning, 3) 'DL': Deep Learning
setupPar['model'] = 'LSTM'                                                                                              # possible models 1) SP: State Space (SS), Transfer Function (TF), 2) ML: RF, KNN, SVM, 3) DL: CNN, LSTM, DNN

# ------------------------------------------
# Framing and Features
# ------------------------------------------
setupPar['lag'] = 0                                                                                                     # lagging between input (X) and output (y) in samples
setupPar['frame'] = 1                                                                                                   # 0) no framing, 1) framing
setupPar['feat'] = 0                                                                                                    # 0) raw data values, 1) statistical features (frame based), 2) statistical features (input based), 3) input and frame based features
setupPar['init'] = 0                                                                                                    # 0) no initial values 1) adding initial values from y
setupPar['window'] = 300                                                                                                # window length (samples)
setupPar['overlap'] = 290                                                                                               # overlap between consecutive windows (no overlap during test if -1)
setupPar['outseq'] = 0                                                                                                  # 0) seq2point, x) length of the subsequence in samples
setupPar['yFocus'] = 299                                                                                                # focus point for seq2point (average if -1)
setupPar['nDim'] = 3                                                                                                    # input dimension for model 2) or 3)

# ------------------------------------------
# Postprocessing
# ------------------------------------------
setupPar['outMin'] = -1e9                                                                                                # limited output values (minimum)
setupPar['outMax'] = +1e9                                                                                                # limited output values (maximum)

# ==============================================================================
# Model Parameters
# ==============================================================================
setupMdl = mdlPara(setupMdl)


#######################################################################################################################
# Calculations
#######################################################################################################################
# ==============================================================================
# Warnings
# ==============================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(setupExp['warn'])

# ==============================================================================
# Model Parameters
# ==============================================================================
if setupExp['sim'] == 0:
    main(setupExp, setupDat, setupPar, setupMdl, setupPath)

# ==============================================================================
# Optimising Hyperparameters
# ==============================================================================
if setupExp['sim'] == 1:
    optiHyp(setupExp, setupDat, setupPar, setupMdl, setupPath)

# ==============================================================================
# Optimising Grid
# ==============================================================================
if setupExp['sim'] == 2:
    optiGrid(setupExp, setupDat, setupPar, setupMdl, setupPath)
