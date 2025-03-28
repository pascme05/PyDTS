#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         mdlParaJournal
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

#######################################################################################################################
# Function
#######################################################################################################################
def mdlParaJournal(setupMdl):
    ###################################################################################################################
    # Init
    ###################################################################################################################
    setupMdl['feat'] = {}
    setupMdl['feat_roll'] = {}

    ###################################################################################################################
    # General Model Parameters
    ###################################################################################################################
    # ==============================================================================
    # Hyperparameters
    # ==============================================================================
    setupMdl['batch'] = 1000                                                                                            # batch size for training and testing
    setupMdl['epoch'] = 100                                                                                             # number of epochs for training
    setupMdl['patience'] = 15                                                                                           # number of epochs as patience during training
    setupMdl['valsteps'] = 50                                                                                           # number of validation steps
    setupMdl['shuffle'] = 'False'                                                                                       # shuffling data before training (after splitting data)
    setupMdl['verbose'] = 2                                                                                             # level of detail for showing training information (0 silent)

    # ==============================================================================
    # Solver Parameters
    # ==============================================================================
    setupMdl['loss'] = 'mae'                                                                                            # loss function 1) mae, 2) mse, 3) BinaryCrossentropy, 4) KLDivergence, 5) accuracy
    setupMdl['metric'] = 'mse'                                                                                          # loss metric 1) mae, 2) mse, 3) BinaryCrossentropy, 4) KLDivergence, 5) accuracy
    setupMdl['opt'] = 'Adam'                                                                                            # solver 1) Adam, 2) RMSprop, 3) SGD
    setupMdl['lr'] = 1e-3                                                                                               # learning rate
    setupMdl['beta1'] = 0.9                                                                                             # first moment decay
    setupMdl['beta2'] = 0.999                                                                                           # second moment decay
    setupMdl['eps'] = 1e-08                                                                                             # small constant for stability
    setupMdl['rho'] = 0.9                                                                                               # discounting factor for the history/coming gradient
    setupMdl['mom'] = 0.0                                                                                               # momentum

    ###################################################################################################################
    # Specific Model Parameters
    ###################################################################################################################
    # ==============================================================================
    # Signal-Processing (SP) Models
    # ==============================================================================
    # ------------------------------------------
    # SS
    # ------------------------------------------
    setupMdl['SP_SS_block'] = 10                                                                                        # number of block rows to use in the block Hankel matrices
    setupMdl['SP_SS_order'] = 3                                                                                         # state space model order

    # ------------------------------------------
    # TF
    # ------------------------------------------
    setupMdl['SP_TF_poles'] = 4                                                                                         # number of poles
    setupMdl['SP_TF_zeros'] = 3                                                                                         # number of zeros
    setupMdl['SP_TF_method'] = 'fft'                                                                                    # estimation method: 'fft' or 'h2'
    setupMdl['SP_TF_l1'] = 0.1                                                                                          # L1 regularisation

    # ==============================================================================
    # Machine Learning (ML) Models
    # ==============================================================================
    # ------------------------------------------
    # RF
    # ------------------------------------------
    setupMdl['SK_RF_depth'] = 10                                                                                        # maximum depth of the tree
    setupMdl['SK_RF_state'] = 0                                                                                         # number of states
    setupMdl['SK_RF_estimators'] = 128                                                                                  # number of trees in the forest

    # ------------------------------------------
    # SVM
    # ------------------------------------------
    setupMdl['SK_SVM_kernel'] = 'rbf'                                                                                   # kernel function of the SVM ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    setupMdl['SK_SVM_C'] = 100                                                                                          # regularization
    setupMdl['SK_SVM_gamma'] = 0.1                                                                                      # kernel coefficient
    setupMdl['SK_SVM_epsilon'] = 0.1

    # ------------------------------------------
    # KNN
    # ------------------------------------------
    setupMdl['SK_KNN_neighbors'] = 140                                                                                  # number of neighbors

    # ==============================================================================
    # Deep Learning (DL) Models
    # ==============================================================================

    # ==============================================================================
    # Short Term Time-Series Forecasting (SF) Models
    # ==============================================================================
    setupMdl['SF_Freq'] = 'H'                                                                                           # Frequency of the data. Must be a valid pandas or polars offset alias, or an integer

    ###################################################################################################################
    # Features
    ###################################################################################################################
    # ==============================================================================
    # Statistical (input based)
    # ==============================================================================
    setupMdl['feat_roll']['EWMA'] = [0]
    setupMdl['feat_roll']['EWMS'] = [0]
    setupMdl['feat_roll']['diff'] = 1

    # ==============================================================================
    # Statistical (frame based)
    # ==============================================================================
    setupMdl['feat']['Mean'] = 1
    setupMdl['feat']['Std'] = 1
    setupMdl['feat']['RMS'] = 1
    setupMdl['feat']['Peak2Rms'] = 1
    setupMdl['feat']['Median'] = 1
    setupMdl['feat']['Min'] = 1
    setupMdl['feat']['Max'] = 1
    setupMdl['feat']['Per25'] = 1
    setupMdl['feat']['Per75'] = 1
    setupMdl['feat']['Energy'] = 1
    setupMdl['feat']['Var'] = 1
    setupMdl['feat']['Range'] = 1
    setupMdl['feat']['3rdMoment'] = 1
    setupMdl['feat']['4thMoment'] = 1

    # ==============================================================================
    # Frequency (frame based)
    # ==============================================================================
    setupMdl['feat']['fft'] = 1

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return setupMdl
