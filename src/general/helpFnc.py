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
from os.path import dirname, join as pjoin
import os
import numpy as np


#######################################################################################################################
# Init Path
#######################################################################################################################
def initPath(nameFolder):
    basePath = pjoin(dirname(os.getcwd()), nameFolder)
    datPath = pjoin(dirname(os.getcwd()), nameFolder, 'data')
    mdlPath = pjoin(dirname(os.getcwd()), nameFolder, 'mdl')
    srcPath = pjoin(dirname(os.getcwd()), nameFolder, 'src')
    resPath = pjoin(dirname(os.getcwd()), nameFolder, 'results')
    setPath = pjoin(dirname(os.getcwd()), nameFolder, 'setup')
    setupPath = {'basePath': basePath, 'datPath': datPath, 'mdlPath': mdlPath, 'srcPath': srcPath, 'resPath': resPath,
                 'setPath': setPath}

    return setupPath


#######################################################################################################################
# Init Setup files
#######################################################################################################################
def initSetup():
    setupExp = {}
    setupDat = {}
    setupPar = {}
    setupMdl = {}

    return [setupExp, setupDat, setupPar, setupMdl]


#######################################################################################################################
# Init Setup files
#######################################################################################################################
def warnMsg(msg, level, flag, setupExp):
    if level == 2:
        setupExp['status']['warnH']['count'] = setupExp['status']['warnH']['count'] + 1
        setupExp['status']['warnH']['msg'][setupExp['status']['warnH']['idx']] = msg
        setupExp['status']['warnH']['idx'] = setupExp['status']['warnH']['idx'] + 1
    else:
        setupExp['status']['warnL']['count'] = setupExp['status']['warnL']['count'] + 1
        setupExp['status']['warnL']['msg'][setupExp['status']['warnL']['idx']] = msg
        setupExp['status']['warnL']['idx'] = setupExp['status']['warnL']['idx'] + 1

    if flag == 1:
        print(msg)


#######################################################################################################################
# Normalisation Values
#######################################################################################################################
def normVal(X, y):
    maxX = np.nanmax(X, axis=0)
    maxY = np.nanmax(y, axis=0)
    minX = np.nanmin(X, axis=0)
    minY = np.nanmin(y, axis=0)
    uX = np.nanmean(X, axis=0)
    uY = np.nanmean(y, axis=0)
    sX = np.nanvar(X, axis=0)
    sY = np.nanvar(y, axis=0)

    return [maxX, maxY, minX, minY, uX, uY, sX, sY]


#######################################################################################################################
# Reshape Mdl Data
#######################################################################################################################
def reshapeMdlData(X, y, setupDat, setupPar, test):
    if len(setupDat['out']) == 1:
        if setupPar['outseq'] >= 1 and test == 0:
            y = y.reshape((y.shape[0], y.shape[1], 1))
        else:
            y = y.reshape((y.shape[0], 1))
    if len(X.shape) == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    elif len(X.shape) == 3:
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
    else:
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    return [X, y]


#######################################################################################################################
# Add Noise
#######################################################################################################################
def addNoise(data, noise):
    noise = noise / 100 * np.random.normal(0, 1, len(data))
    data = data * (1 + noise)

    return data


#######################################################################################################################
# Pad Data
#######################################################################################################################
def pad_dataset(X, y, batch_size):
    """
    Pads the dataset X and y to make sure that the total number of samples is divisible by the batch_size.
    Works for both 2D and 3D data.

    Parameters:
    X: 2D or 3D numpy array of input features
    y: 2D numpy array of labels
    batch_size: The desired batch size

    Returns:
    X_padded: Padded input data
    y_padded: Padded labels
    padding_size: The number of padding samples added
    """

    # Determine the padding size based on the number of samples
    padding_size = batch_size - (X.shape[0] % batch_size)
    padding_data = 0
    padding_labels = 0

    if padding_size == batch_size:  # No padding needed if the size is already divisible
        return X, y, 0  # Return the original data with no padding

    # Padding logic X
    if len(X.shape) == 2:  # 2D array (samples, features)
        padding_data = np.zeros((padding_size, X.shape[1]))  # Padding with zeros, shape should match features

    elif len(X.shape) == 3:  # 3D array (samples, time_steps, features)
        padding_data = np.zeros((padding_size, X.shape[1], X.shape[2]))  # Padding with zeros

    # Padding logic y
    if len(y.shape) == 2:  # 2D array (samples, features)
        padding_labels = np.zeros((padding_size, y.shape[1]))  # Padding the labels with zeros

    elif len(y.shape) == 3:  # 3D array (samples, time_steps, features)
        padding_labels = np.zeros((padding_size, y.shape[1], y.shape[2]))  # Padding labels with zeros

    # Concatenate the original data with the padding
    X_padded = np.concatenate([X, padding_data], axis=0)
    y_padded = np.concatenate([y, padding_labels], axis=0)

    return X_padded, y_padded, padding_size
