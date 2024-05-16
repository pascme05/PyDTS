#######################################################################################################################
#######################################################################################################################
# Title:        PyDTS (Python Deep Timeseries Simulation)
# Topic:        Black-Box Modeling
# File:         models
# Date:         03.11.2023
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.1
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import tensorflow as tf
from tensorflow import keras


#######################################################################################################################
# CNN Models
#######################################################################################################################
# ==============================================================================
# CNN-1
# ==============================================================================
def tfMdlDNN(X_train, outputdim, activation):
    mdl = tf.keras.models.Sequential()
    mdl.add(tf.keras.layers.InputLayer(X_train.shape[1:]))
    mdl.add(tf.keras.layers.Flatten())
    mdl.add(tf.keras.layers.Dense(32, activation='relu'))
    mdl.add(tf.keras.layers.Dense(32, activation='relu'))
    mdl.add(tf.keras.layers.Dense(32, activation='relu'))
    mdl.add(tf.keras.layers.Dense(outputdim, activation=activation))
    mdl.set_weights(mdl.get_weights())

    return mdl


#######################################################################################################################
# CNN Models
#######################################################################################################################
# ==============================================================================
# CNN-1
# ==============================================================================
def tfMdlCNN(X_train, outputdim, activation):
    mdl = tf.keras.models.Sequential([

        tf.keras.layers.Conv1D(filters=30, kernel_size=10, activation='relu', padding="same", strides=1,
                               input_shape=X_train.shape[1:]),
        tf.keras.layers.Conv1D(filters=30, kernel_size=8, activation='relu', padding="same", strides=1),
        tf.keras.layers.Conv1D(filters=40, kernel_size=6, activation='relu', padding="same", strides=1),
        tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu', padding="same", strides=1),
        tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu', padding="same", strides=1),

        tf.keras.layers.MaxPooling1D(pool_size=5, strides=5, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),

        tf.keras.layers.Dense(outputdim, activation=activation)])

    return mdl


# ==============================================================================
# CNN-Opti
# ==============================================================================
def tfMdlCNNopti(X_train, outputdim, activation):
    mdl = tf.keras.models.Sequential([

        tf.keras.layers.Conv1D(filters=48, kernel_size=6, activation='relu', padding="same", strides=1,
                               input_shape=X_train.shape[1:]),
        tf.keras.layers.Conv1D(filters=56, kernel_size=2, activation='relu', padding="same", strides=1),

        tf.keras.layers.MaxPooling1D(pool_size=4, strides=8, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(384, activation='relu'),
        tf.keras.layers.Dense(320, activation='relu'),

        tf.keras.layers.Dense(outputdim, activation=activation)])

    return mdl


#######################################################################################################################
# LSTM Models
#######################################################################################################################
# ==============================================================================
# LSTM-1
# ==============================================================================
def tfMdlLSTM(X_train, outputdim, activation):
    mdl = tf.keras.models.Sequential()
    mdl.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=X_train.shape[1:]))
    mdl.add(tf.keras.layers.LSTM(128))
    mdl.add(tf.keras.layers.Flatten())
    mdl.add(tf.keras.layers.Dense(32, activation='relu'))
    mdl.add(tf.keras.layers.Dense(32, activation='relu'))
    mdl.add(tf.keras.layers.Dense(32, activation='relu'))
    mdl.add(tf.keras.layers.Dense(outputdim, activation=activation))
    mdl.set_weights(mdl.get_weights())

    return mdl


#######################################################################################################################
# Optimal Models
#######################################################################################################################
# ==============================================================================
# Regression
# ==============================================================================
# ------------------------------------------
# DNN
# ------------------------------------------
def tfMdloptiR3(hp):
    # Input
    mdl = keras.Sequential()
    mdl.add(tf.keras.layers.Flatten())

    # Mdl
    hp_units = hp.Int('units', min_value=64, max_value=512, step=64)
    for i in range(hp.Int("dnn_layers", 2, 6, step=1)):
        mdl.add(keras.layers.Dense(units=hp_units, activation='relu'))
    mdl.add(keras.layers.Dense(1, activation='linear'))

    # Learner
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 5e-2, 1e-2])
    mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mae', metrics='mse')

    return mdl


# ------------------------------------------
# LSTM
# ------------------------------------------
def tfMdloptiR2(hp):
    # ------------------------------------------
    # Input
    # ------------------------------------------
    mdl = keras.Sequential()

    # ------------------------------------------
    # Mdl
    # ------------------------------------------
    # LSTM Layers
    for i in range(hp.Int("lstm_layers", 0, 3, step=1)):
        mdl.add(tf.keras.layers.LSTM(hp.Int("nodes_" + str(i), 16, 128, step=16), return_sequences=True))
    mdl.add(tf.keras.layers.LSTM(hp.Int("nodes2", 16, 128, step=16)))

    # DNN Layers
    mdl.add(tf.keras.layers.Flatten())
    for i in range(hp.Int("dnn_layers", 1, 4, step=1)):
        mdl.add(tf.keras.layers.Dense(hp.Int("units_" + str(i), 32, 256, step=32), activation="relu"))

    # ------------------------------------------
    # Output
    # ------------------------------------------
    mdl.add(tf.keras.layers.Dense(1, activation='linear'))

    # Compile
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mae', metrics='mse')

    return mdl


# ------------------------------------------
# CNN
# ------------------------------------------
def tfMdloptiR(hp):
    # ------------------------------------------
    # Input
    # ------------------------------------------
    mdl = keras.Sequential()

    # ------------------------------------------
    # Mdl
    # ------------------------------------------
    # CNN Layers
    for i in range(hp.Int("cnn_layers", 1, 5, step=1)):
        mdl.add(tf.keras.layers.Conv1D(filters=hp.Int("filters_" + str(i), 8, 64, step=8),
                                       kernel_size=(hp.Int("kernel_size_0" + str(i), 2, 10, step=2)),
                                       activation='relu', padding="same", strides=1))

        if hp.Boolean("dropout_opt"):
            mdl.add(tf.keras.layers.Dropout(hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)))

        if hp.Boolean("batch_opt"):
            mdl.add(tf.keras.layers.BatchNormalization())

    # Pooling
    if hp.Boolean("pooling_opt"):
        mdl.add(tf.keras.layers.MaxPooling1D(pool_size=(hp.Int("pool_size", 2, 10, step=2)),
                                             strides=(hp.Int("strides", 2, 10, step=2)),
                                             padding='same'))

    # DNN Layers
    mdl.add(tf.keras.layers.Flatten())
    for i in range(hp.Int("dnn_layers", 1, 4, step=1)):
        mdl.add(tf.keras.layers.Dense(hp.Int("units_" + str(i), 64, 512, step=64), activation="relu"))

    # ------------------------------------------
    # Output
    # ------------------------------------------
    mdl.add(tf.keras.layers.Dense(4, activation='linear'))

    # Compile
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mae', metrics='mse')

    return mdl


# ==============================================================================
# Classification
# ==============================================================================
def tfMdloptiC(hp):
    # Input
    mdl = keras.Sequential()
    mdl.add(tf.keras.layers.Flatten())

    # Mdl
    hp_units = hp.Int('units', min_value=64, max_value=512, step=64)
    for i in range(hp.Int("dnn_layers", 2, 6, step=1)):
        mdl.add(keras.layers.Dense(units=hp_units, activation='relu'))
    mdl.add(keras.layers.Dense(1, activation='sigmoid'))

    # Learner
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 5e-2, 1e-2])
    mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='BinaryCrossentropy',
                metrics='accuracy')

    return mdl
