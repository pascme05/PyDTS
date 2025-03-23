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
import numpy as np
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
# Transformer Models
#######################################################################################################################
# ==============================================================================
# TRAN-1
# ==============================================================================
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def tfMdlTran(X_train, output, activation):
    inputs = tf.keras.layers.Input(shape=X_train.shape[1:])
    x = tf.keras.layers.Dense(32)(inputs)
    transformer_block = TransformerBlock(32, 2, 32)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(output, activation=activation)(x)  # Adjust output layer for your specific task
    mdl = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    mdl.set_weights(mdl.get_weights())

    return mdl


# ==============================================================================
# TRAN-2
# ==============================================================================
# Transformer Encoder Block
class TransformerBlock2(tf.keras.layers.Layer):
    def __init__(self, head_size=64, num_heads=4, ff_dim=64, dropout=0.1):
        super(TransformerBlock2, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(head_size)
        ])
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)  # Residual connection

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)  # Residual connection


# Updated Model Function
def tfMdlTran2(X_train, outputdim, activation):
    inputs = tf.keras.layers.Input(shape=X_train.shape[1:])

    # CNN Feature Extractor
    x = tf.keras.layers.Conv1D(filters=30, kernel_size=10, activation='relu', padding="same", strides=1)(inputs)
    x = tf.keras.layers.Conv1D(filters=30, kernel_size=8, activation='relu', padding="same", strides=1)(x)
    x = tf.keras.layers.Conv1D(filters=40, kernel_size=6, activation='relu', padding="same", strides=1)(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu', padding="same", strides=1)(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu', padding="same", strides=1)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4, strides=2, padding='same')(x)

    # LSTM for Temporal Dependencies
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(x)
    x = tf.keras.layers.Dense(64)(x)

    # Transformer Encoder (Fixed!)
    x = TransformerBlock2()(x)

    # Fully Connected Layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(outputdim, activation=activation)(x)

    mdl = tf.keras.models.Model(inputs=inputs, outputs=x)
    mdl.set_weights(mdl.get_weights())

    return mdl


#######################################################################################################################
# Denoising Models
#######################################################################################################################
# ==============================================================================
# DAE-1
# ==============================================================================
def tfMdlDAE(X_train, output, activation):
    # Encoder
    inputs = tf.keras.layers.Input(shape=X_train.shape[1:])
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    encoded = tf.keras.layers.Dense(32, activation='relu')(x)

    # Decoder
    x = tf.keras.layers.Dense(32 * output, activation='relu')(encoded)  # Map to a larger dense layer
    x = tf.keras.layers.Reshape((output, 32))(x)
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    decoded = tf.keras.layers.Dense(output, activation=activation)(x)  # Produce a 1D output

    # Autoencoder
    mdl = tf.keras.models.Model(inputs, decoded)
    mdl.set_weights(mdl.get_weights())

    return mdl


#######################################################################################################################
# Informer Models
#######################################################################################################################
# ==============================================================================
# INF-1
# ==============================================================================
class ProbSparseSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(ProbSparseSelfAttention, self).__init__()
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    def call(self, inputs, training):
        attn_output = self.multi_head_attention(inputs, inputs)
        return attn_output


class InformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(InformerBlock, self).__init__()
        self.att = ProbSparseSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def tfMdlINF(X_train, output, activation):
    inputs = tf.keras.layers.Input(shape=X_train.shape[1:])

    # Project input features to the embedding dimension
    x = tf.keras.layers.Dense(32)(inputs)

    informer_block = InformerBlock(32, 2, 32)
    x = informer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(output, activation=activation)(x)  # Adjust output layer for your specific task
    mdl = tf.keras.models.Model(inputs=inputs, outputs=outputs)

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
