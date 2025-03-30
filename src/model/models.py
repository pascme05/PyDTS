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
import tensorflow_probability as tfp
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
# Time/Frequency/Statistical (TFS) Model
#######################################################################################################################
# ==============================================================================
# TFS-1
# ==============================================================================
def compute_fft(x):
    x = tf.signal.rfft(x)
    x = tf.abs(x)
    return x


def compute_stats(x):
    # Calculate basic statistical features
    mean = tf.reduce_mean(x, axis=1, keepdims=True)
    min_val = tf.reduce_min(x, axis=1, keepdims=True)
    max_val = tf.reduce_max(x, axis=1, keepdims=True)
    std = tf.math.reduce_std(x, axis=1, keepdims=True)
    median = tfp.stats.percentile(x, 50.0, axis=1, keepdims=True)
    range_val = max_val - min_val
    iqr = tfp.stats.percentile(x, 75.0, axis=1, keepdims=True) - tfp.stats.percentile(x, 25.0, axis=1, keepdims=True)

    # Ensure all features have the same dimensions for concatenation
    stats = tf.concat([mean, min_val, max_val, std, median, range_val, iqr], axis=1)
    return stats


# Multi-Head Attention Fusion Layer
def attention_fusion(inputs, num_heads=4, embed_dim=128):
    inputs = tf.keras.layers.Reshape((1, inputs.shape[-1]))(inputs)  # Ensure 3D shape
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    return attn_output


class DynamicRoutingLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(DynamicRoutingLayer, self).__init__()
        self.output_dim = int(output_dim)  # Ensure it's a single integer
        self.gate = tf.keras.layers.Dense(3, activation="softmax")  # Learnable feature weights

        # Project all inputs to the same dimension
        self.proj_time = tf.keras.layers.Dense(self.output_dim)
        self.proj_freq = tf.keras.layers.Dense(self.output_dim)
        self.proj_stat = tf.keras.layers.Dense(self.output_dim)

    def call(self, inputs):
        x_time, x_freq, x_stat = inputs

        # Ensure all inputs have the same shape
        x_time = self.proj_time(x_time)
        x_freq = self.proj_freq(x_freq)
        x_stat = self.proj_stat(x_stat)

        # Compute soft attention weights
        gates = self.gate(tf.concat([x_time, x_freq, x_stat], axis=-1))

        # Weighted sum of the inputs
        return gates[:, 0:1] * x_time + gates[:, 1:2] * x_freq + gates[:, 2:3] * x_stat


class CapsuleFeatureFusion(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_capsules=3):
        super(CapsuleFeatureFusion, self).__init__()
        self.output_dim = int(output_dim)
        self.num_capsules = num_capsules

        # Linear projection for each feature type
        self.proj_time = tf.keras.layers.Dense(self.output_dim)
        self.proj_freq = tf.keras.layers.Dense(self.output_dim)
        self.proj_stat = tf.keras.layers.Dense(self.output_dim)

        # Dynamic routing weights (learnable)
        self.routing_weights = tf.keras.layers.Dense(num_capsules, activation="softmax")

    def call(self, inputs):
        x_time, x_freq, x_stat = inputs

        # Project features to the same dimension
        x_time = self.proj_time(x_time)
        x_freq = self.proj_freq(x_freq)
        x_stat = self.proj_stat(x_stat)

        # Concatenate and compute capsule routing weights
        combined = tf.concat([x_time, x_freq, x_stat], axis=-1)
        routing_scores = self.routing_weights(combined)

        # Weighted sum of inputs with capsule-like behavior
        return routing_scores[:, 0:1] * x_time + routing_scores[:, 1:2] * x_freq + routing_scores[:, 2:3] * x_stat


class GraphFeatureFusion(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(GraphFeatureFusion, self).__init__()
        self.output_dim = int(output_dim)

        # Project each input to the same size
        self.proj_time = tf.keras.layers.Dense(self.output_dim)
        self.proj_freq = tf.keras.layers.Dense(self.output_dim)
        self.proj_stat = tf.keras.layers.Dense(self.output_dim)

        # Graph weights (learnable adjacency matrix)
        self.adjacency = self.add_weight(
            shape=(3, 3), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        x_time, x_freq, x_stat = inputs

        # Project features
        x_time = self.proj_time(x_time)
        x_freq = self.proj_freq(x_freq)
        x_stat = self.proj_stat(x_stat)

        # Stack features into a graph-like structure
        node_features = tf.stack([x_time, x_freq, x_stat], axis=1)

        # Graph convolution: A * X
        graph_out = tf.einsum("ij,bjk->bik", self.adjacency, node_features)

        # Flatten the output
        return tf.reshape(graph_out, (tf.shape(graph_out)[0], -1))


class TransformerFeatureFusion(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_heads=4):
        super(TransformerFeatureFusion, self).__init__()
        self.output_dim = int(output_dim)
        self.num_heads = num_heads

        # Project each feature type to the same embedding space
        self.proj_time = tf.keras.layers.Dense(self.output_dim)
        self.proj_freq = tf.keras.layers.Dense(self.output_dim)
        self.proj_stat = tf.keras.layers.Dense(self.output_dim)

        # Multi-head attention for fusion
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=self.output_dim
        )

    def call(self, inputs):
        x_time, x_freq, x_stat = inputs

        # Project features
        x_time = self.proj_time(x_time)
        x_freq = self.proj_freq(x_freq)
        x_stat = self.proj_stat(x_stat)

        # Stack features
        feature_stack = tf.stack([x_time, x_freq, x_stat], axis=1)

        # Apply self-attention
        attn_output = self.attention(feature_stack, feature_stack)

        # Flatten the output
        return tf.reshape(attn_output, (tf.shape(attn_output)[0], -1))


class PolynomialFeatureFusion(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(PolynomialFeatureFusion, self).__init__()
        self.output_dim = output_dim

        # Project all features to the same shape
        self.proj_time = tf.keras.layers.Dense(output_dim)
        self.proj_freq = tf.keras.layers.Dense(output_dim)
        self.proj_stat = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        x_time, x_freq, x_stat = inputs

        # Transform to common dimensionality
        x_time = self.proj_time(x_time)
        x_freq = self.proj_freq(x_freq)
        x_stat = self.proj_stat(x_stat)

        # Raw concatenation
        concat_features = tf.concat([x_time, x_freq, x_stat], axis=-1)

        # Interaction terms (pairwise multiplication)
        interaction_1 = x_time * x_freq
        interaction_2 = x_freq * x_stat
        interaction_3 = x_stat * x_time

        # Concatenate all (raw + interactions)
        return tf.concat([concat_features, interaction_1, interaction_2, interaction_3], axis=-1)


def tfMdlTFS3(X_train, outputdim, activation, batch_size):
    # Specify the batch size in batch_input_shape
    inputs = tf.keras.layers.Input(batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]))

    # Time-domain input (CNN + Stateful LSTM)
    x_time = tf.keras.layers.Conv1D(filters=32, kernel_size=6, activation='relu', padding='same')(inputs)
    x_time = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x_time)
    x_time = tf.keras.layers.LSTM(64, return_sequences=False, stateful=True)(x_time)
    x_time = tf.keras.layers.Flatten()(x_time)

    # Compute FFT inside the model
    x_fft = tf.keras.layers.Lambda(compute_fft, name='fft_layer')(inputs)
    x_fft = tf.keras.layers.Conv1D(filters=32, kernel_size=6, activation='relu', padding='same')(x_fft)
    x_fft = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x_fft)
    x_fft = tf.keras.layers.Flatten()(x_fft)

    # Compute statistical features inside the model
    x_stat = tf.keras.layers.Lambda(compute_stats, name='stats_layer')(inputs)
    x_stat = tf.keras.layers.Flatten()(x_stat)
    x_stat = tf.keras.layers.Dense(64, activation='relu')(x_stat)

    # Merging all inputs
    merged = PolynomialFeatureFusion(128)([x_time, x_fft, x_stat])

    # Output
    x = tf.keras.layers.Dense(128, activation='relu')(merged)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(outputdim, activation=activation, name='output')(x)

    mdl = tf.keras.models.Model(inputs=inputs, outputs=x)
    mdl.set_weights(mdl.get_weights())

    return mdl


# ==============================================================================
# TFS-2
# ==============================================================================
class AdvancedFeatureFusion(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(AdvancedFeatureFusion, self).__init__()
        self.output_dim = output_dim

        # Projection layers with explicit initialization
        self.proj_time = tf.keras.layers.Dense(
            output_dim,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
        )
        self.proj_freq = tf.keras.layers.Dense(
            output_dim,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
        )
        self.proj_stat = tf.keras.layers.Dense(
            output_dim,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
        )

        # Cross-feature attention
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=output_dim // 4,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
        )

        # Polynomial interaction terms
        self.interaction_dense = tf.keras.layers.Dense(
            output_dim,
            activation='relu',
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
        )

        # Feature gating
        self.gate = tf.keras.layers.Dense(
            3,
            activation='softmax',
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
        )

    def call(self, inputs):
        x_time, x_freq, x_stat = inputs

        # Project to common space
        p_time = self.proj_time(x_time)
        p_freq = self.proj_freq(x_freq)
        p_stat = self.proj_stat(x_stat)

        # Stack for attention (batch, 3 features, dim)
        features = tf.stack([p_time, p_freq, p_stat], axis=1)

        # Cross-feature attention
        attn_features = self.attention(features, features)

        # Polynomial interactions (Hadamard products)
        interactions = tf.concat([
            p_time * p_freq,
            p_freq * p_stat,
            p_stat * p_time,
            p_time * p_freq * p_stat
        ], axis=-1)
        interactions = self.interaction_dense(interactions)

        # Feature gating
        gate_weights = self.gate(tf.reduce_mean(features, axis=-1))  # (batch, 3)
        gated_features = gate_weights[:, 0:1] * p_time + \
                         gate_weights[:, 1:2] * p_freq + \
                         gate_weights[:, 2:3] * p_stat

        # Combine all features
        combined = tf.concat([
            tf.reduce_mean(attn_features, axis=1),  # Attended features
            interactions,  # Nonlinear interactions
            gated_features  # Weighted combination
        ], axis=-1)

        return combined


def tfMdlTFS2(X_train, outputdim, activation, batch_size):
    inputs = tf.keras.layers.Input(batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]))

    # Time domain processing with proper initialization
    x_time = tf.keras.layers.Conv1D(
        32, 6, activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
    )(inputs)
    x_time = tf.keras.layers.BatchNormalization()(x_time)
    x_time = tf.keras.layers.Conv1D(
        64, 3, activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
    )(x_time)

    # LSTM with proper initialization
    x_time = tf.keras.layers.LSTM(
        64,
        return_sequences=False,
        stateful=True,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in'),
        recurrent_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
    )(x_time)
    x_time = tf.keras.layers.Flatten()(x_time)

    # Frequency domain processing
    x_fft = tf.keras.layers.Lambda(compute_fft)(inputs)
    x_fft = tf.keras.layers.Conv1D(
        32, 6, activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
    )(x_fft)
    x_fft = tf.keras.layers.BatchNormalization()(x_fft)
    x_fft = tf.keras.layers.Conv1D(
        64, 3, activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
    )(x_fft)
    x_fft = tf.keras.layers.Flatten()(x_fft)

    # Statistical processing
    x_stat = tf.keras.layers.Lambda(compute_stats)(inputs)
    x_stat = tf.keras.layers.Flatten()(x_stat)
    x_stat = tf.keras.layers.Dense(
        64, activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
    )(x_stat)

    # Enhanced feature fusion
    merged = AdvancedFeatureFusion(128)([x_time, x_fft, x_stat])

    # Output with skip connection
    x = tf.keras.layers.Dense(
        256, activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
    )(merged)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(
        128, activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
    )(x)

    # Final output
    output = tf.keras.layers.Dense(
        outputdim, activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
    )(x)

    mdl = tf.keras.models.Model(inputs=inputs, outputs=output)
    return mdl


# ==============================================================================
# TFS-3
# ==============================================================================
class StableODEBlock(tf.keras.layers.Layer):
    def __init__(self, units, name=None):
        super(StableODEBlock, self).__init__(name=name)
        self.units = units
        self.proj_in = tf.keras.layers.Dense(units, name=f'{name}_proj_in')
        self.dense1 = tf.keras.layers.Dense(units, activation='tanh', name=f'{name}_dense1')
        self.dense2 = tf.keras.layers.Dense(units, name=f'{name}_dense2')

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]

        # Project input to match output dimension
        x_proj = self.proj_in(x)
        x_flat = tf.reshape(x_proj, [-1, self.units])

        def ode_fn(t, y):
            return self.dense2(self.dense1(y))

        # Solve ODE with explicit naming
        results = tfp.math.ode.DormandPrince().solve(
            ode_fn,
            initial_time=0.0,
            initial_state=x_flat,
            solution_times=[1.0]
        )

        # Reshape back to original dimensions
        return tf.reshape(results.states[-1], [batch_size, seq_length, self.units])


class EfficientFourierFeatures(tf.keras.layers.Layer):
    def __init__(self, output_dim, name=None):
        super(EfficientFourierFeatures, self).__init__(name=name)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.omega = self.add_weight(
            shape=(input_shape[-1], self.output_dim // 2),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=False,
            name=f'{self.name}_omega'
        )

    def call(self, x):
        x_proj = tf.matmul(x, self.omega)
        return tf.concat([tf.sin(x_proj), tf.cos(x_proj)], axis=-1)


def tfMdlTFS(X_train, outputdim, activation, batch_size):
    inputs = tf.keras.layers.Input(
        batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]),
        name='model_input'
    )

    # Time Domain Path
    x_time = tf.keras.layers.Conv1D(
        32, 6, activation='relu', padding='same', name='time_conv1'
    )(inputs)
    x_time = tf.keras.layers.BatchNormalization(name='time_bn1')(x_time)
    x_time = StableODEBlock(64, name='time_ode')(x_time)
    x_time = tf.keras.layers.Flatten(name='time_flatten')(x_time)

    # Frequency Domain Path
    x_freq = EfficientFourierFeatures(64, name='freq_fft')(inputs)
    x_freq = tf.keras.layers.LayerNormalization(name='freq_ln')(x_freq)
    x_freq = tf.keras.layers.Dense(64, activation='relu', name='freq_dense')(x_freq)
    x_freq = tf.keras.layers.Flatten(name='freq_flatten')(x_freq)

    # Statistical Path
    x_stat = tf.keras.layers.Lambda(compute_stats, name='stats_layer')(inputs)
    x_stat = tf.keras.layers.Dense(64, activation='relu', name='stat_dense')(x_stat)
    x_stat = tf.keras.layers.Flatten(name='stat_flatten')(x_stat)

    # Dynamic dimension adjustment with explicit naming
    min_dim = min(x_time.shape[-1], x_freq.shape[-1], x_stat.shape[-1])
    if x_time.shape[-1] != min_dim:
        x_time = tf.keras.layers.Dense(min_dim, name='time_dim_adj')(x_time)
    if x_freq.shape[-1] != min_dim:
        x_freq = tf.keras.layers.Dense(min_dim, name='freq_dim_adj')(x_freq)
    if x_stat.shape[-1] != min_dim:
        x_stat = tf.keras.layers.Dense(min_dim, name='stat_dim_adj')(x_stat)

    merged = tf.keras.layers.Concatenate(name='feature_merge')([x_time, x_freq, x_stat])

    # Output layers with explicit naming
    x = tf.keras.layers.Dense(128, activation='relu', name='dense1')(merged)
    x = tf.keras.layers.Dropout(0.2, name='dropout1')(x)
    output = tf.keras.layers.Dense(outputdim, activation=activation, name='output')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=output, name='TFS_Model')
    return model


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
