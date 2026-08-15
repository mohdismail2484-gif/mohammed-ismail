# lstm_model.py
# Defines the LSTM-based channel estimator model.
#
# Architecture (per project plan):
#   Input  : (100, 16)  -- 100 OFDM symbols, 16 features (real+imag pilot estimates)
#   LSTM 1 : 128 units, return_sequences=True  -- learns temporal channel patterns
#   Dropout: 0.2                               -- reduces overfitting
#   LSTM 2 : 64 units,  return_sequences=True  -- refines temporal features
#   Dropout: 0.2
#   Dense  : 128 units  (linear)               -- predicts real+imag of all 64 subcarriers
#   Output : (100, 128) -- full channel estimate at every time step
#
# Loss     : MSE  (mean squared error)
# Optimiser: Adam (per project plan)

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import config


def build_model(lstm_units1=None, lstm_units2=None, dropout=None, lr=None):
    # use config defaults if not overridden (used during hyperparameter search)
    if lstm_units1 is None:
        lstm_units1 = config.LSTMUnits1
    if lstm_units2 is None:
        lstm_units2 = config.LSTMUnits2
    if dropout is None:
        dropout = config.DropoutRate
    if lr is None:
        lr = config.LearningRate

    model = models.Sequential([
        # input layer — shape is (time steps, features)
        layers.Input(shape=(config.TimeSteps, config.InputFeatures)),

        # first LSTM layer — learns how the channel evolves over 100 symbols
        layers.LSTM(lstm_units1, return_sequences=True),
        layers.Dropout(dropout),

        # second LSTM layer — extracts higher-level temporal patterns
        layers.LSTM(lstm_units2, return_sequences=True),
        layers.Dropout(dropout),

        # output dense layer — one prediction per time step, 128 values
        # (real + imag of 64-subcarrier channel)
        layers.Dense(config.OutputFeatures, activation='linear'),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )

    return model


def print_model_summary():
    model = build_model()
    model.summary()
    total = model.count_params()
    print(f'\nTotal trainable parameters: {total:,}')


if __name__ == '__main__':
    print_model_summary()
