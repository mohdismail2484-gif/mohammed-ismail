# config.py
# Python-side configuration — mirrors MATLAB/config/config.m
# All parameters in one place so nothing has to be hardcoded elsewhere

import os

# project root is one level up from this file (Python/../)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# OFDM parameters
Nfft        = 64       # number of subcarriers
CPLength    = 16       # cyclic prefix length (samples)
NumSymbols  = 100      # OFDM symbols per frame
SampleRate  = 15.36e6  # sample rate in Hz

# Pilot parameters
PilotSpacing    = 8
NumPilots       = 8           # [1, 9, 17, 25, 33, 41, 49, 57]
NumDataSubcarriers = 56

# Dataset dimensions
NumFrames       = 500
InputFeatures   = NumPilots * 2      # 16  (real + imag of 8 pilot estimates)
OutputFeatures  = Nfft * 2           # 128 (real + imag of 64-subcarrier channel)
TimeSteps       = NumSymbols         # 100

# Channel parameters
DopplerShift    = 5000   # Hz
TrainingSNR     = 15     # dB

# Training parameters (per project plan Objective 2)
Epochs          = 100    # project plan says convergence within 100 epochs
BatchSize       = 32
LearningRate    = 1e-3   # Adam default
KFolds          = 5      # k-fold cross-validation
ValidationSplit = 0.1    # within each fold
TestSplit       = 0.1    # held-out test set (taken before k-fold)
RandomSeed      = 42

# LSTM architecture
LSTMUnits1      = 128    # first LSTM layer
LSTMUnits2      = 64     # second LSTM layer
DropoutRate     = 0.2

# Paths — all relative to project root, so scripts work regardless of where you run them from
DatasetPath     = os.path.join(_ROOT, 'MATLAB', 'dataset', 'channel_dataset.mat')
ModelSavePath   = os.path.join(_ROOT, 'models', 'lstm_receiver.keras')
ResultsSavePath = os.path.join(_ROOT, 'results', '')
