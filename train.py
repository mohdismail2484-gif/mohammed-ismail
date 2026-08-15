# train.py
# Main training script for the LSTM channel estimator.
#
# Process:
#   1. Load dataset from MATLAB
#   2. Hold out 10% as test set (never used during training)
#   3. Run 5-fold cross-validation on the remaining 90%
#      -- each fold trains for up to 100 epochs (early stopping)
#      -- records validation MSE for each fold
#   4. Train the final model on ALL training data using the mean fold results
#   5. Save the trained model to models/lstm_receiver.keras
#
# Per project plan: Adam optimiser, MSE loss, k-fold cross-validation,
# convergence within 100 epochs (Objective 2)

import numpy as np
import os
import json
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import config
from load_dataset import load_dataset
from preprocess import train_test_split, normalise, denormalise_Y
from lstm_model import build_model

# fix random seeds for reproducibility
np.random.seed(config.RandomSeed)
tf.random.set_seed(config.RandomSeed)

os.makedirs(os.path.dirname(config.ModelSavePath),  exist_ok=True)
os.makedirs(config.ResultsSavePath, exist_ok=True)


def run_kfold(X_tv, Y_tv):
    """Run k-fold cross-validation. Returns list of val MSE per fold."""

    print(f'\n=== {config.KFolds}-Fold Cross-Validation ===')
    kf = KFold(n_splits=config.KFolds, shuffle=True, random_state=config.RandomSeed)

    fold_val_mse  = []
    fold_val_mae  = []
    fold_histories = []

    fold_num = 1
    for train_idx, val_idx in kf.split(X_tv):
        print(f'\n--- Fold {fold_num}/{config.KFolds} ---')

        # split this fold
        X_tr = X_tv[train_idx]
        Y_tr = Y_tv[train_idx]
        X_vl = X_tv[val_idx]
        Y_vl = Y_tv[val_idx]

        # normalise using training portion only
        X_tr_n, X_vl_n, _, Y_tr_n, Y_vl_n, _, stats = normalise(
            X_tr, X_vl, X_vl,   # pass val twice as placeholder for test
            Y_tr, Y_vl, Y_vl
        )

        # fresh model for each fold
        model = build_model()

        callbacks = [
            # stop early if val_loss stops improving
            EarlyStopping(monitor='val_loss', patience=10,
                          restore_best_weights=True, verbose=0),
            # reduce lr if plateau (helps convergence)
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=1e-6, verbose=0)
        ]

        history = model.fit(
            X_tr_n, Y_tr_n,
            validation_data=(X_vl_n, Y_vl_n),
            epochs=config.Epochs,
            batch_size=config.BatchSize,
            callbacks=callbacks,
            verbose=1
        )

        val_mse = min(history.history['val_loss'])
        val_mae = min(history.history['val_mae'])
        fold_val_mse.append(val_mse)
        fold_val_mae.append(val_mae)
        fold_histories.append({
            'loss':     history.history['loss'],
            'val_loss': history.history['val_loss']
        })

        epochs_run = len(history.history['loss'])
        print(f'  Fold {fold_num} done — epochs: {epochs_run}, '
              f'best val MSE: {val_mse:.6f}, val MAE: {val_mae:.6f}')

        fold_num += 1

    print(f'\n--- K-Fold Summary ---')
    for i, mse in enumerate(fold_val_mse):
        print(f'  Fold {i+1}: val MSE = {mse:.6f}')
    print(f'  Mean val MSE : {np.mean(fold_val_mse):.6f}')
    print(f'  Std  val MSE : {np.std(fold_val_mse):.6f}')

    return fold_val_mse, fold_val_mae, fold_histories


def train_final_model(X_tv, Y_tv, X_test, Y_test):
    """Train final model on all train+val data, evaluate on held-out test set."""

    print('\n=== Training Final Model on Full Training Set ===')

    # normalise using full train+val set
    # split off a small val portion for early stopping
    n = X_tv.shape[0]
    n_val = int(n * 0.1)
    X_tr = X_tv[n_val:]
    Y_tr = Y_tv[n_val:]
    X_vl = X_tv[:n_val]
    Y_vl = Y_tv[:n_val]

    X_tr_n, X_vl_n, X_test_n, Y_tr_n, Y_vl_n, Y_test_n, stats = normalise(
        X_tr, X_vl, X_test,
        Y_tr, Y_vl, Y_test
    )

    # save normalisation stats so we can use them during inference
    np.save('models/norm_stats.npy', stats)

    model = build_model()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1)
    ]

    print(f'Training for up to {config.Epochs} epochs...')
    history = model.fit(
        X_tr_n, Y_tr_n,
        validation_data=(X_vl_n, Y_vl_n),
        epochs=config.Epochs,
        batch_size=config.BatchSize,
        callbacks=callbacks,
        verbose=1
    )

    # evaluate on held-out test set
    print('\n=== Test Set Evaluation ===')
    test_mse, test_mae = model.evaluate(X_test_n, Y_test_n, verbose=0)
    print(f'Test MSE : {test_mse:.6f}')
    print(f'Test MAE : {test_mae:.6f}')

    # compute NMSE on test set (normalised mean square error)
    Y_pred_n = model.predict(X_test_n, verbose=0)
    Y_pred   = denormalise_Y(Y_pred_n, stats)
    Y_true   = denormalise_Y(Y_test_n, stats)
    nmse = np.mean(np.abs(Y_true - Y_pred)**2) / np.mean(np.abs(Y_true)**2)
    nmse_db  = 10 * np.log10(nmse)
    print(f'Test NMSE: {nmse:.6f}  ({nmse_db:.2f} dB)')

    # save model
    model.save(config.ModelSavePath)
    print(f'\nModel saved to: {config.ModelSavePath}')

    # save training history
    hist = {
        'loss':     history.history['loss'],
        'val_loss': history.history['val_loss'],
        'test_mse': float(test_mse),
        'test_mae': float(test_mae),
        'test_nmse_db': float(nmse_db)
    }
    with open(os.path.join(config.ResultsSavePath, 'training_history.json'), 'w') as fout:
        json.dump(hist, fout, indent=2)
    print('Training history saved to: results/training_history.json')

    return model, history, stats, nmse_db


def main():
    print('=== LSTM Channel Estimator Training ===')
    print(f'TensorFlow version : {tf.__version__}')
    print(f'Epochs limit       : {config.Epochs}')
    print(f'K-folds            : {config.KFolds}')
    print(f'Batch size         : {config.BatchSize}')
    print(f'Learning rate      : {config.LearningRate}')
    print()

    # step 1 — load dataset
    X, Y = load_dataset()

    # step 2 — hold out test set
    X_tv, Y_tv, X_test, Y_test = train_test_split(X, Y)

    # step 3 — k-fold cross-validation
    fold_val_mse, fold_val_mae, fold_histories = run_kfold(X_tv, Y_tv)

    # step 4 — train final model on all training data
    model, history, stats, nmse_db = train_final_model(X_tv, Y_tv, X_test, Y_test)

    print('\n=== Training Complete ===')
    print(f'K-fold mean val MSE : {np.mean(fold_val_mse):.6f}')
    print(f'Final test NMSE     : {nmse_db:.2f} dB')
    print(f'Model saved to      : {config.ModelSavePath}')


if __name__ == '__main__':
    main()
