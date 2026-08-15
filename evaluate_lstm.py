# evaluate_lstm.py
# Phase 7: LSTM performance evaluation across multiple SNR values.
#
# Loads the trained LSTM model and the held-out test frames from the
# MATLAB dataset. For each SNR level, it adds noise to the pilot channel
# observations, runs them through the LSTM, and records NMSE.
#
# Also saves the complex channel estimates so MATLAB can compute BER
# (MATLAB handles the equalization and demodulation steps).
#
# How to run:
#   source venv/bin/activate
#   cd Python
#   python evaluate_lstm.py
#
# Output: results/lstm_eval_results.mat
#   Then run MATLAB/evaluation/runEvaluation.m to get full BER+NMSE plots.

import numpy as np
import os
from scipy.io import savemat
import tensorflow as tf

import config
from load_dataset import load_dataset
from preprocess import train_test_split


def main():
    print('=== LSTM Evaluation — Phase 7 ===')
    print(f'TensorFlow version : {tf.__version__}')
    print()

    # ---- load dataset (same as training) ----
    X, Y = load_dataset()

    # reproduce the same train/test split that was used in train.py
    # using the same seed so we get the same 50 test frames
    np.random.seed(config.RandomSeed)
    _, _, X_test, Y_test = train_test_split(X, Y)
    N_test_frames = Y_test.shape[0]
    print(f'Test frames: {N_test_frames}')

    # ---- load saved model ----
    print(f'\nLoading model: {config.ModelSavePath}')
    model = tf.keras.models.load_model(config.ModelSavePath)
    print(f'Model loaded — parameters: {model.count_params():,}')

    # ---- get normalisation stats ----
    # train.py saves norm_stats.npy with a relative path from the Python/ folder,
    # so it ends up at Python/models/norm_stats.npy, not project_root/models/.
    # Try both locations; if neither exists, recompute from the training split.
    norm_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'norm_stats.npy'),
        os.path.join(os.path.dirname(config.ModelSavePath), 'norm_stats.npy'),
    ]
    stats = None
    for p in norm_candidates:
        if os.path.exists(p):
            stats = np.load(p, allow_pickle=True).item()
            print(f'Norm stats loaded from: {p}')
            break

    if stats is None:
        # recompute from training data (same result as what train.py saved)
        print('norm_stats.npy not found — recomputing from training split...')
        X_tv, Y_tv, _, _ = train_test_split(X, Y)
        # match the inner split used in train_final_model: skip first 10% as val
        n_val  = int(X_tv.shape[0] * 0.1)
        X_tr   = X_tv[n_val:]
        Y_tr   = Y_tv[n_val:]
        stats = {
            'X_mean': float(np.mean(X_tr)),
            'X_std':  float(np.std(X_tr)),
            'Y_mean': float(np.mean(Y_tr)),
            'Y_std':  float(np.std(Y_tr)),
        }
        print(f'  X_mean={stats["X_mean"]:.5f}  X_std={stats["X_std"]:.5f}')
        print(f'  Y_mean={stats["Y_mean"]:.5f}  Y_std={stats["Y_std"]:.5f}')

    X_mean = stats['X_mean']
    X_std  = stats['X_std']
    Y_mean = stats['Y_mean']
    Y_std  = stats['Y_std']

    # ---- reconstruct true complex channel from Y_test ----
    # Y_test shape: (N_frames, 100, 128)
    # first 64 features = real(H), last 64 = imag(H)
    H_true_complex = Y_test[:, :, :64] + 1j * Y_test[:, :, 64:]   # (50, 100, 64)

    # pilot subcarrier indices — 0-based (MATLAB uses 1-based [1,9,17,...,57])
    pilot_idx = np.arange(0, 64, 8)   # [0, 8, 16, 24, 32, 40, 48, 56]

    # known pilot value (must match config.m: PilotValue = 1 + 1j)
    pilot_value = 1.0 + 1.0j

    # SNR range to evaluate
    snr_range_dB = np.arange(0, 35, 5, dtype=float)   # [0, 5, 10, 15, 20, 25, 30]
    N_snr = len(snr_range_dB)

    # preallocate result arrays
    nmse_lstm_dB  = np.zeros(N_snr)
    H_est_all_re  = np.zeros((N_snr, N_test_frames, 100, 64), dtype=np.float32)
    H_est_all_im  = np.zeros((N_snr, N_test_frames, 100, 64), dtype=np.float32)
    H_true_all_re = np.tile(np.real(H_true_complex).astype(np.float32),
                            (N_snr, 1, 1, 1))   # same true channel at all SNRs
    H_true_all_im = np.tile(np.imag(H_true_complex).astype(np.float32),
                            (N_snr, 1, 1, 1))

    print(f'\nSNR range: {snr_range_dB} dB')
    print(f'Test frames per SNR: {N_test_frames}\n')
    print(f'{"SNR (dB)":>10}  {"NMSE (dB)":>12}')
    print('-' * 25)

    for si, snr_dB in enumerate(snr_range_dB):
        snr_lin = 10.0 ** (snr_dB / 10.0)

        # noise variance for pilot-divided channel estimate
        # H_ls_pilot = Y_pilot / PilotValue = H_true_pilot + noise/PilotValue
        # noise/PilotValue variance = sigma2 / |PilotValue|^2 = 1/(snr * |1+j|^2) = 1/(2*snr)
        sigma2_eff = 1.0 / (snr_lin * np.abs(pilot_value)**2)

        # extract true channel at pilot positions: (N_frames, 100, 8)
        H_true_pilots = H_true_complex[:, :, pilot_idx]

        # add complex Gaussian noise to simulate noisy pilot observations
        noise_std = np.sqrt(sigma2_eff / 2.0)
        noise = noise_std * (
            np.random.randn(*H_true_pilots.shape) +
            1j * np.random.randn(*H_true_pilots.shape)
        )
        H_noisy_pilots = H_true_pilots + noise   # (N_frames, 100, 8)

        # format as LSTM input: concatenate real and imag along last axis
        # matches how MATLAB built X_data in generateDataset.m:
        #   X_data(:,:,1:8) = real(H_est_pilots)
        #   X_data(:,:,9:16) = imag(H_est_pilots)
        X_input = np.concatenate([
            np.real(H_noisy_pilots),
            np.imag(H_noisy_pilots)
        ], axis=-1)   # (N_frames, 100, 16)

        # normalise using training set statistics
        X_input_n = (X_input - X_mean) / (X_std + 1e-8)

        # run LSTM inference
        Y_pred_n = model.predict(X_input_n, batch_size=config.BatchSize, verbose=0)

        # denormalise predictions
        Y_pred = Y_pred_n * (Y_std + 1e-8) + Y_mean   # (N_frames, 100, 128)

        # reconstruct complex channel estimate
        H_est_complex = Y_pred[:, :, :64] + 1j * Y_pred[:, :, 64:]   # (N_frames, 100, 64)

        # store real and imaginary parts separately (easier for MATLAB to load)
        H_est_all_re[si] = np.real(H_est_complex).astype(np.float32)
        H_est_all_im[si] = np.imag(H_est_complex).astype(np.float32)

        # compute NMSE
        err   = np.mean(np.abs(H_true_complex - H_est_complex) ** 2)
        power = np.mean(np.abs(H_true_complex) ** 2)
        nmse  = err / (power + 1e-12)
        nmse_dB_val = 10.0 * np.log10(nmse)
        nmse_lstm_dB[si] = nmse_dB_val

        print(f'{snr_dB:>10.0f}  {nmse_dB_val:>12.2f}')

    print()
    idx15 = np.where(snr_range_dB == 15)[0]
    if len(idx15) > 0:
        print(f'NMSE at 15 dB SNR : {nmse_lstm_dB[idx15[0]]:.2f} dB  (reference: -22.04 dB)')

    # ---- save results ----
    os.makedirs(config.ResultsSavePath, exist_ok=True)
    save_path = os.path.join(config.ResultsSavePath, 'lstm_eval_results.mat')

    # save real/imaginary parts separately to avoid any MATLAB loading issues
    # runEvaluation.m reconstructs: H_est = H_est_real + 1j*H_est_imag
    savemat(save_path, {
        'snr_range_lstm': snr_range_dB,      # [1 x 7] SNR values
        'nmse_lstm_dB':   nmse_lstm_dB,      # [1 x 7] NMSE at each SNR
        'H_est_real':     H_est_all_re,      # [7 x N_frames x 100 x 64] real part
        'H_est_imag':     H_est_all_im,      # [7 x N_frames x 100 x 64] imag part
        'H_true_real':    H_true_all_re,     # [7 x N_frames x 100 x 64] real part of H_true
        'H_true_imag':    H_true_all_im,     # [7 x N_frames x 100 x 64] imag part of H_true
    })

    print(f'\nResults saved to : {save_path}')
    print(f'Variables saved  :')
    print(f'  snr_range_lstm : {snr_range_dB}')
    print(f'  nmse_lstm_dB   : {np.round(nmse_lstm_dB, 2)}')
    print(f'  H_est_real     : {H_est_all_re.shape}  (float32)')
    print(f'  H_est_imag     : {H_est_all_im.shape}  (float32)')
    print(f'  H_true_real    : {H_true_all_re.shape}')
    print(f'  H_true_imag    : {H_true_all_im.shape}')
    print()
    print('[PASS] evaluate_lstm.py complete.')
    print('       Now run MATLAB/evaluation/runEvaluation.m for BER+NMSE plots.')


if __name__ == '__main__':
    main()
