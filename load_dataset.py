# load_dataset.py
# Loads the MATLAB-generated dataset from channel_dataset.mat
#
# The file was saved with MATLAB's -v7.3 flag which uses HDF5 format.
# h5py reads 3D arrays with reversed dimension order (Fortran vs C ordering),
# so we transpose back to match the original MATLAB shapes:
#   X_data: (500, 100, 16)   -- 500 frames, 100 time steps, 16 features
#   Y_data: (500, 100, 128)  -- 500 frames, 100 time steps, 128 channel values

import h5py
import numpy as np
import config


def load_dataset(path=None):
    if path is None:
        path = config.DatasetPath

    print(f'Loading dataset from: {path}')

    with h5py.File(path, 'r') as f:
        # h5py gives (16, 100, 500) and (128, 100, 500) — need to reverse
        X_raw = np.array(f['X_data'])
        Y_raw = np.array(f['Y_data'])

    # transpose from (features, time, frames) to (frames, time, features)
    X = np.transpose(X_raw, (2, 1, 0))   # -> (500, 100, 16)
    Y = np.transpose(Y_raw, (2, 1, 0))   # -> (500, 100, 128)

    print(f'X_data shape : {X.shape}   (frames, time steps, input features)')
    print(f'Y_data shape : {Y.shape}   (frames, time steps, output values)')
    print(f'X mean abs   : {np.mean(np.abs(X)):.4f}')
    print(f'Y mean abs   : {np.mean(np.abs(Y)):.4f}')
    print(f'Any NaN in X : {np.any(np.isnan(X))}')
    print(f'Any NaN in Y : {np.any(np.isnan(Y))}')

    return X, Y


if __name__ == '__main__':
    X, Y = load_dataset()
    print('\nDataset loaded successfully.')
