# preprocess.py
# Normalises the dataset and splits into train / validation / test sets.
#
# Normalisation: z-score standardisation (zero mean, unit variance).
# We compute mean and std from the training set only, then apply the
# same values to validation and test to avoid data leakage.
#
# Split:
#   10% test  (held out completely, never touched during training)
#   90% used for k-fold cross-validation (train + validation)

import numpy as np
import config


def normalise(X_train, X_val, X_test, Y_train, Y_val, Y_test):
    # compute statistics from training set only
    X_mean = np.mean(X_train)
    X_std  = np.std(X_train)
    Y_mean = np.mean(Y_train)
    Y_std  = np.std(Y_train)

    # apply same transform to all splits
    X_train_n = (X_train - X_mean) / (X_std + 1e-8)
    X_val_n   = (X_val   - X_mean) / (X_std + 1e-8)
    X_test_n  = (X_test  - X_mean) / (X_std + 1e-8)

    Y_train_n = (Y_train - Y_mean) / (Y_std + 1e-8)
    Y_val_n   = (Y_val   - Y_mean) / (Y_std + 1e-8)
    Y_test_n  = (Y_test  - Y_mean) / (Y_std + 1e-8)

    # save stats so we can denormalise predictions later
    stats = {
        'X_mean': X_mean, 'X_std': X_std,
        'Y_mean': Y_mean, 'Y_std': Y_std
    }

    return X_train_n, X_val_n, X_test_n, Y_train_n, Y_val_n, Y_test_n, stats


def denormalise_Y(Y_norm, stats):
    # reverse the normalisation on predictions
    return Y_norm * stats['Y_std'] + stats['Y_mean']


def train_test_split(X, Y, test_frac=None, seed=None):
    if test_frac is None:
        test_frac = config.TestSplit
    if seed is None:
        seed = config.RandomSeed

    np.random.seed(seed)
    n_frames = X.shape[0]
    n_test   = int(n_frames * test_frac)

    # shuffle frame indices
    idx = np.random.permutation(n_frames)
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]

    X_trainval = X[train_idx]
    Y_trainval = Y[train_idx]
    X_test     = X[test_idx]
    Y_test     = Y[test_idx]

    print(f'Train+val frames : {X_trainval.shape[0]}')
    print(f'Test frames      : {X_test.shape[0]}')

    return X_trainval, Y_trainval, X_test, Y_test


if __name__ == '__main__':
    from load_dataset import load_dataset
    X, Y = load_dataset()
    X_tv, Y_tv, X_test, Y_test = train_test_split(X, Y)
    print('Split done.')
