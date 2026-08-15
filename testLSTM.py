# testLSTM.py
# Quick test script to verify the entire Phase 4+5 pipeline works correctly.
# Runs a short 2-epoch mini-train on 20 frames just to check everything
# connects (dataset loading, model building, training, saving).
# This does NOT produce final results — run train.py for full training.

import numpy as np
import os
import tensorflow as tf

import config
from load_dataset import load_dataset
from preprocess import train_test_split, normalise
from lstm_model import build_model

os.makedirs(os.path.dirname(config.ModelSavePath), exist_ok=True)
os.makedirs(config.ResultsSavePath, exist_ok=True)

print('=== LSTM Pipeline Test ===\n')

# ── Step 1: check TensorFlow ─────────────────────────────────────────────────
print('1. Environment check:')
print(f'   TensorFlow : {tf.__version__}')
print(f'   NumPy      : {np.__version__}')

# ── Step 2: load dataset ──────────────────────────────────────────────────────
print('\n2. Dataset loading:')
X, Y = load_dataset()

assert X.shape == (500, 100, 16),  f'X shape wrong: {X.shape}'
assert Y.shape == (500, 100, 128), f'Y shape wrong: {Y.shape}'
assert not np.any(np.isnan(X)), 'NaN in X'
assert not np.any(np.isnan(Y)), 'NaN in Y'
print('   [PASS] Dataset shapes and NaN check OK')

# ── Step 3: train/test split ──────────────────────────────────────────────────
print('\n3. Train/test split:')
X_tv, Y_tv, X_test, Y_test = train_test_split(X, Y)
assert X_tv.shape[0] + X_test.shape[0] == 500, 'Split sizes wrong'
print('   [PASS] Split sizes correct')

# ── Step 4: normalisation ─────────────────────────────────────────────────────
print('\n4. Normalisation:')
n_val = int(X_tv.shape[0] * 0.1)
X_tr, X_vl = X_tv[n_val:], X_tv[:n_val]
Y_tr, Y_vl = Y_tv[n_val:], Y_tv[:n_val]

X_tr_n, X_vl_n, X_te_n, Y_tr_n, Y_vl_n, Y_te_n, stats = normalise(
    X_tr, X_vl, X_test, Y_tr, Y_vl, Y_test
)

# after normalisation, training set should be roughly zero-mean, unit variance
assert abs(np.mean(X_tr_n)) < 0.1, 'X mean not near zero after normalisation'
assert abs(np.std(X_tr_n) - 1.0) < 0.1, 'X std not near 1 after normalisation'
print(f'   X_train mean  : {np.mean(X_tr_n):.4f}  (expected ~0)')
print(f'   X_train std   : {np.std(X_tr_n):.4f}   (expected ~1)')
print('   [PASS] Normalisation correct')

# ── Step 5: model build ───────────────────────────────────────────────────────
print('\n5. Model build:')
model = build_model()
model.summary()
total_params = model.count_params()
assert total_params > 0, 'Model has no parameters'
print(f'   Total parameters : {total_params:,}')
print('   [PASS] Model built OK')

# ── Step 6: mini training (2 epochs, 20 frames only) ─────────────────────────
print('\n6. Mini training (2 epochs, 20 frames — just checks it runs):')
X_mini = X_tr_n[:20]
Y_mini = Y_tr_n[:20]

history = model.fit(
    X_mini, Y_mini,
    validation_data=(X_vl_n[:5], Y_vl_n[:5]),
    epochs=2,
    batch_size=4,
    verbose=1
)

train_loss = history.history['loss']
val_loss   = history.history['val_loss']
assert len(train_loss) == 2, 'Expected 2 epochs'
assert train_loss[0] > 0, 'Loss should be positive'
print(f'   Epoch 1 loss : {train_loss[0]:.6f}')
print(f'   Epoch 2 loss : {train_loss[1]:.6f}')
print('   [PASS] Training loop runs correctly')

# ── Step 7: prediction and output shape ──────────────────────────────────────
print('\n7. Prediction check:')
Y_pred = model.predict(X_te_n[:5], verbose=0)
assert Y_pred.shape == (5, 100, 128), f'Prediction shape wrong: {Y_pred.shape}'
assert not np.any(np.isnan(Y_pred)), 'NaN in predictions'
print(f'   Prediction shape : {Y_pred.shape}  (expected (5, 100, 128))')
print('   [PASS] Predictions shape and NaN check OK')

# ── Step 8: save / reload model ──────────────────────────────────────────────
print('\n8. Model save / reload:')
test_path = os.path.join(os.path.dirname(config.ModelSavePath), 'test_model.keras')
model.save(test_path)
loaded = tf.keras.models.load_model(test_path)
Y_pred2 = loaded.predict(X_te_n[:5], verbose=0)
assert np.allclose(Y_pred, Y_pred2, atol=1e-5), 'Loaded model gives different predictions'
print(f'   Model saved and reloaded OK')
print('   [PASS] Save/reload works correctly')

print('\n=== All tests passed — pipeline is ready for full training ===')
print('\nTo train the full model, run:')
print('   python train.py')
