% testDataset.m
% Quick test to verify the dataset generation pipeline
% before running the full 500-frame generation.
% Runs only 5 frames so it finishes in a few seconds.

clc; clear; close all;
addpath(genpath(pwd));

params = config();

fprintf('=== Dataset Pipeline Test (5 frames) ===\n\n');

Nfft      = params.Nfft;
NumPilots = params.NumPilots;
NumData   = params.NumDataSubcarriers;
NumSymbols = params.NumSymbols;
NumDataBits = NumData * NumSymbols * params.BitsPerSymbol;

%% Step 1 - check pilot insertion
fprintf('1. Pilot insertion:\n');
bits      = randi([0 1], NumDataBits, 1);
data_syms = qpskModulator(bits);
data_syms = reshape(data_syms, NumData, NumSymbols);
X_freq    = insertPilots(data_syms, params);

fprintf('   X_freq size            : %s\n', mat2str(size(X_freq)));
% verify pilot positions have correct value
pilot_vals = X_freq(params.PilotSubcarriers, 1);
if all(pilot_vals == params.PilotValue)
    fprintf('   [PASS] Pilot values correct at all pilot subcarriers\n');
else
    fprintf('   [FAIL] Pilot values wrong\n');
end

%% Step 2 - check receiver chain
fprintf('\n2. Receiver chain:\n');
x_ifft  = ofdmIFFT(X_freq);
x_time  = addCyclicPrefix(x_ifft, params);
fprintf('   Tx signal size         : %s\n', mat2str(size(x_time)));

H       = generateChannel(params);
X_freq_T = X_freq.';
Y_clean  = applyChannel(X_freq_T, H);
Y_noisy  = addAWGN(Y_clean, params.TrainingSNR);

fprintf('   Received signal size   : %s\n', mat2str(size(Y_noisy)));

%% Step 3 - pilot extraction
fprintf('\n3. Pilot extraction:\n');
H_est = extractPilots(Y_noisy, params);
fprintf('   H_est_pilots size      : %s\n', mat2str(size(H_est)));

% correlation between true channel at pilots and estimated
H_true_pilots = H(:, params.PilotSubcarriers);
corr_vals = zeros(1, NumPilots);
for p = 1:NumPilots
    c = corrcoef(abs(H_true_pilots(:,p)), abs(H_est(:,p)));
    corr_vals(p) = c(1,2);
end
fprintf('   Mean correlation (true vs estimated pilots): %.4f\n', mean(corr_vals));
fprintf('   (should be > 0.8 at %d dB SNR)\n', params.TrainingSNR);

%% Step 4 - mini dataset (5 frames)
fprintf('\n4. Mini dataset generation (5 frames):\n');
test_frames = 5;
X_test = zeros(test_frames, NumSymbols, NumPilots*2);
Y_test = zeros(test_frames, NumSymbols, Nfft*2);

for f = 1:test_frames
    bits      = randi([0 1], NumDataBits, 1);
    data_syms = qpskModulator(bits);
    data_syms = reshape(data_syms, NumData, NumSymbols);
    X_freq    = insertPilots(data_syms, params);
    H         = generateChannel(params);
    X_freq_T  = X_freq.';
    Y_clean   = applyChannel(X_freq_T, H);
    Y_noisy   = addAWGN(Y_clean, params.TrainingSNR);
    H_est_p   = extractPilots(Y_noisy, params);

    X_test(f, :, 1:NumPilots)     = real(H_est_p);
    X_test(f, :, NumPilots+1:end) = imag(H_est_p);
    Y_test(f, :, 1:Nfft)     = real(H);
    Y_test(f, :, Nfft+1:end) = imag(H);
end

fprintf('   X_test size : %s  (expected [5 x 100 x 16])\n', mat2str(size(X_test)));
fprintf('   Y_test size : %s  (expected [5 x 100 x 128])\n', mat2str(size(Y_test)));
fprintf('   Any NaN     : %d\n', any(isnan(X_test(:))) || any(isnan(Y_test(:))));

%% Plot - compare true channel vs pilot estimate for frame 1
figure('Name', 'Dataset Test - Channel vs Pilot Estimate');
subplot(2,1,1);
plot(abs(squeeze(Y_test(1, :, 1))), 'b-');   % true H real part at subcarrier 1
hold on;
plot(abs(squeeze(X_test(1, :, 1))), 'r--');  % estimated at pilot 1
legend('True |H| at subcarrier 1', 'Pilot estimate at pilot 1');
xlabel('OFDM Symbol');
ylabel('Magnitude');
title('True channel vs noisy pilot estimate over time');
grid on;

subplot(2,1,2);
% show true channel across all 64 subcarriers at symbol 1
plot(abs(squeeze(Y_test(1, 1, 1:Nfft))), 'b-');
hold on;
% overlay pilot positions
pilot_magnitudes = abs(squeeze(X_test(1, 1, 1:NumPilots)));
plot(params.PilotSubcarriers, pilot_magnitudes, 'ro', 'MarkerSize', 8);
legend('True |H| all subcarriers', 'Pilot estimates');
xlabel('Subcarrier index');
ylabel('|H|');
title('True channel vs pilot estimates across frequency (symbol 1)');
grid on;

fprintf('\n=== Test complete - check plots ===\n');
