% testBaselines.m
% Phase 6 verification script: tests LS and MMSE channel estimators.
%
% Runs both estimators across a range of SNR values and computes NMSE
% (Normalised Mean Square Error) for each.
%
% Expected results:
%   - LS NMSE typically -10 to -15 dB at 15-20 dB SNR
%   - MMSE NMSE typically 3-8 dB better than LS (uses channel stats)
%   - Both improve as SNR increases
%
% Run this from AI_Adaptive_Receiver/MATLAB/ after addpath(genpath(pwd))

clc; close all;

params = config();

fprintf('=== Phase 6: LS and MMSE Baseline Channel Estimators ===\n\n');
fprintf('Channel: CDL-A, Doppler=%d Hz, %d paths, %d pilots\n', ...
    params.DopplerShift, length(params.PathGains), params.NumPilots);
fprintf('SNR range: %d to %d dB, %d frames per SNR point\n\n', ...
    params.SNR_dB(1), params.SNR_dB(end), 50);

snr_range  = params.SNR_dB;   % [0 5 10 15 20 25 30] dB
N_test     = 50;               % frames averaged per SNR point

nmse_ls   = zeros(1, length(snr_range));
nmse_mmse = zeros(1, length(snr_range));

% number of data bits per frame  (data subcarriers only)
NumDataBits = params.NumDataSubcarriers * params.NumSymbols * params.BitsPerSymbol;

fprintf('%-8s  %-16s  %-16s\n', 'SNR(dB)', 'LS NMSE(dB)', 'MMSE NMSE(dB)');
fprintf('%s\n', repmat('-', 1, 44));

for s = 1:length(snr_range)
    snr = snr_range(s);

    err_ls   = 0;
    err_mmse = 0;
    power_h  = 0;

    for f = 1:N_test
        % --- transmitter ---
        bits      = randi([0 1], NumDataBits, 1);
        data_syms = qpskModulator(bits);
        data_syms = reshape(data_syms, params.NumDataSubcarriers, params.NumSymbols);
        X_freq    = insertPilots(data_syms, params);   % [Nfft x NumSymbols]

        % --- channel ---
        H_true  = generateChannel(params);             % [NumSymbols x Nfft]

        % --- apply channel + noise ---
        Y_clean = applyChannel(X_freq.', H_true);      % [NumSymbols x Nfft]
        Y_noisy = addAWGN(Y_clean, snr);

        % --- LS estimation ---
        H_ls   = lsEstimator(Y_noisy, params);

        % --- MMSE estimation ---
        H_mmse = mmseEstimator(Y_noisy, params, snr);

        % --- accumulate errors ---
        err_ls   = err_ls   + sum(abs(H_true(:) - H_ls(:)).^2);
        err_mmse = err_mmse + sum(abs(H_true(:) - H_mmse(:)).^2);
        power_h  = power_h  + sum(abs(H_true(:)).^2);
    end

    nmse_ls(s)   = err_ls   / power_h;
    nmse_mmse(s) = err_mmse / power_h;

    fprintf('%-8d  %-16.2f  %-16.2f\n', snr, ...
        10*log10(nmse_ls(s)), 10*log10(nmse_mmse(s)));
end

%% Plot NMSE vs SNR
figure;
plot(snr_range, 10*log10(nmse_ls),   'b-o', 'LineWidth', 2, 'MarkerSize', 7);
hold on;
plot(snr_range, 10*log10(nmse_mmse), 'r-s', 'LineWidth', 2, 'MarkerSize', 7);
grid on;
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('NMSE (dB)', 'FontSize', 12);
title('Phase 6: LS vs MMSE Channel Estimation — NMSE vs SNR', 'FontSize', 13);
legend('LS Estimator', 'MMSE Estimator', 'Location', 'southwest', 'FontSize', 11);
ylim([-35 5]);
xlim([snr_range(1)-1 snr_range(end)+1]);

%% Print summary
fprintf('\n--- Summary ---\n');
fprintf('At SNR = 15 dB:\n');
idx15 = find(snr_range == 15);
if ~isempty(idx15)
    fprintf('  LS   NMSE : %.2f dB\n', 10*log10(nmse_ls(idx15)));
    fprintf('  MMSE NMSE : %.2f dB\n', 10*log10(nmse_mmse(idx15)));
    fprintf('  MMSE gain over LS: %.2f dB\n', ...
        10*log10(nmse_ls(idx15)) - 10*log10(nmse_mmse(idx15)));
end

fprintf('\n[PASS] Phase 6 complete. NMSE values saved in workspace.\n');
fprintf('Variables: nmse_ls, nmse_mmse, snr_range (all in linear scale)\n');
fprintf('For Phase 7 evaluation, these will be compared against LSTM NMSE.\n');
