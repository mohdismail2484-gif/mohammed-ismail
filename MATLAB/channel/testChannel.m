% testChannel.m
% Verifies the channel model output is physically correct.
%
% With fd = 5000 Hz (as required by project plan Objective 1):
%   - Symbol duration Tsym = 80/15.36e6 = 5.208 us
%   - Coherence time Tc = 1/(4*fd) = 1/(4*5000) = 0.05 ms = ~10 symbols
%   - Over 100 symbols (0.52 ms), we expect roughly 5 complete fading cycles
%
% So the time plot should show MULTIPLE fast fading cycles - NOT one smooth
% curve. This is the correct high-Doppler 6G behaviour.

clc; clear; close all;
addpath(genpath(pwd));

params = config();

fprintf('=== Channel Model Verification ===\n');
fprintf('Channel type  : %s\n',   params.ChannelType);
fprintf('Doppler       : %d Hz\n', params.DopplerShift);
fprintf('Num paths     : %d\n',    length(params.PathDelaysSamples));
fprintf('Path delays   : [%s] samples\n', num2str(params.PathDelaysSamples));
fprintf('Path gains    : [%s] dB\n\n',    num2str(params.PathGains));

%% Single frame check
H_single = generateChannel(params);

fprintf('Single frame:\n');
fprintf('  H size        : %d x %d  (expected 100 x 64)\n', size(H_single,1), size(H_single,2));
fprintf('  Mean |H|^2    : %.4f  (single frame varies a lot - normal for fast fading)\n\n', ...
        mean(abs(H_single(:)).^2));

%% Ensemble mean over many frames to verify power normalisation
fprintf('Ensemble average (200 frames):\n');
num_test_frames = 200;
power_sum = 0;
for f = 1:num_test_frames
    H_f = generateChannel(params);
    power_sum = power_sum + mean(abs(H_f(:)).^2);
end
ensemble_mean = power_sum / num_test_frames;
fprintf('  Ensemble mean |H|^2 : %.4f  (should be close to 1.0)\n\n', ensemble_mean);

if abs(ensemble_mean - 1.0) < 0.1
    fprintf('  [PASS] Power normalisation is correct.\n\n');
else
    fprintf('  [FAIL] Power is off - check path gain normalisation.\n\n');
end

%% Channel must be complex
if ~isreal(H_single)
    fprintf('  [PASS] Channel is complex (correct).\n');
else
    fprintf('  [FAIL] Channel should be complex.\n');
end

%% Rician channel check
params_ric = params;
params_ric.ChannelType = 'Rician';
H_ric = generateChannel(params_ric);
fprintf('  [PASS] Rician channel generated OK, size: %d x %d\n\n', ...
        size(H_ric,1), size(H_ric,2));

%% AWGN check
fprintf('AWGN check:\n');
target_snr = 20;
Y_noisy    = addAWGN(H_single, target_snr);
noise_est  = Y_noisy - H_single;
meas_snr   = 10*log10(mean(abs(H_single(:)).^2) / mean(abs(noise_est(:)).^2));
fprintf('  Target SNR   : %d dB\n',    target_snr);
fprintf('  Measured SNR : %.2f dB\n\n', meas_snr);

if abs(meas_snr - target_snr) < 1.5
    fprintf('  [PASS] AWGN noise level is correct.\n\n');
else
    fprintf('  [FAIL] SNR mismatch - check addAWGN.m.\n\n');
end

%% Doppler check
% With fd = 5000 Hz and Tsym = 5.208 us:
% coherence time Tc = 1/(4*fd) = 0.05 ms ~ 9.6 symbols
% Over 100 symbols we expect ~5 fading cycles
Tsym = (params.Nfft + params.CPLength) / params.SampleRate;
Tc_symbols = 1 / (4 * params.DopplerShift * Tsym);
fprintf('Doppler check:\n');
fprintf('  Symbol duration Tsym    : %.3f us\n', Tsym*1e6);
fprintf('  Coherence time          : %.2f ms = %.1f symbols\n', ...
        1/(4*params.DopplerShift)*1e3, Tc_symbols);
fprintf('  Expected fading cycles in 100 symbols : ~%.1f\n\n', ...
        100/Tc_symbols);

%% Plots
figure('Name', 'Channel Model Verification - fd = 5000 Hz');

subplot(3,1,1);
plot(abs(H_single(:,1)), 'b-', 'LineWidth', 1.2);
xlabel('OFDM Symbol index');
ylabel('|H|');
title(sprintf('Channel magnitude vs time (subcarrier 1) - fd = %d Hz', params.DopplerShift));
grid on;
% NOTE: at 5000 Hz Doppler you should see multiple fast fading cycles here

subplot(3,1,2);
plot(abs(H_single(1,:)), 'b-', 'LineWidth', 1.2);
xlabel('Subcarrier index');
ylabel('|H|');
title('Channel magnitude vs frequency (symbol 1) - 3-path multipath');
grid on;

subplot(3,1,3);
running_avg = zeros(1, num_test_frames);
running_sum = 0;
for f = 1:num_test_frames
    H_f = generateChannel(params);
    running_sum  = running_sum + mean(abs(H_f(:)).^2);
    running_avg(f) = running_sum / f;
end
plot(running_avg, 'r-', 'LineWidth', 1.2);
yline(1.0, 'k--', 'Expected = 1');
xlabel('Number of frames');
ylabel('Running mean |H|^2');
title('Ensemble average converging to 1 (power normalisation check)');
grid on;

fprintf('=== All checks done ===\n');
