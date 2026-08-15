function Y_noisy = addAWGN(Y, snr_dB)
% Add complex AWGN noise to the received signal
%
% SNR is defined as signal power / noise power.
% I calculate the actual signal power from Y itself and then
% work out how much noise to add to hit the target SNR.
%
% Inputs:
%   Y      - received signal [NumSymbols x Nfft], complex
%   snr_dB - target SNR in dB
%
% Output:
%   Y_noisy - signal with noise added

% measure average signal power across all samples
sig_power = mean(abs(Y(:)).^2);

% convert SNR to linear
snr_lin = 10^(snr_dB / 10);

% noise power needed to achieve this SNR
noise_power = sig_power / snr_lin;

% generate complex AWGN - split power equally between real and imag
noise = sqrt(noise_power/2) * (randn(size(Y)) + 1j*randn(size(Y)));

Y_noisy = Y + noise;

end
