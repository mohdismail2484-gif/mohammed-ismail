function H_est = mmseEstimator(Y_rx, params, snr_dB)
% mmseEstimator.m
% MMSE (Minimum Mean Square Error) channel estimator for OFDM.
%
% Uses theoretical channel statistics (covariance derived from CDL-A
% path delays and gains) to compute an optimal linear estimate of the
% channel at all 64 subcarriers from just 8 noisy pilot observations.
%
% MMSE formula:
%   H_mmse = R_fp * inv(R_pp + sigma2_eff * I) * h_ls_pilots
%
% where:
%   R_fp       = cross-covariance between all subcarriers and pilot positions
%   R_pp       = covariance at pilot positions
%   sigma2_eff = effective noise variance after dividing by pilot symbol
%   h_ls_pilots = LS estimate at the 8 pilot positions
%
% Inputs:
%   Y_rx   - received frequency-domain signal  [NumSymbols x Nfft]
%   params - config struct
%   snr_dB - operating SNR in dB
%
% Output:
%   H_est  - estimated channel  [NumSymbols x Nfft]

NumSymbols = params.NumSymbols;
Nfft       = params.Nfft;
pilot_pos  = params.PilotSubcarriers;   % [1 9 17 25 33 41 49 57]
pilot_val  = params.PilotValue;         % 1 + 1j
NumPilots  = params.NumPilots;          % 8

% noise variance per subcarrier (unit signal power assumed)
% effective noise after pilot division: sigma2 / |pilot|^2
snr_lin    = 10^(snr_dB / 10);
sigma2_eff = 1 / (snr_lin * abs(pilot_val)^2);   % = 1 / (snr * 2)

% compute theoretical covariance matrices from channel profile
[~, R_pp, R_fp] = computeChannelCovariance(params);

% MMSE filter matrix  [Nfft x NumPilots]
% W = R_fp * inv(R_pp + sigma2_eff * I)
W = R_fp / (R_pp + sigma2_eff * eye(NumPilots));

H_est = zeros(NumSymbols, Nfft);

for n = 1:NumSymbols
    % step 1: LS estimate at pilot positions
    y_pilots    = Y_rx(n, pilot_pos).';     % [NumPilots x 1] column vector
    h_ls_pilots = y_pilots / pilot_val;     % [NumPilots x 1]

    % step 2: apply MMSE filter to get estimate at all 64 subcarriers
    h_mmse      = W * h_ls_pilots;          % [Nfft x 1]
    H_est(n, :) = h_mmse.';
end

end
