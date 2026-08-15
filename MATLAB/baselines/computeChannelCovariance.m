function [R_full, R_pp, R_fp] = computeChannelCovariance(params)
% computeChannelCovariance.m
% Computes the theoretical frequency-domain channel covariance matrices
% using the CDL-A channel profile (path delays and gains from config).
%
% In OFDM, a multipath channel with path delays [d1, d2, ...] produces
% a correlation between subcarriers k and l:
%
%   R(k,l) = sum_i  P_i * exp(-j * 2*pi * (k-1-l+1) * d_i / Nfft)
%
% where P_i are normalised path powers (sum to 1).
%
% Outputs:
%   R_full  - [Nfft x Nfft]         covariance across all 64 subcarriers
%   R_pp    - [NumPilots x NumPilots] covariance at the 8 pilot positions
%   R_fp    - [Nfft x NumPilots]     cross-covariance: all vs pilots

Nfft       = params.Nfft;
pilot_pos  = params.PilotSubcarriers;   % [1 9 17 25 33 41 49 57]
delays     = params.PathDelaysSamples;  % [0 5 10]
gains_dB   = params.PathGains;          % [0 -3 -8]

% convert path gains to linear power and normalise so total = 1
P = 10 .^ (gains_dB / 10);
P = P / sum(P);

L = length(delays);

% build the full Nfft x Nfft covariance matrix
R_full = zeros(Nfft, Nfft);
for k = 1:Nfft
    for l = 1:Nfft
        for i = 1:L
            R_full(k, l) = R_full(k, l) + P(i) * exp(-1j * 2*pi * (k-l) * delays(i) / Nfft);
        end
    end
end

% covariance at pilot positions only  [NumPilots x NumPilots]
R_pp = R_full(pilot_pos, pilot_pos);

% cross-covariance: all subcarriers vs pilot positions  [Nfft x NumPilots]
R_fp = R_full(:, pilot_pos);

end
