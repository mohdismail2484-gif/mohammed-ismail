function H = generateChannel(params)
% Generates a time-varying frequency-domain channel matrix H
%
% Channel model is based on the 3GPP TR 38.901 CDL-A profile
% (simplified to 3 dominant clusters). Jake's sum-of-sinusoids
% method is used to model each path's time variation due to Doppler.
%
% Reference: 3GPP TR 38.901 V17.0.0, Section 7.7.1, Table 7.7.1-1
%            Rappaport et al. (2019), IEEE Access, 7, pp. 78729-78757
%
% For each OFDM symbol n, the frequency-domain channel at subcarrier k is:
%
%   H(n,k) = sum_l  sqrt(gain_l) * h_l(n) * exp(-j*2*pi*k*d_l/Nfft)
%
% where h_l(n) is the time-varying fading coefficient for path l,
% and d_l is the path delay in samples.
%
% Output:
%   H - [NumSymbols x Nfft] complex channel matrix

Nfft       = params.Nfft;
NumSymbols = params.NumSymbols;
fd         = params.DopplerShift;      % 5000 Hz (from project plan Objective 1)
N          = params.NJakes;
L          = length(params.PathDelaysSamples);

% OFDM symbol duration including cyclic prefix (seconds)
Tsym = (Nfft + params.CPLength) / params.SampleRate;

% Convert path gains from dB to linear power, then normalise
% so that total received power = 1
pathGains_lin = 10 .^ (params.PathGains / 10);
pathGains_lin = pathGains_lin / sum(pathGains_lin);

H = zeros(NumSymbols, Nfft);

for l = 1:L

    % Random initial phases for this path (different per call = different frame)
    phi = 2*pi * rand(1, N);

    % Time-varying fading coefficient using Jake's model
    if strcmp(params.ChannelType, 'Rayleigh')
        h_path = jakesModel(fd, N, Tsym, NumSymbols, phi);

    elseif strcmp(params.ChannelType, 'Rician')
        % Rician = scattered component + dominant LOS component
        % LOS arrives at max Doppler angle (theta = 0)
        K          = params.KFactor;
        h_scatter  = jakesModel(fd, N, Tsym, NumSymbols, phi);
        n_vec      = (0:NumSymbols-1)';
        h_los      = sqrt(K/(K+1)) * exp(1j * 2*pi*fd * n_vec * Tsym);
        h_path     = sqrt(1/(K+1)) * h_scatter + h_los;

    else
        error('ChannelType must be Rayleigh or Rician');
    end

    % Scale by path amplitude (sqrt of normalised power)
    h_path = sqrt(pathGains_lin(l)) * h_path;

    % Add this path's contribution to the frequency-domain channel.
    % Each path introduces a linear phase shift across subcarriers
    % proportional to its delay.
    delay = params.PathDelaysSamples(l);
    for k = 1:Nfft
        phase_shift   = exp(-1j * 2*pi * (k-1) * delay / Nfft);
        H(:, k)       = H(:, k) + h_path * phase_shift;
    end

end

end
