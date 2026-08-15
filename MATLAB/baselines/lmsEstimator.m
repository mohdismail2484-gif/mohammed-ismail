function H_est = lmsEstimator(Y_rx, params, mu)
% lmsEstimator.m
% LMS (Least Mean Squares) adaptive channel estimator for OFDM.
%
% Tracks the time-varying channel at each pilot subcarrier using the
% LMS algorithm, then interpolates across frequency to all subcarriers.
%
% This is one of the "conventional adaptive equalizers" required by the
% project dissertation, alongside RLS and MMSE.
%
% LMS update rule (a-posteriori, corrector form):
%
%   For each symbol n >= 2, for each pilot subcarrier p:
%     error(n)    = Y_rx(n,p) - H_est(n-1,p) * PilotValue
%     H_est(n,p) = H_est(n-1,p) + mu * conj(PilotValue) * error(n)
%
%   This means H_est(n,p) is updated using the CURRENT observation Y_rx(n,:),
%   making it directly comparable to LS which also uses Y_rx(n,:).
%   The result is an exponential smoother:
%     H_est(n,p) = (1 - mu*|PV|^2) * H_est(n-1,p) + mu*|PV|^2 * LS(n,p)
%
% Initialisation: H_est(1,p) = Y_rx(1,p) / PilotValue (LS at symbol 1)
%
% Convergence constraint: 0 < mu < 1/|PilotValue|^2 = 0.5
% mu = 0.1 gives ~20% weight to current LS observation, memory ~5 symbols.
% This is well-matched to 5000 Hz Doppler (coherence ~ 10 symbols).
%
% Inputs:
%   Y_rx   - received freq-domain signal  [NumSymbols x Nfft]
%   params - config struct
%   mu     - LMS step size (default 0.1 if not provided)
%
% Output:
%   H_est  - estimated channel  [NumSymbols x Nfft]

if nargin < 3
    % step size: mu * |PilotValue|^2 = mu * 2 = 0.2 (weight on current obs)
    % stability requires mu < 1/|PilotValue|^2 = 0.5
    mu = 0.1;
end

NumSymbols = params.NumSymbols;
Nfft       = params.Nfft;
pilot_pos  = params.PilotSubcarriers;   % [1 9 17 25 33 41 49 57]
pilot_val  = params.PilotValue;         % 1 + 1j

NumPilots = length(pilot_pos);

% H estimates at pilot positions, one row per OFDM symbol
h_pilots = zeros(NumSymbols, NumPilots);

% symbol 1: initialise with direct LS estimate
h_pilots(1, :) = Y_rx(1, pilot_pos) / pilot_val;

% LMS a-posteriori update: h_pilots(n) uses Y_rx(n) and h_pilots(n-1)
% This ensures the estimate at symbol n incorporates the current pilots,
% just like LS does, while adding temporal smoothing to reduce noise.
for n = 2 : NumSymbols
    for p = 1 : NumPilots
        k = pilot_pos(p);

        % error: how well does the previous estimate predict the current pilot?
        err = Y_rx(n, k) - h_pilots(n-1, p) * pilot_val;

        % update current estimate using current observation
        h_pilots(n, p) = h_pilots(n-1, p) + mu * conj(pilot_val) * err;
    end
end

% interpolate from 8 pilot subcarriers to all 64 subcarriers
% same linear interpolation as LS estimator so comparison is fair
H_est = zeros(NumSymbols, Nfft);
for n = 1 : NumSymbols
    H_est(n, :) = interp1(pilot_pos, h_pilots(n, :), 1:Nfft, 'linear', 'extrap');
end

end
