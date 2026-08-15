function H_est = rlsEstimator(Y_rx, params, lambda)
% rlsEstimator.m
% RLS (Recursive Least Squares) adaptive channel estimator for OFDM.
%
% Tracks the time-varying channel at each pilot subcarrier using the
% scalar RLS algorithm with a forgetting factor. Converges faster than
% LMS and adapts its gain automatically using the inverse correlation P.
%
% This is one of the "conventional adaptive equalizers" required by the
% project dissertation, alongside LMS and MMSE.
%
% Scalar RLS update (a-posteriori corrector form):
%
%   For each symbol n >= 2, for each pilot subcarrier p:
%     k_gain     = P(p) * conj(pv) / (lambda + |pv|^2 * P(p))
%     error(n)   = Y_rx(n,p) - H_est(n-1,p) * pv
%     H_est(n,p) = H_est(n-1,p) + k_gain * error(n)
%     P(p)       = (1/lambda) * ( P(p) - k_gain * pv * P(p) )
%
% The a-posteriori form ensures H_est(n,p) uses the CURRENT observation
% Y_rx(n,:), making it comparable to LS.
%
% lambda = 0.9 gives memory window ~10 symbols (1/(1-lambda)),
% which matches the channel coherence time at 5000 Hz Doppler.
% lambda = 0.99 (100-symbol memory) is too slow for this scenario.
%
% Inputs:
%   Y_rx   - received freq-domain signal  [NumSymbols x Nfft]
%   params - config struct
%   lambda - forgetting factor (default 0.9 if not provided)
%             memory ~ 1/(1-lambda) symbols
%
% Output:
%   H_est  - estimated channel  [NumSymbols x Nfft]

if nargin < 3
    % lambda = 0.9: effective memory ~10 symbols = coherence time at 5000 Hz
    % lambda = 0.99 is too slow for high-Doppler channels
    lambda = 0.9;
end

NumSymbols = params.NumSymbols;
Nfft       = params.Nfft;
pilot_pos  = params.PilotSubcarriers;   % [1 9 17 25 33 41 49 57]
pilot_val  = params.PilotValue;         % 1 + 1j
NumPilots  = length(pilot_pos);

% |PilotValue|^2 = |1+j|^2 = 2 (constant, compute once)
pv2 = abs(pilot_val)^2;

% inverse correlation P — one scalar per pilot subcarrier
% initialise to 1/delta (large) so first update acts like LS
delta = 0.001;
P = (1/delta) * ones(1, NumPilots);

% H estimates at pilot positions
h_pilots = zeros(NumSymbols, NumPilots);

% symbol 1: initialise with direct LS estimate
h_pilots(1, :) = Y_rx(1, pilot_pos) / pilot_val;

% RLS a-posteriori update: h_pilots(n) uses Y_rx(n) and h_pilots(n-1)
% P evolves as the RLS gain adapts from initial LS-like gain to steady state
for n = 2 : NumSymbols
    for p = 1 : NumPilots
        k = pilot_pos(p);

        % Kalman gain (scalar RLS)
        k_gain = P(p) * conj(pilot_val) / (lambda + pv2 * P(p));

        % error: how well does the previous estimate predict current pilot?
        err = Y_rx(n, k) - h_pilots(n-1, p) * pilot_val;

        % update current estimate using current observation
        h_pilots(n, p) = h_pilots(n-1, p) + k_gain * err;

        % update inverse correlation for next step
        P(p) = (1/lambda) * (P(p) - k_gain * pilot_val * P(p));
    end
end

% interpolate from 8 pilot positions to all 64 subcarriers
H_est = zeros(NumSymbols, Nfft);
for n = 1 : NumSymbols
    H_est(n, :) = interp1(pilot_pos, h_pilots(n, :), 1:Nfft, 'linear', 'extrap');
end

end
