function H_est = lsEstimator(Y_rx, params)
% lsEstimator.m
% Least Squares (LS) channel estimator for OFDM.
%
% Steps:
%   1. Extract received signal at the 8 pilot subcarriers
%   2. Divide by known pilot value -> noisy LS channel estimate at pilots
%   3. Linearly interpolate to fill all 64 subcarriers
%
% Inputs:
%   Y_rx   - received frequency-domain signal  [NumSymbols x Nfft]
%   params - config struct
%
% Output:
%   H_est  - estimated channel  [NumSymbols x Nfft]
%
% The LS estimator is the simplest baseline. It has no knowledge of the
% channel statistics, so it is noisier than MMSE but very easy to implement.

NumSymbols = params.NumSymbols;
Nfft       = params.Nfft;
pilot_pos  = params.PilotSubcarriers;   % [1 9 17 25 33 41 49 57]
pilot_val  = params.PilotValue;         % 1 + 1j

H_est = zeros(NumSymbols, Nfft);

for n = 1:NumSymbols
    % step 1: get received values at pilot positions for this symbol
    y_pilots = Y_rx(n, pilot_pos);           % [1 x 8]

    % step 2: LS estimate - divide by known pilot value
    % Y_pilot = H_pilot * X_pilot + noise
    % H_ls    = Y_pilot / X_pilot  (noisy estimate)
    h_ls_pilots = y_pilots / pilot_val;      % [1 x 8]

    % step 3: linear interpolation from 8 pilots to all 64 subcarriers
    % pilot_pos = [1 9 17 25 33 41 49 57], query points = 1:64
    % 'extrap' handles the 7 subcarriers beyond the last pilot (58-64)
    H_est(n, :) = interp1(pilot_pos, h_ls_pilots, 1:Nfft, 'linear', 'extrap');
end

end
