function H_est_pilots = extractPilots(Y_rx, params)
% extractPilots - gets a noisy channel estimate at pilot positions
%
% At each pilot subcarrier:
%   Y[pilot] = H[pilot] * X_pilot + noise
%
% Since X_pilot is known, dividing gives:
%   H_est[pilot] = Y[pilot] / X_pilot  (noisy estimate)
%
% Inputs:
%   Y_rx   - [NumSymbols x Nfft] received freq-domain signal
%   params - config struct
%
% Output:
%   H_est_pilots - [NumSymbols x NumPilots] noisy channel at pilots

% grab the received values at pilot positions
Y_pilots = Y_rx(:, params.PilotSubcarriers);

% divide by known pilot to recover channel estimate
H_est_pilots = Y_pilots / params.PilotValue;

end
