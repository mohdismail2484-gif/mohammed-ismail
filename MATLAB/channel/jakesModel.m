function h = jakesModel(fd, N, Tsym, numSymbols, phi)
% Jake's model to generate time-varying Rayleigh fading coefficients
%
% I'm using Jake's sum-of-sinusoids method here as described in
% Proakis "Digital Communications" and also mentioned in the lecture notes.
% The idea is to approximate the Doppler spectrum by summing N sinusoids
% with uniformly spaced angles.
%
% Inputs:
%   fd         - max Doppler frequency (Hz)
%   N          - number of sinusoids (20 is usually enough)
%   Tsym       - OFDM symbol duration including CP (seconds)
%   numSymbols - how many symbols to generate
%   phi        - random initial phases, vector of size [1 x N]
%
% Output:
%   h  - complex fading coefficients, vector [numSymbols x 1]

h = zeros(numSymbols, 1);

for n = 1:numSymbols
    total = 0;
    for i = 1:N
        % Doppler frequency for the i-th sinusoid
        fi = fd * cos(2*pi*i/N);
        % accumulate
        total = total + exp(1j * (2*pi*fi*(n-1)*Tsym + phi(i)));
    end
    h(n) = total / sqrt(N);
end

% NOTE: the output h is complex Gaussian with unit variance approximately
% You can verify: mean(abs(h).^2) should be close to 1

end
