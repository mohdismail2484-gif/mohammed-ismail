function timeDomainSignal = ofdmIFFT(parallelData)
% ============================================================
% Perform OFDM IFFT
%
% Input:
%   parallelData      : Frequency-domain OFDM symbols
%
% Output:
%   timeDomainSignal  : Time-domain OFDM symbols
% ============================================================

timeDomainSignal = ifft(parallelData, [], 1);

end