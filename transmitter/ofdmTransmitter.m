function tx = ofdmTransmitter(params)
% ============================================================
% OFDM Transmitter
%
% Generates one OFDM frame and returns all intermediate signals.
%
% Input:
%   params : Configuration structure
%
% Output:
%   tx : Structure containing transmitter signals
% ============================================================

% Generate random bits
tx.bits = generateBits(params);

% QPSK modulation
tx.symbols = qpskModulator(tx.bits);

% Serial-to-parallel conversion
tx.parallelData = serialToParallel(tx.symbols, params);

% OFDM IFFT
tx.timeDomainSignal = ofdmIFFT(tx.parallelData);

% Add cyclic prefix
tx.txSignal = addCyclicPrefix(tx.timeDomainSignal, params);

end