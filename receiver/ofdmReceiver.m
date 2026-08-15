function Y_freq = ofdmReceiver(rxSignal, params)
% ofdmReceiver - OFDM receiver chain
%
% Takes the raw received time-domain signal and returns
% the frequency-domain symbols ready for channel estimation.
%
% Steps:
%   1. Remove cyclic prefix
%   2. FFT to get back to frequency domain
%
% Input:
%   rxSignal - [(Nfft+CPLength) x NumSymbols]
%   params   - config struct
%
% Output:
%   Y_freq   - [NumSymbols x Nfft]

rxNoCP = removeCyclicPrefix(rxSignal, params);
Y_freq = ofdmFFT(rxNoCP);

end
