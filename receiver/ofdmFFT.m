function Y_freq = ofdmFFT(rxNoCP)
% ofdmFFT - converts received time-domain signal to frequency domain
%
% This is the FFT step at the receiver, which is the inverse of
% what the transmitter did with IFFT.
%
% Input:
%   rxNoCP  - [Nfft x NumSymbols] time-domain signal (CP already removed)
%
% Output:
%   Y_freq  - [NumSymbols x Nfft] frequency-domain received signal

% FFT along rows (each column is one OFDM symbol in time domain)
% then transpose so rows = symbols, columns = subcarriers
Y_freq = fft(rxNoCP, [], 1).';

end
