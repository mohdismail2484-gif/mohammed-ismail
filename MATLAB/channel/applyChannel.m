function Y = applyChannel(X, H)
% Apply the frequency-domain channel to transmitted OFDM symbols
%
% In OFDM, once we remove the cyclic prefix and take the FFT,
% the channel just multiplies each subcarrier by a complex gain.
% So received = H * transmitted (element-wise per subcarrier).
%
% Inputs:
%   X - transmitted frequency-domain symbols [NumSymbols x Nfft]
%   H - channel matrix [NumSymbols x Nfft]
%
% Output:
%   Y - received signal (before noise) [NumSymbols x Nfft]

if ~isequal(size(X), size(H))
    error('X and H must be the same size');
end

% simple element-wise multiplication
Y = X .* H;

end
