function parallelData = serialToParallel(symbols, params)
% ============================================================
% Convert Serial QPSK Symbols to Parallel OFDM Frames
%
% Input:
%   symbols : Serial QPSK symbol vector
%   params  : Configuration structure
%
% Output:
%   parallelData : OFDM frame matrix
%                  Size = Nfft × NumSymbols
% ============================================================

expectedSymbols = params.Nfft * params.NumSymbols;

if length(symbols) ~= expectedSymbols
    error("Symbol count does not match OFDM frame size.");
end

parallelData = reshape(symbols, params.Nfft, params.NumSymbols);

end