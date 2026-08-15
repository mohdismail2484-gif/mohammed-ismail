function txSignal = addCyclicPrefix(timeDomainSignal, params)
% ============================================================
% Add Cyclic Prefix to OFDM Symbols
%
% Input:
%   timeDomainSignal : Nfft × NumSymbols
%   params           : Configuration structure
%
% Output:
%   txSignal         : (Nfft + CPLength) × NumSymbols
% ============================================================

cp = timeDomainSignal(end - params.CPLength + 1:end, :);

txSignal = [cp; timeDomainSignal];

end