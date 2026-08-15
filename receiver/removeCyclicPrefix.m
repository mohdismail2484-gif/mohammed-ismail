function rxNoCP = removeCyclicPrefix(rxSignal, params)
% removeCyclicPrefix - strips the CP from each received OFDM symbol
%
% The transmitter added CPLength samples at the front of each symbol.
% We just discard those samples here.
%
% Input:
%   rxSignal - [(Nfft + CPLength) x NumSymbols]
%
% Output:
%   rxNoCP   - [Nfft x NumSymbols]

rxNoCP = rxSignal(params.CPLength + 1 : end, :);

end
