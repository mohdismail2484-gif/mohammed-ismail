function X_freq = insertPilots(data_symbols, params)
% insertPilots - puts known pilot symbols into the OFDM frame
%
% I'm using comb-type pilots here (pilots at fixed subcarrier positions
% across all OFDM symbols). This is how it's done in LTE/5G as well.
%
% Inputs:
%   data_symbols - [NumDataSubcarriers x NumSymbols] QPSK data
%   params       - config struct
%
% Output:
%   X_freq - [Nfft x NumSymbols] full frame with pilots + data

X_freq = zeros(params.Nfft, params.NumSymbols);

% put pilot symbols at pilot subcarrier positions
% same pilot value repeated across all symbols (block-type in time)
for p = 1:params.NumPilots
    idx = params.PilotSubcarriers(p);
    X_freq(idx, :) = params.PilotValue;
end

% fill remaining subcarriers with data
for d = 1:params.NumDataSubcarriers
    idx = params.DataSubcarriers(d);
    X_freq(idx, :) = data_symbols(d, :);
end

end
