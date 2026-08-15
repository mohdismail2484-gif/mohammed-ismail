function symbols = qpskModulator(bits)
% ============================================================
% QPSK Gray-Coded Modulator
%
% Input:
%   bits    : Binary column vector
%
% Output:
%   symbols : Complex QPSK symbols
% ============================================================

% Ensure even number of bits
if mod(length(bits),2) ~= 0
    error("Number of bits must be even.");
end

% Group bits into pairs
bitPairs = reshape(bits,2,[])';

% Allocate memory
symbols = zeros(size(bitPairs,1),1);

% Gray-coded QPSK mapping
for k = 1:size(bitPairs,1)

    b1 = bitPairs(k,1);
    b2 = bitPairs(k,2);

    if b1==0 && b2==0
        symbols(k) = (1+1i)/sqrt(2);

    elseif b1==0 && b2==1
        symbols(k) = (-1+1i)/sqrt(2);

    elseif b1==1 && b2==1
        symbols(k) = (-1-1i)/sqrt(2);

    elseif b1==1 && b2==0
        symbols(k) = (1-1i)/sqrt(2);

    end
end

end