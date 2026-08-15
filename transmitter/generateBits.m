function bits = generateBits(params)
% ============================================================
% Generate Random Binary Data
%
% Input:
%   params - Project configuration structure
%
% Output:
%   bits   - Random binary sequence
% ============================================================

bits = randi([0 1], params.NumBits, 1);

end