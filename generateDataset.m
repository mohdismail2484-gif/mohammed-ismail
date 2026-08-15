function generateDataset(params)
% generateDataset - generates the training dataset for the LSTM receiver
%
% For each of the 500 frames (each frame = 100 OFDM symbols):
%   1. Transmit QPSK-modulated OFDM frame with pilots inserted
%   2. Generate a time-varying Rayleigh channel
%   3. Apply channel in frequency domain + add AWGN noise
%   4. Run through receiver to get freq-domain received signal
%   5. Extract noisy pilot channel estimates
%   6. Store: input = noisy pilots, target = true channel
%
% Dataset is saved as channel_dataset.mat in the dataset/ folder.
% Python will load this file for LSTM training.
%
% Total realisations = NumFrames x NumSymbols = 500 x 100 = 50,000

training_SNR = params.TrainingSNR;   % 15 dB

NumFrames  = params.NumFrames;        % 500
NumSymbols = params.NumSymbols;       % 100
Nfft       = params.Nfft;             % 64
NumPilots  = params.NumPilots;        % 8
NumData    = params.NumDataSubcarriers;  % 56

% bits to generate per frame (data subcarriers only, not pilot positions)
NumDataBits = NumData * NumSymbols * params.BitsPerSymbol;

% -------------------------------------------------------------------
% Preallocate dataset arrays
%
% X_data : inputs  -> noisy channel estimate at 8 pilot positions
%          stored as real and imag separately -> 8+8 = 16 features
%          Size: [NumFrames x NumSymbols x NumPilots*2]
%
% Y_data : targets -> true channel at all 64 subcarriers
%          stored as real and imag separately -> 64+64 = 128 values
%          Size: [NumFrames x NumSymbols x Nfft*2]
% -------------------------------------------------------------------
X_data = zeros(NumFrames, NumSymbols, NumPilots * 2);
Y_data = zeros(NumFrames, NumSymbols, Nfft    * 2);

fprintf('=============================================\n');
fprintf('  Dataset Generation\n');
fprintf('=============================================\n');
fprintf('  Frames       : %d\n', NumFrames);
fprintf('  Symbols/frame: %d\n', NumSymbols);
fprintf('  Total samples: %d\n', NumFrames * NumSymbols);
fprintf('  Training SNR : %d dB\n', training_SNR);
fprintf('  Channel type : %s\n\n', params.ChannelType);

tic;

for f = 1:NumFrames

    % progress update every 50 frames
    if mod(f, 50) == 0
        elapsed = toc;
        fprintf('  Frame %d / %d  (%.1f s elapsed)\n', f, NumFrames, elapsed);
    end

    % ----- Transmitter -----
    bits         = randi([0 1], NumDataBits, 1);
    data_syms    = qpskModulator(bits);                          % [NumData*NumSymbols x 1]
    data_syms    = reshape(data_syms, NumData, NumSymbols);      % [56 x 100]
    X_freq       = insertPilots(data_syms, params);              % [64 x 100]
    x_ifft       = ofdmIFFT(X_freq);                            % [64 x 100]
    x_time       = addCyclicPrefix(x_ifft, params);              % [80 x 100]

    % ----- Channel -----
    H = generateChannel(params);   % [100 x 64] - true channel

    % apply channel in frequency domain
    % (equivalent to time-domain convolution + remove CP + FFT)
    X_freq_T = X_freq.';          % [100 x 64]  (transpose to match H)
    Y_clean  = applyChannel(X_freq_T, H);         % [100 x 64]

    % add AWGN noise
    Y_noisy  = addAWGN(Y_clean, training_SNR);    % [100 x 64]

    % ----- Pilot extraction -----
    % get noisy channel estimate at pilot positions
    H_est_pilots = extractPilots(Y_noisy, params);  % [100 x 8]

    % ----- Store in dataset -----
    % inputs: real and imag of pilot channel estimates
    X_data(f, :, 1:NumPilots)          = real(H_est_pilots);
    X_data(f, :, NumPilots+1:end)      = imag(H_est_pilots);

    % targets: real and imag of true full channel
    Y_data(f, :, 1:Nfft)     = real(H);
    Y_data(f, :, Nfft+1:end) = imag(H);

end

total_time = toc;
fprintf('\nDone. Total time: %.1f seconds\n', total_time);

fprintf('\nDataset sizes:\n');
fprintf('  X_data : %s\n', mat2str(size(X_data)));
fprintf('  Y_data : %s\n', mat2str(size(Y_data)));

% basic sanity check on the data
fprintf('\nSanity check:\n');
fprintf('  Mean |X_data|  : %.4f\n', mean(abs(X_data(:))));
fprintf('  Mean |Y_data|  : %.4f\n', mean(abs(Y_data(:))));
fprintf('  Any NaN in X   : %d\n', any(isnan(X_data(:))));
fprintf('  Any NaN in Y   : %d\n', any(isnan(Y_data(:))));

% ----- Save -----
save_dir  = fileparts(mfilename('fullpath'));
save_path = fullfile(save_dir, 'channel_dataset.mat');

save(save_path, 'X_data', 'Y_data', 'params', '-v7.3');

fprintf('\nDataset saved to:\n  %s\n', save_path);
fprintf('=============================================\n');

end
