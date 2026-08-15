function params = config()

% Configuration file for AI-Assisted Adaptive Receiver project
% All system parameters are defined here so I don't have to
% change them in multiple places

%% OFDM Parameters
params.Nfft       = 64;      % number of subcarriers
params.CPLength   = 16;      % cyclic prefix length (samples)
params.NumSymbols = 100;     % OFDM symbols per frame

%% Modulation
params.Modulation    = 'QPSK';
params.ModOrder      = 4;
params.BitsPerSymbol = log2(params.ModOrder);  % = 2 for QPSK

% total bits per frame
params.NumBits = params.Nfft * params.NumSymbols * params.BitsPerSymbol;

%% Channel Parameters
params.ChannelType = 'Rayleigh';   % 'Rayleigh' or 'Rician'

% Doppler shift - project plan Objective 1 states "up to 5 kHz"
% This models high-mobility 6G scenarios (e.g. vehicular links, UAV)
params.DopplerShift = 5000;        % max Doppler frequency (Hz)

params.SampleRate = 15.36e6;       % sample rate (Hz) - standard NR numerology

% Multipath channel profile - simplified 3-path model based on
% 3GPP TR 38.901 CDL-A (Table 7.7.1-1), first three dominant clusters.
%
% CDL-A cluster delays (normalised):
%   Cluster 1: 0 ns       -> 0 samples  at 15.36 MHz
%   Cluster 2: 308.9 ns   -> ~5 samples at 15.36 MHz
%   Cluster 3: 638.9 ns   -> ~10 samples at 15.36 MHz
%
% All delays are within the CP length of 16 samples (1.04 us), so
% there is no inter-symbol interference (ISI).
params.PathDelays        = [0, 3.09e-7, 6.39e-7];   % seconds
params.PathDelaysSamples = [0, 5, 10];               % samples (must be < CPLength)

% Path gains in dB - representative of CDL-A power profile
% (strongest path 0 dB, others attenuated)
params.PathGains = [0, -3, -8];

% Rician K-factor - used only when ChannelType = 'Rician'
% K = 10 means LOS component is 10x stronger than scattered power
params.KFactor = 10;

% Number of sinusoids for Jake's model
% 20 is standard in literature (Proakis & Salehi, Digital Communications)
params.NJakes = 20;

%% Pilot Parameters
% Comb-type pilots: one pilot every 8 subcarriers across the frame.
% Positions (1-indexed): [1, 9, 17, 25, 33, 41, 49, 57]
params.PilotSpacing     = 8;
params.PilotSubcarriers = 1 : params.PilotSpacing : params.Nfft;
params.NumPilots        = length(params.PilotSubcarriers);   % = 8

% Known pilot symbol transmitted at every pilot position
params.PilotValue = 1 + 1j;

% Data subcarriers = all subcarriers not used for pilots
params.DataSubcarriers    = setdiff(1:params.Nfft, params.PilotSubcarriers);
params.NumDataSubcarriers = length(params.DataSubcarriers);   % = 56

%% Noise / SNR
params.SNR_dB      = 0:5:30;   % SNR sweep for evaluation (dB)
params.TrainingSNR = 15;        % SNR used during dataset generation (dB)

%% Dataset Parameters
% 500 frames x 100 symbols = 50,000 channel realisations (meets Objective 1)
params.NumFrames       = 500;
params.NumRealizations = params.NumFrames * params.NumSymbols;   % = 50,000

%% Random Seed
params.RandomSeed = 42;
rng(params.RandomSeed);

end
