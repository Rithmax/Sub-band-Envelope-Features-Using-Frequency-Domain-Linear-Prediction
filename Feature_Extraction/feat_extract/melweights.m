function [wts,idx] = melweights(flen,sr)
% [wts,idx] = melweights(flen,sr,type,par)
%   flen: length of frame (dct or fft length)
%   sr:   sampling rate
%   wts:  the weights in a cell array
%   idx:  the indices to which the weights correspond (for speed up)
%
%   notes: Dan Ellis' code

if nargin < 3; type = 'gauss'; end
if nargin < 4; par  = 1;       end
if nargin < 5; dB   = 48;      end

% DCT length
nfreqs = flen;

% How many output bands? (copied from rasta/init.c)
maxmel = hz2mel(sr/2);
nbands = ceil(maxmel)+1;

% bark per filt
step = maxmel/(nbands - 1);

% mel frequency of every bin in FFT
bin = hz2mel([0:(nfreqs-1)]*(sr/2)/(nfreqs-1));

% Weights to collapse FFT bins into aud channels
wts = cell(nbands,1);

% Initialize idx with full range
idx = repmat([1,flen],nbands,1);

% Defining Gaussian windows
for I = 1:nbands
    f_mel_mid = (I-1) * step;
    wts{I} = exp((-1/2)*((bin - f_mel_mid)).^2);
end

dB = 48;                 % Parameter to specify the band edges
% convert dB to linear scale
lin = 10^(-dB/20);

% Finding the begin and end points
% adjust windows and keep indices
for I = 1:nbands
    tmpidx   = find(wts{I}>=lin);
    idx(I,:) = [min(tmpidx),max(tmpidx)];
    wts{I}   = wts{I}(tmpidx);
end
