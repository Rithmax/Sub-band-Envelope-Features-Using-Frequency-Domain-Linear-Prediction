function ceps = fdlp_feat(samples)

%*******************************************************************
%Congiguration
% Signal Parameters
param.fs = 16000 ;             % Sampling Rate
param.fdlplen = 1 ;           % FDLP frame length in seconds (1 second)
param.order     = 160;        % Model order per sub-band per second
                                                   
% Spectral feature parameters
param.num_spec_ceps   = 13;        % Number of cepstral components
param.fr_len          = 25;        % Frame length for spectral frame (ms)
param.fr_shift        = 10;        % Frame shift for spectral fram (ms)       
%*******************************************************************

if max(abs(samples)) < 1
    samples = samples * 2^15;           % Making the samples to raw format
end

A = samples(:);
sr = param.fs;
param.flen= (param.fr_len/1000)*sr;           % frame length corresponding to 25ms
param.fhop= (param.fr_shift/1000)*sr;         % frame overlap corresponding to 10ms
fnum = floor((length(A)-param.flen)/param.fhop)+1;
send = (fnum-1)*param.fhop + param.flen;
A = A(1:send);

fdlpwin = param.fdlplen*sr;             % FDLP window length

fdlpolap = ceil((param.fr_len-param.fr_shift)/10)*sr/100;
[X,~] = frame_new(A,fdlpwin,fdlpolap);
ceps = [];

for i = 1 :size(X,2)                    % Go over each FDLP window
    x = X(:,i);
    % Now lets dither (make sure the original waves are not normalized!)
    x = ditherit(x);
    x = x - 0.97* [0 ; x(1:end-1)];                 % Pre-emphasis
    % FDLP processing starts here
    temp = do_feats_for_seg(x,param);
    ceps = [ceps temp];
end
ceps = ceps(:,1:fnum);







