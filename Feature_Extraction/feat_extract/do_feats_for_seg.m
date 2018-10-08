function feats = do_feats_for_seg(x,param)

%    Adapted from Sriram Ganapathy code
pad_fr = 2;
padlen=pad_fr*param.fhop;
% Padding the signal with 10 ms on both sides
x = [flipud(x(1:padlen)); x ; flipud(x(end -padlen+1:end))];

fnum = floor((length(x) - param.flen)/param.fhop) + 1 ;
send = (fnum-1)*param.fhop + param.flen ;
factor = 20*(param.fs/8000);           % Downsampling on the envelopes
fdlplen = floor(send/factor);

% FDLP pole extraction
p = fdlpar(x,param);
nb = size(p,1);                     % Number of Sub-bands

%*****************************************************************
%                   Envelope Generation
%*****************************************************************

fdlp_spec =zeros(nb,fdlplen);
for J = 1:(nb)
    % Envelope generation using polynomial interpolation
    ENV  =  fdlpenv(p(J),fdlplen);
    fdlp_spec(J,:) = ENV(:)';
end
clear ENV ;

%*****************************************************************
%        Temporal Average Magnitude (TAM) Feature Extraction     %
%*****************************************************************
energy_bands=zeros(nb,fnum);
wind = hamming(floor(param.flen/factor))';
opt = 'nodelay';
% Energy Integration
for band = 1: size(fdlp_spec,1)
    [band_data,z,opt]=buffer(fdlp_spec(band,:),floor(param.flen/factor),floor((param.flen-param.fhop)/factor),opt);
    opt = 'nodelay';
    energy_bands((band),:) = (wind*band_data);
end

cepstra = spec2cep(energy_bands,param.num_spec_ceps);
cepstra = lifter(cepstra, 0.6);
    
% Delta Double Delta ...
% Append deltas and double-deltas onto the cepstral vectors
del = deltas(cepstra);
        
% Double deltas are deltas applied twice with a shorter window
ddel = deltas(deltas(cepstra,5),5);
        
% Composite, 39-element feature vector, just like we use for speech recognition
feats = [cepstra;del;ddel];

%%% Unpadding
feats = feats(:,pad_fr+1:end-pad_fr);
