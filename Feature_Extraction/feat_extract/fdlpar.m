function p = fdlpar(x,param)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------- DCT and Sub-band Windowing ----------

% Take the DCT
x = dct(x); 

% Get the frame length for FDLP input 
flen = size(x,1);

% Make the sub-band weights
% Mel Scale of Decomposition
[wts,idx] = melweights(flen,param.fs);
    
% Define the model order 
fp = round(param.order*flen/param.fs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -----------------------  Pole Estimation ----------------------

nb = size(idx,1);           % Number of sub-bands ...
p  = cell(nb,1);

% Time envelope estimation per band 
for I = 1:nb
    % Apply the weights and get poles 
      tmpx = full(diag(sparse(wts{I}))*x(idx(I,1):idx(I,2),:));
    %******** Finally, FDLP ********% 
      p{I} =lpc(tmpx,fp)';
end
