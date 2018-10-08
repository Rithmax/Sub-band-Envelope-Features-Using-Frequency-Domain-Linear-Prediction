function y = ditherit(x,db,type)
%   USAGE: x = ditherit(x,db,type)
%   Dithers a signal by db dBs
%   x:    input signal
%   db:   dithering amount, e.g. -96

% Check arguments

if (nargin < 2); db =   -96;  end
y = x + (10^(db/20))*randn(size(x));
