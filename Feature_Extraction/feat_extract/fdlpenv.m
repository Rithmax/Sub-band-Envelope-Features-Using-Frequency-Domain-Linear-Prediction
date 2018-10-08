function env = fdlpenv(p,npts)

% Find the size of the input signal
[na,~] = size(p{1});

% How many fft points to calculate
nfft = 2*(max(npts,na)-1);

% Calculate the LP frequency response
h = fft(p{1},nfft);

% Use the positive half of the spectrum
h = h(1:(nfft/2)+1,:);

% Get the inverse spectrum
h = 1./h;

% Power spectrum 
h = (h.*conj(h));

% Correcting energy factor
env = 2*h;

