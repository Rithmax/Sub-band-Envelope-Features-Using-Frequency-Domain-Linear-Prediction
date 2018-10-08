%clear all;close all;clc;
restoredefaultpath
addpath(genpath('./feat_extract'))

%*****************************************************************
% 08-Oct-2018
% Sarith Fernando
% Speech Processing, Electrical Engineering and Telecommunications
% University of New South Wales
% sarith.fernando@unsw.edu.au
%*****************************************************************

%Feature extraction for 1s duration utterance

[sp,fs]=audioread('100000.wav');
sp=sp(:,1);

D = fdlp_feat(sp)';
