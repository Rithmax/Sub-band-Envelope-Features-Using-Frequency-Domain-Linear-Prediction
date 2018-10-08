clear all;close all;clc;
restoredefaultpath
addpath(genpath('./feat_extract'))

%**************************************************************************
% 08-Oct-2018
% Sarith Fernando
% Speech Processing, Electrical Engineering and Telecommunications
% University of New South Wales
% sarith.fernando@unsw.edu.au
%**************************************************************************

%**************************************************************************
% Title: Sub-band Envelope Features Using Frequency Domain Linear 
%                     Prediction for Short Duration Language Identification
% Cite as: Fernando, S., Sethu, V., Ambikairajah, E. (2018) 
%          Sub-band Envelope Features Using Frequency Domain Linear 
%          Prediction for Short Duration Language Identification. 
%          Proc. Interspeech 2018, 1818-1822, DOI: 10.21437/Interspeech.2018-1805.
% Database: AP17-OLR/AP18-OLR, 'AP17-OLR Challenge: Data, Plan, and Baseline'
%**************************************************************************

%% Feature extraction for Train data
% Data will be stored as H5 file for the use of BLSTM training

%**************************************************************************
% Define the path to your data directory
data_dir='/media/eleceng/E/Sarith/kaldi-caser_olr_2018/egs/ap18_olr_bnf_extract/lre_baseline/data/train/audio/';
% Define the path to your data write location
write_path = '/media/eleceng/E/Sarith/Data_OLR18/FDLP/train.h5';
% Define the path to your training list file
dataList='./Lists/train_list.txt';
%**************************************************************************

fid = fopen(dataList, 'rt');
C = textscan(fid, '%s%s%s');
fclose(fid);
feaFiles = C{1};
[a, ~, labels] = unique(C{2}, 'stable');

for i=1:length(labels)

[sp,fs]=audioread([data_dir feaFiles{i,1} '.wav']);
sp=sp(:,1);
D = fdlp_feat(sp)';

tgt=labels(i);
chunk=98;
start=0;
n=0;
while chunk<=size(D,1)
    data=D(1+start:chunk,:);
    start=chunk;
    chunk=chunk+98;
    n=n+1;
    
    h5create([write_path],['/train_data_',num2str(tgt),'_',num2str(i),'_',num2str(n)],size(data'))
    h5write([write_path], ['/train_data_',num2str(tgt),'_',num2str(i),'_',num2str(n)], data')
    fprintf('Calculated utt %d,chunk %d \n',i,n);
end
end

%% Feature extraction for dev and test
% Data will be stored as H5 file for the use of BLSTM testing

%**************************************************************************
% Define the path to your data directory
read_path='/media/eleceng/E/Sarith/kaldi-caser_olr_2018/egs/ap18_olr_bnf_extract/lre_baseline/';
% Define the path to your data write location
write_dir = '/media/eleceng/E/Sarith/Data_OLR18/FDLP/';
%**************************************************************************

test_dirs={'dev_1s','dev_3s','dev_all','test_1s','test_3s','test_all'};

for j = 1:length(test_dirs)

test_dir=test_dirs{j};
dataList=[read_path,'data/', test_dir, '/wav.scp'];
fid = fopen(dataList, 'rt');
C = textscan(fid, '%s%s%s');
fclose(fid);
feaFiles = C{1};
data_path = C{2};

write_path = [write_dir,test_dir, '.h5'];
n=length(feaFiles);

for i=1:n
    
    [sp,fs]=audioread([read_path,data_path{i,1}]);
    sp=sp(:,1);
    D = fdlp_feat(sp)';
    
    h5create([write_path],['/',feaFiles{i,1}],size(D'))
    h5write([write_path], ['/',feaFiles{i,1}], D')  
    fprintf('Converted %d/%d \n',i,n);
end
end