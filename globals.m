% Set up global variables used throughout the code

% directory for caching models, intermediate data, and results
cachedir = './voccache/';

% directory for LARGE temporary files created during training
tmpdir = './voccache/';

% dataset to use
VOCyear = '2007';

% directory with PASCAL VOC development kit and dataset
VOCdevkit = ['./VOCdevkit/'];

% which development kit is being used
% this does not need to be updated
VOCdevkit2006 = false;
VOCdevkit2007 = false;
VOCdevkit2008 = false;
switch VOCyear
  case '2006'
    VOCdevkit2006=true;
  case '2007'
    VOCdevkit2007=true;
  case '2008'
    VOCdevkit2008=true;
end

% add path
addpath('bin')
addpath('features')
addpath('gdetect')
addpath('io')
addpath('test')
addpath('train')
addpath('vis')
addpath('3rdparty/liblinear-1.94/matlab/')
addpath('wta')
addpath('html')
addpath('vis')
%addpath('3rdparty/libsvm-3.17/matlab')