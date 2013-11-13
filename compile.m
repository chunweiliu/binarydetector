mex -O ./features/resize.cc -o ./bin/resize
mex -O ./gdetect/dt.cc -o ./bin/dt
mex -O ./features/features.cc -o ./bin/features

% use one of the following depending on your setup
% 1 is fastest, 3 is slowest 

% 1) multithreaded convolution using blas
mex -O ./gdetect/fconvblas.cc -lmwblas -o ./bin/fconv
% 2) mulththreaded convolution without blas
% mex -O fconvMT.cc -o fconv
% 3) basic convolution, very compatible
% mex -O fconv.cc -o fconv
