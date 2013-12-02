function writeheader(hdrfile, num, labelsize, model)

% writeheader(file, num, labelssize, model)
% Write training header file.
% Used in the interface with the gradient descent algorithm.

fid = fopen(hdrfile, 'wb');
if model.wta.iswta == 1
    header = [num labelsize model.numblocks model.wta.blocksizes];
else
    header = [num labelsize model.numblocks model.blocksizes];
end
fwrite(fid, header, 'int32');
fwrite(fid, model.regmult, 'single');
fwrite(fid, model.learnmult, 'single');
fclose(fid);
