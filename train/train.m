function model = train(name, model, pos, neg, cs, k)

% model = train(name, model, pos, neg)
% Train LSVM. (For now it's just an SVM)
%

% SVM learning parameters
%C = 0.002*model.numcomponents;
%J = 1; % weight of positive data
globals;
pascal_init;

maxsize = 2^28;

globals;
hdrfile = [tmpdir name '.hdr'];
datfile = [tmpdir name '.dat'];
modfile = [tmpdir name '.mod'];
inffile = [tmpdir name '.inf'];
lobfile = [tmpdir name '.lob'];
datafile = [cachedir name '_data.mat'];
tmpdatfile = [tmpdir name '.dattmp'];
tmphdrfile = [tmpdir name '.hdrtmp'];
tmpmodfile = [tmpdir name '.modtmp'];
tmpinffile = [tmpdir name '.inftmp'];
tmplobfile = [tmpdir name '.lobtmp'];
labelsize = 5;  % [label id level x y]

% Reset some of the tempoaray files, just in case
resetall(hdrfile, inffile, modfile, lobfile, labelsize, model);


if ~exist(datafile, 'file')
    
    % approximate bound on the number of examples used in each iteration
    % dim will reduce due to wta, but we still don't want to use too many
    % negatives.
    dim = 0;
    if model.wta.iswta == 1
        factor = 25;
    else
        factor = 1;
    end
    for i = 1:model.numcomponents
        dim = max(dim, model.components{i}.dim);
    end
    maxnum = floor(maxsize / (dim * 4 * factor));
    
    % Find the positive examples and safe them in the data file
    [posdata, posids, numpos] = poswarp(name, model, 1, pos);
    
    % Add random negatives
    [negdata, negids, numneg] = negrandom(name, model, 1, neg, maxnum-numpos);
    
    data = [posdata negdata]';
    labels = [ones(numpos,1); -ones(numneg,1)];
    ids = [posids; negids];
    datafile = [cachedir name '_data.mat'];
    
    save(datafile, 'data', 'labels', 'ids');
    
else
    load(datafile)
end

if model.wta.iswta == 1
    oridata = data;
    [data, model.wta.Theta] = ...
        wtahash(oridata, model.wta.k, model.wta.m, [], model.wta.iswta);
end    
% if model.wta.iswta == 1
%     fid = fopen(modfile, 'wb');
%     fwrite(fid, zeros(sum(model.wta.blocksizes), 1), 'double');
%     fclose(fid);
%     
%     oridata = data;
%     [data, model.wta.Theta] = ...
%         wtahash(oridata, model.wta.k, model.wta.m, [], model.wta.iswta);
%     
% else
%     
%     fid = fopen(modfile, 'wb');
%     fwrite(fid, zeros(sum(model.blocksizes), 1), 'double');
%     fclose(fid);
% end

% ---
% try to find the bestC by cross-validation using different datfile
% --- cross validation
if k > 0
    [cvids, imname] = wl_cvIds(ids, labels, k);
end

bestap = -inf;
bestparams.c = [];
J = 1; % weight of positive example
for ci=1:length(cs)
    params.c = cs(ci);
    ap = 0;
    for ii=1:k
        
        % for each validation set
        % ONE BUG: the validation ap is jump very serious (maybe the reason
        % is data not enought)
        %bvalids = cvids{ii};
        btrainids = [];
        for jj=1:k
            if jj~=ii
                btrainids = [btrainids; cvids{jj}];
            end
        end
        
        % reset all
        resetall(tmphdrfile, tmpinffile, tmpmodfile, tmplobfile, ...
            labelsize, model);
        
        % set temp data for training
        num = data2dat(name, model, 1, tmpdatfile, data(btrainids,:), labels(btrainids));
        
        % write header
        writeheadermodel(tmpmodfile, tmphdrfile, num, labelsize, model);
        
        
        cmd = sprintf('./bin/learn %.4f %.4f %s %s %s %s %s', ...
            params.c, J, tmphdrfile, tmpdatfile, tmpmodfile, tmpinffile, tmplobfile);
        fprintf('executing: %s\n', cmd);
        status = unix(cmd);
        if status ~= 0
            fprintf('command `%s` failed\n', cmd);
            keyboard;
        end
        
        fprintf('parsing model\n');
        blocks = readmodel(tmpmodfile, model);
        model = parsemodel(model, blocks);
        [labelsii, vals, unique] = readinfo(tmpinffile);
        
        % compute threshold for high recall
        P = find((labelsii == 1) .* unique);
        pos_vals = sort(vals(P));
        model.thresh = pos_vals(ceil(length(pos_vals)*0.05));
        
        % --- perform detection on each image (not box)
        % pascal_eval is limit to evaluate 'train', 'trainval', or 'test'.
        % here we need to evaluate a subset of trainval, need to rewrite
        % the code.
        
        valimage = imname{ii};
        
        boxes = cell(length(valimage),1);
        tic;
        for i = 1:length(valimage)
            if toc > 60
                fprintf('%s: detect: %s %s, %d/%d\n', name, 'valtmp', VOCyear, ...
                    i, length(valimage));
                tic;
            end
            
            im = imread(sprintf(VOCopts.imgpath, valimage{i}));
            b = detect(im, model, model.thresh);
            if ~isempty(b)
                b1 = b(:,[1 2 3 4 end]);
                b1 = clipboxes(im, b1);
                boxes{i} = nms(b1, 0.5);
            else
                boxes{i} = [];
            end
        end
        
        
        % --- compute AP
        % write out detections in PASCAL format and score
        fid = fopen(sprintf(VOCopts.detrespath, 'valtmp',  name), 'w');
        for i = 1:length(valimage);
            bbox = boxes{i};
            for j = 1:size(bbox,1)
                fprintf(fid, '%s %f %d %d %d %d\n', valimage{i}, bbox(j,end), bbox(j,1:4));
            end
        end
        fclose(fid);
        
        % get AP
        [recall, prec, apii] = evaldet(VOCopts, 'valtmp', name, true, valimage);
        ap = ap + apii;
        fprintf('%s AP (part): %f (c: %f)\n', name, apii, params.c);
    end
    
    ap = ap / k;
    fprintf('%s AP: %f (c: %f)\n', name, ap, params.c);
    if ap > bestap
        bestap = ap;
        bestparams.c = params.c;
    end
end

% ---
% learn model
num = data2dat(name, model, 1, datfile, data, labels);
writeheadermodel(modfile, hdrfile, num, labelsize, model);


% Call the SVM learning code
cmd = sprintf('./bin/learn %.4f %.4f %s %s %s %s %s', ...
    bestparams.c, J, hdrfile, datfile, modfile, inffile, lobfile);
fprintf('executing: %s\n', cmd);
status = unix(cmd);
if status ~= 0
    fprintf('command `%s` failed\n', cmd);
    keyboard;
end

fprintf('parsing model\n');
blocks = readmodel(modfile, model);
model = parsemodel(model, blocks);
[labels, vals, unique] = readinfo(inffile);

% compute threshold for high recall
P = find((labels == 1) .* unique);
pos_vals = sort(vals(P));
model.thresh = pos_vals(ceil(length(pos_vals)*0.05));

% cache model
save([cachedir name '_model'], 'model');


function writeheadermodel(modfile, hdrfile, num, labelsize, model)
writeheader(hdrfile, num, labelsize, model);

% reset initial model
fid = fopen(modfile, 'wb');
if model.wta.iswta == 1
    fwrite(fid, zeros(sum(model.wta.blocksizes), 1), 'double');
else
    fwrite(fid, zeros(sum(model.blocksizes), 1), 'double');
end
fclose(fid);


function model = resetall(hdrfile, inffile, modfile, lobfile, labelsize, model)
% reinitialize a model here? No need.

% reset data file
%fid = fopen(datfile, 'wb');
%fclose(fid);

% reset header file
writeheader(hdrfile, 0, labelsize, model);
% reset info file
fid = fopen(inffile, 'w');
fclose(fid);
% reset initial model
fid = fopen(modfile, 'wb');
if model.wta.iswta == 1
    fwrite(fid, zeros(sum(model.wta.blocksizes), 1), 'double');
else
    fwrite(fid, zeros(sum(model.blocksizes), 1), 'double');
end
fclose(fid);
% reset lower bounds
writelob(lobfile, model)

% get positive examples by warping positive bounding boxes
% we create virtual examples by flipping each image left to right
function numdata = data2dat(name, model, c, datfile, data, labels)
% reset datfile
fid = fopen(datfile, 'wb');
fclose(fid);

fid = fopen(datfile, 'w');

numdata = size(data,1);
ridx = model.components{c}.rootindex;
oidx = model.components{c}.offsetindex;
rblocklabel = model.rootfilters{ridx}.blocklabel;
oblocklabel = model.offsets{oidx}.blocklabel;
dim = model.components{c}.dim;
width1 = ceil(model.rootfilters{ridx}.size(2)/2);

tic;
for i = 1:numdata
    if toc>1
        fprintf('%s: data2dat: %d/%d\n', name, i, numdata);
        tic;
    end    
    % get example
    %feat = reshape(data(i, :), [model.rootfilters{ridx}.size(1), width1, 31]);
    feat = data(i, :);
    fwrite(fid, [labels(i) i 0 0 0 2 dim], 'int32');
    fwrite(fid, [oblocklabel 1], 'single');
    fwrite(fid, rblocklabel, 'single');
    fwrite(fid, feat, 'single');       
end
fclose(fid);

function [cvIds, imname] = wl_cvIds(ids, labels, k)
% wl_cvIds() will partition the labels into k parts with equally
% number of images
% Input:
%	ids: the image name for all the features
%	labels: the label for all the features
%	k: the number of partitions
%
% Output:
%   trai

% step 1: hash the image names
hash = VOChash_init(ids);

% step 2: get the unique name of the images
imgNames = unique(ids);

% step 2.1: get the positive image names
posImgNames = unique(ids(labels==1));
nPosImgs = length(posImgNames);

% step 2.2: get the negative image names
negImgNames = setdiff(imgNames, posImgNames);
nNegImgs = length(negImgNames);

% step 3: randomly split the positive image names
if nPosImgs ~= 0
    % step 3.1: randomly permute the postive image names
    posImgNames = posImgNames(randperm(nPosImgs));
    % step 3.2: split the image names into k parts
    n = floor(nPosImgs/k);
    count = 0;
    i = 1;
    cvIds{i} = [];
    imname{i} = [];
    for d=1:nPosImgs
        imgName = posImgNames{d};
        count = count + 1;
        % step 4.1: get the indices for the image name
        idx = VOChash_lookup(hash, imgName);
        if count < n || i==k
            cvIds{i} = [cvIds{i}; idx'];
            imname{i} = [imname{i}; {imgName}];
        else
            cvIds{i} = [cvIds{i}; idx'];
            imname{i} = [imname{i}; {imgName}];
            i = i+1;
            count = 0;
            cvIds{i} = [];
            imname{i} = [];
        end
    end
end

% step 4: randomly split the negative image names
if nNegImgs ~= 0
    % step 4.1: randomly permute the negative image names
    negImgNames = negImgNames(randperm(nNegImgs));
    % step 3.2: split the image names into k parts
    n = floor(nNegImgs/k);
    count = 0;
    i = 1;
    for d=1:nNegImgs
        imgName = negImgNames{d};
        count = count + 1;
        % step 4.1: get the indices for the image name
        idx = VOChash_lookup(hash, imgName);
        if count < n || i==k
            cvIds{i} = [cvIds{i}; idx'];
            imname{i} = [imname{i}; {imgName}];
        else
            cvIds{i} = [cvIds{i}; idx'];
            imname{i} = [imname{i}; {imgName}];
            i = i+1;
            count = 0;
        end
    end
end
