function model = train(name, model, pos, neg)

% model = train(name, model, pos, neg)
% Train LSVM. (For now it's just an SVM)
% 

% SVM learning parameters
%C = 0.002*model.numcomponents;
%J = 1;

maxsize = 2^28;

globals;
pascal_init;

% approximate bound on the number of examples used in each iteration
dim = 0;
for i = 1:model.numcomponents
  dim = max(dim, model.components{i}.dim);
end
maxnum = floor(maxsize / (dim * 4));


% Find the positive examples and save them in the data file
[posdata, posids, numpos] = poswarp(name, model, 1, pos);

% Add random negatives
[negdata, negids, numneg] = negrandom(name, model, 1, neg, maxnum-numpos);

data = [posdata negdata]';
labels = [ones(numpos,1); -ones(numneg,1)];
ids = [posids; negids];


        
% Call the SVM learning code
% --- cross validation
k = 3;
cvids = wl_cvIds(ids, labels, k);

bestparams.c = [];

cs = [0.01 0.1 1 10 100];
for ci=1:length(cs)
    c = cs(ci);
    for ii=1:k
        
        % for each validation set
        valids = cvids{ii};
        trainids = [];
        for jj=1:k
            if jj~=i
                trainids = [trainids; cvids{jj}];
            end
        end
        
        % liblinear train
        w1 = sqrt(sum(labels(trainids)~=1)/sum(labels(trainids)==1));
        op = sprintf('-s 3 -c %f -w1 %f -w-1 1 -B 1', c, w1);
        linearmodel = lineartrain(labels(trainids), sparse(data(trainids,:)), op);
        
        % liblinear predict
        op = sprintf('-b 0');
        [~,~,vals] = linearpredict(labels(valids), sparse(data(valids,:)), linearmodel, op);
        unique = ones(size(vals,1),1);
        
        % --- update model.rootfilters{1}.w
        % compute threshold for high recall
        P = find((labels(valids) == 1) .* unique);
        pos_vals = sort(vals(P));
        model.thresh = pos_vals(ceil(length(pos_vals)*0.05));
        model.rootfilters{1}.w = reshape(linearmodel.w(1:end-1)...
            ,size(model.rootfilters{1}.w));
        
        
        % --- perform detection
        % pascal_eval is limit to evaluate 'train', 'trainval', or 'test'.
        % here we need to evaluate a subset of trainval, need to rewrite
        % the code.
        boxes = cell(length(valids),1);
        for i = 1:length(valids)
            fprintf('%s: detect for val: %s %s, %d/%d\n', cls, 'valtmp', VOCyear, ...
                i, length(valids));
            im = imread(sprintf(VOCopts.imgpath, ids{valids(i)}));  
            b = detect(im, model, model.thresh); %need nmx...
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
        fid = fopen(sprintf(VOCopts.detrespath, 'valtmp', cls), 'w');
        for i = 1:length(valids);
            bbox = boxes{i};
            for j = 1:size(bbox,1)
                fprintf(fid, '%s %f %d %d %d %d\n', ids{valids(i)}, bbox(j,end), bbox(j,1:4));
            end
        end
        fclose(fid);
        % get AP
        [recall, prec, ap] = evaldet(VOCopts, 'valtmp', cls, true, ids(valids));
        fprintf('%s AP: %f (c: %f)\n', cls, ap, c);
        if ap > bestap
            bestap = ap;
            bestparams.c = c;
        end
      
    end
end



    
% --- get labels, vals (w*x+b), unique (all ones?)
fprintf('parsing model\n');
%blocks = readmodel(modfile, model);
%model = parsemodel(model, blocks);
%[labels, vals, unique] = readinfo(inffile);
    



% --- compute AP
%  for each val image
%  b = detect(im, model, thresh)

%  because pascal_test and pascal_eavl's id is one-to-one map
%  they don't need to remeber the id, but we need to know the
%  id information in val set

% ---------------------------------


% cache model
save([cachedir name '_model'], 'model');


% get positive examples by warping positive bounding boxes
% we create virtual examples by flipping each image left to right
function [data, ids, num]= poswarp(name, model, c, pos)
numpos = length(pos);
warped = warppos(name, model, c, pos);
ridx = model.components{c}.rootindex;
pixels = model.rootfilters{ridx}.size * model.sbin;
minsize = prod(pixels);
num = 0;
data = zeros(numel(model.rootfilters{1}.w), 2*numpos); % column major faster
ids = cell(2*numpos, 1);
for i = 1:numpos
    if mod(i,100)==0
        fprintf('%s: warped positive: %d/%d\n', name, i, numpos);
    end
    bbox = [pos(i).x1 pos(i).y1 pos(i).x2 pos(i).y2];
    % skip small examples
    if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
      continue
    end    
    % get example
    im = warped{i};
    feat = features(im, model.sbin); % size(feat)=(size(im)/8)-2
    data(:,1+num) = feat(:);
    ids(1+num) = {pos(i).id};
    
    % get flipped example
    feat = features(im(:,end:-1:1,:), model.sbin);    
    data(:,2+num) = feat(:);
    ids(2+num) = {pos(i).id};
    
    num = num+2;    
end
data = data(:,1:num);
ids = ids(1:num);

% get random negative examples
function [data, ids, num] = negrandom(name, model, c, neg, maxnum)
numneg = length(neg);
rndneg = floor(maxnum/numneg);
ridx = model.components{c}.rootindex;
rsize = model.rootfilters{ridx}.size;
num = 0;
data = zeros(numel(model.rootfilters{1}.w), numneg*rndneg);
ids = cell(numneg*rndneg,1);
for i = 1:numneg
  if mod(i,100)==0
    fprintf('%s: random negatives: %d/%d\n', name, i, numneg);
  end
  im = color(imread(neg(i).im));
  feat = features(double(im), model.sbin);  
  if size(feat,2) > rsize(2) && size(feat,1) > rsize(1)
    for j = 1:rndneg
      x = random('unid', size(feat,2)-rsize(2)+1);
      y = random('unid', size(feat,1)-rsize(1)+1);
      f = feat(y:y+rsize(1)-1, x:x+rsize(2)-1,:);
      data(:,rndneg*(i-1)+j) = f(:);
      ids(rndneg*(i-1)+j) = {neg(i).id};
    end
    num = num+rndneg;
  end
end

function cvIds = wl_cvIds(ids, labels, k)
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
    for d=1:nPosImgs
        imgName = posImgNames{d};
        count = count + 1;
        % step 4.1: get the indices for the image name
        idx = VOChash_lookup(hash, imgName);
        if count < n || i==k
            cvIds{i} = [cvIds{i}; idx'];
        else
            cvIds{i} = [cvIds{i}; idx'];
            i = i+1;
            count = 0;
            cvIds{i} = [];
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
        else
            cvIds{i} = [cvIds{i}; idx'];
            i = i+1;
            count = 0;
        end
    end
end