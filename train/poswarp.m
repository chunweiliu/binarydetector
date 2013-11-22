function [data, ids, num]= poswarp(name, model, c, pos)
% get positive examples by warping positive bounding boxes
% we create virtual examples by flipping each image left to right
numpos = length(pos);
warped = warppos(name, model, c, pos);
ridx = model.components{c}.rootindex;

width1 = ceil(model.rootfilters{ridx}.size(2)/2);
width2 = floor(model.rootfilters{ridx}.size(2)/2);

pixels = model.rootfilters{ridx}.size * model.sbin;
minsize = prod(pixels);
num = 0;

data = zeros(prod([model.rootfilters{ridx}.size(1) width1 31]), 2*numpos); % column major faster
ids = cell(2*numpos, 1);
tic;
for i = 1:numpos
    if toc > 1
        fprintf('%s: warped positive: %d/%d\n', name, i, numpos);
        tic;
    end
    bbox = [pos(i).x1 pos(i).y1 pos(i).x2 pos(i).y2];
    % skip small examples
    if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
      continue
    end    
    % get example
    im = warped{i};
    feat = features(im, model.sbin); % size(feat)=(size(im)/8)-2
    feat(:,1:width2,:) = feat(:,1:width2,:) + flipfeat(feat(:,width1+1:end,:));
    feat = feat(:,1:width1,:);
    data(:,1+num) = feat(:);
    ids(1+num) = {pos(i).id};
    
    % get flipped example
    feat = features(im(:,end:-1:1,:), model.sbin); 
    feat(:,1:width2,:) = feat(:,1:width2,:) + flipfeat(feat(:,width1+1:end,:));
    feat = feat(:,1:width1,:);
    data(:,2+num) = feat(:);
    ids(2+num) = {pos(i).id};
    
    num = num+2;    
end
data = data(:,1:num);
ids = ids(1:num);
