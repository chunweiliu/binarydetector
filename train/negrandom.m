function [data, ids, num] = negrandom(name, model, c, neg, maxnum)
% get random negative examples
numneg = length(neg);
rndneg = floor(maxnum/numneg);
ridx = model.components{c}.rootindex;
rsize = model.rootfilters{ridx}.size;
width1 = ceil(rsize(2)/2);
width2 = floor(rsize(2)/2);

num = 0;

data = zeros(prod([model.rootfilters{ridx}.size(1) width1 31]), numneg*rndneg);
ids = cell(numneg*rndneg,1);
tic;
for i = 1:numneg
  if toc > 1
    fprintf('%s: random negatives: %d/%d\n', name, i, numneg);
    tic;
  end
  im = color(imread(neg(i).im));
  feat = features(double(im), model.sbin);  
  if size(feat,2) > rsize(2) && size(feat,1) > rsize(1)
    for j = 1:rndneg
      x = random('unid', size(feat,2)-rsize(2)+1);
      y = random('unid', size(feat,1)-rsize(1)+1);
      f = feat(y:y+rsize(1)-1, x:x+rsize(2)-1,:);
      f(:,1:width2,:) = f(:,1:width2,:) + flipfeat(f(:,width1+1:end,:));
      f = f(:,1:width1,:);
      data(:,rndneg*(i-1)+j) = f(:);
      ids(rndneg*(i-1)+j) = {neg(i).id};
    end
    num = num+rndneg;
  end
end