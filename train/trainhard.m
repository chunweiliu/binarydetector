function [model, ap] = trainhard(name, model, pos, neg)

% [model, ap] = trainroot(name, model, pos, neg)
% Given the initial model, positive and negative examples, 
% train the root filter 

globals;
pascal_init;

% setting
nrelabel = 2;
ndatamine = 2;
maxsize = 2^28;
params.c = 1;

% define constant
npos = length(pos);
nneg = length(neg);
fsize = prod([model.rootfilters{1}.size 31]);
dim = 0;
for i = 1:model.numcomponents
    dim = max(dim, model.components{i}.dim);
end
maxnum = floor(maxsize / (dim * 4));

hardneg = [];
for relabel = 1:nrelabel
    
    newpos = pos;
    for i = 1:npos
        newpos(i) = detectbest(VOCopts, model, pos(i));
    end
    [posdata, posids, numpos] = poswarp(name, model, 1, newpos);
    
    % add hard negative
    for datamine = 1:ndatamine
        
        for j = 1:length(neg)
            if nneg > maxnum, break; end
            fprintf('%s: datamine: %d/%d\n', name, j, length(neg));
            [newneg, nnewneg] = detectall(VOCopts, model, neg(j));
            
            hardneg = [hardneg newneg];
            nneg = nneg + nnewneg;
        end
        
        [negdata, negids, numneg] = poswarp(name, model, 1, hardneg);
        
        data = [posdata negdata]';
        labels = [ones(numpos,1); -ones(numneg,1)];
        ids = [posids; negids];
        
        model = trainlinear(labels, data, params, model);
    end
end


function newpos = detectbest(VOCopts, model, pos)

im = imread(sprintf(VOCopts.imgpath, pos.id));

b = detect(im, model, model.thresh);
if ~isempty(b)
    b1 = b(:,[1 2 3 4 end]);
    b1 = clipboxes(im, b1);
    boxes = nms(b1, 0.5);
    box = boxes(1,:);
    
    % add a new positive example
    newpos.im = pos.im;
    newpos.x1 = box(1);
    newpos.y1 = box(2);
    newpos.x2 = box(3);
    newpos.y2 = box(4);
    newpos.id = pos.id;
else
    newpos = pos;
end

function [newneg, nnew] = detectall(VOCopts, model, neg)

im = imread(sprintf(VOCopts.imgpath, neg.id));

b = detect(im, model, model.thresh);
if ~isempty(b)
    b1 = b(:,[1 2 3 4 end]);
    b1 = clipboxes(im, b1);
    boxes = nms(b1, 0.5);
    
    % add all false positive examples
    nnew = size(boxes,1);
    for i = 1:nnew
        newneg(i).im = neg.im;
        newneg(i).x1 = boxes(i,1);
        newneg(i).y1 = boxes(i,2);
        newneg(i).x2 = boxes(i,3);
        newneg(i).y2 = boxes(i,4);
        newneg(i).id = neg.id;
    end

else
    % if this negative image cannot detect any hypothesis, let it go
    newneg = [];
    nnew = 0;
end