function highscoreboxes(cls, testset, suffix, k, istop)

% boxes = highscoresboxes(boxes, ids, k)
% return top k score boxes

% load ids and boxes
globals;
pascal_init;
ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');
load([cachedir cls '_boxes_' testset '_' suffix]);

nim = length(boxes);
allboxes = [];
boxesid = [];
for i = 1:nim
    allboxes = [allboxes; boxes{i}];
    boxesid = [boxesid; i*ones(size(boxes{i},1),1)];
end

if istop == 1
    [sortedscores, sortedidx] = sort(allboxes(:,5),'descend');
else
    [sortedscores, sortedidx] = sort(allboxes(:,5));
end
topscores = sortedscores(1:k);
topboxes = allboxes(sortedidx(1:k),:);
topboxesids = boxesid(sortedidx(1:k));

for i = 1:k
    figure(i)
    im = imread(sprintf(VOCopts.imgpath, ids{topboxesids(i)}));
    showboxes(im, topboxes(i,:));
    
    text(topboxes(i,1)+2,topboxes(i,2)+9,num2str(topscores(i)),...
        'BackgroundColor','red');
end

end

