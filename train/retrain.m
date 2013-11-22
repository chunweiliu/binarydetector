function model = retrain(labels, data, params, model)

% linearmodel = retrain(labels, data, params)
% try to train the linear model using the hard examples

% setting
svmop = '-s 3 -c %f -w1 %f -w-1 1 -B 1';
ndatamine = 1;
initratio = 1;

% train the model using the subset of data in the first time
pos = find(labels == 1);
neg = find(labels ~= 1);
sub = [pos; randsample(neg, initratio*length(pos))];

for datamine = 1:ndatamine
    
    % train on the subset
    w1 = sqrt(sum(labels(sub)~=1)/sum(labels(sub)==1));
    op = sprintf(svmop, params.c, w1);
    linearmodel = train(labels(sub), sparse(data(sub,:)), op);
    
    % predict all
    op = sprintf('-b 0');
    [~,~,vals] = predict(labels(neg), sparse(data(neg,:)), linearmodel, op);
    
    % distinguish the hard and the easy examples
    %% BUG you have to prevent the good examples
    hard = find(labels.*vals < 1 | labels == 1);
    easy = find(labels.*vals > 1 & labels ~= 1); % always keep positve
    
    % if the hard set belong to current subset then return the model
    if sum(ismember(hard, sub)) == length(hard)
        break
    end
    
    % remove the easy examples from the negative set
    sub(ismember(sub, easy)) = [];
    
    % add the hard examples which are not include in current subset
    sub = [sub; hard(~ismember(hard, sub))];
    
end

% compute threshold for high recall
uniqueidx = zeros(size(vals,1),1);
[~,ia,~] = unique(vals);
uniqueidx(ia) = 1;
P = find((labels == 1) .* uniqueidx);
pos_vals = sort(vals(P));

% update model
model.thresh = pos_vals(ceil(length(pos_vals)*0.05));
model.rootfilters{1}.w = reshape(linearmodel.w(1:end-1),...
    size(model.rootfilters{1}.w));
model.offsets{1}.w = linearmodel.w(end);
