function model = trainlinear(labels, data, params, model)

% linearmodel = trainlinear(labels, data, params)
% try to train the linear model

% setting
svmop = '-s 3 -c %f -w1 %f -w-1 1 -B 1';
    
% train on the subset
w1 = sqrt(sum(labels~=1)/sum(labels==1));
op = sprintf(svmop, params.c, w1);
linearmodel = train(labels, sparse(data), op);

% predict all
op = sprintf('-b 0');
[~,~,vals] = predict(labels, sparse(data), linearmodel, op);


% compute threshold for high recall
uniqueidx = zeros(size(vals,1),1);
[~,ia,~] = unique(vals);
uniqueidx(ia) = 1;
P = find((labels == 1) .* uniqueidx);
pos_vals = sort(vals(P));
model.thresh = pos_vals(ceil(length(pos_vals)*0.05));

% update the filter of model
blocks{1} = linearmodel.w(end);
blocks{2} = linearmodel.w(1:end-1);
model = parsemodel(model, blocks);


%model.rootfilters{1}.w = reshape(linearmodel.w(1:end-1),...
%    size(model.rootfilters{1}.w));
%model.offsets{1}.w = linearmodel.w(end);
