function model = trainlinear(labels, data, params, model)

% linearmodel = trainlinear(labels, data, params)
% try to train the linear model

% setting
svmop = '-s 3 -c %f -w1 %f -w-1 %f -B 1';
    
% train on the subset
wp = sqrt(sum(labels~=1)/sum(labels==1));
wn = 1;

% nw = ones(length(labels), 1);
% nw(labels==1) = w1;
% nw = nw / norm(nw, 2);
% 
% a = unique(nw);
% wp = max(a);
% wn = min(a);



op = sprintf(svmop, params.c, wp, wn);
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
w = linearmodel.w(1:end-1);
b = linearmodel.w(end);

if linearmodel.Label(1) == -1
  w = -w;
  b = -b;
end

blocks{1} = b;
blocks{2} = w;
model = parsemodel(model, blocks);


%model.rootfilters{1}.w = reshape(linearmodel.w(1:end-1),...
%    size(model.rootfilters{1}.w));
%model.offsets{1}.w = linearmodel.w(end);
