function model = trainsvm(labels, data, params, model)

% linearmodel = trainlinear(labels, data, params)
% try to train the linear model

% setting
svmop = '-t 0 -c %f -w1 %f -w-1 %f -h 0'; % svm

    
% train on the subset
wp = sqrt(sum(labels~=1)/sum(labels==1));
wn = 1;

op = sprintf(svmop, params.c, wp, wn);
linearmodel = svmtrain(labels, data, op);


% predict all
op = sprintf('-b 0');
[~,~,vals] = svmpredict(labels, data, linearmodel, op);


% compute threshold for high recall
uniqueidx = zeros(size(vals,1),1);
[~,ia,~] = unique(vals);
uniqueidx(ia) = 1;
P = find((labels == 1) .* uniqueidx);
pos_vals = sort(vals(P));
model.thresh = pos_vals(ceil(length(pos_vals)*0.05));

% update the filter of model
w = linearmodel.SVs' * linearmodel.sv_coef;
b = -linearmodel.rho;

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
