function [obj, reg, lp, ln] = evalobj(labels, data, model, params)

% a = evalobj(labels, data, model, params)
% 0.5 * |W|^2 + C_pos * \sum_{i in pos} max(0, 1-y_i*<w, x_i>) + ...
%               C_neg * \sum_{i in neg} max(0, 1-y_i*<w, x_i>)

w = model.rootfilters{1}.w;
w = w(:,1:ceil(model.rootfilters{1}.size(2)/2),:);
b = model.offsets{1}.w;

y = labels;
x = data;
w1 = sqrt(sum(labels~=1)/sum(labels==1));
ip = find(labels==1);
in = find(labels~=1);

lp = 0;
for i=1:length(ip)
    lp = lp + w1 * max(0, 1 - y(ip(i)) * (x(ip(i),:)*w(:)) + b);
end
lp = lp * params.c;

ln = 0;
for i=1:length(in)
    ln = ln + max(0, 1 - y(in(i)) * (x(in(i),:)*w(:)) + b);
end
ln = ln * params.c;

reg = 0.5 * (norm(w(:),2) + b^2);

obj = lp + ln + reg;