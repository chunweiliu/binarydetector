function [ rootmatch ] = wtaconv( feat, rootsize, model )
%WTACONV Summary of this function goes here
%   Detailed explanation goes here

wta = model.wta;

featr = size(feat,1); % size of feature map
featc = size(feat,2);
r = rootsize(1); % size of filter
c = rootsize(2);

nr = featr-r+1;
nc = featc-c+1;

rootmatch = zeros(nr, nc);
for j = 1:nc
    for i = 1:nr
        f = feat(i:i+r-1, j:j+c-1, :);
        f = reshape(f, 1, []);
        d = wtahash(f, wta.k, wta.m, wta.Theta, wta.iswta);
        %wta.w = reshape(wta.w, 1, []);
        rootmatch(i,j) = d*wta.w';
    end
end

rootmatch = {rootmatch};

end

