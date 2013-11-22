function resp = wtaconv(feat, filt, wta)

% resp = wtaconv(feat, filt)
% compute filter respond using hamming distance

x1 = wta.w;

nr = size(feat,1);
nc = size(feat,2);
fr = size(filt,1);
fc = size(filt,2);

resp = zeros(nr-fr+1, nc-fc+1);
for i = 1:nc-fc+1
    for j = 1:nr-fr+1
        x2 = reshape(feat(i:i+fr,j:j+fc,:), 1, []);
        x2 = wtahash(x2, wta.params.k, wta.params.m, wta.params.Theta, wta.params.iswta);
        
        % which response is better? Hamming dist, l2 norm?
        %resp(i,j) = sum(x1~=x2);
        resp(i,j) = norm(x1-x2);
    end
end


end

