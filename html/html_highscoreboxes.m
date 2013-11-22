function html_highscoreboxes(cls, testset, year, k, istop)
% Top 100 results of detection
highscoreboxes(cls, testset, year, k, istop);

% Write an html to display the images
imwidth = 200;

% open a file
filename = sprintf('html/%s_%s_%s_%d_%d.html', cls, testset, year, k, istop);
fid = fopen(filename, 'w');

% create the html header
line = sprintf(['<!DOCTYPE html>\n',...
    '<html>\n',...
    '<head>\n',...
    '<title>Top 100</title>\n',...
    '</head>\n',...
    '<body>\n']);
fprintf(fid, line);

% create title
line = sprintf('<h1>Top %d examples in "%s%s"</h1>\n', k, cls, year);
fprintf(fid, line);

% plot images
for i = 1:k
    line = sprintf('<img src="images/%s_%s_%s_%03d.png" width="%d">\n',...
        cls, testset, year, i, imwidth);
    fprintf(fid, line);
end

% create the end
line = sprintf(['</body>\n',...
    '</html>\n']);
fprintf(fid, line);

fclose(fid);