% get the baseline of the root filter for all classes
globals;
pascal_init;

aps = zeros(VOCopts.nclasses,1);
for i=1:VOCopts.nclasses
    cls = VOCopts.classes{i};
    if strcmp(cls,'bottle')
        continue;
    end
    aps(i) = pascal(cls);
end