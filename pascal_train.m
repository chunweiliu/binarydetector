function model = pascal_train(cls)%, n)

% model = pascal_train(cls)
% Train a model using the PASCAL dataset.

globals; 
[pos, neg] = pascal_data(cls);

% train root filter using warped positives & random negatives
try
  load([cachedir cls '_random']);
catch
  model = initmodel(pos);
  model = train(cls, model, pos, neg);
  %model = trainroot(cls, model, pos, neg);
  %model = trainhard(cls, model, pos, neg);
  save([cachedir cls '_random'], 'model');
end

% PUT YOUR CODE HERE
% TODO: Train the rest of the DPM (latent root position, part filters, ...)
