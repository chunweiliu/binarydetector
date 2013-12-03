function model = pascal_train(cls)%, n)

% model = pascal_train(cls)
% Train a model using the PASCAL dataset.

globals; 
[pos, neg] = pascal_data(cls);

trainop;

% train root filter using warped positives & random negatives
try
  load([cachedir cls '_random']);
catch
  model = initmodel(pos, wta);
  switch trainmode
      case 'train'
        model = train(cls, model, pos, neg, Cs, k);
      case 'trainroot'
        model = trainroot(cls, model, pos, neg, Cs, k);
      case 'trainhard'  
        model = trainhard(cls, model, pos, neg, Cs, k);
  end
  save([cachedir cls '_random'], 'model');
end

% PUT YOUR CODE HERE
% TODO: Train the rest of the DPM (latent root position, part filters, ...)
