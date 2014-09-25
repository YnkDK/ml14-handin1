% % Prepare input data

% Load training file
s = load('../dat/auTrain.mat');
% Set input matrix
X = s.au_train_digits;
% Add bias term
X = [ones(size(X,1),1) X];
% Set target vector
y = s.au_train_labels;
% % Make it a binary classification
targets = unique(y);
% Initialize (d + 1) x K result
logThetaAu = nan(size(X, 2), numel(targets));

for i = 1:numel(targets)
   t = targets(i);
   fprintf('%d vs all in progress!\n', t);
   % All vs one
   yPrime = double(( y(:) == t ));
   % Save the k'th best parameters
   logThetaAu(:,t+1) = Kfold(X, yPrime, 10);
end
save('logBestParams.mat', 'logThetaAu', '-append')