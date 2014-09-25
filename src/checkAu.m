load('logBestParams.mat', 'logThetaAu');
s = load('../dat/auTrain.mat');
y = s.au_train_labels;
X = s.au_train_digits;
% Add bias term
X = [ones(size(X,1),1) X];
guess = zeros(size(logThetaAu, 2), 2);

for j=1:length(y)
   xi = X(j, :)';
   bestPred = 0.0;
   bestTarget = -1;
   for i=1:size(logThetaAu, 2)
      theta = logThetaAu(:, i);
      pred = 1/(1+exp(-theta'*xi));
      if bestPred < pred
          bestPred = pred;
          bestTarget = i-1;
      end
   end
   
   if y(j) == bestTarget
       guess((y(j) + 1), 1) = guess((y(j) + 1), 1) + 1;
   else
       guess((y(j) + 1), 2) = guess((y(j) + 1), 2) + 1;
   end
end

fprintf('Number of correct predictions on auTest: %f', 100*sum(guess(:,1))/sum(guess(:)));