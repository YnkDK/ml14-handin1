load('logBestParams.mat', 'logThetaMnist');
s = load('../dat/mnistTest.mat');
y = s.labels;
X = s.images;
% Add bias term
X = [ones(size(X,1),1) X];
guess = zeros(size(logThetaMnist, 2), 2);

for j=1:length(y)
   xi = X(j, :)';
   bestPred = 0.0;
   bestTarget = -1;
   for i=1:size(logThetaMnist, 2)
      theta = logThetaMnist(:, i);
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

fprintf('Number of correct predictions on mnistTest: %f', 100*sum(guess(:,1))/sum(guess(:)));