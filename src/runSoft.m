% % Prepare input data

% Load training file
s = load('../dat/mnistTrain.mat');
% Set input matrix
%X = s.au_train_digits;
X= s.images;
% Add bias term
X = [ones(size(X,1),1) X];
% Set target vector
%y = s.au_train_labels;
y = s.labels;
t = 2;
y = double(( y(:) == t) + (y(:) == 5 ));

% Set initial weights (d+1) vector
theta = rand(size(X,2), 1);

% % Clear unused variable
clear s;

% % Run
fprintf('started at: %s\n', datestr(clock, 0));
%best = logRun(X, y, theta);
%runLinear();
best = softRun(X,y,theta);

fprintf('ended at  : %s\n', datestr(clock, 0));


% Load training file
s = load('../dat/mnistTest.mat');
% Set input matrix
%X = s.au_train_digits;
X= s.images;
% Add bias term
X = [ones(size(X,1),1) X];
% Set target vector
y = s.labels;
t = 2;
y = double(( y(:) == t) + (y(:) == 5 ));
%y = s.au_train_labels;



% Set initial weights (d+1) vector
theta = rand(size(X,2), numOfClasses);
correct = 0;

for i=1:length(y)
   xi = X(i, :)';
   %pred = 1/(1+exp(-best'*xi));
   %if(max(pred)== pred(y(i)+1))
   M = exp(best' * xi);
   M = bsxfun(@rdivide, M, sum(M));
   [~,pred] = max(M, [], 1);
   if( pred==(y(i)+1))
       correct = correct +1;
   else
      asd =   2+2;
   end
end

fprintf('Correct %f  | (num correct: %d | of : %d) | Wrong: %f\n', 100*correct/length(y),correct,length(y), 100-(100*correct/length(y)));



