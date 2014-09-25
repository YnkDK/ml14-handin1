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


% Set initial weights (d+1) vector
numOfClasses=2;
theta = rand(size(X,2), numOfClasses);
yy = zeros(size(y,1),numOfClasses);

yy(:) = [double(( y(:) == 2 )),double(( y(:) ~=2))];


% % Clear unused variable
clear s;

% % Run
fprintf('started at: %s\n', datestr(clock, 0));
%best = logRun(X, y, theta);
%runLinear();
softThetaMnist = softRun(X,yy,theta);

fprintf('ended at  : %s\n', datestr(clock, 0));
save('softBestParams.mat', 'softThetaMnist')

% Load training file
s = load('../dat/mnistTest.mat');
% Set input matrix
%X = s.au_train_digits;
X= s.images;
% Add bias term
X = [ones(size(X,1),1) X];
% Set target vector
y = s.labels;
%y = s.au_train_labels;
yy = zeros(size(y,1),numOfClasses);

yy(:) = [double(( y(:) == 2 )),double(( y(:) ~=2))];

% Set initial weights (d+1) vector
theta = rand(size(X,2), numOfClasses);
correct = 0;

for i=1:length(y)
   xi = X(i, :)';
   %pred = 1/(1+exp(-best'*xi));
   %if(max(pred)== pred(y(i)+1))
   M = exp(softThetaMnist' * xi);
   M = bsxfun(@rdivide, M, sum(M));
   [~,pred] = max(M, [], 1);
   [~,yRes] = max(yy(i), [], 1);
   if( yy(i,pred)==1)
       correct = correct +1;
   end
end

fprintf('Correct %f  | (num correct: %d | of : %d) | Wrong: %f\n', 100*correct/length(y),correct,length(y), 100-(100*correct/length(y)));

% % % %

% % Prepare input data

% Load training file
s = load('../dat/auTrain.mat');
% Set input matrix
%X = s.au_train_digits;
X= s.au_train_digits;
% Add bias term
X = [ones(size(X,1),1) X];
% Set target vector
%y = s.au_train_labels;
y = s.au_train_labels;


% Set initial weights (d+1) vector
numOfClasses=2;
theta = rand(size(X,2), numOfClasses);
yy = zeros(size(y,1),numOfClasses);

yy(:) = [double(( y(:) == 2 )),double(( y(:) ~=2))];


% % Clear unused variable
clear s;

% % Run
fprintf('started at: %s\n', datestr(clock, 0));
%best = logRun(X, y, theta);
%runLinear();
softThetaAu = softRun(X,yy,theta);

fprintf('ended at  : %s\n', datestr(clock, 0));
save('softBestParams.mat', 'softThetaMnist', '-append')

% Load training file
s = load('../dat/auTrain.mat');
% Set input matrix
%X = s.au_train_digits;
X= s.au_train_digits;
% Add bias term
X = [ones(size(X,1),1) X];
% Set target vector
y = s.au_train_labels;
%y = s.au_train_labels;
yy = zeros(size(y,1),numOfClasses);

yy(:) = [double(( y(:) == 2 )),double(( y(:) ~=2))];

% Set initial weights (d+1) vector
theta = rand(size(X,2), numOfClasses);
correct = 0;

for i=1:length(y)
   xi = X(i, :)';
   %pred = 1/(1+exp(-best'*xi));
   %if(max(pred)== pred(y(i)+1))
   M = exp(softThetaAu' * xi);
   M = bsxfun(@rdivide, M, sum(M));
   [~,pred] = max(M, [], 1);
   [~,yRes] = max(yy(i), [], 1);
   if( yy(i,pred)==1)
       correct = correct +1;
   end
end

fprintf('Correct %f  | (num correct: %d | of : %d) | Wrong: %f\n', 100*correct/length(y),correct,length(y), 100-(100*correct/length(y)));

