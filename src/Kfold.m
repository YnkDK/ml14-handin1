function parameters = Kfold(X, y, k)
    n = size(X, 1);
    % Permute X and y
    idx = randperm(n);
    X = X(idx, :);
    y = y(idx);    
    
    % Training size
    idx = 1:n;
    ts = round(n/k);
    
    % Best pctCorrect
    bestCorrect = 0.0;
    
    for i = 1:k
        % Indices for the validation set
        validIdx = (i*ts - ts) + 1 : i*ts;
        % Indices for the training set
        trainIdx = setdiff(idx, validIdx);
        % Split X and y according to the indices
        trainX = X(trainIdx, :);
        trainY = y(trainIdx);
        validX = X(validIdx, :);
        validY = y(validIdx);
        % Start at some random point        
        theta = rand(size(X,2), 1);
        % Run the training algorithm
        best = logRun(trainX, trainY, theta);
        % Count the number of correct classifications
        % on the valdiation set using the best parameters
        correct = 0;
        for j=1:length(validY)
           xi = validX(j, :)';
           pred = 1/(1+exp(-best'*xi));
           if validY(j) == 1 && pred >= 0.5
               correct = correct + 1;
           elseif validY(j) == 0 && pred < 0.5
               correct = correct + 1;
           end
        end
        % Calculate the percentage of correct classifications
        correct = correct/length(validY);
        fprintf('Percentage correct: %f\n', correct);
        % Save the parameters if we did better than previous
        if correct > bestCorrect
            bestCorrect = correct;
            parameters = best;
        end      
    end
end