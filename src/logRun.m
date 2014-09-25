function [bestTheta] = logRun(X, y, theta)
    % Stop conditions
    conditionVal = 0.5;
    maxIter = 500;
    
    % Initialize the result
    bestCost = Inf;
    bestTheta = theta;
    
    % Initialize states   
    cost_current = 0;
    cost_prev = logCost(X, y, theta);
    theta_current = theta;
    
    
    i = 0;
    while i < maxIter && (cost_current > 10000 || i < 10 || abs(cost_prev - cost_current) > conditionVal)
        cost_prev = cost_current;
        theta_prev = theta_current;
        % Get cost and grad from previous step
        [cost_current, grad] = logCost(X,y,theta_prev);
        if cost_current < bestCost
            bestCost = cost_current;
            bestTheta = theta_prev;
        end
        % Step direction is the unit vector opposite of 
        % the gradient
        %direction = -(grad/norm(grad));
        % Learning rate is variable as suggested in [YMH] p. 94
        %lr = 10 * norm(grad);
        % Update theta in currect step (lr*direction was reduced)
        theta_current = theta_prev + 10 * grad;
        
        %fprintf('Diff: %f | Cost: %f\n', abs(cost_prev - cost_current), cost_current);
        i = i + 1;
    end
    fprintf('Stopped after %d iterations with cost %f (Diff: %f)\n', i, bestCost, abs(cost_prev - cost_current));
end
