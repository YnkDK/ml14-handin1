function [bestTheta] = softRun(X, y, theta)
%SOFTRUN.M Summary of this function goes here
%   Detailed explanation goes here
%
%      X is:  n * (d+1) matrix,
%      y is:  n * K matrix
% theata is: (d+1) * K matrix
   % Stop conditions
    conditionVal = 0.0002; % 94 pression can be achived by letting it run for ever.. just append 2 more 0's
    maxIter = 50000;
     
    % Initialize the result
    bestCost = Inf;
    bestTheta = theta;
    
    % Initialize states   
    cost_current = Inf;
    cost_prev  = Inf;
    theta_current = theta';
    
    
    i = 0;
    while i<10 || cost_prev ==Inf || ((i < maxIter ) && abs(cost_prev - cost_current) > conditionVal)
        cost_prev = cost_current;
        theta_prev = theta_current;

        [cost_current, grad] = softCost(X,y,theta_prev);
        if cost_current <= bestCost
            bestCost = cost_current;
            bestTheta = theta_prev;
        end
        % Step direction is the unit vector opposite of 
        % the gradient
        theta_current = (theta_prev + 0.05 * grad); %0.05
        fprintf('Diff: %f | Cost: %f\n', abs(cost_prev - cost_current), cost_current);
        i = i + 1;
    end
    bestTheta = bestTheta';
    fprintf('Stopped after %d iterations with cost %f (Diff: %f)\n', i, bestCost, abs(cost_prev - cost_current));
end

