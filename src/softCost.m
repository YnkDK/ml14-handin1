function [ cost , grad ] = softCost(  X , y ,theta )
%cost function for the softmax.. softly.
%
% K is : num of classes.
%      X is:  n * (d+1) matrix,
%      y is:  n * K matrix
% theata is: (d+1) * K matrix
%  (+1 is for the bias).
    
    numSamplesPerClass = size(X,1)/(size(y,2));
    % M = theta' * X
    %Find the hypothesis
    M = theta * X'; % our hypotehsis, before softmax. 
    M = bsxfun(@minus, M,max(M,[],1)); %expotential of M may overflow, so do this. solution 2.
    
    softmaxBot = repmat(sum(exp(M)),size(M,1),1); %softmax function bottom
    hypothesis = exp(M)./softmaxBot;
    cost = (-1/numSamplesPerClass) * sum(sum(y' .* log(hypothesis)));
    grad = ((y' - hypothesis) * X)/numSamplesPerClass;
end


