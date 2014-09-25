function [cost,grad] = logCost(X, y, theta)
    cost = costCalc(X, y, theta);
    grad = X'*(y-arrayfun(@sig, X*theta));
    grad = grad/length(y);
    
end

function res = sig(z)
    res = (1+exp(-z)).^(-1);
end

function [res] = costCalc(X, y, theta)
  
  % res = 0;
   % for i=1:size(X, 1)
        % Get the i'th x' row (and make it a column)
     %   xi = X(i,:)';
        % Get value of the sigmoid function
   %     sig = 1/(1+exp(-theta'*xi));
   %   tst = 1+exp(-theta'*xi);
    %   sig = abs(1/(1+exp(-theta'*xi))-0.00000001) ;
        % log(1) and log(0) are not desireable
    %    if sig == 1
    %        sig = 0.99999999;
    %    elseif sig == 0
     %       sig = 0.00000001;
     %   end
        %fprintf('tx = %f\n', theta'*x);
        % Update the log likelihood
     %   aa = y(i) * log(sig) ;
     %   bb=  (1-y(i))*log(1-sig);
     %  res = res + y(i) * log(sig) + (1-y(i))*log(1-sig);
    %end
    
    y = transpose(y);
    part1 = abs(((1+(exp(X*-theta))).^-1)-0.00000001);
    MatSig1 = y*(log(part1));
    MatSig2 = (1-y)*(log((1-part1))); %might become imaginary not sure if has effect
    res =MatSig1+MatSig2;
    
    %first step, mult the whole matrix with 
    % replicate v twice in the row dimension
    %m([1,3],:) = repmat(v,2,1)
    %mt = repmat(theta, 
%    Mat1 = X * -theta;
%    Mat1 = 1/(1+exp(Mat1));
 %   Mat1 = abs(Mat1-00000001);
    
%    Mat2 = y*log(Mat1);
  %  Mat3 = (1-y)*log(1-Mat1);
    
    %sum that up.
%    TestRes = sum(Mat2) + sum(Mat3);
    
 %   ResMatrix = (log(abs(1/(1+exp(X * -theta))- 00000001)));
    
  %  Part1Matrix = 
    % Make it the negative log likelihood
    res = -res;
end