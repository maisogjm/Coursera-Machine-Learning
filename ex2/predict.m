function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

disp(sprintf('X    is %d by %d',size(X,1),size(X,2)))
disp(sprintf('theta is %d by %d',size(theta,1),size(theta,2)))
p_temp = sigmoid(X * theta);
p      = ( p_temp >= 0.5 );

% =========================================================================


end
