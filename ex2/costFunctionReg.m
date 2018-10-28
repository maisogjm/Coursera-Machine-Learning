function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Compute cost function.
% Note that you should not regularize the parameter THETA_0.
h           = sigmoid(X * theta);
term1       = -y' * log(h);
term2       = ( 1 - y )' * log( 1 - h );
theta2      = theta .* theta;
thetaSum    = sum(theta2(2:length(theta2))); % Don't regularize THETA_0.
regTerm     = ( lambda * thetaSum / ( 2 * m ) );
J           = ( sum( term1 - term2 ) / m ) + regTerm;

% Compute gradient.
% Note that you should not regularize the parameter THETA_0.
regTerm    = ( lambda * theta' / m);
regTerm(1) = 0; % Don't regularize THETA_0.
grad       = ( ( h - y )' * X / m ) + regTerm;

% =============================================================

end
