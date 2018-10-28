function [J grad ] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

% First generate the output of the input layer.
% This passes to the hidden layer as input.
% Using predict.m from Exercise 3 as a model.
hHidden = sigmoid(X * Theta1');

% Add ones to the hHidden data matrix
hHidden = [ ones(m, 1) hHidden];

% Next pass the output of the input layer to the hidden layer.
% The output is used to determine the final predicted class.
% Again using predict.m from Exercise 3 as a model.
h = sigmoid(hHidden * Theta2');

% "You may use a for-loop over the examples to compute the cost."
% We need to recode the labels as vectors containing only values
% 0 or 1.
% Using lrCostFunction.m from Exercise 3 as a model.
J = 0; % Initialize J to 0.
for t = 1:m
    yk       = zeros(num_labels,1);
    yk(y(t)) = 1;
    hk       = h(t,:)';
    term1    = -yk' * log(hk);
    term2    = ( 1 - yk )' * log( 1 - hk );
    J        = J + sum( term1 - term2 ) / m; % Cumulate J.
end

% Add regularization term.
% Note that you should not be regularizing the terms that
% correspond to the bias.
Theta1SumSqr = sum( sum( Theta1(:,2:end) .* Theta1(:,2:end) ) );
Theta2SumSqr = sum( sum( Theta2(:,2:end) .* Theta2(:,2:end) ) );
regTerm      = ( Theta1SumSqr + Theta2SumSqr) * lambda / (2*m);
J            = J + regTerm;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

D2 = zeros(num_labels,hidden_layer_size + 1);       % Initialize accumulator
D1 = zeros(hidden_layer_size,input_layer_size + 1); % Initialize accumulator

for t=1:m
    % Set d3k for the output layer.
    y3k = [ 1:num_labels ]' == y(t); % Another way to set yk.
    h3k = h(t,:)';
    d3k = h3k - y3k;

    % Set d2k for the hidden layer.
    z2  = Theta1 * X(t,:)';
    d2k = ( Theta2(:,2:end)' * d3k ) .* ( sigmoidGradient(z2) ) ;

    % Accumulate the gradient.
    D2 = D2 + d3k * hHidden(t,:);
    D1 = D1 + d2k * X(t,:);
end
Theta1_grad  = D1 / m;
Theta2_grad  = D2 / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Note that you should *not* be regularizing the first
% column of Theta(l) which is used for the bias term.
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda*Theta1(:,2:end)/m);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda*Theta2(:,2:end)/m);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
