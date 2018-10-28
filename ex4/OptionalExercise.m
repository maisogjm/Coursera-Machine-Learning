% Octave script implementing optional exercise in Exercise 4.

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

% Load Training Data
load('ex4data1.mat');
m = size(X, 1);

% Randomly select 4500 data points to be the training set
% and the rest will be the test set.
sel          = randperm(m);
trainIndices = sel(1:4500);
testIndices  = sel(4501:m);
Xtrain       = X(trainIndices,:);
Xtest        = X(testIndices,:);
yTrain       = y(trainIndices);
yTest        = y(testIndices);

% Initializing Neural Network Parameters
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Define ranges of lambda and MaxIter to try.
% Pre-allocate memory for matrix to store accuracy results.
lambdaRange     = [ 0:10 ];
maxItRange      = [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ];
numLambda       = length(lambdaRange);
numMaxIt        = length(maxItRange);
accuracyResults = zeros(numLambda,numMaxIt);

% Create "short hand" for the cost function to be minimized.
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xtrain, yTrain, lambda);

% Loop over values of lambda.
for lambdaIndex = 1:numLambda
    lambda = lambdaRange(lambdaIndex);

    % Loop over values of maximum number of iterations.
    for maxItIndex = 1:numMaxIt
        maxIt = maxItRange(maxItIndex);
        fprintf('lambda = %g, maxIter = %d',lambda,maxIt);

        % Set MaxIter option for FMINCG().
        options = optimset('MaxIter', maxIt);

        % Optimize neural network parameters.
        [nn_params, cost] = fmincg(@(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xtrain, yTrain, lambda), ...
                                   initial_nn_params, options);

        % Obtain Theta1 and Theta2 back from nn_params.
        Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                         hidden_layer_size, (input_layer_size + 1));

        Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                         num_labels, (hidden_layer_size + 1));

        % Predict the labels of the TEST set.
        % This lets you compute the TEST set accuracy.
        pred     = predict(Theta1, Theta2, Xtest);
        accuracy = mean(double(pred == yTest)) * 100;
        fprintf('\nTraining Set Accuracy: %f\n',accuracy);
        accuracyResults(lambdaIndex,maxItIndex) = accuracy;
    end
end

% Save accuracy results.
save('c:/Documents and Settings/admin/My Documents/StanfordML/ex4/accuracyResults.mat','accuracyResults')

% Plot accuracy as an image.
h = imagesc(accuracyResults);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Do not show axis
axis image off

drawnow;
