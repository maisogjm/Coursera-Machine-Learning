%% Initialization
clear ; close all; clc

data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);
X = mapFeature(X(:,1), X(:,2));

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

for lambda = [ 0, 1, 10, 100 ]
    % Initialize fitting parameters
    initial_theta = zeros(size(X, 2), 1);

    % Optimize
    [theta, J, exit_flag] = ...
    	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

    % Compute accuracy on our training set
    p = predict(theta, X);

    % Plot Boundary
    plotDecisionBoundary(theta, X, y);
    hold on;
    title(sprintf('lambda = %g, Train Accurac: %f', lambda, mean(double(p == y)) * 100))

    % Labels and Legend
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')
    legend('y = 1', 'y = 0', 'Decision boundary')
    hold off;

    fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
end


