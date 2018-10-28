lambda = 3;
[theta] = trainLinearReg([ones(12, 1) X_poly], y, lambda);
[ error_test, grad ] = ...
        linearRegCostFunction([ones(21, 1) X_poly_test], ytest, theta, 0);
