function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% FOR Loop over the examples.
m = size(X,1);
for i = 1:m
    % Make an m-by-numCentroids matrix whose rows
    % are repeats of the i-th example.
    Y = ones(K,1) * X(i,:);

    % Compute distance between the i-th example to each centroid.
    delta  = Y - centroids;
    delta2 = delta .* delta;
    dist2  = sum(delta2,2);

    % We wouldd take the square root of dist2 to find the distance,
    % but if we don't we still get the same answer anyway since the
    % square root function is monotonic.
    % I.e., finding the index of the minimum distance squared gives
    % the same answer as finding the index of the minimum distance.
    % So let's save on computations and forgo the square root.

    % Find index of minimum distance squared.
    % This is the closest centroid.
    [ dx idx(i) ] = min(dist2);
end

% =============================================================

end
