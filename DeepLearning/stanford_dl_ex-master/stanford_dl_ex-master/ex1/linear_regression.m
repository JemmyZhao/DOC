function [f,g] = linear_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  
  m=size(X,2);
  n=size(X,1);

  f=0;
  g=zeros(size(theta));

  %
  % TODO:  Compute the linear regression objective by looping over the examples in X.
  %        Store the objective function value in 'f'.
  %
  % TODO:  Compute the gradient of the objective with respect to theta by looping over
  %        the examples in X and adding up the gradient for each example.  Store the
  %        computed gradient in 'g'.
  
%%% YOUR CODE HERE %%%
% 
% for j = 1:m
%     f = f + 0.5*(dot(X(:,j),theta)-y(j))^2;
%     g = g + (dot(X(:,j),theta)-y(j))*X(:,j);
% end

y_h = theta'*X;
f = 0.5 * sum((y_h-y).^2, 2);
g = X*(y_h-y)';

epsilon = 0.0001;
d = epsilon * eye(n);
tp = theta + d;
tm = theta - d;
yhp = tp'*X;
yhm = tm'*X;
fp = 0.5 * sum((yhp-y).^2, 2);
fm = 0.5 * sum((yhm-y).^2, 2);
gd = (fp - fm)/2/epsilon;
fprintf('g: %f \n',g-gd);
    
