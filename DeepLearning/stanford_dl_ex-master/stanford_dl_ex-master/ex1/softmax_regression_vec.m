function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
% tz = zeros(size(theta,1),1);
% for i = 1:m
%     if(y(i) ~= 10)
%         t = theta(:,y(i));
%         psum = sum(exp(theta'*X(:,i)));
%         pk = exp(t'*X(:,i));
%         p = pk/psum;
%         f = f - log(p);
%         g(:,y(i)) = g(:,y(i)) - X(:,i)*(1-p);
%     end
% end
theta = [theta, zeros(n, 1)];
e = exp(theta'*X);
% e = [e;zeros(1,m)];
esum = sum(e,1);
i = sub2ind(size(e), y, 1:m);
ei = e(i);
p = bsxfun(@rdivide, ei, esum);
f = -sum(log(p));

id = [1:1:10]';
one = bsxfun(@eq, y, id);
p_ = bsxfun(@minus, one, p);
for i = 1:num_classes-1
    g(:,i) = -sum(bsxfun(@times, X, p_(i,:)),2);
end







    


  
  g=g(:); % make gradient a vector for minFunc

