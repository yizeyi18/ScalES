function y = mae (A, B)
% MAE computes the mean average error of A and B
[m1, n1] = size(A);
[m2, n2] = size(B);
assert(m1 == m2 & n1 == n2);

y = mean(sqrt(sum((A-B).^2,2)));
