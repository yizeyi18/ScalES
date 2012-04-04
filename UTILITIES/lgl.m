function [x,w,P,D] = lgl(n)
[x,w,P] = lglnodes(n);  %w: weights, matrix: P(x_loc, order)
xi = x;
xj = x';
Lnxi = P(:,end);
Lnxj = P(:,end)';
% Construct differentiation matrix on parent element.
D = 1./(xi*ones(1,n+1)-ones(n+1,1)*xj) .* (Lnxi*ones(1,n+1)) ./ (ones(n+1,1)*Lnxj);
D(1,1) = n*(n+1)/4;
for j=2:n
  D(j,j) = 0;
end
D(end,end) = -n*(n+1)/4; %derivative matrix

x = x(end:-1:1);  w = w(end:-1:1);  P = P(end:-1:1,:);
D = D(end:-1:1,end:-1:1);

