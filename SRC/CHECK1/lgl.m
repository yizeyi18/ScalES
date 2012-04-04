function [x,w,P,D,S,T] = lgl(n)

  [x,w,P] = lglnodes(n);  %w: weights, matrix: P(x_loc, order)
  
  %D: derivative Lag basis
  xi = x;
  xj = x';
  Lnxi = P(:,end);
  Lnxj = P(:,end)';
  D = 1./(xi*ones(1,n+1)-ones(n+1,1)*xj) .* (Lnxi*ones(1,n+1)) ./ (ones(n+1,1)*Lnxj);
  D(1,1) = n*(n+1)/4;
  for j=2:n
    D(j,j) = 0;
  end
  D(end,end) = -n*(n+1)/4;
  
  x = x(end:-1:1);  w = w(end:-1:1);  P = P(end:-1:1,:);
  D = D(end:-1:1,end:-1:1);
  
  %inner product, Lag basis
  IP = inv(P);
  g = 1./([0:n]+1/2);
  g(end) = 2/n;   % LL: The weight is modified for Gauss-Lobatto
  S = IP' * diag(g)* IP; 
  
  %triple product, Lag basis, in tensor
  A = zeros(n+1,n+1,3*n+1);
  g = 1./([0:n]+1/2);
  %i=0;
  for j=0:n
    A(1,j+1,j+1) = g(j+1);
  end
  %j=0;
  for i=0:n
    A(i+1,1,i+1) = g(i+1);
  end
  %k=0;
  for i=0:n
    A(i+1,i+1,1) = g(i+1);
  end
  [ii,jj,kk] = ndgrid(1:n,1:n,1:3*n);
  for ss=3:3*n
    gud = find(ii+jj+kk==ss);
    for c=gud(end:-1:1)'
      i = ii(c);
      j = jj(c);
      k = kk(c);
      rs = [1:ceil(i/2)]';
      tmp = A(1+i+1-2*rs,1+j,1+k+1)-A(1+i+1-2*rs,1+j,1+k-1);
      sum1 = sum( (2*i+3-4*rs).*tmp(:) );
      rs = [1:ceil(j/2)]';
      tmp = A(1+i,1+j+1-2*rs,1+k+1)-A(1+i,1+j+1-2*rs,1+k-1);
      sum2 = sum( (2*j+3-4*rs).*tmp(:) );
      A(1+i,1+j,1+k) = -1/(2*k+1) * (sum1+sum2);
    end
  end
  T = A(1:n+1,1:n+1,1:n+1);
  n1 = n+1;
  tmp = reshape(T,n1*n1,n1) * IP;  T = reshape(tmp,[n1,n1,n1]);  T = shiftdim(T,1);
  tmp = reshape(T,n1*n1,n1) * IP;  T = reshape(tmp,[n1,n1,n1]);  T = shiftdim(T,1);
  tmp = reshape(T,n1*n1,n1) * IP;  T = reshape(tmp,[n1,n1,n1]);  T = shiftdim(T,1);