% READCOEF reads the coefficient of the nonlocal adaptive local basis
% functions.
% 
% The code assumes that each element is occupied by one processor
%
% Last revision: 12/9/2011

fprintf('The code assumes coef_xx_xx and basesglb_xx_xx exist under the current directory.\n\n');

Nelem = input('Number of elements ([1 1 4]): ');
if( isempty(Nelem) )
  Nelem = [1 1 4];
end
Nelem = reshape(Nelem, [], 1);
nproc = prod(Nelem);

% Read the coefficients 
coef = [];
for i = 1 : nproc
  fname = strcat('coef_',num2str(i-1),'_',num2str(nproc));
  fid = fopen(fname,'r');
  string = {'DblNumMat'};
  if ( isempty(coef) )
    coef = deserialize(fid, string);
  else
    tmp =  deserialize(fid, string);
    coef = [coef tmp];
  end
  fclose(fid);
end
