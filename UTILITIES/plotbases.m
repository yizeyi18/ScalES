% PLOTBASES plots the nonorthogonal adaptive local basis functions
% outputed from MDDG. 
% 
% The code assumes that each element is occupied by one processor
%
% NOTE: This subroutine is now obsolete and substituted by
% SRC/convcube.cpp
%
% Last revision: 12/8/2011

fprintf('The code assumes coef_xx_xx and basesglb_xx_xx exist under the current directory.\n\n');

Nelem = input('Number of elements ([1 1 4]): ');
if( isempty(Nelem) )
  Nelem = [1 1 4];
end
Nelem = reshape(Nelem, [], 1);
nproc = prod(Nelem);


ielem = input('Index of element to plot ([1 1 2]): ');
if( isempty(ielem) )
  ielem = [1 1 2];
end
idx = ielem(1) + (ielem(2)-1) * Nelem(1) + (ielem(3)-1) * Nelem(1) * Nelem(2);

% Read the coefficients 
fname = strcat('coef_',num2str(idx-1),'_',num2str(nproc));
fid = fopen(fname,'r');
string = {'DblNumMat'};
coef = deserialize(fid, string);
fclose(fid);

% Read the adaptive local basis functions defined on the global domain
basesglb = cell(nproc,2);
Nsglb    = cell(nproc,1);
posidx   = cell(nproc,1);
for i = 1 : nproc
  fname = strcat('basesglb_',num2str(i-1),'_',num2str(nproc));
  fid = fopen(fname,'r');
  Nsglbtmp  = deserialize(fid, {'Index3'});
  posidxtmp = deserialize(fid, {'Index3'});
  string = {'vector',{'DblNumTns'}};
  basestmp  = deserialize(fid, string); 
  basesglb{i} = basestmp;
  Nsglb{i}    = Nsglbtmp;
  posidx{i}   = posidxtmp;
  fclose(fid);
end

Nsglb = Nsglb{1};
Nsglbtot = Nsglb .* Nelem;

% Compute the nonorthogonal adaptive local basis functions
nb = size(coef,2);
basesnalb = cell(nb,1);
for i = 1 : nb
  basesnalb{i} = zeros(Nsglbtot(1),Nsglbtot(2),Nsglbtot(3));
  cnt = 1;
  for j = 1 : nproc
    nbases = numel(basesglb{j});
    for k = 1 : nbases
      i1 = 1:Nsglb(1);
      i2 = 1:Nsglb(2);
      i3 = 1:Nsglb(3);
      i1sh = posidx{j}(1) + i1;
      i2sh = posidx{j}(2) + i2;
      i3sh = posidx{j}(3) + i3; 
      basesnalb{i}(i1sh, i2sh, i3sh) = basesnalb{i}(i1sh,i2sh,i3sh) + ...
	basesglb{j}{k}(i1,i2,i3) * coef(cnt,i);
      cnt = cnt + 1;
    end % k
  end % j
  % Normalize the basis functions
  basesnalb{i} = basesnalb{i} / sqrt(sum(basesnalb{i}(:).^2));
end % i


