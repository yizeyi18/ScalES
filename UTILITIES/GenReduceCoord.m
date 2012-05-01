function [C, xyzmat, xyzmatReduce] = GenReduceCoord(nreps, asize, nat, coefs, ran);
% Generate the atomic position for a give crystal structure with
% potential randomness

nrepx = nreps(1); nrepy = nreps(2); nrepz = nreps(3);

fprintf('Constructing cells with size %3d*%3d*%3d\n', ...
  nreps(1), nreps(2), nreps(3));
fprintf('Total number of atoms:   %6d\n\n', nat * prod(nreps));

if( numel(asize) == 1 )
  asize = ones(3,1)*asize;
elseif( numel(asize) == 3 )
  % do nothing
else
  error('asize is not in the correct form');
end
C = diag(asize);

xyzmat = coefs*C';
%
% repeat the cell nrep times along the z direction
%
for krep = 1 : nrepz
  for jrep = 1 : nrepy
    for irep = 1 : nrepx
      if( irep ~= 1 || jrep ~= 1 || krep ~= 1 ) 
	xyzpick = xyzmat(1:nat,:);
	xyzpick(:,1) = xyzpick(:,1) + (irep-1)*asize(1);
	xyzpick(:,2) = xyzpick(:,2) + (jrep-1)*asize(2);
	xyzpick(:,3) = xyzpick(:,3) + (krep-1)*asize(3);
	xyzmat = [xyzmat; xyzpick];
      end
    end
  end
end
% Add randomness
xyzmat = xyzmat + ran * (randn(size(xyzmat)));
%
% modify the supercell
%
C(1,1) = nrepx*asize(1);
C(2,2) = nrepy*asize(2);
C(3,3) = nrepz*asize(3);

fprintf('C\n%15.6f    %15.6f    %15.6f\n', C(1,1), C(2,2), C(3,3));
fprintf('xyzmat\n');
fprintf('%15.6f    %15.6f    %15.6f\n', xyzmat'); % Transpose is important!

disp('Reduced coordinate')
xyzmatReduce = xyzmat ./ repmat([C(1,1), C(2,2), C(3,3)], ...
  size(xyzmat,1), 1)
