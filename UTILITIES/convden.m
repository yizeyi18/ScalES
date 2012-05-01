% CONVDEN converts the density into the Gaussian cube format (Z-Y-X)
%
% It should be combined with a head file.  
%
% CONVDEN can also be used to print out the Gaussian cube file for
% potential.
%
% Lin Lin
% Last revision: 4/12/2012

fprintf('The code assumes DEN exist under the current directory.\n\n');

% Read the coefficients 
Ns = input('Number of grid points ([20 20 80]): ');
if( isempty(Ns) )
  Ns = [20 20 80];
end

Ntot = prod(Ns);


fname = 'DEN';
fid = fopen(fname,'r');
string = {'DblNumVec'};  % LL: Later should change it to DblNumTns.
rho = deserialize(fid, string);
assert(numel(rho) == Ntot);
fclose(fid);

rho3D = reshape(rho, Ns);
rhoCube3D = permute(rho3D, [3 2 1]);
rhoCube = rhoCube3D(:);

fid = fopen('rho.cub','w');
fprintf(fid, '%12.5e\n', rhoCube);
fclose(fid);
