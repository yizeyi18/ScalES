function volData3D = ReadCube(fn);
% Read the Gaussian cube format into the MATLAB volume format
%
% Lin Lin
% 04/13/2012

if( nargin < 1 )
  fn = input('Input Gaussian cube file name: ','s');
end

fid = fopen(fn, 'r');
% The first two lines are omitted
fgetl(fid);
fgetl(fid);
Natoms = fscanf(fid, '%d', 1);
fgetl(fid);
Ns = zeros(1,3);
for i = 1 : 3
  Ns(i) = fscanf(fid, '%d', 1);
  fgetl(fid);
end
for i = 1 : Natoms
  fgetl(fid);
end
Ntot = prod(Ns);
volData = fscanf(fid,'%g',inf);
volData3D = reshape(volData, [Ns(1) Ns(2) Ns(3)]);
volData3D = permute(volData3D, [3 2 1]);

fclose(fid);
