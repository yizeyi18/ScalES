%% Plot the electron density obtained from the DGDFT calculation.
%
%  This is just a temporary code for post processing.  In future this
%  shall be addressed by cut3d like code.
%
%  Lin Lin
%  2013/08/15

% Density must start with "DEN_"
numProc = 4;
rho = cell(numProc, 1);
gridPos = cell(numProc, 3);
numGrid = cell(numProc, 1);

for l = 1 : numProc
	fname = sprintf('DENLGL_%d_%d',l-1,numProc);
	fid = fopen(fname, 'r');
	gridPos{l, 1} = deserialize( fid, {'DblNumVec'} );
	gridPos{l, 2} = deserialize( fid, {'DblNumVec'} );
	gridPos{l, 3} = deserialize( fid, {'DblNumVec'} );
	numGrid{l} = [numel(gridPos{l,1}) numel(gridPos{l,2}) numel(gridPos{l,3})];
	rho{l} = deserialize( fid, {'DblNumVec'} );
	assert( numel( rho{l} ) == prod( numGrid{l} ) );
	rho{l} = reshape( rho{l}, numGrid{l} );
	fclose( fid );
end

d = 3;
xi = 16;
yi = 1;
grid1D = zeros( numGrid{1}(d), numProc );
rho1D  = zeros( numGrid{1}(d), numProc );
for l = 1 : numProc
	grid1D(:, l) = gridPos{l,d};
	rho1D(:,l)   = squeeze(rho{l}(xi,yi,:));
end
% Reflecting the periodic boundary condition
grid1D = [grid1D gridPos{1,d}+max(gridPos{numProc,d})];
rho1D  = [rho1D  squeeze(rho{1}(xi,yi,:))];

plot(grid1D, rho1D,'-o');
