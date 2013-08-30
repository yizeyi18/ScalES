%% Plot the potential obtained from the DGDFT calculation.
%
%  This is just a temporary code for post processing.  In future this
%  shall be addressed by cut3d like code.
%
%  Lin Lin
%  2013/08/21

% Potential must start with "POTEXT_"
numProc = 4;
vtot = cell(numProc, 1);
vext = cell(numProc, 1);
gridPos = cell(numProc, 3);
numGrid = cell(numProc, 1);

for l = 1 : numProc
	fname = sprintf('POTEXT_%d_%d',l-1,numProc);
	fid = fopen(fname, 'r');
	gridPos{l, 1} = deserialize( fid, {'DblNumVec'} );
	gridPos{l, 2} = deserialize( fid, {'DblNumVec'} );
	gridPos{l, 3} = deserialize( fid, {'DblNumVec'} );
	numGrid{l} = [numel(gridPos{l,1}) numel(gridPos{l,2}) numel(gridPos{l,3})];
	vtot{l} = deserialize( fid, {'DblNumVec'} );
	vtot{l} = reshape( vtot{l}, numGrid{l} );
	vext{l} = deserialize( fid, {'DblNumVec'} );
	vext{l} = reshape( vext{l}, numGrid{l} );
	fclose( fid );
end

d = 3;
xi = 16;
yi = 16;
l  = 2;

close all
figure
hold on
plot(gridPos{l,3}, squeeze(vtot{l}(xi,yi,:)), 'b-o');
plot(gridPos{l,3}, squeeze(vext{l}(xi,yi,:)), 'r-^');
plot(gridPos{l,3}, squeeze(vtot{l}(xi,yi,:)) - ...
	squeeze(vext{l}(xi,yi,:)), 'k-d');
hold off
legend('Vtot','Vext','Vorig');

