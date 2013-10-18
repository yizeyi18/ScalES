%% Plot the adaptive local basis functions obtained from the DGDFT calculation.
%
%  This is just a temporary code for post processing.  In future this
%  shall be addressed by cut3d like code.
%
%  Lin Lin
%  2013/08/16

% ALB must start with "WFNEXT_"
numProc = 4;
wavefun = cell(numProc, 1);
gridPos = cell(numProc, 3);
numGrid = cell(numProc, 1);
numWavefun = cell(numProc, 1);

for l = 1 : numProc
	fname = sprintf('WFNEXT_%d_%d',l-1,numProc);
	fid   = fopen(fname, 'r');
	gridPos{l, 1} = deserialize( fid, {'DblNumVec'} );
	gridPos{l, 2} = deserialize( fid, {'DblNumVec'} );
	gridPos{l, 3} = deserialize( fid, {'DblNumVec'} );
	numGrid{l} = [numel(gridPos{l,1}) numel(gridPos{l,2}) numel(gridPos{l,3})];
	wavefun{l} = deserialize( fid, {'DblNumTns'} );
	numWavefun{l} = size(wavefun{l},3);
	wavefun{l} =  reshape( wavefun{l}, [numGrid{l} numWavefun{l}] );
	fclose( fid );
end


d = 3;
xi = 48;
yi = 1;
l  = 1;
n  = 1;
close all
figure
hold on
plot(gridPos{l,3}, squeeze(wavefun{l}(xi,yi,:,n)), 'b-o');
hold off

