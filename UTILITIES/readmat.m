fileName = 'DGMAT_FULL';
fh = fopen(fileName, 'rb');
sizeA = fread(fh,1,'int');
sizeTriplet = fread(fh, 1, 'int');
RowVec = fread(fh, sizeTriplet, 'int');
ColVec = fread(fh, sizeTriplet, 'int');
ValVec = fread(fh, sizeTriplet, 'double');
disp('Finish reading');

A = sparse(RowVec+1, ColVec+1, ValVec, sizeA, sizeA);
D = eigs(A+10*speye(size(A)),6,'sm')-10;

fclose(fh);

