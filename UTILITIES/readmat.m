fileName = 'DGMAT_FULL';
fh = fopen(fileName, 'rb');
sizeA = fread(fh,1,'int');
sizeTriplet = fread(fh, 1, 'int');
RowVec = fread(fh, sizeTriplet, 'int');
ColVec = fread(fh, sizeTriplet, 'int');
ValVec = fread(fh, sizeTriplet, 'double');

A = sparse(RowVec+1, ColVec+1, ValVec, sizeA, sizeA);
D = eig(full(A));

fclose(fh);

