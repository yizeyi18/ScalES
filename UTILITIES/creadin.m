fid = fopen('dump.dat','rb');
nsize = fread(fid,1,'int');
tvec = fread(fid,2*nsize,'double');
vec = tvec(1:2:end) + 1i*tvec(2:2:end);
fclose(fid);
