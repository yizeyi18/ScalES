fid = fopen(filename,'rb');
nsize = fread(fid,1,'int');
vec = fread(fid,nsize,'double');
fclose(fid);
