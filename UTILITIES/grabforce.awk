# Grab the first component of the force from the reduced statfiles
# generated from PW and DG calculation.
#
# See also grabegy.awk
#
# DATE: 06/03/2011

BEGIN{
  print "# Tstep  PW             DG            PW-DG";
} 
/FORCES/{
  getline;
  f[istep[ARGIND]++, ARGIND] = $6;
}
END{
  if(istep[1]!= istep[2]){
    print istep[1], istep[2];
    print "Wrong number of records!";
  }
  else{
    for(i=0;i<istep[1];i++){
      printf("%6d %12.4e  %12.4e   %12.4e\n", 
	     i*NSTEP, f[i,1], f[i,2], f[i,1]-f[i,2]);
    }
  }
} 
