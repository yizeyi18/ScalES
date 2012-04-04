# This is a template to generate a series of RUN directories to test
# various combinations of parameters.
#
# This is for the case with 2 tuning parameters. 
#
# Lin Lin
# 12/04/2011

BEGIN{
  # Change the parameters below
  fname = "md_Na_dg.in";
  relfile = "HGH.bin"
  runcnt  = 2;
  nvars = 2;
  nvarptr[1] = 2;
  nvarptr[2] = 3;
  attrb[1] = "ElePerEle :";
  attrb[2] = "OrbPerEle :";
  val[1,1] = 4;
  val[1,2] = 7;
  val[2,1] = 1;
  val[2,2] = 4;
  val[2,3] = 7;

  # Generate RUN directories
  for(i = 1; i <= nvarptr[1]; i++){
    for(j = 1; j <= nvarptr[2]; j++){
      rundir = "RUN" runcnt;
      outputfname = rundir "/" fname;
      system("mkdir " rundir);
      print "#Tuning parameters" > outputfname;
      printf("%s %s\n", attrb[1], val[1,i]) >> outputfname;
      printf("%s %s\n", attrb[2], val[2,j]) >> outputfname;
      printf("\n#From the template file\n") >> outputfname;
      system("cat " fname  " >> " outputfname);
      system("cp " relfile " " rundir);
      runcnt = runcnt + 1;
    }
  }
}
