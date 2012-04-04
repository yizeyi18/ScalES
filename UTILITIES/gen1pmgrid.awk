# This is a template to generate a series of RUN directories to test
# various combinations of parameters.
#
# This is for the case with 1 tuning parameters. 
#
# Lin Lin
# 12/04/2011

BEGIN{
  # Change the parameters below
  fname = "md_Na_dg.in";
  relfile = "HGH.bin"
  runcnt  = 2;
  nvars = 1;
  nvarptr[1] = 2;
  attrb[1] = "ElePerEle :";
  val[1,1] = 4;
  val[1,2] = 7;

  # Generate RUN directories
  for(i = 1; i <= nvarptr[1]; i++){
    rundir = "RUN" runcnt;
    outputfname = rundir "/" fname;
    system("mkdir " rundir);
    print "#Tuning parameters" > outputfname;
    printf("%s %s\n", attrb[1], val[1,i]) >> outputfname;
    printf("\n#From the template file\n") >> outputfname;
    system("cat " fname  " >> " outputfname);
    system("cp " relfile " " rundir);
    runcnt = runcnt + 1;
  }
}
