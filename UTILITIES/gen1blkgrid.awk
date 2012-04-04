# This is a template to generate a series of RUN directories to test
# various combinations of parameters.
#
# This is for the case with 1 tuning parameter but of block form, e.g.
#
# begin Grid_Size
#   40 40 40
# end Grid_Size
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
  attrb[1] = "Grid_Size";
  val[1,1] = "40  40  40";
  val[1,2] = "80  80  80";

  # Generate RUN directories
  for(i = 1; i <= nvarptr[1]; i++){
    rundir = "RUN" runcnt;
    outputfname = rundir "/" fname;
    system("mkdir " rundir);
    print "#Tuning parameters" > outputfname;
    printf(" begin %s\n%s\n end %s\n", attrb[1], val[1,i], attrb[1]) >> outputfname;
    printf("\n#From the template file\n") >> outputfname;
    system("cat " fname  " >> " outputfname);
    system("cp " relfile " " rundir);
    runcnt = runcnt + 1;
  }
}
