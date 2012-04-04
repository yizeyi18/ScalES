# GENQSUB generates the qsub script for a series of RUNxx directories.
#
# Lin Lin
# 12/04/2011

BEGIN{
  # Change the parameters below
  fname = "runhead.pbs";  # template, without the name PBS -N"
  projhead = "Na4_DG"
  nrun  = 6;
  runsta = 2;
  procopt = " -n 4 -S 2 ~/force/MDnonorth/mddg md_Na_dg.in";

  for( i = runsta; i < runsta+nrun; i++){
    outfname = "run" i ".pbs";
    system("touch " outfname);
    print "#!/bin/bash" >> outfname;
    print "#PBS -N " projhead "_RUN"i >> outfname;
    system("cat " fname  " >> " outfname);
    print "cd RUN" i  >> outfname;
    print "aprun " procopt >> outfname; 
    print "cd .." >> outfname;
  }

  
}
