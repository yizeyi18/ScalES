# From the statfile of DG calculation, grab the coordinates and generate
# different configuration files for PW simulation.  The reduced statfile
# for DG is also generated.
#
# See also grabforce.awk
#
# DATE: 06/03/2011
BEGIN{
  istep=0; 
  ifile=10000;    
  # LL: IT IS VERY IMPORTANT to have the files ordered.
  statname="shortstat_dg";
  system("rm -f "statname);
  system("touch "statname);
}

/Time step/{
  if(istep % NSTEP == 0){
    while($0!~/Molglobal/){
      print $0 >> statname;
      if($0~/FORCES/){
	filename = "mdpw."ifile".in";
	print "statfile = statfile."ifile"\n" > filename;
	print "begin Atom_Coord\n" >> filename;
	for(i = 0; i < NATOM; i++){
	  getline;
	  print $0 >> statname;
	  printf("%10.4f %10.4f %10.4f\n", $3, $4, $5) >> filename;
	}
	print "end Atom_Coord\n" >> filename;
	echocmd = "cat mdpw_temp.in >> "filename;
	system(echocmd);
	ifile++;
      }
      getline;
    }
  }
  istep++;
}

