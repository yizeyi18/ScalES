# The number of atoms appear in the first time reading a file.
BEGIN{
  first_file_ind = 0;
}
/Total number of atoms/{
  if( first_file_ind == 0 )
    first_file_ind = ARGIND;
  i = ARGIND - first_file_ind;
  if( flag_natom[i]++ == 0 )
    natom[i] = $NF;
}
/Number of occupied states/{
  i = ARGIND - first_file_ind;
  if( flag_occupied[i]++ == 0 )
    noccupied[i] = $NF-1;
}

# Compare the CBM/VBM energies last appeared in the stat file.
($1~/eig\[/) && ($2 == noccupied[i]"]"){
  i = ARGIND - first_file_ind;
  E_VBM[i] = $4;
  getline;
  E_CBM[i] = $4;
}

# Compare total energy, kinetic energy, Coulomb energy and
# exchange-correlation energy last appeared in the stat file.
/Total Energy/{
  i = ARGIND - first_file_ind;
  ETOT[i] = $4;
}
/Ekin/{
  i = ARGIND - first_file_ind;
  EKIN[i] = $3;
}
/Ecoul/{
  i = ARGIND - first_file_ind;
  ECOUL[i] = $3;
}
/Exc/{
  i = ARGIND - first_file_ind;
  EXC[i] = $3;
}


END{
  if( natom[0] != natom[1] )
    print "Number of atoms do not match";
  if( noccupied[0] != noccupied[1] )
    print "Occupied state number do not match";

  printf("Number of atoms             : %6d \n", natom[0]);
  printf("Number of occupied states   : %6d \n", noccupied[0]+1);
  printf("\n"); 
  printf("$E_{\\text{VBM}}$   & %10.4f & %10.4f & %10.4f\\\\\n",
	 E_VBM[0], E_VBM[1], abs(E_VBM[0]-E_VBM[1]));
  printf("$E_{\\text{CBM}}$   & %10.4f & %10.4f & %10.4f\\\\\n",
	 E_CBM[0], E_CBM[1], abs(E_CBM[0]-E_CBM[1]));
  printf("$E_{\\text{TOT}}$    & %10.4f & %10.4f & %10.4f\\\\\n", 
	 ETOT[0], ETOT[1],
	 abs(ETOT[0]-ETOT[1])/natom[0]);
  printf("$E_{\\text{KIN}}$    & %10.4f & %10.4f & %10.4f\\\\\n", 
	 EKIN[0], EKIN[1],
	 abs(EKIN[0]-EKIN[1])/natom[0]);
  printf("$E_{\\text{COUL}}$   & %10.4f & %10.4f & %10.4f\\\\\n", 
	 ECOUL[0], ECOUL[1],
	 abs(ECOUL[0]-ECOUL[1])/natom[0]);
  printf("$E_{\\text{XC}}$     & %10.4f & %10.4f & %10.4f\\\\\n", 
	 EXC[0], EXC[1],
	 abs(EXC[0]-EXC[1])/natom[0]);
}


function abs(x){
  return (x > 0.0) ? x : -x;
}
