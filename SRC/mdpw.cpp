#include "esdf.h"
#include "debug.hpp"
#include "scfpw.hpp"


FILE * fhstat;
int CONTXT;

// NVE Molecular dynamics simulation.
int main(int argc, char **argv) 
{
  MPI_Init(&argc, &argv);
  int myid;  
  int nprocs;  
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  clock_t tottime_sta, tottime_end;

  /* Initialize the total time */
  tottime_sta = clock();
  
  /* BLACS initialization */
  char colorder = 'C';
  int nprow = 0, npcol = 0;
  for(int i = round(sqrt(double(nprocs))); i <= nprocs; i++){
    nprow = i; npcol = nprocs / nprow;
    if( nprow * npcol == nprocs) break;
  } 
  if(myid==MASTER) {
    cerr<<"nprow "<<nprow<<" npcol "<<npcol<<endl;
  }
  Cblacs_get(0,0, &CONTXT);
  Cblacs_gridinit(&CONTXT, &colorder, nprow, npcol);
  
  
  //---------
  char inputfile[80];
  strcpy(inputfile, argv[1]);
  if(myid == MASTER )  esdf_init(inputfile);
  esdf_bcast(myid, MASTER);
  //---------
  char strtmp[80];
  int inttmp;
  //---------
  //statfile
  esdf_string((char*)("statfile"), (char*)("statfile"), strtmp);
  /* Initialize output file */
  char statfile[128];
  sprintf(statfile, strcat(strtmp,".%d"), 100000+myid);
  fhstat = fopen(statfile,"w");
  if(fhstat == NULL){
    cerr << "myid :" << myid << " cannot open status file" << endl;
    ABORT("",1);
  }
 
  //---------
  //Input format

  esdf_string((char*)("Input_Format"), (char*)("v1.2"), strtmp);
  string inputformat(strtmp);

  
  //---------
  //scfpw
  ScfPW scf;
  vector<double> rhoinput;
  vector<double> psiinput; 
  //MD parameters
  double dt = esdf_double((char*)("Time_Step"), 100.0);
  double max_step = esdf_integer((char*)("Max_Step"), 1);
 
  if( inputformat == "v1.0" ){

    //***************************************************************
    //Input format  v1.0
    //***************************************************************

    //---------
    //domain
    Domain dm;
    {
      int nlines;
      //
      Point3 _Ls;
      Index3 _Ns;
      Point3 _pos;
      if (esdf_block((char*)("Super_Cell"),&nlines)) {
	sscanf(block_data[0],"%lf %lf %lf",
	       &_Ls[0],&_Ls[1],&_Ls[2]);
      } else {
	ABORT("ERROR: bad specification of 'Super_Cell'",1);
      }
      if (esdf_block((char*)("Grid_Size"),&nlines)) {
	sscanf(block_data[0],"%d %d %d",
	       &_Ns[0],&_Ns[1],&_Ns[2]);
      }  else {
	ABORT("ERROR: bad specification of 'Grid_Size'",1);
      } /* end of if:esdf_block statement */
      _pos = Point3(0,0,0); //LEXING: VERY IMPORTANT
      dm.Ls() = _Ls;    dm.Ns() = _Ns;    dm.pos() = _pos;
    }

    //---------
    //ptable
    PeriodTable ptable;
    {
      esdf_string((char*)("PeriodTable"), (char*)("../ptable.bin"), strtmp);
      string ptablefile(strtmp);
      iC( ptable.setup(ptablefile) );
    }


    //---------
    //atoms
    vector<Atom> atomvec_in;
    {
      int natp = esdf_integer((char*)("Atom_Types_Num"),0);
      if (natp == 0) {
	ABORT("ERROR: unknown number of atom types.\r\n"
	      "'Atom_Types_Num' not found in inputfile!\r\n",1);
      } /* end of if:natp statement */

      for(int ityp = 0; ityp < natp; ityp++){
	char astring[3];
	esdf_string((char*)("Atom_Type"), (char*)("H"), astring);
	//strupr(astring);
	//LEXING: string to integer
	string tmp(astring);
	int atype = 0;
	if(     tmp=="H")
	  atype=1;
	else if(tmp=="Li")
	  atype=3;
	else if(tmp=="Be")
	  atype=4;
	else if(tmp=="B")
	  atype=5;
	else if(tmp=="C")
	  atype=6;
	else if(tmp=="N")
	  atype=7;
	else if(tmp=="O")
	  atype=8;
	else if(tmp=="F")
	  atype=9;
	else if(tmp=="Na")
	  atype=11;
	else if(tmp=="Mg")
	  atype=12;
	else if(tmp=="Al")
	  atype=13;
	else if(tmp=="Si")
	  atype=14;
	else if(tmp=="P")
	  atype=15;
	else if(tmp=="S")
	  atype=16;
	else if(tmp=="Cl")
	  atype=17;
	else {
	  cerr<<tmp<<endl;
	  ABORT("ERROR: atom type not supported",1);
	}
	iA(atype>0);
	int natoms;
	if (!esdf_block((char*)("Atom_Coord"),&natoms)) {
	  printf("ERROR: no atom coordinates were found for atom type"
		 " %s\r\n",astring);
	}
	double xx,yy,zz;


	double mass = ptable.ptemap()[atype].params()(PeriodTable::i_mass);
	mass *= amu2au;  // LL: VERY IMPORTANT. The mass in PeriodTable 
	// is in atomic mass unit (amu), but the mass in
	// atomvec is in atomic unit (au).
	for (int j=0;j<natoms;j++) {
	  sscanf(block_data[j],"%lf %lf %lf", &xx,&yy,&zz);
	  atomvec_in.push_back( Atom(atype, mass, Point3(xx,yy,zz), 
				     Point3(0,0,0), Point3(0,0,0)) );
	}
      }
    }

    {
      scf._dm = dm;
      scf._atomvec = atomvec_in;   // LL: Initial position
      scf._ptable = ptable;
      scf._posidx = Index3(0,0,0); //LY: VERY IMPORTANT

      double temperature;
      double AU2K = 315774.67;

      scf._mixdim      = esdf_integer((char*)("Max_Mixing"), 9);
      esdf_string((char*)("Mixing_Type"), (char*)("anderson"), strtmp);
      scf._mixtype     = strtmp;
      scf._alpha       = esdf_double((char*)("Mixing_Alpha"), 0.8);
      temperature        = esdf_double((char*)("Temperature"), 80.0);
      scf._Tbeta       = AU2K / temperature;   

      scf._scftol      = esdf_double((char*)("SCF_Tolerance"), 1e-4);
      scf._scfmaxiter  = esdf_integer((char*)("SCF_Maxiter"), 30);

      scf._eigtol      = esdf_double((char*)("Eig_Tolerance"), 1e-5);
      scf._eigmaxiter  = esdf_integer((char*)("Eig_Maxiter"), 10);

      scf._nExtraStates = esdf_integer((char*)("Extra_States"), 0);

      esdf_string((char*)("Pseudo_Type"), (char*)("GTH"), strtmp);
      scf._pseudotype   = strtmp;
    }
    //----------

    //----------
    iC( scf.setup() );  //cerr<<"LY "<<scf._scfmaxiter<<" "<<scf._scftol<<endl;
    //----------
    //inital guess?
    {
      esdf_string((char*)("Restart_Mode"), (char*)("from_scratch"), strtmp);
      string restartmode = strtmp;

      if(restartmode == string("from_scratch") )
	rhoinput = scf._rho0;
      if(restartmode == string("restart") ){
	ifstream rhoid;
	Index3 Ns;
	Point3 Ls;      //DblNumVec rho(scf.ntot(), false, rhoinput
	int ntot;
	//
	esdf_string((char*)("Restart_Density"), (char*)("rho_dg.dat"), strtmp);
	string restart_density = strtmp;
	rhoid.open(restart_density.c_str(), ios::in);
	iA(rhoid.fail() == false);
	rhoid >> Ns >> Ls;
	iA(scf._Ns == Ns);
	//read vector
	rhoid >> ntot;      iA( ntot==Ns(0)*Ns(1)*Ns(2) );
	rhoinput.resize(ntot);      for(int i=0; i<ntot; i++)	rhoid >> rhoinput[i];
	rhoid.close();
      }
    }
    // initial guess for wave functions?
    {
      esdf_string((char*)("Restart_Wave_Mode"), (char*)("from_scratch"), strtmp);
      string restartmode = strtmp;

      if(restartmode == string("from_scratch") ){
	psiinput.resize(scf._npsi*scf._ntot);
	for (int i=0; i<scf._npsi*scf._ntot; i++) psiinput[i] = dunirand();
      }
      if(restartmode == string("restart") ){
	ifstream psiid;
	int npsi, ntot, tempg;
	Point3 Ls;
	Index3 Ns;
	vector<double> occ;


	esdf_string((char*)("Restart_Wave"), (char*)("wfn_dg.dat"), strtmp);
	string restart_wave = strtmp;
	psiid.open(restart_wave.c_str(), ios::in);
	iA(psiid.fail() == false);
	psiid >> npsi >> Ns >> Ls;
	iA(scf._Ns == Ns);
	iA(scf._npsi == npsi);

	occ.resize(npsi);
	for(int i=0; i<npsi; i++) psiid >> occ[i];
	//read vector
	psiinput.resize(scf._npsi*scf._ntot);
	for(int i=0; i<npsi; i++){
	  psiid >> tempg;  iA(tempg == i);
	  psiid >> ntot;  iA(ntot == Ns(0)*Ns(1)*Ns(2) );
	  for(int j=0; j<ntot; j++)	psiid >> psiinput[i*ntot+j];
	}
	psiid.close();
      }
    }
  }


  if( inputformat == "v1.2" ){

    //***************************************************************
    //Input format  v1.2
    //  Support reduced coordinate (Atom_Red)
    //***************************************************************

    //---------
    //domain
    Domain dm;
    {
      int nlines;
      //
      Point3 _Ls;
      Index3 _Ns;
      Point3 _pos;
      if (esdf_block((char*)("Super_Cell"),&nlines)) {
	sscanf(block_data[0],"%lf %lf %lf",
	       &_Ls[0],&_Ls[1],&_Ls[2]);
      } else {
	ABORT("ERROR: bad specification of 'Super_Cell'",1);
      }
      if (esdf_block((char*)("Grid_Size"),&nlines)) {
	sscanf(block_data[0],"%d %d %d",
	       &_Ns[0],&_Ns[1],&_Ns[2]);
      }  else {
	ABORT("ERROR: bad specification of 'Grid_Size'",1);
      } /* end of if:esdf_block statement */
      _pos = Point3(0,0,0); //LEXING: VERY IMPORTANT
      dm.Ls() = _Ls;    dm.Ns() = _Ns;    dm.pos() = _pos;
    }

    //---------
    //ptable
    PeriodTable ptable;
    {
      esdf_string((char*)("PeriodTable"), (char*)("../ptable.bin"), strtmp);
      string ptablefile(strtmp);
      iC( ptable.setup(ptablefile) );
    }


    //---------
    //atoms
    vector<Atom> atomvec_in;
    {
      int natp = esdf_integer((char*)("Atom_Types_Num"),0);
      if (natp == 0) {
	ABORT("ERROR: unknown number of atom types.\r\n"
	      "'Atom_Types_Num' not found in inputfile!\r\n",1);
      } /* end of if:natp statement */

      for(int ityp = 0; ityp < natp; ityp++){
	char astring[3];
	esdf_string((char*)("Atom_Type"), (char*)("H"), astring);
	//strupr(astring);
	//LEXING: string to integer
	string tmp(astring);
	int atype = 0;
	if(     tmp=="H")
	  atype=1;
	else if(tmp=="Li")
	  atype=3;
	else if(tmp=="Be")
	  atype=4;
	else if(tmp=="B")
	  atype=5;
	else if(tmp=="C")
	  atype=6;
	else if(tmp=="N")
	  atype=7;
	else if(tmp=="O")
	  atype=8;
	else if(tmp=="F")
	  atype=9;
	else if(tmp=="Na")
	  atype=11;
	else if(tmp=="Mg")
	  atype=12;
	else if(tmp=="Al")
	  atype=13;
	else if(tmp=="Si")
	  atype=14;
	else if(tmp=="P")
	  atype=15;
	else if(tmp=="S")
	  atype=16;
	else if(tmp=="Cl")
	  atype=17;
	else {
	  cerr<<tmp<<endl;
	  ABORT("ERROR: atom type not supported",1);
	}
	iA(atype>0);
	int natoms;
	
	double mass = ptable.ptemap()[atype].params()(PeriodTable::i_mass);
	mass *= amu2au;  // LL: VERY IMPORTANT. The mass in PeriodTable 
	// is in atomic mass unit (amu), but the mass in
	// atomvec is in atomic unit (au).
        
	if (esdf_block((char*)("Atom_Cart"),&natoms)) {
	  // Cartesian coordinate (in the unit of Bohr)
	  double xx,yy,zz;
	  for (int j=0;j<natoms;j++) {
	    sscanf(block_data[j],"%lf %lf %lf", &xx,&yy,&zz);
	    atomvec_in.push_back( Atom(atype, mass, Point3(xx,yy,zz), 
				       Point3(0,0,0), Point3(0,0,0)) );
	  }
	}
	else if (esdf_block((char*)("Atom_Red"),&natoms)) {
	  // Reduce coordinate (in the unit of Super_Cell)
	  double xx,yy,zz;
	  Point3 Ls = dm.Ls();
	  for (int j=0;j<natoms;j++) {
	    sscanf(block_data[j],"%lf %lf %lf", &xx,&yy,&zz);
	    atomvec_in.push_back( Atom(atype, mass, 
				       Point3(xx*Ls[0],yy*Ls[1],zz*Ls[2]), 
				       Point3(0,0,0), Point3(0,0,0)) );
	  }
	}
	else{
	  fprintf(fhstat, "ERROR: no atom coordinates were found for atom type"
		 " %s\r\n",astring);
	  fprintf(stderr, "ERROR: no atom coordinates were found for atom type"
		 " %s\r\n",astring);
	}
      }
    }

    {
      scf._dm = dm;
      scf._atomvec = atomvec_in;   // LL: Initial position
      scf._ptable = ptable;
      scf._posidx = Index3(0,0,0); //LY: VERY IMPORTANT

      double temperature;
      double AU2K = 315774.67;

      scf._mixdim      = esdf_integer((char*)("Max_Mixing"), 9);
      esdf_string((char*)("Mixing_Type"), (char*)("anderson"), strtmp);
      scf._mixtype     = strtmp;
      scf._alpha       = esdf_double((char*)("Mixing_Alpha"), 0.8);
      temperature        = esdf_double((char*)("Temperature"), 80.0);
      scf._Tbeta       = AU2K / temperature;   

      scf._scftol      = esdf_double((char*)("SCF_Tolerance"), 1e-4);
      scf._scfmaxiter  = esdf_integer((char*)("SCF_Maxiter"), 30);

      scf._eigtol      = esdf_double((char*)("Eig_Tolerance"), 1e-5);
      scf._eigmaxiter  = esdf_integer((char*)("Eig_Maxiter"), 10);

      scf._nExtraStates = esdf_integer((char*)("Extra_States"), 0);

      esdf_string((char*)("Pseudo_Type"), (char*)("GTH"), strtmp);
      scf._pseudotype   = strtmp;
    }
    //----------

    //----------
    iC( scf.setup() );  //cerr<<"LY "<<scf._scfmaxiter<<" "<<scf._scftol<<endl;
    //----------
    //inital guess?
    {
      esdf_string((char*)("Restart_Mode"), (char*)("from_scratch"), strtmp);
      string restartmode = strtmp;

      if( restartmode == string("from_scratch") )
	rhoinput = scf._rho0;
      if( restartmode == string("restart") ){
	ifstream rhoid;
	Index3 Ns;
	Point3 Ls;      //DblNumVec rho(scf.ntot(), false, rhoinput
	int ntot;
	//
	esdf_string((char*)("Restart_Density"), (char*)("rho_dg.dat"), strtmp);
	string restart_density = strtmp;
	rhoid.open(restart_density.c_str(), ios::in);
	iA(rhoid.fail() == false);
	rhoid >> Ns >> Ls;
	iA(scf._Ns == Ns);
	//read vector
	rhoid >> ntot;      iA( ntot==Ns(0)*Ns(1)*Ns(2) );
	rhoinput.resize(ntot);      for(int i=0; i<ntot; i++)	rhoid >> rhoinput[i];
	rhoid.close();
      }
    }
    // initial guess for wave functions?
    {
      esdf_string((char*)("Restart_Wave_Mode"), (char*)("from_scratch"), strtmp);
      string restartmode = strtmp;

      if(restartmode == string("from_scratch") ){
	psiinput.resize(scf._npsi*scf._ntot);
	for (int i=0; i<scf._npsi*scf._ntot; i++) psiinput[i] = dunirand();
      }
      if(restartmode == string("restart") ){
	ifstream psiid;
	int npsi, ntot, tempg;
	Point3 Ls;
	Index3 Ns;
	vector<double> occ;


	esdf_string((char*)("Restart_Wave"), (char*)("wfn_dg.dat"), strtmp);
	string restart_wave = strtmp;
	psiid.open(restart_wave.c_str(), ios::in);
	iA(psiid.fail() == false);
	psiid >> npsi >> Ns >> Ls;
	iA(scf._Ns == Ns);
	iA(scf._npsi == npsi);

	occ.resize(npsi);
	for(int i=0; i<npsi; i++) psiid >> occ[i];
	//read vector
	psiinput.resize(scf._npsi*scf._ntot);
	for(int i=0; i<npsi; i++){
	  psiid >> tempg;  iA(tempg == i);
	  psiid >> ntot;  iA(ntot == Ns(0)*Ns(1)*Ns(2) );
	  for(int j=0; j<ntot; j++)	psiid >> psiinput[i*ntot+j];
	}
	psiid.close();
      }
    }
  
    // Reading external potential
    {
      esdf_string((char*)("Vext"), (char*)("null"), strtmp);
      string vextfile = strtmp;

      if( vextfile != string("null") ){
	// Read external potential from file
	ifstream vextid;
        vextid.open(vextfile.c_str(), ios::in);
	iA(vextid.fail() == false);
	//read vector
	Index3 Ns;
	int ntot;
	vextid >> Ns;  iA( Ns == scf._Ns);
	vextid >> ntot;      iA( ntot==scf.ntot() );
	scf._vext.resize(ntot);
	for(int i=0; i<ntot; i++)	vextid >> scf._vext[i];
	vextid.close();
      }
      else{
	int ntot = scf.ntot();
	scf._vext.resize(ntot);
	for(int i=0; i<ntot; i++)	scf._vext[i] = 0.0;
      }
    }
 
    // Output the external potential. Test purpose
    if( 0 ){
      ofstream vextid;
      string strtmp = string("vext_glb.dat");
      vextid.open(strtmp.c_str(), ios::out | ios::trunc);
      vextid << scf._Ns << endl;
      vextid << scf._ntot<<endl;
      for(int i=0; i<scf._ntot; i++)	  vextid<<scf._vext[i]<<" ";
      vextid << endl;
      vextid.close();
      fclose(fhstat);
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
    }
  
  }


  //---------
  //Initialze Molecular dynamics simulation
  double T_init = 300.0 / au2K;
  
  
  //NVE simulation. The initial velocity of each atom is only nonzero at
  //the first component with temperature equal to 300K.
  vector<Atom>& atomvec = scf._atomvec;

  //Initialize the state variables
  double Eatomkin, Hconv, Hconv0, drift;

  // The velocity of the centroid is 0.  Formula to be improved later.
  for(vector<Atom>::iterator mi=atomvec.begin(); 
      mi != atomvec.end(); mi++){
    int j = mi - atomvec.begin();
    double mass = (*mi).mass();
    Point3& vel = (*mi).vel();
    if( j%2 == 0 )
      vel = Point3(sqrt(2.0*T_init/mass), 0.0, 0.0);
    else
      vel = Point3(-sqrt(2.0*T_init/mass), 0.0, 0.0);
    if( (atomvec.size()%2 == 1) && (j == atomvec.size()-1) )
      vel = Point3(0.0, 0.0, 0.0);
  }

  //---------
  //Initial SCF step
  {
    clock_t start,end;
    double cpu_time_used;

    start = clock();


    iC( scf.scf(rhoinput, psiinput) );
    iC( scf.force() );  

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    Eatomkin = 0.0;
    for(vector<Atom>::iterator mi=atomvec.begin(); 
	mi != atomvec.end(); mi++){
      double mass = (*mi).mass();
      Point3& vel = (*mi).vel();
      for(int i = 0; i < 3; i++){
	Eatomkin += 0.5 * mass * vel[i] * vel[i];
      }
    }
    Hconv0 = Eatomkin + scf._Efree;
    fprintf(fhstat, "---------------------------------------------\n");
    fprintf(fhstat, "      Initial step\n");
    fprintf(fhstat, "---------------------------------------------\n");
    fprintf(fhstat, "Eatomkin      = %12.6e (au)\n", Eatomkin);
    fprintf(fhstat, "DFTEtot       = %12.6e (au)\n", scf._Etot);
    fprintf(fhstat, "DFTEfree      = %12.6e (au)\n", scf._Efree);
    fprintf(fhstat, "Hconv         = %12.6e (au)\n", Hconv0);
    fprintf(fhstat, "Fermi         = %12.6e (au)\n", scf._Fermi);
    fprintf(fhstat, "\nTIME ELAPSED : %10.1f sec \n",cpu_time_used);
    fprintf(fhstat, "\n-------------------------------------------------------------------------------\n");
    fprintf(fhstat, "     ATOM          COORDINATES                        FORCES\n");
    for(vector<Atom>::iterator mi=atomvec.begin(); 
	mi != atomvec.end(); mi++){
      int type = (*mi).type();
      Point3& coord = (*mi).coord();
      Point3& fs    = (*mi).force();
      fprintf(fhstat, "%6d%3d %10.4f%10.4f%10.4f %12.3e%12.3e%12.3e\n",
	      int(mi-atomvec.begin()), type, coord(0), coord(1), coord(2),
	      fs(0), fs(1), fs(2));
    }


    fprintf(fhstat, "\n-------------------------------------------------------------------------------\n");
    fprintf(fhstat, "     ATOM          VELOCITIES\n");
    for(vector<Atom>::iterator mi=atomvec.begin(); 
	mi != atomvec.end(); mi++){
      int type = (*mi).type();
      Point3& vel = (*mi).vel();
      fprintf(fhstat, "%6d%3d %12.3e%12.3e%12.3e\n",
	      int(mi-atomvec.begin()), type, vel(0), vel(1), vel(2));
    }
  } 
  
  //---------
  //Start the MD simulation
  for(int istep = 0; istep < max_step; istep++){
    clock_t start,end;
    double cpu_time_used;

    start = clock();



    for(vector<Atom>::iterator mi=atomvec.begin(); 
	mi != atomvec.end(); mi++){
      Point3& fs  = (*mi).force();
      Point3& vel = (*mi).vel();
      Point3& coord = (*mi).coord();
      double mass = (*mi).mass();
      vel += fs / mass * dt * 0.5;
      coord += vel * dt;
    }

    iC( scf.update() );  
    iC( scf.scf(scf._rho, scf._psi) );
    iC( scf.force() );  
   

    for(vector<Atom>::iterator mi=atomvec.begin(); 
	mi != atomvec.end(); mi++){
      Point3& fs  = (*mi).force();
      Point3& vel = (*mi).vel();
      double mass = (*mi).mass();
      vel += fs / mass * dt * 0.5;
    }

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    //Output the state variables
    //Kinetic energy
    Eatomkin = 0.0;
    for(vector<Atom>::iterator mi=atomvec.begin(); 
	mi != atomvec.end(); mi++){
      double mass = (*mi).mass();
      Point3& vel = (*mi).vel();
      for(int i = 0; i < 3; i++){
	Eatomkin += 0.5 * mass * vel[i] * vel[i];
      }
    }
    Hconv = Eatomkin + scf._Efree;
    drift = (Hconv-Hconv0) / Hconv0;

    fprintf(fhstat, "---------------------------------------------\n");
    fprintf(fhstat, "     Time step %10d \n", istep);
    fprintf(fhstat, "---------------------------------------------\n");
    fprintf(fhstat, "Eatomkin      = %12.6e (au)\n", Eatomkin);
    fprintf(fhstat, "DFTEtot       = %12.6e (au)\n", scf._Etot);
    fprintf(fhstat, "DFTEfree      = %12.6e (au)\n", scf._Efree);
    fprintf(fhstat, "Hconv         = %12.6e (au)\n", Hconv);
    fprintf(fhstat, "drift         = %12.6e (au)\n", drift);
    fprintf(fhstat, "Fermi         = %12.6e (au)\n", scf._Fermi);
    fprintf(fhstat, "\nTIME ELAPSED : %10.1f sec \n",cpu_time_used);
    fprintf(fhstat, "\n-------------------------------------------------------------------------------\n");
    fprintf(fhstat, "     ATOM          COORDINATES                        FORCES\n");
    for(vector<Atom>::iterator mi=atomvec.begin(); 
	mi != atomvec.end(); mi++){
      int type = (*mi).type();
      Point3& coord = (*mi).coord();
      Point3& fs    = (*mi).force();
      fprintf(fhstat, "%6d%3d %10.4f%10.4f%10.4f %12.3e%12.3e%12.3e\n",
	      int(mi-atomvec.begin()), type, coord(0), coord(1), coord(2),
	      fs(0), fs(1), fs(2));
    }


    fprintf(fhstat, "\n-------------------------------------------------------------------------------\n");
    fprintf(fhstat, "     ATOM          VELOCITIES\n");
    for(vector<Atom>::iterator mi=atomvec.begin(); 
	mi != atomvec.end(); mi++){
      int type = (*mi).type();
      Point3& vel = (*mi).vel();
      fprintf(fhstat, "%6d%3d %12.3e%12.3e%12.3e\n",
	      int(mi-atomvec.begin()), type, vel(0), vel(1), vel(2));
    }

  } 

  {
    esdf_string((char*)("Output_Dir"), (char*)("./"), strtmp);
    string outputdir = strtmp;
    inttmp          = esdf_integer((char*)("Output_Density"), 0 );
    int output_density  = (bool)inttmp;
    inttmp          = esdf_integer((char*)("Output_Wfn"), 0 );
    int output_wfn      = (bool)inttmp;
    inttmp          = esdf_integer((char*)("Output_Vtot"), 0 );
    int output_vtot      = (bool)inttmp;
    //
    if( output_wfn == true ){
      ofstream wfnid;
      string strtmp = outputdir + string("wfn_glb.dat");
      wfnid.open(strtmp.c_str(), ios::out | ios::trunc);
      wfnid << scf._npsi << endl;
      wfnid << scf._Ns << endl;
      wfnid << scf._Ls << endl;
      for(int g=0; g < scf._npsi; g++)
	wfnid << scf._occ[g] << " " << endl;
      for(int g=0; g < scf._npsi; g++){
	wfnid << g << endl;
	wfnid << scf._ntot<<endl;
	for(int i=0; i<scf._ntot; i++)	  wfnid<<scf._psi[g*scf._ntot+i]<<" ";
	wfnid<<endl;
      }
      wfnid.close();
    }
    if( output_density == true ){
      ofstream rhoid;
      string strtmp = outputdir + string("rho_glb.dat");
      rhoid.open(strtmp.c_str(), ios::out | ios::trunc);
      rhoid << scf._Ns << endl;
      rhoid << scf._Ls << endl;
      rhoid << scf._ntot<<endl;
      for(int i=0; i<scf._ntot; i++)	  rhoid<<scf._rho[i]<<" ";
      rhoid << endl;
      rhoid.close();
    }
    
    if( output_vtot == true ){
      ofstream vtotid;
      string strtmp = outputdir + string("vtot_glb.dat");
      vtotid.open(strtmp.c_str(), ios::out | ios::trunc);
      vtotid << scf._Ns << endl;
      vtotid << scf._Ls << endl;
      vtotid << scf._ntot<<endl;
      for(int i=0; i<scf._ntot; i++)	  vtotid<<scf._vtot[i]<<" ";
      vtotid << endl;
      vtotid.close();
    }

  }
  
  tottime_end = clock();
  
  fprintf(fhstat, "\n-----------------------------------------------------------\n");
  fprintf(fhstat, "TOTAL TIME ELAPSED : = %10.1f sec\n",((double) (tottime_end-tottime_sta)) / CLOCKS_PER_SEC);

  fclose(fhstat);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  
  return 0;
}
