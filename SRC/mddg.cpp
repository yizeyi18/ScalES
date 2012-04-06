#include "esdf.h"
#include "debug.hpp"
#include "scfdg.hpp"
#include "parallel.hpp"

FILE * fhstat;
int CONTXT;

//------------------------------------------------------------------------
// NVE Molecular dynamics simulation.
//------------------------------------------------------------------------
//LEXING: ALL PROCESSOR PROCEED IN PARALLEL
int main(int argc, char **argv) 
{
  MPI_Init(&argc, &argv);
  int myid;  
  int nprocs;  
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  int mpirank = myid;
  int mpisize = nprocs;

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
  char inputfile[128];
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
    
  /* Parameters used later */
  ScfDG scf;
  vector<double> rhoinput;
  double dt;
  int max_step;

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
      _pos = Point3(0,0,0); //LY: VERY IMPORTANT
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
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "done ptable\n"); }



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

    //---------
    //scfdg
    {
      scf._inputformat = inputformat;

      int INTONE = 1;
      int nlines;
      double xx, yy, zz;
      int ixx, iyy, izz;
      //----------------------------
      //Control parameters
      inttmp            = esdf_integer((char*)("Output_Bases"), 0 );
      scf._output_bases = inttmp;
       
      //----------------------------
      //
      scf._dm = dm;
      scf._atomvec = atomvec_in;
      scf._ptable = ptable;
      scf._posidx = Index3(0,0,0); //LY: IMPORTANT

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

      //EXTRA STUFF NEEDED BY DG
      //LLIN: Nonorthogonal basis use
      esdf_string((char*)("DG_Solver"), (char*)("nonorth"), strtmp);
      scf._dgsolver    = strtmp;
      scf._delta       = esdf_double((char*)("WallWidth"), 0.1);
      scf._gamma       = esdf_double((char*)("WeightRatio"),    0.01);
      scf._Neigperele  = esdf_integer((char*)("EigPerEle"), 4);
      scf._Norbperele  = esdf_integer((char*)("OrbPerEle"), 7);

      //Standard DG 
      scf._nBuffUpdate = esdf_integer((char*)("Buff_Update"), 1);
      scf._dgalpha     = esdf_double((char*)("DG_Alpha"), 500.0);
      scf._dgndeg      = esdf_integer((char*)("DG_Degree"), 3);
      scf._nenrich     = esdf_integer((char*)("Enrich_Number"), 30 );
      scf._MB          = esdf_integer((char*)("MB"), 128 );
      Index3 NElems(1,1,1);
      if (esdf_block((char*)("Element_Size"),&nlines)) {
	sscanf(block_data[0],"%d %d %d", &NElems[0],&NElems[1],&NElems[2]);
      }
      scf._NElems = NElems;
      //
      NumTns<Domain> bufftns(NElems[0],NElems[1],NElems[2]);
      for(int k=0; k<NElems[2]; k++)
	for(int j=0; j<NElems[1]; j++)
	  for(int i=0; i<NElems[0]; i++) {
	    Domain& buff = bufftns(i,j,k);
	    if (!esdf_block((char*)("Buffer_Cell"), &INTONE)) {
	      printf("ERROR: no cell length were found for buffer %4d %4d %4d.\n", i,j,k);
	    }
	    sscanf(block_data[0],"%lf %lf %lf", &xx,&yy,&zz);
	    buff._Ls = Point3(xx, yy, zz);
	    if (!esdf_block((char*)("Buffer_Grid_Size"), &INTONE)) {
	      printf("ERROR: no grid size were found for buffer %4d %4d %4d.\n", i,j,k);
	    }
	    sscanf(block_data[0],"%d %d %d", &ixx,&iyy,&izz);
	    buff._Ns = Index3(ixx, iyy, izz);
	    if (!esdf_block((char*)("Buffer_Position_Start"), &INTONE)) {
	      printf("ERROR: no starting positions were found for buffer %4d %4d %4d.\n", i,j,k);
	    }
	    sscanf(block_data[0],"%lf %lf %lf", &xx,&yy,&zz);
	    buff._pos = Point3(xx, yy, zz);
	  }
      scf._bufftns = bufftns;
      //
      NumTns<Domain> elemtns(NElems[0],NElems[1],NElems[2]);
      for(int k=0; k<NElems[2]; k++)
	for(int j=0; j<NElems[1]; j++)
	  for(int i=0; i<NElems[0]; i++) {
	    Domain& elem = elemtns(i,j,k);
	    if (!esdf_block((char*)("Element_Cell"), &INTONE)) {
	      printf("ERROR: no cell length were found for element %4d %4d %4d.\n", i,j,k);
	    }
	    sscanf(block_data[0],"%lf %lf %lf", &xx,&yy,&zz);
	    elem._Ls = Point3(xx, yy, zz);
	    if (!esdf_block((char*)("Element_Grid_Size"), &INTONE)) {
	      printf("ERROR: no grid size were found for element %4d %4d %4d.\n", i,j,k);
	    }
	    sscanf(block_data[0],"%d %d %d", &ixx,&iyy,&izz);
	    elem._Ns = Index3(ixx, iyy, izz);
	    if (!esdf_block((char*)("Element_Position_Start"), &INTONE)) {
	      printf("ERROR: no starting positions were found for element %4d %4d %4d.\n", i,j,k);
	    }
	    sscanf(block_data[0],"%lf %lf %lf", &xx,&yy,&zz);
	    elem._pos = Point3(xx, yy, zz);
	  }
      scf._elemtns = elemtns;
      //
      int elemtot = NElems[0] * NElems[1] * NElems[2];
      iA(mpisize==elemtot);
      IntNumTns elemptninfo(NElems[0],NElems[1],NElems[2]);
      int cnt = 0;
      for(int k=0; k<NElems[2]; k++)
	for(int j=0; j<NElems[1]; j++)
	  for(int i=0; i<NElems[0]; i++) {
	    elemptninfo(i,j,k) = cnt;
	    cnt++;
	  }
      iA(cnt==elemtot);
      scf._elemptninfo = elemptninfo;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf read done\n"); }

    //----------
    //MD parameters
    dt = esdf_double((char*)("Time_Step"), 100.0);
    max_step = esdf_integer((char*)("Max_Step"), 1);

    //----------
    iC( scf.setup() );
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf setup done\n"); }
    //----------
    //inital guess?
    {
      esdf_string((char*)("Restart_Mode"), (char*)("from_scratch"), strtmp);
      string restartmode = strtmp;

      if(restartmode == string("from_scratch") )
	rhoinput = scf._rho0;
      if(restartmode == string("restart") ){
	Index3 Ns;
	Point3 Ls;      //DblNumVec rho(scf.ntot(), false, rhoinput
	int ntot;
	//
	esdf_string((char*)("Restart_Density"), (char*)("rho_dg.dat"), strtmp);
	string restart_density = strtmp;
	istringstream rhoid;      iC( Shared_Read(restart_density, rhoid) );
	rhoid >> Ns >> Ls;      //cerr<<Ns<<endl<<Ls<<endl;
	iA(scf._Ns == Ns);
	//read vector
	rhoid >> ntot;
	rhoinput.resize(ntot);
	for(int i=0; i<ntot; i++)
	  rhoid >> rhoinput[i];
	//fprintf(stderr, "ntot %d\n", ntot);
      }
    }
    fprintf(stderr, "proc %d, rhoinput val %f\n", myid, rhoinput[myid]);
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf guess done\n"); }
  }

  if( inputformat == "v1.1" ){

    //***************************************************************
    //Input format  v1.1: Simplified input format
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
      _pos = Point3(0,0,0); //LY: VERY IMPORTANT
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
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "done ptable\n"); }


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

    //---------
    //scfdg
    {
      scf._inputformat = inputformat;

      int INTONE = 1;
      int nlines;
      double xx, yy, zz;
      int ixx, iyy, izz;
      //----------------------------
      //Control parameters
      inttmp            = esdf_integer((char*)("Output_Bases"), 0 );
      scf._output_bases = inttmp;
      
      //
      scf._dm = dm;
      scf._atomvec = atomvec_in;
      scf._ptable = ptable;
      scf._posidx = Index3(0,0,0); //LY: IMPORTANT

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

      //EXTRA STUFF NEEDED BY DG
      //LLIN: Nonorthogonal basis use
      esdf_string((char*)("DG_Solver"), (char*)("nonorth"), strtmp);
      scf._dgsolver    = strtmp;
      scf._delta       = esdf_double((char*)("WallWidth"), 0.1);
      scf._gamma       = esdf_double((char*)("WeightRatio"),    0.00);
      scf._Neigperele  = esdf_integer((char*)("EigPerEle"), 4);
      scf._Norbperele  = esdf_integer((char*)("OrbPerEle"), 7);

      //Standard DG 
      scf._nBuffUpdate = esdf_integer((char*)("Buff_Update"), 1);
      scf._dgalpha     = esdf_double((char*)("DG_Alpha"), 500.0);
      scf._dgndeg      = esdf_integer((char*)("DG_Degree"), 3);
      scf._nenrich     = esdf_integer((char*)("Enrich_Number"), 30 );
      scf._MB          = esdf_integer((char*)("MB"), 128 );
      Index3 NElems(1,1,1);
      if (esdf_block((char*)("Element_Size"),&nlines)) {
	sscanf(block_data[0],"%d %d %d", &NElems[0],&NElems[1],&NElems[2]);
      }
      scf._NElems = NElems;

      if( (dm._Ns[0]%NElems[0]) || (dm._Ns[1]%NElems[1]) || 
	  (dm._Ns[2]%NElems[2]) ){
	ABORT("ERROR: Adjusted the Grid_size so it can be divided exactly by NElems!", 1);
      }

      Point3 ExtRatio(2.0, 2.0, 2.0); 
      if (esdf_block((char*)("Extended_Element_Ratio"),&nlines)) {
	sscanf(block_data[0],"%lf %lf %lf", &ExtRatio[0],&ExtRatio[1],&ExtRatio[2]);
      }
      scf._ExtRatio = ExtRatio;

      if(mpirank == 0){ cerr << ExtRatio<< endl; }
      
      //-----
      //Setup the elements, uniformly distributed
      Index3 ElemNs(1, 1, 1);
      if (!esdf_block((char*)("Element_Grid_Size"), &INTONE)) {
	printf("ERROR: no grid size for element was found.\n");
      }
      sscanf(block_data[0],"%d %d %d", &ElemNs[0],&ElemNs[1],&ElemNs[2]);


      Point3 ElemLs;
      ElemLs[0] = dm._Ls[0] / NElems[0];
      ElemLs[1] = dm._Ls[1] / NElems[1];
      ElemLs[2] = dm._Ls[2] / NElems[2];

      NumTns<Domain> elemtns(NElems[0],NElems[1],NElems[2]);
      for(int k=0; k<NElems[2]; k++)
	for(int j=0; j<NElems[1]; j++)
	  for(int i=0; i<NElems[0]; i++) {

	    Domain& elem = elemtns(i,j,k);
	    elem._Ns = ElemNs;
	    elem._Ls = ElemLs;
      
      	    xx = i * ElemLs[0];
	    yy = j * ElemLs[1];
	    zz = k * ElemLs[2];
	    elem._pos = Point3(xx, yy, zz);
	  }
      scf._elemtns = elemtns;
      
      // LLIN: FIXME for more general partition strategy
      int elemtot = NElems[0] * NElems[1] * NElems[2];
      iA(mpisize==elemtot);
      IntNumTns elemptninfo(NElems[0],NElems[1],NElems[2]);
      int cnt = 0;
      for(int k=0; k<NElems[2]; k++)
	for(int j=0; j<NElems[1]; j++)
	  for(int i=0; i<NElems[0]; i++) {
	    elemptninfo(i,j,k) = cnt;
	    cnt++;
	  }
      iA(cnt==elemtot);
      scf._elemptninfo = elemptninfo;

      // Printout the information for elements
      for(int k=0; k<NElems[2]; k++)
	for(int j=0; j<NElems[1]; j++)
	  for(int i=0; i<NElems[0]; i++) {
	    if( scf._elemptninfo(i,j,k) == mpirank ){
	      Domain& elem = scf._elemtns(i,j,k);
	      fprintf(fhstat, "Element [%4d, %4d, %4d]\n", 
		      i, j, k);
	      fprintf(fhstat,"Elem.Ls               = %10.5f %10.5f %10.5f\n",
		      elem._Ls[0], elem._Ls[1], elem._Ls[2] );
	      fprintf(fhstat,"Elem.Ns               = %10d %10d %10d\n",
		      elem._Ns[0], elem._Ns[1], elem._Ns[2] );
	      fprintf(fhstat,"Elem.pos              = %10.5f %10.5f %10.5f\n",
		      elem._pos[0], elem._pos[1], elem._pos[2] );
	    }
	  }

      //------
      //Setup the buffer
      
      NumTns<Domain> bufftns(NElems[0],NElems[1],NElems[2]);

      //Grid spacing for global domain along each dimension
      Point3 dh;
      dh[0] = dm._Ls[0] / dm._Ns[0];
      dh[1] = dm._Ls[1] / dm._Ns[1];
      dh[2] = dm._Ls[2] / dm._Ns[2];

      Point3 BufferLs;
      BufferLs[0] = ElemLs[0] * ExtRatio[0];
      BufferLs[1] = ElemLs[1] * ExtRatio[1];
      BufferLs[2] = ElemLs[2] * ExtRatio[2];

      // Adjust the ratio of extended element if not divided by dh
      Point3 adjratio;
      adjratio[0] = round(BufferLs[0]/dh[0]) * dh[0] / BufferLs[0];
      adjratio[1] = round(BufferLs[1]/dh[1]) * dh[1] / BufferLs[1];
      adjratio[2] = round(BufferLs[2]/dh[2]) * dh[2] / BufferLs[2];

      ExtRatio = ewmul(ExtRatio, adjratio);
      BufferLs = ewmul(ElemLs, ExtRatio);

      if(mpirank==0) { 
	fprintf(stderr, "Adjusted ExtRatio: [%10.5f %10.5f %10.5f]\n",
		ExtRatio[0], ExtRatio[1], ExtRatio[2]); 
      }

      Point3 BufferRatio;
      BufferRatio[0] = (ExtRatio[0] - 1.0) / 2.0;
      BufferRatio[1] = (ExtRatio[1] - 1.0) / 2.0;
      BufferRatio[2] = (ExtRatio[2] - 1.0) / 2.0;

      Index3 BufferNs;
      BufferNs[0] = round(BufferLs[0]/dh[0]);
      BufferNs[1] = round(BufferLs[1]/dh[1]);
      BufferNs[2] = round(BufferLs[2]/dh[2]);

      
      //
      for(int k=0; k<NElems[2]; k++)
	for(int j=0; j<NElems[1]; j++)
	  for(int i=0; i<NElems[0]; i++) {
	    Domain& buff = bufftns(i,j,k);
	    buff._Ls = BufferLs;
	    buff._Ns = BufferNs;

            xx = (-BufferRatio[0]+i)*ElemLs[0];
	    yy = (-BufferRatio[1]+j)*ElemLs[1];
	    zz = (-BufferRatio[2]+k)*ElemLs[2];

	    xx = round(xx/dh[0])*dh[0];
	    yy = round(yy/dh[1])*dh[1];
	    zz = round(zz/dh[2])*dh[2];

	    buff._pos = Point3(xx, yy, zz);
	  }
      scf._bufftns = bufftns;
      
      // Printout the information for buffer
      for(int k=0; k<NElems[2]; k++)
	for(int j=0; j<NElems[1]; j++)
	  for(int i=0; i<NElems[0]; i++) {
	    if( scf._elemptninfo(i,j,k) == mpirank ){
	      Domain& buff = scf._bufftns(i,j,k);
	      fprintf(fhstat, "Extended element [%4d, %4d, %4d]\n", 
		      i, j, k);
	      fprintf(fhstat,"Buff.Ls               = %10.5f %10.5f %10.5f\n",
		      buff._Ls[0], buff._Ls[1], buff._Ls[2] );
	      fprintf(fhstat,"Buff.Ns               = %10d %10d %10d\n",
		      buff._Ns[0], buff._Ns[1], buff._Ns[2] );
	      fprintf(fhstat,"Buff.pos              = %10.5f %10.5f %10.5f\n",
		      buff._pos[0], buff._pos[1], buff._pos[2] );
	    }
	  }
    }
    
   
  
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf read done\n"); }


    //----------
    //MD parameters
    dt = esdf_double((char*)("Time_Step"), 100.0);
    max_step = esdf_integer((char*)("Max_Step"), 1);

    //----------
    iC( scf.setup() );
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf setup done\n"); }
    //----------
    //inital guess?
    {
      esdf_string((char*)("Restart_Mode"), (char*)("from_scratch"), strtmp);
      string restartmode = strtmp;

      if(restartmode == string("from_scratch") )
	rhoinput = scf._rho0;
      if(restartmode == string("restart") ){
	Index3 Ns;
	Point3 Ls;      //DblNumVec rho(scf.ntot(), false, rhoinput
	int ntot;
	//
	esdf_string((char*)("Restart_Density"), (char*)("rho_dg.dat"), strtmp);
	string restart_density = strtmp;
	istringstream rhoid;      iC( Shared_Read(restart_density, rhoid) );
	rhoid >> Ns >> Ls;      //cerr<<Ns<<endl<<Ls<<endl;
	iA(scf._Ns == Ns);
	//read vector
	rhoid >> ntot;
	rhoinput.resize(ntot);
	for(int i=0; i<ntot; i++)
	  rhoid >> rhoinput[i];
	//fprintf(stderr, "ntot %d\n", ntot);
      }
    }
    fprintf(stderr, "proc %d, rhoinput val %f\n", myid, rhoinput[myid]);
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf guess done\n"); }
  }

  if( inputformat == "v1.2" ){

    //***************************************************************
    //Input format  v1.2: 
    //  Support reduced coordinate (Atom_Red)
    //  WallWidth parameter can also be specified by the absolute value
    //  of  Basis_Radius
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
      _pos = Point3(0,0,0); //LY: VERY IMPORTANT
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
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "done ptable\n"); }


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

    //---------
    //scfdg
    {
      scf._inputformat = inputformat;

      int INTONE = 1;
      int nlines;
      double xx, yy, zz;
      int ixx, iyy, izz;
      //----------------------------
      //Control parameters
      inttmp            = esdf_integer((char*)("Output_Bases"), 0 );
      scf._output_bases = inttmp;
      
      //
      scf._atomvec = atomvec_in;
      scf._ptable = ptable;
      scf._posidx = Index3(0,0,0); //LY: IMPORTANT

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

      //EXTRA STUFF NEEDED BY DG
      //LLIN: Nonorthogonal basis use
      esdf_string((char*)("DG_Solver"), (char*)("nonorth"), strtmp);
      scf._dgsolver    = strtmp;
      scf._delta       = esdf_double((char*)("WallWidth"), 0.0);
      scf._basisradius = esdf_double((char*)("Basis_Radius"), 0.0);
      scf._gamma       = esdf_double((char*)("WeightRatio"),    0.00);
      scf._Neigperele  = esdf_integer((char*)("EigPerEle"), 4);
      scf._DeltaFermi  = esdf_double((char*)("DeltaFermi"), 0.0);
      scf._Norbperele  = esdf_integer((char*)("OrbPerEle"), 7);
      scf._bufdual     = esdf_integer((char*)("Buf_Dual"), 0); 


      //Standard DG 
      scf._nBuffUpdate = esdf_integer((char*)("Buff_Update"), 1);
      scf._dgalpha     = esdf_double((char*)("DG_Alpha"), 500.0);
      scf._dgndeg      = esdf_integer((char*)("DG_Degree"), 3);
      scf._nenrich     = esdf_integer((char*)("Enrich_Number"), 30 );
      scf._MB          = esdf_integer((char*)("MB"), 128 );
      Index3 NElems(1,1,1);
      if (esdf_block((char*)("Element_Size"),&nlines)) {
	sscanf(block_data[0],"%d %d %d", &NElems[0],&NElems[1],&NElems[2]);
      }
      scf._NElems = NElems;

      // LLIN: Readjust the number of grids in the global domain suitable for
      // DG calculation
      if( scf._bufdual == 0 ){
	// Buffer grid size == Global grid size.  The global grid size
	// should be a multiple of NElems along each dimension
	dm._Ns[0] = int(ceil((double)dm._Ns[0]/(double)NElems[0]))*NElems[0];
	dm._Ns[1] = int(ceil((double)dm._Ns[1]/(double)NElems[1]))*NElems[1];
	dm._Ns[2] = int(ceil((double)dm._Ns[2]/(double)NElems[2]))*NElems[2];
      }
      else{
	// Buffer grid size == 2* Global grid size.  The global grid size
	// should be a multiple of (2*NElems) along each dimension
	dm._Ns[0] = int(ceil((double)dm._Ns[0]/(double)(2.0*NElems[0])))*(2.0*NElems[0]);
	dm._Ns[1] = int(ceil((double)dm._Ns[1]/(double)(2.0*NElems[1])))*(2.0*NElems[1]);
	dm._Ns[2] = int(ceil((double)dm._Ns[2]/(double)(2.0*NElems[2])))*(2.0*NElems[2]);
      }
      scf._dm = dm;

      if(mpirank==0) { 
	fprintf(stderr, "Adjusted grid size in the global domain: [%10d %10d %10d]\n",
		dm._Ns[0], dm._Ns[1], dm._Ns[2]);
      }

      Point3 ExtRatio(2.0, 2.0, 2.0); 
      if (esdf_block((char*)("Extended_Element_Ratio"),&nlines)) {
	sscanf(block_data[0],"%lf %lf %lf", &ExtRatio[0],&ExtRatio[1],&ExtRatio[2]);
      }
      scf._ExtRatio = ExtRatio;

      if(mpirank == 0){ cerr << ExtRatio<< endl; }
      
      //-----
      //Setup the elements, uniformly distributed
      Index3 ElemNs(1, 1, 1);
      if (!esdf_block((char*)("Element_Grid_Size"), &INTONE)) {
	printf("ERROR: no grid size for element was found.\n");
      }
      sscanf(block_data[0],"%d %d %d", &ElemNs[0],&ElemNs[1],&ElemNs[2]);


      Point3 ElemLs;
      ElemLs[0] = dm._Ls[0] / NElems[0];
      ElemLs[1] = dm._Ls[1] / NElems[1];
      ElemLs[2] = dm._Ls[2] / NElems[2];

      NumTns<Domain> elemtns(NElems[0],NElems[1],NElems[2]);
      for(int k=0; k<NElems[2]; k++)
	for(int j=0; j<NElems[1]; j++)
	  for(int i=0; i<NElems[0]; i++) {

	    Domain& elem = elemtns(i,j,k);
	    elem._Ns = ElemNs;
	    elem._Ls = ElemLs;
      
      	    xx = i * ElemLs[0];
	    yy = j * ElemLs[1];
	    zz = k * ElemLs[2];
	    elem._pos = Point3(xx, yy, zz);
	  }
      scf._elemtns = elemtns;
      
      // LLIN: FIXME for more general partition strategy
      int elemtot = NElems[0] * NElems[1] * NElems[2];
      iA(mpisize==elemtot);
      IntNumTns elemptninfo(NElems[0],NElems[1],NElems[2]);
      int cnt = 0;
      for(int k=0; k<NElems[2]; k++)
	for(int j=0; j<NElems[1]; j++)
	  for(int i=0; i<NElems[0]; i++) {
	    elemptninfo(i,j,k) = cnt;
	    cnt++;
	  }
      iA(cnt==elemtot);
      scf._elemptninfo = elemptninfo;

      // Printout the information for elements
      for(int k=0; k<NElems[2]; k++)
	for(int j=0; j<NElems[1]; j++)
	  for(int i=0; i<NElems[0]; i++) {
	    if( scf._elemptninfo(i,j,k) == mpirank ){
	      Domain& elem = scf._elemtns(i,j,k);
	      fprintf(fhstat, "Element [%4d, %4d, %4d]\n", 
		      i, j, k);
	      fprintf(fhstat,"Elem.Ls               = %10.5f %10.5f %10.5f\n",
		      elem._Ls[0], elem._Ls[1], elem._Ls[2] );
	      fprintf(fhstat,"Elem.Ns               = %10d %10d %10d\n",
		      elem._Ns[0], elem._Ns[1], elem._Ns[2] );
	      fprintf(fhstat,"Elem.pos              = %10.5f %10.5f %10.5f\n",
		      elem._pos[0], elem._pos[1], elem._pos[2] );
	    }
	  }

      //------
      //Setup the buffer
      NumTns<Domain> bufftns(NElems[0],NElems[1],NElems[2]);
      
      if( scf._bufdual == 0 ){ // No dual grid
	//Grid spacing for global domain along each dimension
	Point3 dh;
	dh[0] = dm._Ls[0] / dm._Ns[0];
	dh[1] = dm._Ls[1] / dm._Ns[1];
	dh[2] = dm._Ls[2] / dm._Ns[2];

	Point3 BufferLs;
	BufferLs[0] = ElemLs[0] * ExtRatio[0];
	BufferLs[1] = ElemLs[1] * ExtRatio[1];
	BufferLs[2] = ElemLs[2] * ExtRatio[2];

	// Adjust the ratio of extended element if not divided by dh
	Point3 adjratio;
	adjratio[0] = round(BufferLs[0]/dh[0]) * dh[0] / BufferLs[0];
	adjratio[1] = round(BufferLs[1]/dh[1]) * dh[1] / BufferLs[1];
	adjratio[2] = round(BufferLs[2]/dh[2]) * dh[2] / BufferLs[2];

	ExtRatio = ewmul(ExtRatio, adjratio);
	BufferLs = ewmul(ElemLs, ExtRatio);

	if(mpirank==0) { 
	  fprintf(stderr, "Adjusted ExtRatio: [%10.5f %10.5f %10.5f]\n",
		  ExtRatio[0], ExtRatio[1], ExtRatio[2]); 
	}

	Point3 BufferRatio;
	BufferRatio[0] = (ExtRatio[0] - 1.0) / 2.0;
	BufferRatio[1] = (ExtRatio[1] - 1.0) / 2.0;
	BufferRatio[2] = (ExtRatio[2] - 1.0) / 2.0;

	Index3 BufferNs;
	BufferNs[0] = round(BufferLs[0]/dh[0]);
	BufferNs[1] = round(BufferLs[1]/dh[1]);
	BufferNs[2] = round(BufferLs[2]/dh[2]);

	for(int k=0; k<NElems[2]; k++)
	  for(int j=0; j<NElems[1]; j++)
	    for(int i=0; i<NElems[0]; i++) {
	      Domain& buff = bufftns(i,j,k);
	      buff._Ls = BufferLs;
	      buff._Ns = BufferNs;

	      xx = (-BufferRatio[0]+i)*ElemLs[0];
	      yy = (-BufferRatio[1]+j)*ElemLs[1];
	      zz = (-BufferRatio[2]+k)*ElemLs[2];

	      xx = round(xx/dh[0])*dh[0];
	      yy = round(yy/dh[1])*dh[1];
	      zz = round(zz/dh[2])*dh[2];

	      buff._pos = Point3(xx, yy, zz);
	    }
      }
      else{  // Use dual grid for the buffer
	//Grid spacing for buffer along each dimension (dual grid or
	//reduced grid)
	Point3 dh;
	dh[0] = dm._Ls[0] / dm._Ns[0] * 2.0;
	dh[1] = dm._Ls[1] / dm._Ns[1] * 2.0;
	dh[2] = dm._Ls[2] / dm._Ns[2] * 2.0;

	Point3 BufferLs;
	BufferLs[0] = ElemLs[0] * ExtRatio[0];
	BufferLs[1] = ElemLs[1] * ExtRatio[1];
	BufferLs[2] = ElemLs[2] * ExtRatio[2];

	// Adjust the ratio of extended element if not divided by dh
	Point3 adjratio;
	adjratio[0] = round(BufferLs[0]/dh[0]) * dh[0] / BufferLs[0];
	adjratio[1] = round(BufferLs[1]/dh[1]) * dh[1] / BufferLs[1];
	adjratio[2] = round(BufferLs[2]/dh[2]) * dh[2] / BufferLs[2];

	ExtRatio = ewmul(ExtRatio, adjratio);
	BufferLs = ewmul(ElemLs, ExtRatio);

	if(mpirank==0) { 
	  fprintf(stderr, "Adjusted ExtRatio: [%10.5f %10.5f %10.5f]\n",
		  ExtRatio[0], ExtRatio[1], ExtRatio[2]); 
	}

	Point3 BufferRatio;
	BufferRatio[0] = (ExtRatio[0] - 1.0) / 2.0;
	BufferRatio[1] = (ExtRatio[1] - 1.0) / 2.0;
	BufferRatio[2] = (ExtRatio[2] - 1.0) / 2.0;

	Index3 BufferNs;
	BufferNs[0] = round(BufferLs[0]/dh[0]);
	BufferNs[1] = round(BufferLs[1]/dh[1]);
	BufferNs[2] = round(BufferLs[2]/dh[2]);


	//
	for(int k=0; k<NElems[2]; k++)
	  for(int j=0; j<NElems[1]; j++)
	    for(int i=0; i<NElems[0]; i++) {
	      Domain& buff = bufftns(i,j,k);
	      buff._Ls = BufferLs;
	      buff._Ns = BufferNs;

	      xx = (-BufferRatio[0]+i)*ElemLs[0];
	      yy = (-BufferRatio[1]+j)*ElemLs[1];
	      zz = (-BufferRatio[2]+k)*ElemLs[2];

	      xx = round(xx/dh[0])*dh[0];
	      yy = round(yy/dh[1])*dh[1];
	      zz = round(zz/dh[2])*dh[2];

	      buff._pos = Point3(xx, yy, zz);
	    }
      }
      scf._bufftns = bufftns;
      
      // Printout the information for buffer
      for(int k=0; k<NElems[2]; k++)
	for(int j=0; j<NElems[1]; j++)
	  for(int i=0; i<NElems[0]; i++) {
	    if( scf._elemptninfo(i,j,k) == mpirank ){
	      Domain& buff = scf._bufftns(i,j,k);
	      fprintf(fhstat, "Extended element [%4d, %4d, %4d]\n", 
		      i, j, k);
	      fprintf(fhstat,"Buff.Ls               = %10.5f %10.5f %10.5f\n",
		      buff._Ls[0], buff._Ls[1], buff._Ls[2] );
	      fprintf(fhstat,"Buff.Ns               = %10d %10d %10d\n",
		      buff._Ns[0], buff._Ns[1], buff._Ns[2] );
	      fprintf(fhstat,"Buff.pos              = %10.5f %10.5f %10.5f\n",
		      buff._pos[0], buff._pos[1], buff._pos[2] );
	    }
	  }
    }
    
   
  
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf read done\n"); }


    //----------
    //MD parameters
    dt = esdf_double((char*)("Time_Step"), 100.0);
    max_step = esdf_integer((char*)("Max_Step"), 1);

    //----------
    iC( scf.setup() );
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf setup done\n"); }
    //----------
    //inital guess?
    {
      esdf_string((char*)("Restart_Mode"), (char*)("from_scratch"), strtmp);
      string restartmode = strtmp;

      if(restartmode == string("from_scratch") )
	rhoinput = scf._rho0;
      if(restartmode == string("restart") ){
	Index3 Ns;
	Point3 Ls;      //DblNumVec rho(scf.ntot(), false, rhoinput
	int ntot;
	//
	esdf_string((char*)("Restart_Density"), (char*)("rho_dg.dat"), strtmp);
	string restart_density = strtmp;
	istringstream rhoid;      iC( Shared_Read(restart_density, rhoid) );
	rhoid >> Ns >> Ls;      //cerr<<Ns<<endl<<Ls<<endl;
	iA(scf._Ns == Ns);
	//read vector
	rhoid >> ntot;
	rhoinput.resize(ntot);
	for(int i=0; i<ntot; i++)
	  rhoid >> rhoinput[i];
	//fprintf(stderr, "ntot %d\n", ntot);
      }
    }
    fprintf(stderr, "proc %d, rhoinput val %f\n", myid, rhoinput[myid]);
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf guess done\n"); }
    
    // Reading external potential
    {
      esdf_string((char*)("Vext"), (char*)("null"), strtmp);
      string vextfile = strtmp;

      if( vextfile != string("null") ){
	// Read external potential from file
	istringstream vextid;   iC( Shared_Read(vextfile, vextid) );
	// read vector
	Index3 Ns;
	int ntot;
	vextid >> Ns;  iA( Ns == scf._Ns);
	vextid >> ntot;      iA( ntot==scf.ntot() );
	scf._vext.resize(ntot);
	for(int i=0; i<ntot; i++)	vextid >> scf._vext[i];
      }
      else{
	int ntot = scf.ntot();
	scf._vext.resize(ntot);
	for(int i=0; i<ntot; i++)	scf._vext[i] = 0.0;
      }
    }
  
 
  }



  //---------
  //Initialze Molecular dynamics simulation
  double T_init = 300.0 / au2K;
  
  
  //NVE simulation. The initial velocity of each atom is only nonzero at
  //the first component with temperature equal to 300K.
  //LL: Since the force is computed in parallel, all processors
  //have the state variables such as position, velocity and force
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

    iC( scf.scf(rhoinput) );
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf scf done\n"); }
    //---------
    iC( scf.force() );  
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf force done\n"); }
    
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
    if(mpirank == 0 ){
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
		mi-atomvec.begin(), type, coord(0), coord(1), coord(2),
		fs(0), fs(1), fs(2));
      }


      fprintf(fhstat, "\n-------------------------------------------------------------------------------\n");
      fprintf(fhstat, "     ATOM          VELOCITIES\n");
      for(vector<Atom>::iterator mi=atomvec.begin(); 
	  mi != atomvec.end(); mi++){
	int type = (*mi).type();
	Point3& vel = (*mi).vel();
	fprintf(fhstat, "%6d%3d %12.3e%12.3e%12.3e\n",
		mi-atomvec.begin(), type, vel(0), vel(1), vel(2));
      }
    }
  }
  
  //--------
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
    iC( scf.scf(scf._rho) );
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

    if(mpirank == 0 ){
      fprintf(fhstat, "---------------------------------------------\n");
      fprintf(fhstat, "     Time step %10d \n", istep);
      fprintf(fhstat, "---------------------------------------------\n");
      fprintf(fhstat, "Eatomkin      = %12.6e (au)\n", Eatomkin);
      fprintf(fhstat, "DFTEtot       = %12.6e (au)\n", scf._Etot);
      fprintf(fhstat, "DFTEfree      = %12.6e (au)\n", scf._Efree);
      fprintf(fhstat, "Hconv         = %12.6e (au)\n", Hconv);
      fprintf(fhstat, "drift         = %12.6e (au)\n", drift);
      fprintf(fhstat, "Fermi         = %12.6e (au)\n", scf._Fermi);
      fprintf(fhstat, "\nTIME ELAPSED : %10.1f sec\n",cpu_time_used);
      fprintf(fhstat, "\n-------------------------------------------------------------------------------\n");
      fprintf(fhstat, "     ATOM          COORDINATES                        FORCES\n");
      for(vector<Atom>::iterator mi=atomvec.begin(); 
	  mi != atomvec.end(); mi++){
	int type = (*mi).type();
	Point3& coord = (*mi).coord();
	Point3& fs    = (*mi).force();
	fprintf(fhstat, "%6d%3d %10.4f%10.4f%10.4f %12.3e%12.3e%12.3e\n",
		mi-atomvec.begin(), type, coord(0), coord(1), coord(2),
		fs(0), fs(1), fs(2));
      }


      fprintf(fhstat, "\n-------------------------------------------------------------------------------\n");
      fprintf(fhstat, "     ATOM          VELOCITIES\n");
      for(vector<Atom>::iterator mi=atomvec.begin(); 
	  mi != atomvec.end(); mi++){
	int type = (*mi).type();
	Point3& vel = (*mi).vel();
	fprintf(fhstat, "%6d%3d %12.3e%12.3e%12.3e\n",
		mi-atomvec.begin(), type, vel(0), vel(1), vel(2));
      }
    }
  } 

  //---------
  //output
  if(mpirank == 0)
  {
    esdf_string((char*)("Output_Dir"), (char*)("./"), strtmp);
    string outputdir = strtmp;
    inttmp          = esdf_integer((char*)("Output_Density"), 0 );
    int output_density  = (bool)inttmp;
    inttmp          = esdf_integer((char*)("Output_Wfn"), 0 );
    int output_wfn      = (bool)inttmp;
    inttmp          = esdf_integer((char*)("Output_Vtot"), 0 );
    int output_vtot      = (bool)inttmp;
    /*
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
    */
    if( output_density == true ){
      ofstream rhoid;
      string strtmp = outputdir + string("rho_dg.dat");
      rhoid.open(strtmp.c_str(), ios::out | ios::trunc);
      rhoid << scf._Ns << endl;
      rhoid << scf._Ls << endl;
      rhoid << scf._ntot<<endl;
      for(int i=0; i<scf._ntot; i++)	  rhoid<<scf._rho[i]<<" ";
      rhoid << endl;
      rhoid.close();
    }

    if( output_vtot == true ){
      cerr << "Outputing vtot" << endl;
      ofstream vtotid;
      string strtmp = outputdir + string("vtot_dg.dat");
      vtotid.open(strtmp.c_str(), ios::out | ios::trunc);
      vtotid << scf._Ns << endl;
      vtotid << scf._Ls << endl;
      vtotid << scf._ntot<<endl;
      for(int i=0; i<scf._ntot; i++)	  vtotid<<scf._vtot[i]<<" ";
      vtotid << endl;
      vtotid.close();
    }

  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpirank==0) { fprintf(stderr, "scf output done\n"); }
  //---------
  
  tottime_end = clock();
  
  if(mpirank == 0){
    fprintf(fhstat, "\n-----------------------------------------------------------\n");
    fprintf(fhstat, "TOTAL TIME ELAPSED : = %10.1f sec\n",((double) (tottime_end-tottime_sta)) / CLOCKS_PER_SEC);
  }
  
  fclose(fhstat);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  
  return 0;
}
