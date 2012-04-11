#include "scfdg.hpp"
#include "eigdg.hpp"
#include "interp.hpp"
#include "eigpw.hpp"
#include "parallel.hpp"
#include "vecmatop.hpp"

extern FILE* fhstat;

//--------------------------------------
int ScfDG::scf()
{
  /* MPI variables */
  int myid = mpirank();
  int nprocs = mpisize(); 
  int mpirank; MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize; MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  
  int iterflag; //LY: =1 then exit
  
  /* timing */
  double TimeSta(0), TimeEnd(0), t0(0), t1(0);
  
  /* counters */
  int iter;
  
  /* potential */
  vector<double> vtotnew(_ntot); 

  /* For Anderson mixing.  dfmat and dvmat will be initialized in the
   * first iteration in Anderson mixing */
  ParVec<Index3, DblNumMat, ElemPtn> dfmat, dvmat;
  // Test part for Anderson mixing (Sequential)
  //  vector<double> df(_ntot*_mixdim), dv(_ntot*_mixdim);
  
  
  if( myid == MASTER ) {
    iC( scf_Print(fhstat) );
  }
  
  // ONLY MASTER PROCESSOR works
  MPI_Barrier(MPI_COMM_WORLD);

  iC( scf_CalXC() );
  MPI_Barrier(MPI_COMM_WORLD);

  iC( scf_CalHartree() );
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpirank==0) { 
    fprintf(stderr, "scf scf_CalHartree done\n"); fflush(stderr); 
  }

  iC( scf_CalVtot(&_vtot[0]) );
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpirank==0) { fprintf(stderr, "scf scf_CalVtot done\n"); fflush(stderr); }
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpirank==0) { fprintf(stderr, "scf scf prelim done\n"); }
  
  //LEXING: no preparation, nonlocal pseudopot already done
  iterflag = 0;
  //---------------
  // LLIN: IMPORTANT for mixing reason, iter must start from 1!
  for (iter=1;iter<=_scfmaxiter;iter++) {
    TimeSta = time(0);   // LLIN (TimeSta, TimeEnd) records the wall clock 
                         // computational time for each SCF iteration.
    //--------------------
    if( iterflag == 1 )  break;
    //--------------------
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf scf iter %d start\n", iter); }
    
    /* ===============================================================
       =     Performing each iteration
       =============================================================== */
    //LY: IMPORTANT: Broadcast vtot at global level from MASTER processor to all processors
    if( myid != MASTER )  _vtot.resize(_ntot);
    MPI_Bcast(&_vtot[0], _ntot, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf scf iter %d vtot done\n", iter); }
    
    //----------------
    //Solve the buffer problem every few SCF steps
    // LL: The vtot should be updated at each step.
    for(int i2=0; i2<_NElems[2]; i2++)
      for(int i1=0; i1<_NElems[1]; i1++)
	for(int i0=0; i0<_NElems[0]; i0++) {
	  if(_elemptn.owner(Index3(i0,i1,i2))==myid) {
	    //--------------
	    Buff& buff = _buffvec(i0,i1,i2); 	    //Buff& buff = _buffvec.lclmap()[Index3(i,j,k)];
	    Elem& elem = _elemvec(i0,i1,i2);
	    //--------------
	    //PARA: interpolative vtot from global to buffer and from
	    //buffer to element

	    // LLIN: Whether the reduced grid (dual grid) is used for buffer
	    // calculation. Now it seems that the dual grid does not
	    // help.
	    if( _bufdual == 0 ){
	      xinterp(&(buff._vtot)[0], &_vtot[0], buff, *this);
	    }
	    else{
	      xinterp_dual(&(buff._vtot)[0], &_vtot[0], buff, *this);
	    }
	    
	    _vtotvec.lclmap()[Index3(i0,i1,i2)].resize(elem.ntot());
	    xlinterp(&(_vtotvec.lclmap()[Index3(i0,i1,i2)][0]), &(buff._vtot)[0], elem, buff); 	    
	  }
	}
    
    // LLIN: Usually choose nBuffUpdate = 1
    if( ((iter-1) % _nBuffUpdate) == 0 ){
      for(int i2=0; i2<_NElems[2]; i2++)
	for(int i1=0; i1<_NElems[1]; i1++)
	  for(int i0=0; i0<_NElems[0]; i0++) {
	    if(_elemptn.owner(Index3(i0,i1,i2))==myid) {
	      //--------------
	      Buff& buff = _buffvec(i0,i1,i2); 	    
	      Elem& elem = _elemvec(i0,i1,i2);
	      //-------------
	      //PARA: solve buffer problem
	      EigPW eigpw;
	      eigpw._Ls = buff._Ls;
	      eigpw._Ns = buff._Ns;
	      eigpw._eigmaxiter = _eigmaxiter;
	      eigpw._eigtol = _eigtol;
	      iC( eigpw.setup() );
	      iC( eigpw.solve(buff._vtot, buff._vnls, _nenrich + _nbufextra, 
			      buff._psi, buff._ev, buff._nactive, buff._active_indices) );
	      //MPI_Barrier(MPI_COMM_WORLD);
	      //if(mpirank==0) { fprintf(stderr, "scf scf iter %d eigpw done\n", iter); }
	      //-------------
	      //PARA: interp from buffer to element
	      vector<DblNumTns>& elembases = _basesvec.lclmap()[Index3(i0,i1,i2)];
	      elembases.resize(_nenrich);		//elem._bases.resize(_nenrich);
	      for(int g=0; g<_nenrich; g++) {
		double* psibufptr = &(buff._psi[0]) + g*(buff._ntot);
		Index3 Nlbls = elem._Ns; //LY: LBL GRIDS
		elembases[g].resize(Nlbls(0),Nlbls(1),Nlbls(2));
		xlinterp(elembases[g].data(), psibufptr, elem, buff); //LEXING: this stores the local basis functions
	      }
	      //MPI_Barrier(MPI_COMM_WORLD);
	      //if(mpirank==0) { fprintf(stderr, "scf scf iter %d xlinterp done\n", iter); }
	      //-------------
	      //PARA: add poly
	      Index3 Nlbls = elem._Ns; //Number of LBL grid points
	      int Nlbl1 = Nlbls(0);  int Nlbl2 = Nlbls(1);  int Nlbl3 = Nlbls(2);
	      int Nlbltot = Nlbl1*Nlbl2*Nlbl3;
	      Point3 hs = elem._Ls;
	      double h1 = hs(0);	    double h2 = hs(1);	    double h3 = hs(2);
	      DblNumVec x1(Nlbl1), w1(Nlbl1);
	      lglnodes(x1.data(), w1.data(), Nlbl1-1);
	      for(int g=0; g<x1.m(); g++)    x1(g) = x1(g)/2.0*h1;
	      for(int g=0; g<w1.m(); g++)    w1(g) = w1(g)/2.0*h1;
	      DblNumVec x2(Nlbl2), w2(Nlbl2);
	      lglnodes(x2.data(), w2.data(), Nlbl2-1);
	      for(int g=0; g<x2.m(); g++)    x2(g) = x2(g)/2.0*h2;
	      for(int g=0; g<w2.m(); g++)    w2(g) = w2(g)/2.0*h2;
	      DblNumVec x3(Nlbl3), w3(Nlbl3);
	      lglnodes(x3.data(), w3.data(), Nlbl3-1);
	      for(int g=0; g<x3.m(); g++)    x3(g) = x3(g)/2.0*h3;
	      for(int g=0; g<w3.m(); g++)    w3(g) = w3(g)/2.0*h3;
	      DblNumTns www(Nlbl1,Nlbl2,Nlbl3);
	      for(int g1=0; g1<Nlbl1; g1++)
		for(int g2=0; g2<Nlbl2; g2++)
		  for(int g3=0; g3<Nlbl3; g3++) {
		    www(g1,g2,g3) = w1(g1)*w2(g2)*w3(g3);
		  }
	      DblNumTns sqrtwww(Nlbl1,Nlbl2,Nlbl3);
	      for(int g1=0; g1<Nlbl1; g1++)
		for(int g2=0; g2<Nlbl2; g2++)
		  for(int g3=0; g3<Nlbl3; g3++) {
		    sqrtwww(g1,g2,g3) = sqrt(www(g1,g2,g3));
		  }
	      DblNumTns recsqrtwww(Nlbl1,Nlbl2,Nlbl3);
	      for(int g1=0; g1<Nlbl1; g1++)
		for(int g2=0; g2<Nlbl2; g2++)
		  for(int g3=0; g3<Nlbl3; g3++) {
		    recsqrtwww(g1,g2,g3) = 1.0/sqrt(www(g1,g2,g3));
		  }
	      //LEXING: NO POLYNOMIALS ADDED FOR NOW
	      /*
	      DblNumTns xx1(Nlbl1,Nlbl2,Nlbl3);
	      DblNumTns xx2(Nlbl1,Nlbl2,Nlbl3);
	      DblNumTns xx3(Nlbl1,Nlbl2,Nlbl3);
	      for(int g1=0; g1<Nlbl1; g1++)
		for(int g2=0; g2<Nlbl2; g2++)
		  for(int g3=0; g3<Nlbl3; g3++) {
		    xx1(g1,g2,g3) = x1(g1);	  xx2(g1,g2,g3) = x2(g2);	  xx3(g1,g2,g3) = x3(g3);
		  }
	      //
	      vector<DblNumTns>& bases = elembases;
	      int Ndeg = _dgndeg;
	      for(int d1=0; d1<=Ndeg; d1++)
		for(int d2=0; d2<=Ndeg; d2++)
		  for(int d3=0; d3<=Ndeg; d3++) 
		    if(d1+d2+d3<=Ndeg) { //do work if sum of degrees is small
		      DblNumTns cur;		    cur.resize(Nlbl1,Nlbl2,Nlbl3);		
		      for(int g1=0; g1<Nlbl1; g1++)
			for(int g2=0; g2<Nlbl2; g2++)
			  for(int g3=0; g3<Nlbl3; g3++) {
			    cur(g1,g2,g3) = pow(xx1(g1,g2,g3),d1) * pow(xx2(g1,g2,g3),d2) * pow(xx3(g1,g2,g3),d3);
			  }
		      bases.push_back(cur);
		    }
	      */
	      //LEXING: NEED TO SVD THESE BASIS FUNCS FIRST?
	      //-----------
	      //PARA: svd
	      vector<DblNumTns>& bases = elembases;
	      int INTONE = 1.0;
	      int nbases = bases.size();
	      int Ngrid = Nlbltot;
	      DblNumMat VLtmp(Ngrid,nbases);
	      double *VLptr;
	      for(int g=0; g<nbases; g++) {
		VLptr = VLtmp.data()+g*Nlbltot;
		memcpy(VLptr, bases[g].data(), sizeof(double)*Nlbltot); 
		XScaleByY(VLptr, sqrtwww.data(), Nlbltot);
		double egy = dnrm2_(&Nlbltot, VLptr, &INTONE);
		egy = 1.0/egy;
		dscal_(&Nlbltot, &egy, VLptr, &INTONE);
	      }
	      bases.clear();
	      DblNumMat tmp(VLtmp);
	      int m = Ngrid;
	      int n = nbases;
	      int k = min(m,n);
	      DblNumMat U(m,k);
	      DblNumVec S(k);
	      DblNumMat VT(k,n);
	      {
		char jobu  = 'N';
		char jobvt = 'S';
		int lwork = 20*max(m,n);
		DblNumVec work(lwork);
		int info;
		dgesvd_(&jobu, &jobvt, &m, &n, tmp.data(), &m, S.data(), U.data(), 
			&m, VT.data(), &k, work.data(), &lwork, &info);
		iA(info==0);
	      }
	      //truncate
	      double EPS = 1e-10;
	      int c = 0;
	      for(int g=0; g<k; g++)	      if(S(g)>S(0)*EPS) c++;
	      //fprintf(stderr, "(i0,i1,i2) = (%4d, %4d, %4d): S(0) = %8.2e, S(end) = %8.2e\n", i0, i1, i2, S(0), S(k-1));
	      DblNumVec SA(c);
	      DblNumMat VAT(c,n);
	      for(int i=0; i<c; i++)
		SA(i) = S(i);
	      for(int i=0; i<c; i++)
		for(int j=0; j<n; j++)
		  VAT(i,j) = VT(i,j);
	      //rotate and put back
	      DblNumMat G(n,c);
	      for(int i=0; i<n; i++)
		for(int j=0; j<c; j++)
		  G(i,j) = VAT(j,i) / SA(j);
	      tmp = VLtmp;	VLtmp.resize(Ngrid,c);	iC( dgemm(1.0, tmp, G, 0.0, VLtmp) );
	      nbases = c;
	      bases.resize(nbases);
	      for(int g=0; g<nbases; g++) {
		bases[g].resize(Nlbl1,Nlbl2,Nlbl3);
		VLptr = VLtmp.data()+g*Nlbltot;
		//LY:
		XScaleByY(VLptr, recsqrtwww.data(), Nlbltot);
		memcpy(bases[g].data(), VLptr, sizeof(double)*Nlbltot); 
	      }
	    }
	  }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf scf iter %d buffer done\n", iter); }
    
    
    //-----------------------------
    //call eigdg
    EigDG eigdg;
    eigdg._inputformat  = _inputformat;


    eigdg._hs = _elemtns(0,0,0).Ls(); //length of element
    eigdg._Ns = _NElems; //number of elements in each direction
    eigdg._Nlbls = _elemtns(0,0,0).Ns(); //number of LBL points
    eigdg._alpha = _dgalpha;
    eigdg._MB = _MB;
    // LLIN: For non-orthogonal basis
    eigdg._dgsolver = _dgsolver;
    eigdg._delta = _delta;
    eigdg._basisradius = _basisradius;
    eigdg._gamma = _gamma;
    eigdg._Neigperele = _Neigperele;
    eigdg._Norbperele = _Norbperele;
    eigdg._DeltaFermi = _DeltaFermi;

    
    // LLIN: Get the energy cutoff for the candidate functions
    DblNumTns Ecut(_NElems[0], _NElems[1], _NElems[2]); setvalue(Ecut, 0.0);
    for(int i2=0; i2<_NElems[2]; i2++)
      for(int i1=0; i1<_NElems[1]; i1++)
	for(int i0=0; i0<_NElems[0]; i0++) {
	  if(_elemptn.owner(Index3(i0,i1,i2))==myid) {
	      Buff& buff = _buffvec(i0,i1,i2); 	    
	      Ecut(i0,i1,i2) = buff._ev[buff._npsi-1] + _DeltaFermi; 
	      //LLIN: _DeltaFermi usually is just set to be 0.
	      //(04/11/2012)
	  }
	}

    eigdg._EcutCnddt.resize(_NElems[0],_NElems[1],_NElems[2]);
    setvalue(eigdg._EcutCnddt, 0.0);
    MPI_Allreduce(Ecut.data(), eigdg._EcutCnddt.data(), _NElems[0]*_NElems[1]*_NElems[2], 
		  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if(mpirank == 0 ){
      fprintf(stderr, "The Fermi energy cutoff for all the extended elements\n");
      cerr << eigdg._EcutCnddt << endl;
    }
    
    iC( eigdg.setup() );
    t0 = time(0);
    iC( eigdg.solve(_vtotvec, _basesvec, _psdovec, _npsi, _ev, _eigvecsvec, _EOcoef) );
    t1 = time(0);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) 
    { 
      fprintf(fhstat, 
	      "scf scf iter %d eigdg done. EigDG::solve time = %15.3f secs\n", 
	      iter, difftime(t1, t0)); 
      fprintf(stderr, 
	      "scf scf iter %d eigdg done. EigDG::solve time = %15.3f secs\n", 
	      iter, difftime(t1, t0)); 
    }
    
    //-----------------------------
    //PARA: calocc first
    iC( scf_CalOcc(_Tbeta) );
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf scf iter %d occ done\n", iter); }


    //-----------------------------
    //PARA: interpolate wave fun back to global and form density
    // Interpolate wavefunctions from element back to global and form
    // density directly.
    iC(scf_EvaluateDensity());

    MPI_Barrier(MPI_COMM_WORLD);
    if(mpirank==0) { fprintf(stderr, "scf iter %d rho done\n", iter); }


    //-------------------------------------------------------
    //           Post processing
    //-------------------------------------------------------
    t0 = time(0);   
    if( myid == MASTER ) {
      fprintf(stderr, "Post processing (Hartree & XC)...\n");
      fprintf(fhstat, "Post processing (Hartree & XC)...\n");
    }
    //
    iC( scf_CalXC() );
    iC( scf_CalHartree() );
    iC( scf_CalEnergy() );
    iC( scf_PrintState(fhstat) );
    t1 = time(0);
    if(myid==MASTER) {
      fprintf(stderr, "Finish post processing (Hartree & XC). Time = %10.2f secs\n", difftime(t1,t0));
      fprintf(fhstat, "Finish post processing (Hartree & XC). Time = %10.2f secs\n", difftime(t1,t0));
    }
    

    // Update new total potential
    iC( scf_CalVtot(&vtotnew[0]) );
    // convergence check
    if( mpirank == MASTER ){
      double verr;
      vector<double> vdif(_ntot);
      for(int i = 0; i < _ntot; i++){
	vdif[i] = vtotnew[i] - _vtot[i];
      }
      verr = norm(&vdif[0], _ntot) / norm(&_vtot[0], _ntot);

      //LLIN: Accelerate SCF by fixing basis functions
//      if(verr<1e-2) _nBuffUpdate = 20;

      t1 = time(0);
      fprintf(stderr, "norm(vout-vin) = %10.3e\n", verr);
      fprintf(fhstat, "norm(vout-vin) = %10.3e\n", verr);
      if( verr < _scftol ){
	// converged 
	fprintf(stderr, "SCF convergence reached!\n");
	fprintf(fhstat, "SCF convergence reached!\n");
	iterflag = 1;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Mixing and update potential      //WILL set _vtot in the MASTER PROCESSOR
    t0 = time(0);   
    if(myid==MASTER) {
      fprintf(stderr, "Start mixing...\n");
      fprintf(fhstat, "Start mixing...\n");
    }
    if( _mixtype == "anderson" ){
      iC( scf_PAndersonMix(vtotnew, dfmat, dvmat, iter) );
    } else { //"kerker"
      if( mpirank == MASTER )
	cerr << "Start Kerker mixing" << endl;
      iC( scf_KerkerMix(vtotnew, 0.8) );  // LL: Fixed mixing parameter 
					  // (0.8) for Kerker mixing.
					  // Kerker mixing must be
					  // performed before Anderson
					  // mixing.
      if( mpirank == MASTER )
	cerr << "Start Anderson mixing" << endl;
      iC( scf_PAndersonMix(vtotnew, dfmat, dvmat, iter) );
      //      iC( scf_AndersonMix(vtotnew, df, dv, iter) );
    }
    t1 = time(0);
    if( mpirank == MASTER ){
      fprintf(stderr, "Finish mixing. Time = %10.2f secs\n", 
	      iter, difftime(t1,t0));
      fprintf(fhstat, "Finish mixing. Time = %10.2f secs\n", 
	      iter, difftime(t1,t0));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast( &iterflag, 1, MPI_INT, MASTER, MPI_COMM_WORLD ); //LEXING: VERY IMPORTANT
    
    
    TimeEnd = time(0);   
    if( myid == MASTER ){
      fprintf(stderr, "Finish iteration %4d. Time = %10.2f secs\n", 
	      iter, difftime(TimeEnd,TimeSta));
      fprintf(stderr, "**************************************************************\n\n");
    }
    fprintf(fhstat, "Finish iteration %4d. Time = %10.2f secs\n", 
	    iter, difftime(TimeEnd,TimeSta));
    fprintf(fhstat, "**************************************************************\n\n");
    fflush(fhstat);
  }
  
  if(myid == MASTER) {
    iC( scf_PrintState(stderr) );
  }

  //LLIN: Output data to hard disk
  scf_FileIO();

  return 0;
}


//--------------------------------------
// Perform File IO after the SCF iterations
int ScfDG::scf_FileIO()
{
  time_t t0, t1;

  //LLIN: File IO part
  t0 = time(0);   
  int mpirank; MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize; MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

  // Adaptive local basis functions
  if( _isOutputBases ){
    int N1 = _NElems(0);	int N2 = _NElems(1);	int N3 = _NElems(2);
    Index3 Nlbls = _elemtns(0,0,0).Ns();
    int ntotelem = Nlbls[0] * Nlbls[1] * Nlbls[2];      

    //int ntot_nsglb = Nsglb[0] * Nsglb[1] * Nsglb[2];

    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if( _elemptn.owner(curkey)==mpirank ) {
	    Elem& curdat = _elemvec(i1,i2,i3);
	    Index3 Nsglb = curdat._Nsglb;
	    Index3 posidx = curdat.posidx();

	    vector<DblNumTns>& bases = _basesvec.lclmap()[curkey];		//curdat.bases();
	    int nbases = bases.size();
	    vector<DblNumTns> basesglb(nbases);

	    for(int g = 0; g < nbases; g++){

	      basesglb[g].resize(Nsglb[0], Nsglb[1], Nsglb[2]);
	      setvalue(basesglb[g], 0.0);

	      lxinterp_local(basesglb[g].data(), bases[g].data(), curdat);

	    }

	    // Output
	    ostringstream oss;
	    vector<int> all(1);

	    serialize(Nsglb, oss, all);
	    serialize(posidx, oss, all);
	    serialize(basesglb, oss, all);
	    Separate_Write("ALB", oss);
	    basesglb.clear();
	  }
	}
  }

  // Coefficients for element orbitals
  if(_isOutputBases && _dgsolver == "nonorth" ){
    int N1 = _NElems(0);	int N2 = _NElems(1);	int N3 = _NElems(2);
    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if( _elemptn.owner(curkey)==mpirank ) {
	    DblNumMat& coef = _EOcoef.lclmap()[curkey];
	    ostringstream oss;
	    vector<int> all(1);

	    serialize(Index3(i1,i2,i3), oss, all);
	    serialize(coef, oss, all);

	    Separate_Write("EOcoef", oss);
	  }
	}
  }

  // The information of the basis functions, only for the master
  // processor
  if(_isOutputBases){

    // Everyone collect the number of adaptive local basis functions first.
    
    int N1 = _NElems(0);	int N2 = _NElems(1);	int N3 = _NElems(2);
    IntNumTns numLocalBasis(N1, N2, N3);  
    setvalue(numLocalBasis, 0);

    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if( _elemptn.owner(curkey)==mpirank ) {
	    Elem& curdat = _elemvec(i1,i2,i3);
	    vector<DblNumTns>& bases = _basesvec.lclmap()[curkey];		
	    numLocalBasis(i1, i2, i3) += bases.size();
	  }
	}

    IntNumTns tempBasis(N1, N2, N3);  setvalue(tempBasis, 0);
    MPI_Reduce(numLocalBasis.data(), tempBasis.data(), N1*N2*N3, MPI_INT, 
	       MPI_SUM, 0, MPI_COMM_WORLD);
    numLocalBasis = tempBasis;

    if(mpirank == 0){
       ofstream fhout("BASISINFO"); iA(fhout.good());
       int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
       vector<int> noMask(1);
       //LL: Now, every element occupies a processor
       
       serialize(mpisize, fhout, noMask);
       serialize(_NElems, fhout, noMask);
       serialize(_Ns,     fhout, noMask);
       serialize(_Ls,     fhout, noMask);
       IntNumVec elementToProcessorMap(N1*N2*N3);

       for(int i3=0; i3<N3; i3++)
	 for(int i2=0; i2<N2; i2++)
	   for(int i1=0; i1<N1; i1++){
	     int iElem = i1 + i2 * N1 + i3 * N1 * N2;
	     elementToProcessorMap[iElem] = iElem;   
	   } 

       int numTotalBasis = 0;
       for(int i3=0; i3<N3; i3++)
	 for(int i2=0; i2<N2; i2++)
	   for(int i1=0; i1<N1; i1++) {
	     numTotalBasis += numLocalBasis(i1,i2,i3);
	   }

       IntNumVec adaptiveLocalBasisToProcessorMap(numTotalBasis);
       IntNumVec adaptiveLocalBasisGlobalToLocalMap(numTotalBasis);
       int countBasis = 0, countBasisLocal;
       for(int i3=0; i3<N3; i3++)
	 for(int i2=0; i2<N2; i2++)
	   for(int i1=0; i1<N1; i1++){
	     countBasisLocal = 0;
	     int iElem = i1 + i2 * N1 + i3 * N1 * N2;
	     for(int l = 0; l < numLocalBasis(i1,i2,i3); l++){
	       adaptiveLocalBasisToProcessorMap[countBasis] = 
		 elementToProcessorMap[iElem];
	       adaptiveLocalBasisGlobalToLocalMap[countBasis] = 
		 countBasisLocal;
               countBasisLocal++;
	       countBasis++;
	     }
	   }
       

       serialize(elementToProcessorMap, fhout, noMask);
       serialize(adaptiveLocalBasisToProcessorMap, fhout, noMask);
       serialize(adaptiveLocalBasisGlobalToLocalMap, fhout, noMask);

       int natoms = _atomvec.size();
       IntNumVec atomTypeVector(natoms);
       DblNumMat atomPositionMatrix(natoms,3);
       for(int i = 0; i < natoms; i ++){
	 atomTypeVector(i) = _atomvec[i].type();
	 Point3 coord = _atomvec[i].coord();
	 atomPositionMatrix(i,0) = coord(0);
	 atomPositionMatrix(i,1) = coord(1);
	 atomPositionMatrix(i,2) = coord(2);
       }

       serialize(natoms, fhout, noMask);
       serialize(atomTypeVector, fhout, noMask);
       serialize(atomPositionMatrix, fhout, noMask);

       fhout.close();
    }
  }  

  // Output Wavefunctions in the extended element
  if( _isOutputBufWfn ){
    vector<int> noMask(1);
    ostringstream oss;
    int N1 = _NElems(0);	int N2 = _NElems(1);	int N3 = _NElems(2);

    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if( _elemptn.owner(curkey)==mpirank ) {
	    Buff& buff = _buffvec(i1,i2,i3);
	    DblNumMat psiMatrix = DblNumMat(buff._ntot, buff._npsi,
					    false, &buff._psi[0]);
        
            serialize(curkey, oss, noMask);
	    serialize(psiMatrix, oss, noMask);
	    Separate_Write("BUFWFN", oss);
	  }
	}
  }
  
  // Output the density in the global domain
  if( _isOutputDensity ){
    if( mpirank == 0 ){
      vector<int> noMask(1);
      ofstream outputFileStream("DEN");  iA(outputFileStream.good());
      DblNumVec rho = DblNumVec(_ntot,false, &_rho[0]);
      serialize(rho, outputFileStream, noMask);
      outputFileStream.close();
    }
  }

  // Output the total potential in the global domain
  if( _isOutputVtot ){
    if( mpirank == 0 ){
      vector<int> noMask(1);
      ofstream outputFileStream("VTOT");  iA(outputFileStream.good());
      DblNumVec vtot = DblNumVec(_ntot,false, &_vtot[0]);
      serialize(vtot, outputFileStream, noMask);
      outputFileStream.close();
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if(mpirank==0) { fprintf(stderr, "scf output done\n"); }
  //---------

  t1 = time(0);
  if( mpirank == MASTER ){
    fprintf(stderr, "Finish IO. Time = %10.2f secs\n", 
	    difftime(t1,t0));
    fprintf(fhstat, "Finish IO. Time = %10.2f secs\n", 
	    difftime(t1,t0));
  }

  return 0;
}

//--------------------------------------
int ScfDG::scf_EvaluateDensity()
{
  int mpirank; MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize; MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  time_t t0, t1;
  
  t0 = time(0);

  DblNumTns rhotmp(_Ns[0],_Ns[1],_Ns[2]);    setvalue(rhotmp, 0.0);

  Index3 Nlbls = _elemtns(0,0,0).Ns();
  int N1=_NElems[0];    int N2=_NElems[1];    int N3=_NElems[2];
  double rhofac;
  int ntotglb = _ntot;
  int ntotelem = Nlbls[0] * Nlbls[1] * Nlbls[2];      //int ntot_nsglb = Nsglb[0] * Nsglb[1] * Nsglb[2];
  DblNumVec psielemtmp(ntotelem);


  for(int g=0; g<_npsi; g++){
    /* local and global normalization factor for each orbital */
    double l2psi_loc = 0.0;
    double l2psi_glb = 0.0;
    NumTns<DblNumTns> eigfun_glb;	eigfun_glb.resize(N1,N2,N3);
    //eigfun_glb.resize(Nsglb[0],Nsglb[1],Nsglb[2]);
    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if( _elemptn.owner(curkey)==mpirank ) {
	    Elem& curdat = _elemvec(i1,i2,i3);
	    DblNumMat& eigvecs = _eigvecsvec.lclmap()[curkey];
	    vector<DblNumTns>& bases = _basesvec.lclmap()[curkey];		
	    int nbases = bases.size();
	    setvalue(psielemtmp, 0.0);
	    int I_ONE = 1;
	    for(int a=0; a<nbases; a++){
	      daxpy_(&ntotelem, &eigvecs(a,g), bases[a].data(), &I_ONE, psielemtmp.data(), &I_ONE);
	    }
	    // the data in eigfun_glb(i1,i2,i3) will be rewritten by
	    // lxinterp_local for next orbital 
	    Index3 Nsglb = curdat._Nsglb;
	    eigfun_glb(i1,i2,i3).resize(Nsglb[0],Nsglb[1],Nsglb[2]);
	    lxinterp_local(eigfun_glb(i1,i2,i3).data(), psielemtmp.data(), curdat);
	    l2psi_loc += energy(eigfun_glb(i1,i2,i3));
	  }
	}
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(mpirank==0) { fprintf(stderr, "scf scf iter %d rho func %d part a\n", iter, g); }

    /* All processors get the normalization factor */
    MPI_Allreduce(&l2psi_loc, &l2psi_glb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    /* pre-constant in front of psi^2 for density */
    rhofac = (2.0 * _ntot / _vol) * _occ[g];
    /* Normalize the wavefunctions, and Add the normalized wavefunction to local density function denfun_glb */
    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if(_elemptn.owner(curkey)==mpirank) {
	    //map<Index3,Elem>::iterator mi=_elemvec.lclmap().find(curkey);	
	    //iA(mi!=_elemvec.lclmap().end());
	    Elem& curdat = _elemvec(i1,i2,i3);		//Elem& curdat = (*mi).second;
	    Index3 Nsglb = curdat._Nsglb;
	    //LEXING: IMPORTANT CHECK FOLLOWING
	    //DblNumTns& eigfun_glb = curdat.eigfun_glb();
	    //DblNumTns& denfun_glb = curdat.denfun_glb();
	    //double *eigptr = eigfun_glb.data();
	    //double *rhoptr = denfun_glb.data();
	    Index3 posidx = curdat.posidx();
	    for(int k = 0; k < Nsglb[2]; k++){		  int ksh = posidx[2]+ k; 
	      for(int j = 0; j < Nsglb[1]; j++){		    int jsh = posidx[1] + j; 
		for(int i = 0; i < Nsglb[0]; i++){		      int ish = posidx[0] + i;
		  double tmp = eigfun_glb(i1,i2,i3)(i,j,k) /sqrt(l2psi_glb);
		  eigfun_glb(i1,i2,i3)(i,j,k) = tmp;
		  rhotmp(ish, jsh, ksh) += tmp*tmp*rhofac;
		}
	      }
	    }
	  }
	}
  }
  t1 = time(0);
  if( mpirank == MASTER ){
    fprintf(fhstat,"Interpolating from element to global used %15.3f secs\n", difftime(t1,t0)); 
    fprintf(stderr,"Interpolating from element to global used %15.3f secs\n", difftime(t1,t0)); 
  }
  //LEXING: all processors get the total density
  t0 = time(0);
  //LEXING: REPLACED
  iC( MPI_Allreduce(rhotmp.data(), &(_rho[0]), _Ns[0]*_Ns[1]*_Ns[2],
		    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ); 
  
  t1 = time(0);
  if( mpirank == MASTER ){
    fprintf(fhstat,"Master processor to get total density used %15.3f secs\n", difftime(t1,t0)); 
    fprintf(stderr,"Master processor to get total density used %15.3f secs\n", difftime(t1,t0)); 
  }
  if( mpirank == MASTER ){
    double sumval = 0.0;
    for(int i = 0; i < ntotglb; i++){
      sumval += _rho[i] * _vol/_ntot;
    }
    fprintf(fhstat,"Sum rho = %15.5f\n", sumval); 
    fprintf(stderr,"Sum rho = %15.5f\n", sumval); 
  }

  return 0;
}


//--------------------------------------
int ScfDG::scf_CalOcc(double Tbeta)
{
  /* For a given finite temperature, update the occupation number */
  double tol = 1e-10;
  int maxiter = 100;
  double lb, ub, flb, fub, occsum;
  int ilb, iub, iter, j;

  if(_npsi > _nOccStates)  {
    /* use bisection to find efermi such that 
     * sum_i fermidirac(ev(i)) = nocc
     */
    ilb = _nOccStates-1;
    iub = _nOccStates+1;
    lb = _ev[ilb-1];
    ub = _ev[iub-1];
    /* Calculate Fermi-Dirac function and make sure that
     * flb < nocc and fub > nocc
     */
    flb = 0.0;
    fub = 0.0;
    for(j = 0; j < _npsi; j++){
      flb += 1.0 / (1.0 + exp(Tbeta*(_ev[j]-lb)));
      fub += 1.0 / (1.0 + exp(Tbeta*(_ev[j]-ub))); 
    }
    while( (_nOccStates-flb)*(fub-_nOccStates) < 0 ) {
      if( flb > _nOccStates ){
	if(ilb > 0){
	  ilb--;
	  lb = _ev[ilb-1];
	  flb = 0.0;
	  for(j = 0; j < _npsi; j++){
	    flb += 1.0 / (1.0 + exp(Tbeta*(_ev[j]-lb)));
	  }
	}
	else{
	  ABORT("CalOcc: cannot find a lower bound for efermi, something is wrong", 1);
	}
      }
      if( fub < _nOccStates ){
	if( iub < _npsi ){
	  iub++;
	  ub = _ev[iub-1];
	  fub = 0.0;
	  for(j = 0; j < _npsi; j++){
	    fub += 1.0 / (1.0 + exp(Tbeta*(_ev[j]-ub)));
	  }
	}
	else{
	  ABORT("getocc: cannot find an upper bound for efermi, something is wrong, try increasing the number of wavefunctions in X0", 1);
	}
      }
    }  /* while */
    _Fermi = (lb+ub)*0.5;
    occsum = 0.0;
    for(j = 0; j < _npsi; j++){
      _occ[j] = 1.0 / (1.0 + exp(Tbeta*(_ev[j] - _Fermi)));
      occsum += _occ[j];
    }
    /* Start bisection iteration */
    iter = 1;
    while( (fabs(occsum - _nOccStates) > tol) && (iter < maxiter) ){
      if( occsum < _nOccStates )
	lb = _Fermi;
      else
	ub = _Fermi;

      _Fermi = (lb+ub)*0.5;
      occsum = 0.0;
      for(j = 0; j < _npsi; j++){
	_occ[j] = 1.0 / (1.0 + exp(Tbeta*(_ev[j]-_Fermi)));
	occsum += _occ[j];
      }
      iter++;
    }
  }
  else if(_npsi == _nOccStates ){
    for(j = 0; j < _npsi; j++)
      _occ[j] = 1.0;
    _Fermi = _ev[_npsi-1];
  }
  else{
    ABORT("The number of eigenvalues in ev should be larger than nocc", 1);
  }
  // LLIN: temporary for sodium 2*2*2 system
  //  for(j = 0; j < 7; j++)
  //    _occ[j] = 1.0;
  //  for(j = 7; j < 13; j++)
  //    _occ[j] = 1.0/6.0;
  //  _Fermi = _ev[_npsi-1]; 
  return 0;
}


//--------------------------------------
// Exchange-correlation function by Goedecker-Teter-Hutter Tested compared to Ceperley-Alder ()
int ScfDG::scf_CalXC()
{
  double a0 = 0.4581652932831429;
  double a1 = 2.217058676663745;
  double a2 = 0.7405551735357053;
  double a3 = 0.01968227878617998;
  double b1 = 1.0;
  double b2 = 4.504130959426697;
  double b3 = 1.110667363742916;
  double b4 = 0.02359291751427506;


  double third = 1.0/3.0; 
  double pi = 3.14159265354;
  double p75vpi = 0.75/pi;
  int ntot = this->ntot();

  double exc; 
  double rho, rs, drsdrho, ec;
  double *vxc = &(_vxc[0]);

  double auxnum, auxdenom, auxdnum, auxddenom; 

  exc = 0.0;

  for(int i = 0; i < ntot; i++){
    rho = _rho[i];
    vxc[i] = 0.0;
    if( rho > 1e-8 ){
      rs = pow((p75vpi/rho), third);
      drsdrho = - rs / (3*rho);

      auxnum = a0 + a1*rs + a2*pow(rs, 2) + a3*pow(rs, 3); 
      auxdnum = a1 + 2*a2*rs + 3*a3*pow(rs, 2);
      auxdenom = b1*rs + b2*pow(rs,2) + b3*pow(rs,3) + b4*pow(rs,4);
      auxddenom = b1 + 2*b2*rs + 3*b3*pow(rs,2) + 4*b4*pow(rs,3);

      ec = - auxnum/auxdenom;
      exc += rho*ec;
      vxc[i] = ec + rho * (- auxdnum / auxdenom + auxnum * auxddenom / pow(auxdenom, 2)) * drsdrho;

    }
  }

  exc *= (vol()) / (this->ntot());

  _Exc = exc;


  return 0;
}


//--------------------------------------
int ScfDG::scf_CalHartree()
{
  // Old version
  if(0){
    double pi = 4.0 * atan(1.0);
    vector<cpx> crho;
    vector<cpx> rhotemp;
    crho.resize(_ntot);
    rhotemp.resize(_ntot);

    for(int i = 0; i < _ntot; i++)
      crho[i] = cpx(_rho[i] - _rho0[i], 0.0); //LY: rho-rho0

    {
      fprintf(stderr, "crho OK\n");
    }

    fftw_execute_dft(_planpsiforward, 
		     reinterpret_cast<fftw_complex*>(&crho[0]), 
		     reinterpret_cast<fftw_complex*>(&rhotemp[0]));

    /* 2.0*pi rather than 4.0*pi is because gkk = 1/2 k^2 */
    for(int i = 0; i < _ntot; i++){
      if(_gkk[i] != 0 )
	rhotemp[i] *= 2.0*pi/_gkk[i];
      else
	rhotemp[i] = cpx(0.0, 0.0);
    }

    {
      fprintf(stderr, "fft1 OK\n");
    }

    fftw_execute_dft(_planpsibackward, 
		     reinterpret_cast<fftw_complex*>(&rhotemp[0]), 
		     reinterpret_cast<fftw_complex*>(&crho[0]));

    for(int i = 0; i < _ntot; i++ ){
      _vhart[i] = crho[i].real()/(double)_ntot;
    }

    {
      fprintf(stderr, "fft2 OK\n");
    }

    // LEXING: Ecoul commented out
    //  _Ecoul = 0.0;
    //   for(int i = 0; i < _ntot; i++)
    //     _Ecoul += (_vhart[i] + _vhart0[i]) * _rho[i]; 
    //   _Ecoul *= _vol / _ntot * 0.5;


    return 0;
  }

  // New version that ensures alignment
  if(1){
    double pi = 4.0 * atan(1.0);
    fftw_complex *crho, *rhotemp;
    crho    = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*_ntot);
    rhotemp = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*_ntot); 

    for(int i = 0; i < _ntot; i++){
      crho[i][0] = _rho[i] - _rho0[i];
      crho[i][1] = 0.0;
    }

    fftw_execute_dft(_planpsiforward, &crho[0], &rhotemp[0]);

    /* 2.0*pi rather than 4.0*pi is because gkk = 1/2 k^2 */
    for(int i = 0; i < _ntot; i++){
      if(_gkk[i] != 0 ){
	rhotemp[i][0]   *= 2.0*pi/_gkk[i];
	rhotemp[i][1] *= 2.0*pi/_gkk[i];
      }
      else{
	rhotemp[i][0] = 0.0;
	rhotemp[i][1] = 0.0;
      }

    }

    fftw_execute_dft(_planpsibackward, (&rhotemp[0]), (&crho[0]));

    for(int i = 0; i < _ntot; i++ ){
      _vhart[i] = crho[i][0]/(double)_ntot;
    }

    // LEXING: Ecoul commented out
    //  _Ecoul = 0.0;
    //   for(int i = 0; i < _ntot; i++)
    //     _Ecoul += (_vhart[i] + _vhart0[i]) * _rho[i]; 
    //   _Ecoul *= _vol / _ntot * 0.5;

    fftw_free(rhotemp);
    fftw_free(crho);

    return 0;
  }

}


//--------------------------------------
int ScfDG::scf_CalVtot(double* vtot)
{
  double vtotsum = 0.0;
  for(int i = 0; i < _ntot; i++){
    vtot[i] = _vext[i] + _vhart[i] + _vxc[i];
    vtotsum += vtot[i];
  }
  /* 
     vtotsum /= (double)_ntot; for(int i = 0; i < _ntot; i++){ vtot[i]
     = vtot[i] - vtotsum; }*/
  return 0;
}



//--------------------------------------
int ScfDG::scf_CalEnergy()
{
  //-----------
  _Ekin = 0.0;
  for(int l = 0; l < _npsi; l++)
    _Ekin += 2.0*(_ev[l] * _occ[l]); //LEXING: THIS IS CORRECT

  //-----------
  _Ecor = 0.0;
  vector<double> srho(_ntot);
  for(int i = 0; i < _ntot; i++)    srho[i] = _rho[i]+_rho0[i];
  for(int i = 0; i < _ntot; i++)    _Ecor += (-_vxc[i]*_rho[i] - 0.5*_vhart[i]*srho[i]);
  _Ecor *= _vol/double(_ntot);

  _Ecor += _Exc;

  double Es = 0;
  for(int a=0; a<_atomvec.size(); a++) {
    int type = _atomvec[a].type();
    Es = Es + _ptable.ptemap()[type].params()(PeriodTable::i_Es);
  }
  _Ecor -= Es;

  //-----------
  _Etot = _Ekin + _Ecor ;

  //-----------
  //LL: Calculate the free energy functional at finite temperature
  _Efree = 0.0;
  for(int l = 0; l < _npsi; l++)
  {
    if(_Tbeta*(_ev[l]-_Fermi)<15.0)
      _Efree += -2.0/_Tbeta*log(1+exp(-_Tbeta*(_ev[l]-_Fermi)));
  }
  _Efree += _Ecor + _Fermi * (_nOccStates * 2.0); 


  return 0;
}


//--------------------------------------
int ScfDG::scf_Print(FILE *fh)
{
  Point3 coord;
  fprintf(fh, "**********************************************\n");
  fprintf(fh, "                  Molglobal\n");
  fprintf(fh, "**********************************************\n\n");
  fprintf(fh,"Super_Cell            = %10.5f %10.5f %10.5f\n", 	  _Ls[0], _Ls[1], _Ls[2] );
  fprintf(fh,"Grid_Size             = %10d %10d %10d\n", 	  _Ns[0], _Ns[1], _Ns[2] );
  fprintf(fh,"Total number of atoms = %10d\n", 	 _atomvec.size() );
  fprintf(fh, "\n");

  fprintf(fh, "Atom Type and Coordinates:\n");
  for(int i=0; i<_atomvec.size(); i++) {
    int type = _atomvec[i].type();
    Point3 coord = _atomvec[i].coord();
    fprintf(fh,"%d     %10.5f   %10.5f   %10.5f\n", 
	    type, coord(0), coord(1), coord(2) );
  }
  fprintf(fh, "\n");


  fprintf(fh,"Number of occupied states   = %10d\n", _nOccStates);
  fprintf(fh,"Number of extra states      = %10d\n", _nExtraStates);
  fprintf(fh,"Number of eigenvalues       = %10d\n", _npsi);
  fprintf(fh,"Inverse temperature         = %10.5e\n", _Tbeta);
  fprintf(fh, "\n");

  fprintf(fh,"Mixing method               = %s\n", _mixtype.c_str());
  fprintf(fh,"SCF tolerance               = %10.5e\n", _scftol);
  fprintf(fh,"LOBPCG tolerance            = %10.5e\n", _eigtol);
  fprintf(fh,"Maximum number of SCF       = %10d\n", _scfmaxiter);
  fprintf(fh,"Maximum number of LOBPCG    = %10d\n", _eigmaxiter);
  fprintf(fh, "\n");

  fprintf(fh,"DG Solver                   = %s\n", _dgsolver.c_str());
  fprintf(fh,"DG alpha                    = %10.5e\n", _alpha);
  fprintf(fh,"DG basis radius             = %10.5e\n", _basisradius);
  fprintf(fh,"DG delta                    = %10.5e\n", _delta);
  fprintf(fh,"DG gamma                    = %10.5e\n", _gamma);
  fprintf(fh,"Nenrich (ALB)               = %10d\n", _nenrich);
  fprintf(fh,"Neigperele                  = %10d\n", _Neigperele);
  fprintf(fh,"Norbperele                  = %10d\n", _Norbperele);
  fprintf(fh,"Buffer dual grid            = %10d\n", _bufdual);
  fprintf(fh,"Buffer update frequency     = %10d\n", _nBuffUpdate);
  fprintf(fh,"MB                          = %10d\n", _MB);

  //  fprintf(fh," nvnl(global) = %d\n", _nvnl);
  fprintf(fh, "\n");
  fflush(fh);
  return 0;
}

//--------------------------------------
int ScfDG::scf_PrintState(FILE *fh)
{
  for(int i = 0; i < _npsi; i++){
    fprintf(fh, "eig[%5d] :  %25.15e,   occ[%5d] : %15.5f\n", 
	    i, _ev[i], i, _occ[i]); 
  }

  fprintf(fh, "Total Energy = %25.15e [Ry]\n", _Etot*2);
  fprintf(fh, "Helmholtz    = %25.15e [Ry]\n", _Efree*2);
  fprintf(fh, "Ekin         = %25.15e [Ry]\n", _Ekin*2);
  fprintf(fh, "Ecor         = %25.15e [Ry]\n", _Ecor*2);
  //fprintf(fh, "Ecoul        = %25.15e [Ry]\n", _Ecoul*2);
  fprintf(fh, "Exc          = %25.15e [Ry]\n", _Exc*2);
  fflush(fh);
  return 0;
}

int ScfDG::scf_PAndersonMix(vector<double>& vtotnew,
			    ParVec<Index3, DblNumMat, ElemPtn>& df, 
			    ParVec<Index3, DblNumMat, ElemPtn>& dv, 
			    int iter)
{
  ParVec<Index3, DblNumTns, ElemPtn> vin, vout, vinsave, voutsave;
  int iterused, ipos;
  int mixdim;
  double alpha;
  int mpirank; MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

  alpha = _alpha;
  mixdim = _mixdim;

  int N1=_NElems[0];    int N2=_NElems[1];    int N3=_NElems[2];
  int NTOT = _Ns[0] * _Ns[1] * _Ns[2];

  {
    vin.prtn()     = _elemptn;
    vout.prtn()    = _elemptn;
    vinsave.prtn() = _elemptn;
    voutsave.prtn()= _elemptn;

    /*	 vin = _vtot;
     *	 vout = vtotnew - _vtot;
     *	 vinsave = vin;
     *	 voutsave = vout;
     */
    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if( _elemptn.owner(curkey)==mpirank ) {
	    Elem& curdat = _elemvec(i1,i2,i3);	    //Elem& curdat = _elemvec.lclmap()[curkey];
	    Index3& Nsglb = curdat.Nsglb();
	    Index3& posidx = curdat.posidx();

	    DblNumTns blank(Nsglb[0], Nsglb[1], Nsglb[2]);
	    setvalue(blank, 0.0);
	    vin.lclmap()[curkey] = blank;
	    vout.lclmap()[curkey] = blank;
	    vinsave.lclmap()[curkey] = blank;
	    voutsave.lclmap()[curkey] = blank;
	    DblNumTns& vincur = vin.lclmap()[curkey];
	    DblNumTns& voutcur = vout.lclmap()[curkey];
	    DblNumTns& vinsavecur = vinsave.lclmap()[curkey];
	    DblNumTns& voutsavecur = voutsave.lclmap()[curkey];

	    for(int k = 0; k < Nsglb[2]; k++){ int ksh = posidx[2]+ k; 
	      for(int j = 0; j < Nsglb[1]; j++){ int jsh = posidx[1] + j; 
		for(int i = 0; i < Nsglb[0]; i++){ int ish = posidx[0] + i;
		  int idxsh = ish + jsh * _Ns[0] + ksh * _Ns[0] * _Ns[1];
		  vincur(i,j,k) = _vtot[idxsh];
		  voutcur(i,j,k) = vtotnew[idxsh] - _vtot[idxsh];
		  vinsavecur(i,j,k) = vincur(i,j,k);
		  voutsavecur(i,j,k) = voutcur(i,j,k);
		}
	      }
	    }
	  } // if( owner(curkey) == mpirank)
	}
  }


  // LLIN: iter must start from 1!
  iterused = MIN(iter-1,mixdim);

  ipos = iter - 1 - ((iter-2)/mixdim)*mixdim;

  //cerr << "Anderson mixing = " << iterused << endl;
  //cerr << "iterused = " << iterused << endl;
  //cerr << "ipos = " << ipos << endl;

  // Initialize df and dv for Anderson mixing
  // iter starts from 1. VERY IMPORTANT
  if( iter == 1 ){
    df.prtn() = _elemptn;
    dv.prtn() = _elemptn;
    {
      for(int i3=0; i3<N3; i3++)
	for(int i2=0; i2<N2; i2++)
	  for(int i1=0; i1<N1; i1++) {
	    Index3 curkey = Index3(i1,i2,i3);
	    if( _elemptn.owner(curkey)==mpirank ) {
	      Elem& curdat = _elemvec(i1,i2,i3);//Elem& curdat = _elemvec.lclmap()[curkey];
	      Index3 Nsglb = curdat._Nsglb;
	      int lclNsglb = Nsglb[0] * Nsglb[1] * Nsglb[2];
	      DblNumMat blank(lclNsglb, mixdim);
	      setvalue(blank, 0.0);  // LLIN: IMPORTANT
	      df.lclmap()[curkey] = blank;
	      dv.lclmap()[curkey] = blank;
	    }
	  }
    }
  }

  // LLIN: Perform Anderson mixing by solving a least square problem
  // using PDGELSS

  if( iter > 1 ){
    /* df(:, ipos) -= vout;
       dv(:, ipos) -= vin */

    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if( _elemptn.owner(curkey)==mpirank ) {
	    double *pdf, *pdv, *pvout, *pvin;
	    int lclsize = df.lclmap()[curkey].m();
	    pdf = df.lclmap()[curkey].clmdata(ipos-1);
	    pdv = dv.lclmap()[curkey].clmdata(ipos-1);
	    pvin  =  vin.lclmap()[curkey].data();   
	    pvout = vout.lclmap()[curkey].data();
	    for(int l = 0; l < lclsize; l++){
	      *pdf -= *pvout;
	      *pdv -= *pvin;
	      pdf++; pdv++; pvin++; pvout++;
	    }
	  }
	}

    /* Calculate the normal equation
       A gammas = b
       Here A = df'*df
       b = df'*vout
       b is also denoted by gammas and is updated in-place in the
       least-square solving process.  */

    DblNumMat A(iterused, iterused);
    DblNumVec gammas(iterused);
    setvalue(A, 0.0); setvalue(gammas, 0.0);
    {
      DblNumMat tA(iterused, iterused);
      DblNumVec tgammas(iterused);
      setvalue(tA, 0.0);   setvalue(tgammas, 0.0);
      char trans = 'T', ntrans = 'N';
      for(int i3=0; i3<N3; i3++)
	for(int i2=0; i2<N2; i2++)
	  for(int i1=0; i1<N1; i1++) {
	    Index3 curkey = Index3(i1,i2,i3);
	    if( _elemptn.owner(curkey)==mpirank ) {
	      DblNumMat& dfcur = df.lclmap()[curkey];
	      DblNumTns& voutcur = vout.lclmap()[curkey];
	      int lclsize = dfcur.m();
	      double one = 1.0;
	      int ione = 1;
	      dgemm_(&trans, &ntrans, &iterused, &iterused,
		     &lclsize, &one, dfcur.data(), &lclsize, 
		     dfcur.data(), &lclsize, &one, tA.data(),
		     &iterused);
	      dgemv_(&trans, &lclsize, &iterused, &one, 
		     dfcur.data(), &lclsize, voutcur.data(), &ione,
		     &one, tgammas.data(), &ione);
	    }
	  }
      iC( MPI_Allreduce(tA.data(), A.data(), iterused*iterused, 
			MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );
      iC( MPI_Allreduce(tgammas.data(), gammas.data(), iterused, 
			MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );

    }



    /* LLIN: 
     * Every processor solves the least square problem for the normal
     * equation.  DGELSS is used since it is based on SVD and can avoid
     * the numerical instability problem. DGELSS is more
     * stable than DGELS which is based on QR factorization */
    {
      int nrhs = 1 , LWORK, info, rank;
      double rcond = 1e-12;  // LLIN: Condition number threshold
      // criterion for A=df'*df vector<double> WORK;
      /* Choice of LWORK
       * The dimension of the array WORK. LWORK >= 1, and also: 
       * LWORK >= 3*min(M,N) + max( 2*min(M,N), max(M,N), NRHS )
       * For good performance, LWORK should generally be larger. */
      LWORK = iterused * 20;

      DblNumVec WORK(LWORK), S(iterused);

      if( mpirank == MASTER ){
	cerr << "Start least square problem " << endl;
      }
      dgelss_(&iterused, &iterused, &nrhs, A.data(), &iterused,
	      gammas.data(), &iterused, S.data(), &rcond, &rank,
	      WORK.data(), &LWORK, &info);
      iC( info );
      if( mpirank == MASTER ){
	cerr << "End least square problem" << endl;
	cerr << "Rank of (df'*df) (criterion " << rcond << " ) = " << rank << endl;
      }
    }

    /* update vin, vout 
     * vin  -= dv * gammas
     * vout -= df * gammas */

    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if( _elemptn.owner(curkey)==mpirank ) {
	    int lclsize = df.lclmap()[curkey].m();
	    char ntrans = 'N';
	    double one = 1.0, mone = -1.0;
	    int ione = 1;
	    DblNumMat& dfcur = df.lclmap()[curkey];
	    DblNumMat& dvcur = dv.lclmap()[curkey];
	    DblNumTns& vincur = vin.lclmap()[curkey];
	    DblNumTns& voutcur = vout.lclmap()[curkey];

	    dgemv_(&ntrans, &lclsize, &iterused, &mone,
		   dfcur.data(), &lclsize, gammas.data(), &ione,
		   &one, voutcur.data(), &ione);

	    dgemv_(&ntrans, &lclsize, &iterused, &mone, 
		   dvcur.data(), &lclsize, gammas.data(), &ione,
		   &one, vincur.data(), &ione);

	  } // if( owner(curkey) == mpirank )
	}

  }


  /* Update dv and df locally 
   * df(:, inext) = vout
   * dv(:, inext) = vin */
  int inext = iter - ((iter-1)/mixdim)*mixdim;

  for(int i3=0; i3<N3; i3++)
    for(int i2=0; i2<N2; i2++)
      for(int i1=0; i1<N1; i1++) {
	Index3 curkey = Index3(i1,i2,i3);
	if( _elemptn.owner(curkey)==mpirank ) {
	  double *pdf, *pdv, *pvoutsave, *pvinsave;
	  int lclsize = df.lclmap()[curkey].m();
	  pdf = df.lclmap()[curkey].clmdata(inext-1);
	  pdv = dv.lclmap()[curkey].clmdata(inext-1);
	  pvinsave = vinsave.lclmap()[curkey].data();   
	  pvoutsave = voutsave.lclmap()[curkey].data();
	  for(int l = 0; l < lclsize; l++){
	    *pdf = *pvoutsave;
	    *pdv = *pvinsave;
	    pdf++; pdv++; pvinsave++; pvoutsave++;
	  }
	}
      }


  /* Update vtot by summing up all the local contributions i
   * vtot = vin + alpha * vout */

  DblNumTns vtottmp(_Ns[0], _Ns[1], _Ns[2]);
  setvalue(vtottmp, 0.0);

  for(int i3=0; i3<N3; i3++)
    for(int i2=0; i2<N2; i2++)
      for(int i1=0; i1<N1; i1++) {
	Index3 curkey = Index3(i1,i2,i3);
	if(_elemptn.owner(curkey)==mpirank) {
	  Elem& curdat = _elemvec(i1,i2,i3);	    //Elem& curdat = _elemvec.lclmap()[curkey];
	  Index3 Nsglb = curdat._Nsglb;
	  Index3 posidx = curdat.posidx();
	  DblNumTns& vincur = vin.lclmap()[curkey];
	  DblNumTns& voutcur = vout.lclmap()[curkey];
	  for(int k = 0; k < Nsglb[2]; k++){  int ksh = posidx[2]+ k; 
	    for(int j = 0; j < Nsglb[1]; j++){  int jsh = posidx[1] + j; 
	      for(int i = 0; i < Nsglb[0]; i++){  int ish = posidx[0] + i;
		vtottmp(ish, jsh, ksh) = vincur(i,j,k) + alpha * 
		  voutcur(i,j,k);
	      }
	    }
	  } // for(k)
	} // if( owner(curkey) == mpirank )
      } // for(i1)

  for(int i = 0; i < NTOT; i++){
    _vtot[i] = 0.0;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  // LLIN: Every processor updates the local pseudopotential 

  iC( MPI_Allreduce(vtottmp.data(), &(_vtot[0]), 
		    NTOT, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );


  return 0;
}



//--------------------------------------
// Sequential version (Obsolete).  Substituted by the parallel version
// PAndersonMix to reduce the memory cost.
//
int ScfDG::scf_AndersonMix(vector<double>& vtotnew,
			   vector<double>& df, vector<double>& dv, int iter)
{
  int N1, N2, N3, NTOT;
  vector<double> vin, vout, vinsave, voutsave;
  int iterused, ipos;
  int mixdim;
  double alpha;

  alpha = _alpha;
  mixdim = _mixdim;

  N1 = _Ns[0]; N2 = _Ns[1]; N3 = _Ns[2]; NTOT = _ntot;
  vin.resize(NTOT);
  vout.resize(NTOT);
  vinsave.resize(NTOT);
  voutsave.resize(NTOT);

  for(int i = 0; i < NTOT; i++){
    vin[i] = _vtot[i];
    vout[i] = vtotnew[i] - _vtot[i];
  }

  for(int i = 0; i < NTOT; i++){
    vinsave[i] = vin[i];
    voutsave[i] = vout[i];
  }

  iterused = MIN(iter-1,mixdim);

  ipos = iter - 1 - ((iter-2)/mixdim)*mixdim;

  //cerr << "Anderson mixing = " << iterused << endl;
  //cerr << "iterused = " << iterused << endl;
  //cerr << "ipos = " << ipos << endl;

  // clear df and dv
  if( iter == 1 ){
    for(int i = 0; i < NTOT * mixdim; i++){
      df[i] = 0.0; dv[i] = 0.0;
    }
  }

  // LLIN: Calculating the pseudoinverse  using DGELSS

  if(1){
    if( iter > 1 ){
      for(int i = 0; i < NTOT; i++){
	df[(ipos-1)*NTOT+i] -= vout[i];
	dv[(ipos-1)*NTOT+i] -= vin[i];
      }

      /* Calculating pseudoinverse */
      vector<double> gammas, dftemp;
      int m, n, nrhs, lda, ldb, rank, LWORK, info;
      double rcond;
      vector<double> S, WORK;

      S.resize(iterused);

      m = NTOT;
      n = iterused;
      nrhs = 1;
      lda = m;
      ldb = m;
      rcond = 1e-6;

      LWORK =(3*MIN(m,n) + MAX(MAX(2*MIN(m,n), MAX(m,n)), nrhs));
      //cerr << "LWORK = " << LWORK << endl;
      WORK.resize(LWORK);

      gammas.resize(NTOT);
      for(int i = 0; i < NTOT; i++){
	gammas[i] = vout[i];
      }
      dftemp.resize(NTOT*iterused);
      for(int i = 0; i < NTOT*iterused; i++){
	dftemp[i] = df[i];
      }
      cerr << "Start calculating pseudoinverse" << endl;
      dgelss_(&m, &n, &nrhs, &dftemp[0], &lda, &gammas[0], &ldb, 
	      &S[0], &rcond, &rank, &WORK[0], &LWORK,
	      &info);
      cerr << "End calculating pseudoinverse" << endl;
      //cerr << " rank of dfmat = " << rank << endl;
      if(info){
	cerr << "DGELSS ERROR! INFO = " << info << endl;
	ABORT("DGELSS ERROR",1);
      }

      /* update vin, vout*/

      for( int l = 0; l < iterused; l++){
	for(int i = 0; i < NTOT; i++){
	  vin[i]  -= gammas[l] * dv[i+l*NTOT];
	  vout[i] -= gammas[l] * df[i+l*NTOT];
	}
      }
    }
  }

  // LLIN: Calculating the pseudoinverse  using DGELS

  if(0){
    if( iter > 1 ){
      for(int i = 0; i < NTOT; i++){
	df[(ipos-1)*NTOT+i] -= vout[i];
	dv[(ipos-1)*NTOT+i] -= vin[i];
      }

      /* Calculating pseudoinverse */
      char trans='N';
      vector<double> gammas, dftemp;
      int m, n, nrhs, lda, ldb, LWORK, info, MN, NB;
      vector<double> WORK;

      m = NTOT;
      n = iterused;
      nrhs = 1;
      lda = m;
      ldb = m;
      MN = MIN(m,n);
      NB = 20;


      /* Choice of LWORK
	 For optimal performance,
	 LWORK >= max( 1, MN + max( MN, NRHS )*NB ).
	 where MN = min(M,N) and NB is the optimum block size.  */
      LWORK = MN + MAX( MN, nrhs ) * NB; 
      cerr << "  m = " << m << "  n = " << n << "  MN = " << MN << endl;
      cerr << "LWORK = " << LWORK << endl;

      WORK.resize(LWORK);

      gammas.resize(NTOT);
      for(int i = 0; i < NTOT; i++){
	gammas[i] = vout[i];
      }
      dftemp.resize(NTOT*iterused);
      for(int i = 0; i < NTOT*iterused; i++){
	dftemp[i] = df[i];
      }
      cerr << "Start calculating pseudoinverse" << endl;
      dgels_(&trans, &m, &n, &nrhs, &dftemp[0], &lda, &gammas[0], &ldb, 
	     &WORK[0], &LWORK, &info);
      cerr << "End calculating pseudoinverse" << endl;
      if(info){
	cerr << "DGELS ERROR! INFO = " << info << endl;
	ABORT("DGELS ERROR",1);
      }

      /* update vin, vout*/

      for( int l = 0; l < iterused; l++){
	for(int i = 0; i < NTOT; i++){
	  vin[i]  -= gammas[l] * dv[i+l*NTOT];
	  vout[i] -= gammas[l] * df[i+l*NTOT];
	}
      }
    }
  }


  int inext = iter - ((iter-1)/mixdim)*mixdim;

  for(int i = 0; i < NTOT; i++){
    df[(inext-1)*NTOT+i] = voutsave[i];
    dv[(inext-1)*NTOT+i] = vinsave[i];
  }

  for(int i = 0; i < NTOT; i++){
    _vtot[i] = vin[i] + alpha * vout[i];
  }

  return 0;
}

//--------------------------------------
int ScfDG::scf_KerkerMix(vector<double>& vtotnew, double alpha)
{
  // Old version without forcing the alignment
  if(0){
    int NTOT = _ntot;

    vector<cpx> vres, rfft, vcor;
    vres.resize(NTOT);
    rfft.resize(NTOT);
    vcor.resize(NTOT);

    for(int i = 0; i < NTOT; i++){
      vres[i] = cpx(vtotnew[i]-_vtot[i],0.0);
    } 
    fftw_execute_dft(_planpsiforward, 
		     reinterpret_cast<fftw_complex*>(&vres[0]), 
		     reinterpret_cast<fftw_complex*>(&rfft[0]) );
    for(int i = 0; i < NTOT; i++){
      if( _gkk[i] == 0 )
	rfft[i] = -rfft[i] / 2.0;
      else
	rfft[i] *= alpha * _gkk[i] / (_gkk[i]+0.5) - alpha;
    } 

    fftw_execute_dft(_planpsibackward, 
		     reinterpret_cast<fftw_complex*>(&rfft[0]), 
		     reinterpret_cast<fftw_complex*>(&vcor[0]) );
    for(int i = 0; i < NTOT; i++){
      vcor[i] /= double(NTOT);
    } 
    for(int i = 0; i < NTOT; i++){
      _vtot[i] += vcor[i].real() + alpha * vres[i].real();
    } 

    return 0;
  }

  // New version forcing the alignment
  if(1){
    int NTOT = _ntot;

    fftw_complex *vres, *rfft, *vcor;
    vres = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*NTOT);
    rfft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*NTOT);
    vcor = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*NTOT);

    for(int i = 0; i < NTOT; i++){
      vres[i][0] = vtotnew[i]-_vtot[i];
      vres[i][1] = 0.0;
    } 
    fftw_execute_dft(_planpsiforward, 
		     (&vres[0]), 
		     (&rfft[0]) );

    for(int i = 0; i < NTOT; i++){
      if( _gkk[i] == 0 ){
	rfft[i][0] = -rfft[i][0] / 2.0;
	rfft[i][1] = -rfft[i][1] / 2.0;
      }
      else{
	rfft[i][0] *= alpha * _gkk[i] / (_gkk[i]+0.5) - alpha;
	rfft[i][1] *= alpha * _gkk[i] / (_gkk[i]+0.5) - alpha;
      }
    } 

    fftw_execute_dft(_planpsibackward, 
		     (&rfft[0]), 
		     (&vcor[0]) );
    for(int i = 0; i < NTOT; i++){
      vcor[i][0] /= double(NTOT);
      vcor[i][1] /= double(NTOT);
    } 
    for(int i = 0; i < NTOT; i++){
      _vtot[i] += vcor[i][0] + alpha * vres[i][0];
    } 

    fftw_free(vres);
    fftw_free(rfft);
    fftw_free(vcor);

    return 0;
  }

}

//--------------------------------------
int ScfDG::force_innerprod(SparseVec& cur, double* ful, double* www, double* res)
{
  IntNumVec& iv = cur.first;
  DblNumMat& dv = cur.second;
  //clear
  for(int g=0; g<4; g++) {
    res[g] = 0;
    for(int k=0; k<iv.m(); k++)
      res[g] += dv(g,k) * ful[iv(k)] * www[iv(k)];
  }
  return 0;
}

//--------------------------------------
//Auxilliary subroutine for the force calculation.  Calculates the inner
//product of all the four components (values and three derivatives) of a
//sparse vector with another full vector.
int ScfDG::force_innerprod(SparseVec& cur, double* ful, double wgt, double* res)
{
  IntNumVec& iv = cur.first;
  DblNumMat& dv = cur.second;
  //clear
  for(int g=0; g<4; g++) {
    res[g] = 0;
    for(int k=0; k<iv.m(); k++)
      res[g] += dv(g,k) * ful[iv(k)];
    res[g] *= wgt;
  }
  return 0;
}

//--------------------------------------
//Auxilliary subroutine for the force calculation.  Calculates the inner
//product of a sparse vector (no derivatives)  with another full vector.
int ScfDG::force_innerprod_val(SparseVec& cur, double* ful, double wgt,
			       double& res)
{
  IntNumVec& iv = cur.first;
  DblNumMat& dv = cur.second;
  //clear
  res = 0.0;
  for(int k=0; k<iv.m(); k++)
    res += dv(0,k) * ful[iv(k)];  //LL: value only
  res *= wgt; // (_vol/_ntot);
  return 0;
}


//--------------------------------------
int ScfDG::force()
{
  // Old version without forcing alignment
  if(0){
    int myid = mpirank();
    int natoms = _atomvec.size();

    //EACH PROCESSOR DOES THE FOLLOWING COMPUTATION
    //calculate Vrho
    vector<cpx> crho(_ntot);
    for(int i = 0; i<_ntot; i++)    crho[i] = cpx(_rho[i]-_rho0[i]);
    vector<cpx> rhotemp(_ntot);
    fftw_execute_dft(_planpsiforward, 
		     reinterpret_cast<fftw_complex*>(&crho[0]), 
		     reinterpret_cast<fftw_complex*>(&rhotemp[0]));
    /* 2.0*pi rather than 4.0*pi is because gkk = 1/2 k^2 */
    for(int i = 0; i < _ntot; i++){
      if(_gkk[i] != 0 )
	rhotemp[i] *= 2.0*M_PI/_gkk[i];
      else
	rhotemp[i] = cpx(0.0, 0.0);
    }
    fftw_execute_dft(_planpsibackward, 
		     reinterpret_cast<fftw_complex*>(&rhotemp[0]), 
		     reinterpret_cast<fftw_complex*>(&crho[0]));
    vector<double> vrho(_ntot, 0.0);
    for(int i = 0; i < _ntot; i++ ){
      vrho[i] = crho[i].real()/(double)_ntot;
    }
    //For Method II of calculate the force using the derivative on the
    //electrostatic potential to improve accuracy.

    DblNumMat vhartdev(_ntot, 3); // three derivatives of vhart
    CpxNumVec cpxtmp1(_ntot), cpxtmp2(_ntot);
    //rhotemp is already the Fourier transform of the electrostatic
    //potential vrho.

    for(int g = 0; g < 3; g++){
      cpx *vik = _ik.clmdata(g);
      for(int i = 0; i < _ntot; i++){
	cpxtmp1[i] = rhotemp[i]*vik[i];
      }
      fftw_execute_dft(_planpsibackward, 
		       reinterpret_cast<fftw_complex*>(&cpxtmp1[0]), 
		       reinterpret_cast<fftw_complex*>(&cpxtmp2[0]) );
      double *vtmp = vhartdev.clmdata(g);
      for(int i = 0; i < _ntot; i++){
	vtmp[i] = cpxtmp2[i].real()/(double)_ntot; //LL: VERY IMPORTANT
      }
    }


    //--------------------
    Index3 Nlbls = _elemtns(0,0,0).Ns();
    Point3 hs = _elemtns(0,0,0).Ls(); //length of element
    int Nlbl1 = Nlbls(0);  int Nlbl2 = Nlbls(1);  int Nlbl3 = Nlbls(2);
    double h1 = hs(0);  double h2 = hs(1);  double h3 = hs(2);
    DblNumVec x1(Nlbl1), w1(Nlbl1);
    lglnodes(x1.data(), w1.data(), Nlbl1-1);
    for(int g=0; g<x1.m(); g++)    x1(g) = x1(g)/2.0*h1;
    for(int g=0; g<w1.m(); g++)    w1(g) = w1(g)/2.0*h1;
    DblNumVec x2(Nlbl2), w2(Nlbl2);
    lglnodes(x2.data(), w2.data(), Nlbl2-1);
    for(int g=0; g<x2.m(); g++)    x2(g) = x2(g)/2.0*h2;
    for(int g=0; g<w2.m(); g++)    w2(g) = w2(g)/2.0*h2;
    DblNumVec x3(Nlbl3), w3(Nlbl3);
    lglnodes(x3.data(), w3.data(), Nlbl3-1);
    for(int g=0; g<x3.m(); g++)    x3(g) = x3(g)/2.0*h3;
    for(int g=0; g<w3.m(); g++)    w3(g) = w3(g)/2.0*h3;
    DblNumTns www(Nlbl1,Nlbl2,Nlbl3);
    for(int g1=0; g1<Nlbl1; g1++)
      for(int g2=0; g2<Nlbl2; g2++)
	for(int g3=0; g3<Nlbl3; g3++) {
	  www(g1,g2,g3) = w1(g1)*w2(g2)*w3(g3);
	}
    //--------------------
    vector<Point3> force(natoms, Point3(0,0,0));
    for(int a=0; a<natoms; a++) {
      if(_psdoptn.owner(a)==myid) {
	int VL=0, DX=1, DY=2, DZ=3;
	//-----------------------------------
	//1.
	Point3 part1(0,0,0);
	//Method I: Calculate the force using the derivative on the
	//pseudo-charge
	if(0){
	  double res[4];
	  SparseVec& rho0now = _psdovec.lclmap()[a].rho0();
	  double wgt = _vol/double(_ntot);
	  iC( force_innerprod(rho0now, &(vrho[0]), wgt, res) );
	  part1(0) = -res[DX];  //LEXING: MINUS SIGN VERY IMPORTANT
	  part1(1) = -res[DY];
	  part1(2) = -res[DZ];
	}
	//Method II: Calculate the force using the derivative on the
	//electrostatic potential to improve accuracy.
	if(1){
	  double res[3];
	  double wgt = _vol/double(_ntot);
	  SparseVec& rho0now = _psdovec.lclmap()[a].rho0();
	  iC( force_innerprod_val(rho0now, vhartdev.clmdata(0), wgt, res[0]) );
	  iC( force_innerprod_val(rho0now, vhartdev.clmdata(1), wgt, res[1]) );
	  iC( force_innerprod_val(rho0now, vhartdev.clmdata(2), wgt, res[2]) );
	  //LL: NOTE There is no MINUS sign here due to integration by
	  //parts.
	  part1(0) = res[0];
	  part1(1) = res[1];
	  part1(2) = res[2];
	}
	//-----------------------------------
	//2.
	Point3 part2(0,0,0);
	vector< pair<NumTns<SparseVec>,double> >& vnls = _psdovec.lclmap()[a].vnls();
	for(int e=0; e<vnls.size(); e++) {
	  double gamma = vnls[e].second;
	  NumTns<SparseVec>& be = vnls[e].first;
	  Index3 Ns = _NElems; //LEXING: CHANGE OF DEF FOR Ns
	  int N1 = Ns(0);	int N2 = Ns(1);	int N3 = Ns(2);
	  //get alpha matrix
	  NumTns<DblNumMat> alpha;
	  alpha.resize(N1,N2,N3);
	  for(int i3=0; i3<N3; i3++)
	    for(int i2=0; i2<N2; i2++)
	      for(int i1=0; i1<N1; i1++) {
		if( be(i1,i2,i3).first.m()>0 ) {
		  //map<Index3,Elem>::iterator ei = _elemvec.lclmap().find(Index3(i1,i2,i3)); iA(ei!=_elemvec.lclmap().end());
		  vector<DblNumTns>& bases = _basesvec.lclmap()[Index3(i1,i2,i3)];		//vector<DblNumTns>& bases = (*ei).second.bases();
		  int nbases = bases.size();
		  alpha(i1,i2,i3).resize(4,nbases);
		  for(int a=0; a<nbases; a++) {
		    iC( force_innerprod(be(i1,i2,i3), bases[a].data(), www.data(), alpha(i1,i2,i3).clmdata(a)) );
		  }
		} else {
		  //if empty
		  alpha(i1,i2,i3).resize(0,0);		//setvalue(alpha(i1,i2,i3), 0.0);
		}
	      }
	  //go through all eigenvecs
	  for(int g=0; g<_npsi; g++) {
	    DblNumVec res(4); 	  //accumulate results
	    setvalue(res, 0.0);
	    for(int i3=0; i3<N3; i3++)
	      for(int i2=0; i2<N2; i2++)
		for(int i1=0; i1<N1; i1++) {
		  if(be(i1,i2,i3).first.m()>0) {
		    //get coefficient
		    //map<Index3,Elem>::iterator ei = _elemvec.lclmap().find(Index3(i1,i2,i3)); iA(ei!=_elemvec.lclmap().end());
		    //DblNumMat& eigvecs = (*ei).second.eigvecs();
		    DblNumMat& eigvecs = _eigvecsvec.lclmap()[Index3(i1,i2,i3)];
		    DblNumVec tmp(eigvecs.m(), false, eigvecs.clmdata(g));
		    iA( alpha(i1,i2,i3).m()>0 && alpha(i1,i2,i3).n()>0 );
		    iC( dgemv(1.0, alpha(i1,i2,i3), tmp, 1.0, res) );
		  }
		}
	    part2(0) += 4*_occ[g]*(gamma* res(VL) * res(DX));
	    part2(1) += 4*_occ[g]*(gamma* res(VL) * res(DY));
	    part2(2) += 4*_occ[g]*(gamma* res(VL) * res(DZ));
	  }
	}
	//
	force[a] = part1+part2;
      }
    }
    vector<double> tmpfs(3*natoms);
    for(int a=0; a<natoms; a++) {
      tmpfs[3*a+0] = force[a](0);    tmpfs[3*a+1] = force[a](1);    tmpfs[3*a+2] = force[a](2);
    }
    //GLOBAL SUM
    vector<double> fstot(3*natoms);
    iC( MPI_Allreduce(&(tmpfs[0]), &(fstot[0]), 3*natoms, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );
    MPI_Barrier(MPI_COMM_WORLD);

    //LL: All processors have access to the force of all atoms
    for(int a=0; a<natoms; a++){
      Point3& fs = _atomvec[a].force();
      fs = Point3(fstot[3*a], fstot[3*a+1], fstot[3*a+2]);
    }
    return 0;
  }

  // New version forcing alignment
  if(1){
    int myid = mpirank();
    int natoms = _atomvec.size();

    //EACH PROCESSOR DOES THE FOLLOWING COMPUTATION
    //calculate Vrho
    fftw_complex *crho, *rhotemp;
    crho    = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*_ntot);
    rhotemp = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*_ntot);

    for(int i = 0; i<_ntot; i++){
      crho[i][0] = _rho[i]-_rho0[i];
      crho[i][1] = 0.0;
    }

    fftw_execute_dft(_planpsiforward, 
		     (&crho[0]), 
		     (&rhotemp[0]));

    /* 2.0*pi rather than 4.0*pi is because gkk = 1/2 k^2 */
    for(int i = 0; i < _ntot; i++){
      if(_gkk[i] != 0 ){
	rhotemp[i][0] *= 2.0*M_PI/_gkk[i];
	rhotemp[i][1] *= 2.0*M_PI/_gkk[i];
      }
      else{
	rhotemp[i][0] = 0.0;
	rhotemp[i][1] = 0.0;
      }
    }

    fftw_execute_dft(_planpsibackward, (&rhotemp[0]), (&crho[0]));

    vector<double> vrho(_ntot, 0.0);
    for(int i = 0; i < _ntot; i++ ){
      vrho[i] = crho[i][0]/(double)_ntot;
    }


    //For Method II of calculate the force using the derivative on the
    //electrostatic potential to improve accuracy.

    DblNumMat vhartdev(_ntot, 3); // three derivatives of vhart

    fftw_complex *cpxtmp1, *cpxtmp2;
    cpxtmp1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*_ntot);
    cpxtmp2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*_ntot);

    //rhotemp is already the Fourier transform of the electrostatic
    //potential vrho.

    for(int g = 0; g < 3; g++){
      cpx *vik = _ik.clmdata(g);
      for(int i = 0; i < _ntot; i++){
	cpxtmp1[i][0] = rhotemp[i][0]*vik[i].real() - rhotemp[i][1]*vik[i].imag();
	cpxtmp1[i][1] = rhotemp[i][0]*vik[i].imag() + rhotemp[i][1]*vik[i].real();
      }
      fftw_execute_dft(_planpsibackward, 
		       (&cpxtmp1[0]), 
		       (&cpxtmp2[0]) );
      double *vtmp = vhartdev.clmdata(g);
      for(int i = 0; i < _ntot; i++){
	vtmp[i] = cpxtmp2[i][0]/(double)_ntot; //LL: VERY IMPORTANT
      }
    }

    fftw_free(rhotemp);
    fftw_free(crho);
    fftw_free(cpxtmp1);
    fftw_free(cpxtmp2);

    //--------------------
    Index3 Nlbls = _elemtns(0,0,0).Ns();
    Point3 hs = _elemtns(0,0,0).Ls(); //length of element
    int Nlbl1 = Nlbls(0);  int Nlbl2 = Nlbls(1);  int Nlbl3 = Nlbls(2);
    double h1 = hs(0);  double h2 = hs(1);  double h3 = hs(2);
    DblNumVec x1(Nlbl1), w1(Nlbl1);
    lglnodes(x1.data(), w1.data(), Nlbl1-1);
    for(int g=0; g<x1.m(); g++)    x1(g) = x1(g)/2.0*h1;
    for(int g=0; g<w1.m(); g++)    w1(g) = w1(g)/2.0*h1;
    DblNumVec x2(Nlbl2), w2(Nlbl2);
    lglnodes(x2.data(), w2.data(), Nlbl2-1);
    for(int g=0; g<x2.m(); g++)    x2(g) = x2(g)/2.0*h2;
    for(int g=0; g<w2.m(); g++)    w2(g) = w2(g)/2.0*h2;
    DblNumVec x3(Nlbl3), w3(Nlbl3);
    lglnodes(x3.data(), w3.data(), Nlbl3-1);
    for(int g=0; g<x3.m(); g++)    x3(g) = x3(g)/2.0*h3;
    for(int g=0; g<w3.m(); g++)    w3(g) = w3(g)/2.0*h3;
    DblNumTns www(Nlbl1,Nlbl2,Nlbl3);
    for(int g1=0; g1<Nlbl1; g1++)
      for(int g2=0; g2<Nlbl2; g2++)
	for(int g3=0; g3<Nlbl3; g3++) {
	  www(g1,g2,g3) = w1(g1)*w2(g2)*w3(g3);
	}
    //--------------------
    vector<Point3> force(natoms, Point3(0,0,0));
    for(int a=0; a<natoms; a++) {
      if(_psdoptn.owner(a)==myid) {
	int VL=0, DX=1, DY=2, DZ=3;
	//-----------------------------------
	//1.
	Point3 part1(0,0,0);
	//Method I: Calculate the force using the derivative on the
	//pseudo-charge
	if(0){
	  double res[4];
	  SparseVec& rho0now = _psdovec.lclmap()[a].rho0();
	  double wgt = _vol/double(_ntot);
	  iC( force_innerprod(rho0now, &(vrho[0]), wgt, res) );
	  part1(0) = -res[DX];  //LEXING: MINUS SIGN VERY IMPORTANT
	  part1(1) = -res[DY];
	  part1(2) = -res[DZ];
	}
	//Method II: Calculate the force using the derivative on the
	//electrostatic potential to improve accuracy.
	if(1){
	  double res[3];
	  double wgt = _vol/double(_ntot);
	  SparseVec& rho0now = _psdovec.lclmap()[a].rho0();
	  iC( force_innerprod_val(rho0now, vhartdev.clmdata(0), wgt, res[0]) );
	  iC( force_innerprod_val(rho0now, vhartdev.clmdata(1), wgt, res[1]) );
	  iC( force_innerprod_val(rho0now, vhartdev.clmdata(2), wgt, res[2]) );
	  //LL: NOTE There is no MINUS sign here due to integration by
	  //parts.
	  part1(0) = res[0];
	  part1(1) = res[1];
	  part1(2) = res[2];
	}
	//-----------------------------------
	//2.
	Point3 part2(0,0,0);
	vector< pair<NumTns<SparseVec>,double> >& vnls = _psdovec.lclmap()[a].vnls();
	for(int e=0; e<vnls.size(); e++) {
	  double gamma = vnls[e].second;
	  NumTns<SparseVec>& be = vnls[e].first;
	  Index3 Ns = _NElems; //LEXING: CHANGE OF DEF FOR Ns
	  int N1 = Ns(0);	int N2 = Ns(1);	int N3 = Ns(2);
	  //get alpha matrix
	  NumTns<DblNumMat> alpha;
	  alpha.resize(N1,N2,N3);
	  for(int i3=0; i3<N3; i3++)
	    for(int i2=0; i2<N2; i2++)
	      for(int i1=0; i1<N1; i1++) {
		if( be(i1,i2,i3).first.m()>0 ) {
		  //map<Index3,Elem>::iterator ei = _elemvec.lclmap().find(Index3(i1,i2,i3)); iA(ei!=_elemvec.lclmap().end());
		  vector<DblNumTns>& bases = _basesvec.lclmap()[Index3(i1,i2,i3)];		//vector<DblNumTns>& bases = (*ei).second.bases();
		  int nbases = bases.size();
		  alpha(i1,i2,i3).resize(4,nbases);
		  for(int a=0; a<nbases; a++) {
		    iC( force_innerprod(be(i1,i2,i3), bases[a].data(), www.data(), alpha(i1,i2,i3).clmdata(a)) );
		  }
		} else {
		  //if empty
		  alpha(i1,i2,i3).resize(0,0);		//setvalue(alpha(i1,i2,i3), 0.0);
		}
	      }
	  //go through all eigenvecs
	  for(int g=0; g<_npsi; g++) {
	    DblNumVec res(4); 	  //accumulate results
	    setvalue(res, 0.0);
	    for(int i3=0; i3<N3; i3++)
	      for(int i2=0; i2<N2; i2++)
		for(int i1=0; i1<N1; i1++) {
		  if(be(i1,i2,i3).first.m()>0) {
		    //get coefficient
		    //map<Index3,Elem>::iterator ei = _elemvec.lclmap().find(Index3(i1,i2,i3)); iA(ei!=_elemvec.lclmap().end());
		    //DblNumMat& eigvecs = (*ei).second.eigvecs();
		    DblNumMat& eigvecs = _eigvecsvec.lclmap()[Index3(i1,i2,i3)];
		    DblNumVec tmp(eigvecs.m(), false, eigvecs.clmdata(g));
		    iA( alpha(i1,i2,i3).m()>0 && alpha(i1,i2,i3).n()>0 );
		    iC( dgemv(1.0, alpha(i1,i2,i3), tmp, 1.0, res) );
		  }
		}
	    part2(0) += 4*_occ[g]*(gamma* res(VL) * res(DX));
	    part2(1) += 4*_occ[g]*(gamma* res(VL) * res(DY));
	    part2(2) += 4*_occ[g]*(gamma* res(VL) * res(DZ));
	  }
	}
	//
	force[a] = part1+part2;
      }
    }
    vector<double> tmpfs(3*natoms);
    for(int a=0; a<natoms; a++) {
      tmpfs[3*a+0] = force[a](0);    tmpfs[3*a+1] = force[a](1);    tmpfs[3*a+2] = force[a](2);
    }
    //GLOBAL SUM
    vector<double> fstot(3*natoms);
    iC( MPI_Allreduce(&(tmpfs[0]), &(fstot[0]), 3*natoms, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );
    MPI_Barrier(MPI_COMM_WORLD);

    //LL: All processors have access to the force of all atoms
    for(int a=0; a<natoms; a++){
      Point3& fs = _atomvec[a].force();
      fs = Point3(fstot[3*a], fstot[3*a+1], fstot[3*a+2]);
    }
    return 0;
  }

}
