#include "scfpw.hpp"
#include "eigpw.hpp"

extern FILE* fhstat;

//--------------------------------------
ScfPW::ScfPW()
{
  //---------------------------------------------------------------
  _mixtypeset.insert("anderson"); //LY: USED
  _mixtypeset.insert("kerker");
  
  _pseudotypeset.insert("GTH"); //LL: Default
  _pseudotypeset.insert("HGH");
  _pseudotypeset.insert("TM");
  /*
  _solvemodeset.insert("global");
  _solvemodeset.insert("dg");
  _solvemodeset.insert("dgv2"); //LY: not used
  
  _mappingmodeset.insert("self"); //LY: sequential, everything in one proc
  _mappingmodeset.insert("uniform"); //LY: n elements to n procs
  
  _bufferatommodeset.insert("auto"); //LY: used, get buffer automatically
  _bufferatommodeset.insert("manual"); //LY: Don't use
  
  _restartmodeset.insert("from_scratch"); //LY: random
  _restartmodeset.insert("restart"); //LY: read from outside
  */
  
  //------------------------
  _planpsibackward = NULL;
  _planpsiforward  = NULL;
}

//--------------------------------------
ScfPW::~ScfPW()
{
  if( _planpsibackward ){    fftw_destroy_plan(_planpsibackward);  }
  if( _planpsiforward ){    fftw_destroy_plan(_planpsiforward);  }
}

//--------------------------------------
int ScfPW::setup()
{
  int myid = mpirank();
  
  //---------------------------------------------------------
  _Ls = _dm.Ls();
  _Ns = _dm.Ns();
  _pos = _dm.pos();
  _vol = _Ls[0] * _Ls[1] * _Ls[2];
  _ntot = _Ns[0] * _Ns[1] * _Ns[2];
  double pi = 4.0 * atan(1.0);
  int ndim = 3;
  _gridpos.resize(ndim);
  for(int d = 0; d < ndim; d++){
    _gridpos[d].resize(_Ns[d]);
    double h = _Ls[d] / _Ns[d];
    for(int i = 0; i < _Ns[d]; i++){
      _gridpos[d](i) = _pos[d] + i*h;
    }
  }
  
  _rho.resize(_ntot,0.0);
  _rho0.resize(_ntot,0.0);
  
  _vtot.resize(_ntot,0.0);
  _vhart.resize(_ntot,0.0);
  _vxc.resize(_ntot,0.0);
  
  /* Ewald and Ealphat, CURRENTLY set to ZERO */
  //_Ewald = 0.0;
  //_Ealphat = 0.0;
  
  int nelec = 0;
  int indtyp;
  for(int i=0; i<_atomvec.size(); i++) {
    int type = _atomvec[i].type();
    Point3 coord = _atomvec[i].coord();
    iA( _ptable.ptemap().find(type)!=_ptable.ptemap().end() ); //MAKE SURE THAT THIS TYPE OF ATOM IS AVAILABLE
    nelec = nelec + _ptable.ptemap()[type].params()(PeriodTable::i_Zion);
  }
  if( nelec % 2 == 1 ){
    fprintf(stderr,"odd number of electrons is not supported in the current spin-unpolarized code!\n");
    exit(1);
  }
  _nOccStates = ceil((double)nelec/2.0); //LEXING: NUMBER OF OCC STATES
  _npsi = _nOccStates + _nExtraStates;
  
  /* Initialize eigenvalues and wavefunctions */
  _ev.resize(_npsi);
  _occ.resize(_npsi);
  
  //----------------------------------------------------
  int VL=0, DX=1, DY=2, DZ=3;
  //DIAG PSEUDOPOTENTIAL
  int ntot = this->ntot();
  double* rho0 = &_rho0[0];
  _rho0s.resize(_atomvec.size());
  for(int a=0; a<_atomvec.size(); a++) {
    iC( _ptable.pseudoRho0(_atomvec[a], _Ls, _pos, _Ns, _rho0s[a]) );
    //accumulate
    IntNumVec& idx = _rho0s[a].first;
    DblNumMat& val = _rho0s[a].second;
    for(int k=0; k<idx.m(); k++)
      rho0[idx(k)] += val(VL,k);
  }
  double sumrho = 0; 
  for(int i = 0; i < ntot; i++){
    sumrho += rho0[i];
  }
  sumrho *= vol()/ntot;
  //
  fprintf(stderr, "sumrho %e %e:\n", sumrho, _nOccStates*2.0);
  //LEXING: MAKE IT EQUAL TO THE # of ELECTRONS BY SHIFTING
  double diff = (_nOccStates*2-sumrho)/(vol());
  for(int i=0; i<ntot; i++)    rho0[i] += diff;
  //-----------------------------
  //NONLOCAL PSEUDOPOTENTIAL
  _vnlss.resize(_atomvec.size());
  int cnt = 0;
  for(int a=0; a<_atomvec.size(); a++) {
    iC( _ptable.pseudoNL(_atomvec[a], _Ls, _pos, _gridpos, _vnlss[a]) );
    cnt = cnt + _vnlss[a].size();
  }
  cerr<<"num of nl "<<cnt<<endl;
  for(int a=0; a<_vnlss.size(); a++) {
    for(int k=0; k<_vnlss[a].size(); k++) {
      SparseVec& now = _vnlss[a][k].first;
      IntNumVec& iv = now.first;
      DblNumMat& dv = now.second;
      double sum = 0;
      for(int a=0; a<dv.n(); a++)
        sum += dv(VL,a)*dv(VL,a);
      sum *= vol()/ntot;
      cerr<<a<<" "<<k<<" energy "<<sum<<endl;
    }
  }  
  // Old plan without guarantee of aligning
  if(0){
    vector<cpx> psitemp1, psitemp2;
    psitemp1.resize(_ntot);
    psitemp2.resize(_ntot);
    _planpsibackward = fftw_plan_dft_3d(_Ns[2], _Ns[1], _Ns[0],
					reinterpret_cast<fftw_complex*>(&psitemp1[0]),
					reinterpret_cast<fftw_complex*>(&psitemp2[0]),
					FFTW_BACKWARD, FFTW_MEASURE);
    _planpsiforward = fftw_plan_dft_3d(_Ns[2], _Ns[1], _Ns[0],
				       reinterpret_cast<fftw_complex*>(&psitemp2[0]),
				       reinterpret_cast<fftw_complex*>(&psitemp1[0]),
				       FFTW_FORWARD, FFTW_MEASURE);
  }

  // New plan using fftw_malloc with aligning.
  if(1){
    fftw_complex* psitemp1, *psitemp2;
    psitemp1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*_ntot);
    psitemp2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*_ntot);

    _planpsibackward = fftw_plan_dft_3d(_Ns[2], _Ns[1], _Ns[0],
					(&psitemp1[0]),
					(&psitemp2[0]),
					FFTW_BACKWARD, FFTW_MEASURE);
    _planpsiforward = fftw_plan_dft_3d(_Ns[2], _Ns[1], _Ns[0],
				       (&psitemp2[0]),
				       (&psitemp1[0]),
				       FFTW_FORWARD, FFTW_MEASURE);

    fftw_free(psitemp1);
    fftw_free(psitemp2);
  }
  
  {
    double pi = 4.0 * atan(1.0);
    _gkk.resize(_ntot);
    vector<double> k1, k2, k3;
    k1.resize(_Ns[0]); k2.resize(_Ns[1]); k3.resize(_Ns[2]); 
    for (int i=0;i<=_Ns[0]/2;i++){      k1[i] = i * 2*pi / _Ls[0];    }
    for (int i=_Ns[0]/2 + 1; i<_Ns[0]; i++){      k1[i] = (i - _Ns[0]) * 2*pi / _Ls[0];    }
    for (int i=0;i<=_Ns[1]/2;i++){      k2[i] = i * 2*pi / _Ls[1];    }
    for (int i=_Ns[1]/2 + 1; i<_Ns[1]; i++){      k2[i] = (i - _Ns[1]) * 2*pi / _Ls[1];    }
    for (int i=0;i<=_Ns[2]/2;i++){      k3[i] = i * 2*pi / _Ls[2];    }
    for (int i=_Ns[2]/2 + 1; i<_Ns[2]; i++){      k3[i] = (i - _Ns[2]) * 2*pi / _Ls[2];    }
    int cnt = 0;
    for (int k=0; k<_Ns[2]; k++)
      for (int j=0; j<_Ns[1]; j++)
	for (int i=0; i<_Ns[0]; i++){
	  _gkk[cnt++] = (k1[i]*k1[i] + k2[j]*k2[j] + k3[k]*k3[k])/2.;
	}
  }
  
  //-----------------------------
  //INIT _psi
  _psi.resize(_npsi*_ntot); 
  // for(vector<double>::iterator vi = _psi.begin(); vi != _psi.end();
  // vi++) (*vi) = dunirand();
 
  //-----------------------------
  //LL: Construct ik for derivative purpose
  {
    double pi = 4.0 * atan(1.0);
    
    vector<double> k1, k2, k3;
    k1.resize(_Ns[0]); k2.resize(_Ns[1]); k3.resize(_Ns[2]); 
    for (int i=0;i<=_Ns[0]/2;i++){      k1[i] = i * 2*pi / _Ls[0];    }
    for (int i=_Ns[0]/2 + 1; i<_Ns[0]; i++){      k1[i] = (i - _Ns[0]) * 2*pi / _Ls[0];    }
    for (int i=0;i<=_Ns[1]/2;i++){      k2[i] = i * 2*pi / _Ls[1];    }
    for (int i=_Ns[1]/2 + 1; i<_Ns[1]; i++){      k2[i] = (i - _Ns[1]) * 2*pi / _Ls[1];    }
    for (int i=0;i<=_Ns[2]/2;i++){      k3[i] = i * 2*pi / _Ls[2];    }
    for (int i=_Ns[2]/2 + 1; i<_Ns[2]; i++){      k3[i] = (i - _Ns[2]) * 2*pi / _Ls[2];    }
   
    _ik.resize(_ntot,3);
    int cnt = 0;
    for (int k=0; k<_Ns[2]; k++)
      for (int j=0; j<_Ns[1]; j++)
	for (int i=0; i<_Ns[0]; i++){
	  _ik(cnt,0) = cpx(0, k1[i]);
	  _ik(cnt,1) = cpx(0, k2[j]);
	  _ik(cnt,2) = cpx(0, k3[k]);
	  cnt++;
	}
  }
  
  return 0;
}

//--------------------------------------
int ScfPW::update()
{
  int myid = mpirank();
  
  //---------------------------------------------------------
  double pi = 4.0 * atan(1.0);
  int ndim = 3;
  //----------------------------------------------------
  int VL=0, DX=1, DY=2, DZ=3;
  int ntot = this->ntot();
  
  if(1){
    _rho0.clear();
    _rho0.resize(_ntot); //LL: resize DOESNOT assign the values if the size 
                         //does not change, VERY IMPORTANT


    //DIAG PSEUDOPOTENTIAL
    double* rho0 = &_rho0[0];
    _rho0s.resize(_atomvec.size());
    for(int a=0; a<_atomvec.size(); a++) {
      iC( _ptable.pseudoRho0(_atomvec[a], _Ls, _pos, _Ns, _rho0s[a]) );
      //accumulate
      IntNumVec& idx = _rho0s[a].first;
      DblNumMat& val = _rho0s[a].second;
      for(int k=0; k<idx.m(); k++)
	rho0[idx(k)] += val(VL,k);
    }
    double sumrho = 0; 
    for(int i = 0; i < ntot; i++){
      sumrho += rho0[i];
    }
    sumrho *= vol()/ntot;
    //
    fprintf(stderr, "sumrho %e %e:\n", sumrho, _nOccStates*2.0);
    //LEXING: MAKE IT EQUAL TO THE # of ELECTRONS BY SHIFTING
    double diff = (_nOccStates*2-sumrho)/(vol());
    for(int i=0; i<ntot; i++)    rho0[i] += diff;
  }
  
  if(1){
    //-----------------------------
    //NONLOCAL PSEUDOPOTENTIAL
    _vnlss.resize(_atomvec.size());
    int cnt = 0;
    for(int a=0; a<_atomvec.size(); a++) {
      iC( _ptable.pseudoNL(_atomvec[a], _Ls, _pos, _gridpos, _vnlss[a]) );
      cnt = cnt + _vnlss[a].size();
    }
    cerr<<"num of nl "<<cnt<<endl;
    for(int a=0; a<_vnlss.size(); a++) {
      for(int k=0; k<_vnlss[a].size(); k++) {
	SparseVec& now = _vnlss[a][k].first;
	IntNumVec& iv = now.first;
	DblNumMat& dv = now.second;
	double sum = 0;
	for(int a=0; a<dv.n(); a++)
	  sum += dv(VL,a)*dv(VL,a);
	sum *= vol()/ntot;
	cerr<<a<<" "<<k<<" energy "<<sum<<endl;
      }
    }  
  }  
  
  return 0;
}

//--------------------------------------
int ScfPW::scf(vector<double>& rhoinput, vector<double>& psiinput)
{
  /* MPI variables */
  int myid = mpirank();
  int nprocs = mpisize(); 
  int iterflag; //LY: =1 then exit
  
  /* potential */
  vector<double> vtotnew, vdif; //LY: 
  
  /* timing */
  double TimeSta(0), TimeEnd(0), t0(0), t1(0);
  
  /* counters */
  int i,j,k, iter;
  vector<double> dfmat, dvmat; //LY: used for anderson mixing
  int ierr;
  
  vtotnew.resize(_ntot);
  vdif.resize(_ntot);
  dfmat.resize(_ntot * _mixdim);
  dvmat.resize(_ntot * _mixdim);
  
  /* Initialize density, hartree, exchange and vtot */
  _rho = rhoinput;
  _psi = psiinput;
  /*
  if( _restartmode == string("from_scratch") )
    _rho = _rho0;
  if( _restartmode == string("restart") ){
    ifstream rhoid;
    Index3 tNs;
    Point3 Ls;
    DblNumVec rho(ntot(), false, &_rho[0]); 
    int ntot;
    rhoid.open(_restart_density.c_str(), ios::in);
    iA(rhoid.fail() == false);
    rhoid >> tNs >> Ls >> ntot;
    iA( (Ns(0) == tNs(0)) &&
	(Ns(1) == tNs(1)) && 
	(Ns(2) == Ns(2)) &&
	(this->ntot() == ntot) );
    rhoid >> rho;
    rhoid.close();
  }
  */
  //--------------------
  if( myid == MASTER ) {
    iC( scf_Print(fhstat) );
  }
  
  iC( scf_CalXC() );
  iC( scf_CalHartree() );
  iC( scf_CalVtot(&_vtot[0]) );
  
  //PREPARE vnl
  vector< pair<SparseVec,double> > tmpvnls;
  for(int a=0; a<_vnlss.size(); a++)
    tmpvnls.insert(tmpvnls.end(), _vnlss[a].begin(), _vnlss[a].end());
  
  //LEXING: no preparation, nonlocal pseudopot already done
  iterflag = 0;
  for (iter=1;iter<=_scfmaxiter;iter++) {
    TimeSta = time(0);   // LLIN (TimeSta, TimeEnd) records the wall clock 
                         // computational time for each SCF iteration.
    if( iterflag == 1 )  break;
    
    /* ===============================================================
       =     Performing each iteration
       =============================================================== */
    
    //LEXING: result back into _npsi, _ev, _vtot, _rho
    EigPW eigpw;
    eigpw._Ls = _Ls;
    eigpw._Ns = _Ns;
    eigpw._eigmaxiter = _eigmaxiter;
    eigpw._eigtol = _eigtol;
    //
    //--------------------
    iC( eigpw.setup() );
    //--------------------
    int nactive = _npsi;
    vector<int> active_indices;
    iC( eigpw.solve(_vtot, tmpvnls, _npsi, _psi, _ev, nactive, active_indices) ); //IMPORTANT: results back into _psi and _ev
    //LLIN: nactive and active_indices are just place holder here.
    
    //--------------------
    //BELOW FROM GlobalSolver
    vector<double>::iterator mi;
    /* Normalize the wavefunctions */
    for(int j=0; j < _npsi; j++){
      mi = _psi.begin() + j*_ntot;
      double sumval = 0.0;
      for(int i = 0; i < _ntot; i++){
	sumval += (*mi)*(*mi);
	mi++;
      }
      sumval = sqrt(sumval);
      mi = _psi.begin() + j*_ntot;
      for(int i = 0; i < _ntot; i++){
	(*mi) /= sumval;
	mi++;
      }
    }
    //
    //--------------------
    iC( scf_CalOcc(_Tbeta) );
    //--------------------
    iC( scf_CalCharge() );
    
    /* ===============================================================
       =     Post processing
       =============================================================== */
    if( myid == MASTER ) {
      t0 = time(0);   
      fprintf(fhstat, "Post processing (Hartree & XC)...\n");
      fprintf(stderr, "Post processing (Hartree & XC)...\n");
    }
    
    //--------------------
    iC( scf_CalXC() );
    iC( scf_CalHartree() );
    iC( scf_CalEnergy() );
    if(myid == MASTER) {
      iC( scf_PrintState(fhstat) );
    }
    //--------------------
    
    if( myid == MASTER ) {
      t1 = time(0);
      fprintf(fhstat, "Finish post processing (Hartree & XC). Time = %10.2f secs\n", difftime(t1,t0));
      fprintf(stderr, "Finish post processing (Hartree & XC). Time = %10.2f secs\n", difftime(t1,t0));
    }
    
    /* Update new total potential */
    double vtotsum = 0.0;
    iC( scf_CalVtot(&vtotnew[0]) );
    
    /* convergence check */
    
    double verr;
    for(int i = 0; i < _ntot; i++){
      vdif[i] = vtotnew[i] - _vtot[i];
    }
    verr = norm(&vdif[0], _ntot) / norm(&_vtot[0], _ntot);
    fprintf(fhstat, "SCF iter %4d:\n", iter);
    fprintf(fhstat, "norm(vout-vin) = %10.3e\n", verr);
    if( verr < _scftol ){
      /* converged */
      fprintf(fhstat, "SCF convergence reached!\n");
      iterflag = 1;
    }
    fflush(fhstat);
    
    /* Mixing and update potential */
    //WILL set _vtot in the MASTER PROCESSOR
    if( myid == MASTER ) {
      t0 = time(0);   
      fprintf(fhstat, "Start mixing...\n");
      fprintf(stderr, "Start mixing...\n");
    }
    if( _mixtype == "anderson" ){
      iC( scf_AndersonMix(vtotnew, dfmat, dvmat, iter) );
    }
    if( _mixtype == "kerker" ){
      iC( scf_KerkerMix(vtotnew, 0.8) );  // LL: Fixed alpha, Kerker must be at first
      iC( scf_AndersonMix(vtotnew, dfmat, dvmat, iter) );
    }
    if( myid == MASTER ) {
      t1 = time(0);
      fprintf(fhstat, "Finish mixing. Time = %10.2f secs\n", difftime(t1,t0));
      fprintf(stderr, "Finish mixing. Time = %10.2f secs\n", difftime(t1,t0));
    }
    TimeEnd = time(0);   
    if( myid == MASTER ){
      fprintf(stderr, "Finish iteration %4d. Time = %10.2f secs\n", 
	      iter, difftime(TimeEnd,TimeSta));
      fprintf(fhstat, "Finish iteration %4d. Time = %10.2f secs\n", 
	      iter, difftime(TimeEnd,TimeSta));
      fprintf(fhstat, "**************************************************************\n\n");
      fprintf(stderr, "**************************************************************\n\n");
      fflush(fhstat);
    }
    /* Check convergence */
    //MPI_Bcast( &iterflag, 1, MPI_INT, MASTER, MPI_COMM_WORLD );
  }
  
  //LEXING: EASY TO COMPARE
  if(myid == MASTER) {
    iC( scf_PrintState(stderr) );
  }
  
  return 0;
}

//--------------------------------------
int ScfPW::scf_CalOcc(double Tbeta)
{
  /* For a given finite temperature, update the occupation number */
  double tol = 1e-10;
  int maxiter = 100;
  int i, j, k;
  double lb, ub, flb, fub, occsum;
  int ilb, iub, iter;

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
int ScfPW::scf_CalCharge()
{
  int i, j;
  double sumval;
  vector<double>::iterator mi;
  for(i = 0; i < _ntot; i++)
    _rho[i] = 0.0;

  for(j=0; j < _npsi; j++){
    mi = _psi.begin() + j*_ntot;
    for(i = 0; i < _ntot; i++){
      _rho[i] += _occ[j] * (*mi)*(*mi); //LEXING: PSI CONTAIN SQRT OF WEIGHTS (THEY ARE NORMALIZED)
      mi++;
    }
  }
  sumval = 0.0;
  for(i = 0; i < _ntot; i++){
    _rho[i] *= 2.0*_ntot/_vol; //LEXING: TRUE DENSITY WITH SPIN
    sumval += _rho[i] * _vol/_ntot;
  }
  cerr << "Sum rho = " << sumval << endl;
  return 0;
}

//--------------------------------------
// Exchange-correlation function, currently support the LDA
// exchange-correlation potential used  by Goedecker-Teter-Hutter, and
// the usual form of LDA by Ceperley-Alder.
int ScfPW::scf_CalXC()
{
  // Exchange-correlation function by Goedecker-Teter-Hutter 
  if( _pseudotype == string("GTH") || _pseudotype == string("HGH") ) {
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
      if( rho > 0.0 ){
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

  }
  
  // Exchange-correlation function by Ceperley-Alder
  if( _pseudotype == string("TM") ){
    double g  =-0.2846,  b1 = 1.0529,
	   b2 = 0.3334, c1 = 0.0622,
	   c2 = 0.096,  c3 = 0.004,
	   c4 = 0.0232, c5 = 0.0192;
    double zero = 0.0, one = 1.0, two = 2.0, four = 4.0,  
	   nine = 9.0, third = 1.0/3.0; 
    double pi = 3.14159265;
    double a0 = pow((four/(nine*pi)),third);
    double twovpia0 = two/(pi*a0);
    double p75vpi = 0.75/pi;
    int ntot = this->ntot();

    double exc; 
    double rho, rs, sqrs, ec, alpha;
    double *vxc = &(_vxc[0]);

    exc = 0.0;

    for(int i = 0; i < ntot; i++){
      rho = _rho[i];
      vxc[i] = 0.0;
      if( rho > zero ){
	rs = pow((p75vpi/rho), third);
	vxc[i] = -twovpia0/rs;
	exc = exc + 0.75*rho*vxc[i];
	if (rs >= one){
	  sqrs = sqrt(rs);
	  ec = g/(one + b1*sqrs + b2*rs);
	  vxc[i] = vxc[i] + ec*ec*(one+3.5*b1*sqrs*third+four*b2*rs*third)/g;
	}
	else{
	  alpha = log(rs);
	  ec = c1*alpha - c2 + (c3*alpha - c4)*rs;
	  vxc[i] = vxc[i] + ec - (c1 + (c3*alpha - c5)*rs)*third;
	}
	exc = exc + rho*ec;
      }
    }

    exc *= (vol()) / (this->ntot());

    /* To avoid the burden later, vxc and exc are multiplied by 0.5 here
     * for the Rydberg->Hartree conversion */
    for(int i = 0; i < ntot; i++){
      vxc[i] *= 0.5;
    }
    _Exc = 0.5 * exc;
  }
  
  
  return 0;
  
}

int ScfPW::scf_CalHartree()
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
int ScfPW::scf_CalVtot(double* vtot)
{
  double vtotsum = 0.0;
  for(int i = 0; i < _ntot; i++){
    vtot[i] = _vext[i] + _vhart[i] + _vxc[i];
    vtotsum += vtot[i];
  }
  /*vtotsum /= (double)_ntot;
    for(int i = 0; i < _ntot; i++){
    vtot[i] = vtot[i] - vtotsum;
    }*/
  return 0;
}


//--------------------------------------
int ScfPW::scf_CalEnergy()
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
      _Efree += -2.0/_Tbeta*log(1.0+exp(-_Tbeta*(_ev[l]-_Fermi)));
  }
  _Efree += _Ecor + _Fermi * (_nOccStates * 2.0); 

  //-----------
  //debugging info
  _Evxcrho = 0;
  for(int i=0; i<_ntot; i++)    _Evxcrho += (_vxc[i]*_rho[i]);
  _Evxcrho *= _vol/double(_ntot);
  
  _Ehalfmm = 0;
  for(int i=0; i<_ntot; i++)    _Ehalfmm += 0.5*_vhart[i]*(_rho[i]-_rho0[i]);
  _Ehalfmm *= _vol/double(_ntot);
  
  _Ehalfmp = 0;
  for(int i=0; i<_ntot; i++)    _Ehalfmp += 0.5*_vhart[i]*(_rho[i]+_rho0[i]);
  _Ehalfmp *= _vol/double(_ntot);
  
  _Es = Es;
  return 0;
}



//--------------------------------------
int ScfPW::scf_Print(FILE *fh)
{
  Point3 coord;
  fprintf(fh, "**********************************************\n");
  fprintf(fh, "                  Molglobal\n");
  fprintf(fh, "**********************************************\n\n");
  fprintf(fh,"Super_Cell            = %6.2f %6.2f %6.2f\n", 	  _Ls[0], _Ls[1], _Ls[2] );
  fprintf(fh,"Grid_Size             = %6d %6d %6d\n", 	  _Ns[0], _Ns[1], _Ns[2] );
  fprintf(fh,"Total number of atoms = %6d\n", 	 _atomvec.size() );
  fprintf(fh, "\n");

  fprintf(fh, "Atom Type and Coordinates:\n");
  for(int i=0; i<_atomvec.size(); i++) {
    int type = _atomvec[i].type();
    Point3 coord = _atomvec[i].coord();
    fprintf(fh,"%d     %6.2f   %6.2f   %6.2f\n", 
	    type, coord(0), coord(1), coord(2) );
  }
  fprintf(fh, "\n");

  
  fprintf(fh,"Number of occupied states   = %10d\n", _nOccStates);
  fprintf(fh,"Number of extra states      = %10d\n", _nExtraStates);
  fprintf(fh,"Number of eigenvalues       = %10d\n", _npsi);
  fprintf(fh, "\n");
  
  fprintf(fh, "\n");
  fflush(fh);
  return 0;
}

//--------------------------------------
int ScfPW::scf_PrintState(FILE *fh)
{
  for(int i = 0; i < _npsi; i++){
    fprintf(fh, "eig[%5d] :  %25.15e,   occ[%5d] : %15.5f\n", 
	    i, _ev[i], i, _occ[i]); 
  }
  
  fprintf(fh, "Total Energy = %25.15e [Ry]\n", _Etot*2);
  fprintf(fh, "Helmholtz    = %25.15e [Ry]\n", _Efree*2);
  fprintf(fh, "Ekin         = %25.15e [Ry]\n", _Ekin*2);
  fprintf(fh, "Ecor         = %25.15e [Ry]\n", _Ecor*2);
  fprintf(fh, "Exc          = %25.15e [Ry]\n", _Exc*2);
  fprintf(fh, "Evxcrho      = %25.15e [Ry]\n", _Evxcrho*2);
  fprintf(fh, "Ehalfmm      = %25.15e [Ry]\n", _Ehalfmm*2);
  fprintf(fh, "Ehalfmp      = %25.15e [Ry]\n", _Ehalfmp*2);
  fprintf(fh, "Es           = %25.15e [Ry]\n", _Es*2);
  //fprintf(fh, "Ewald        = %25.15e [Ry]\n", _Ewald*2);
  //fprintf(fh, "EalphaT      = %25.15e [Ry]\n\n", _Ealphat*2);
  fflush(fh);
  return 0;
}

//--------------------------------------
int ScfPW::scf_AndersonMix(vector<double>& vtotnew,
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
  
  cerr << "Anderson mixing = " << iter << endl;
  cerr << "iterused = " << iterused << endl;
  cerr << "ipos = " << ipos << endl;
  
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
    
    LWORK =3*MIN(m,n) + MAX(MAX(2*MIN(m,n), MAX(m,n)), nrhs);
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
    dgelss_(&m, &n, &nrhs, &dftemp[0], &lda, &gammas[0], &ldb, 
	    &S[0], &rcond, &rank, &WORK[0], &LWORK,
	    &info);
    cerr << " rank of dfmat = " << rank << endl;
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
//Kerker mixing: default alpha=0.8
//Reference: KSSOLV/kerkmix.m
int ScfPW::scf_KerkerMix(vector<double>& vtotnew, double alpha)
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
//Auxilliary subroutine for the force calculation.  Calculates the inner
//product of all the four components (values and three derivatives) of a
//sparse vector with another full vector.
int ScfPW::force_innerprod(SparseVec& cur, double* ful, double wgt,
			   double* res)
{
  IntNumVec& iv = cur.first;
  DblNumMat& dv = cur.second;
  //clear
  for(int g=0; g<4; g++) {
    res[g] = 0;
    for(int k=0; k<iv.m(); k++)
      res[g] += dv(g,k) * ful[iv(k)];
    res[g] *= wgt; // (_vol/_ntot);
  }
  return 0;
}

//--------------------------------------
//Auxilliary subroutine for the force calculation.  Calculates the inner
//product of a sparse vector (no derivatives)  with another full vector.
int ScfPW::force_innerprod_val(SparseVec& cur, double* ful, double wgt,
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
int ScfPW::force()
{
  // Old version without forcing alignment
  if(0){
    int natoms = _atomvec.size();

    /*
    //calculate Vrho
    vector<cpx> crho(_ntot);
    for(int i = 0; i<_ntot; i++)    crho[i] = cpx(_rho[i]-_rho0[i]);
    vector<cpx> rhotemp(_ntot);
    fftw_execute_dft(_planpsiforward, 
    reinterpret_cast<fftw_complex*>(&crho[0]), 
    reinterpret_cast<fftw_complex*>(&rhotemp[0]));
    for(int i = 0; i < _ntot; i++){
    if(_gkk[i] != 0 )
    rhotemp[i] *= 2.0*M_PI/_gkk[i];
    else
    rhotemp[i] = cpx(0.0, 0.0);
    }
    fftw_execute_dft(_planpsibackward, 
    reinterpret_cast<fftw_complex*>(&rhotemp[0]), 
    reinterpret_cast<fftw_complex*>(&crho[0]));
    vector<double> vrho(_ntot);
    for(int i = 0; i < _ntot; i++ ){
    vrho[i] = crho[i].real()/(double)_ntot;
    }
    */

    vector<Point3> force(natoms, Point3(0,0,0));
    for(int a=0; a<natoms; a++) {
      int VL=0, DX=1, DY=2, DZ=3;
      //1.
      Point3 part1(0,0,0);
      //Method I: Calculate the force using the derivative on the
      //pseudo-charge
      if(0){
	double res[4];
	double wgt = _vol/double(_ntot);
	iC( force_innerprod(_rho0s[a], &(_vhart[0]), wgt, res) );
	part1(0) = -res[DX]; //LEXING: MINUS SIGN VERY IMPORTANT
	part1(1) = -res[DY];
	part1(2) = -res[DZ];
      }
      //Method II: Calculate the force using the derivative on the
      //electrostatic potential to improve accuracy.
      if(1){
	DblNumMat vhartdev(_ntot, 3); // three derivatives of vhart
	double res[3];
	double wgt = _vol/double(_ntot);
	// Calculate the gradient of vhart
	CpxNumVec cpxFTvhart(_ntot), cpxtmp1(_ntot), cpxtmp2(_ntot);
	for(int i = 0; i < _ntot; i++){
	  cpxtmp1[i] = cpx(_vhart[i], 0.0);
	}
	fftw_execute_dft(_planpsiforward, 
			 reinterpret_cast<fftw_complex*>(cpxtmp1.data()), 
			 reinterpret_cast<fftw_complex*>(cpxFTvhart.data()) );
	for(int g = 0; g < 3; g++){
	  cpx *vik = _ik.clmdata(g);
	  for(int i = 0; i < _ntot; i++){
	    cpxtmp1[i] = cpxFTvhart[i]*vik[i];
	  }
	  fftw_execute_dft(_planpsibackward, 
			   reinterpret_cast<fftw_complex*>(&cpxtmp1[0]), 
			   reinterpret_cast<fftw_complex*>(&cpxtmp2[0]) );
	  double *vtmp = vhartdev.clmdata(g);
	  for(int i = 0; i < _ntot; i++){
	    vtmp[i] = cpxtmp2[i].real()/(double)_ntot; //LL: VERY IMPORTANT
	  }
	}

	iC( force_innerprod_val(_rho0s[a], vhartdev.clmdata(0), wgt, res[0]) );
	iC( force_innerprod_val(_rho0s[a], vhartdev.clmdata(1), wgt, res[1]) );
	iC( force_innerprod_val(_rho0s[a], vhartdev.clmdata(2), wgt, res[2]) );
	//LL: NOTE There is no MINUS sign here due to integration by
	//parts.
	part1(0) = res[0];
	part1(1) = res[1];
	part1(2) = res[2];
      }
      //2.
      Point3 part2(0,0,0);
      vector< pair< SparseVec,double > >& vnls = _vnlss[a];
      for(int e=0; e<vnls.size(); e++) {
	SparseVec& be = vnls[e].first;
	double gamma = vnls[e].second;
	for(int i=0; i<_npsi; i++) {
	  double res[4];
	  double wgt = _vol/double(_ntot);
	  iC( force_innerprod(be, &(_psi[i*_ntot]), sqrt(wgt), res) ); //LEXING: IMPORTANT _psi already contains half of the weight
	  part2(0) += 4*_occ[i]*(gamma* res[VL] * res[DX]); //LEXING: COEFFICIENT IS 4 here due to spin degeneracy
	  part2(1) += 4*_occ[i]*(gamma* res[VL] * res[DY]); //LEXING: COEFFICIENT IS 4 here due to spin degeneracy
	  part2(2) += 4*_occ[i]*(gamma* res[VL] * res[DZ]); //LEXING: COEFFICIENT IS 4 here due to spin degeneracy
	}
      }
      //    //cerr<<"atom "<<a<<" "<<part1<<" "<<part2<<endl;
      force[a] = part1+part2;
    }

    for(int a=0; a<natoms; a++){
      Point3& fs = _atomvec[a].force();
      fs = force[a];
    }

    return 0;
  }

  // New version forcing alignment
  if(1){
    int natoms = _atomvec.size();

    /*
    //calculate Vrho
    vector<cpx> crho(_ntot);
    for(int i = 0; i<_ntot; i++)    crho[i] = cpx(_rho[i]-_rho0[i]);
    vector<cpx> rhotemp(_ntot);
    fftw_execute_dft(_planpsiforward, 
    reinterpret_cast<fftw_complex*>(&crho[0]), 
    reinterpret_cast<fftw_complex*>(&rhotemp[0]));
    for(int i = 0; i < _ntot; i++){
    if(_gkk[i] != 0 )
    rhotemp[i] *= 2.0*M_PI/_gkk[i];
    else
    rhotemp[i] = cpx(0.0, 0.0);
    }
    fftw_execute_dft(_planpsibackward, 
    reinterpret_cast<fftw_complex*>(&rhotemp[0]), 
    reinterpret_cast<fftw_complex*>(&crho[0]));
    vector<double> vrho(_ntot);
    for(int i = 0; i < _ntot; i++ ){
    vrho[i] = crho[i].real()/(double)_ntot;
    }
    */

    vector<Point3> force(natoms, Point3(0,0,0));
    for(int a=0; a<natoms; a++) {
      int VL=0, DX=1, DY=2, DZ=3;
      //1.
      Point3 part1(0,0,0);
      //Method I: Calculate the force using the derivative on the
      //pseudo-charge
      if(0){
	double res[4];
	double wgt = _vol/double(_ntot);
	iC( force_innerprod(_rho0s[a], &(_vhart[0]), wgt, res) );
	part1(0) = -res[DX]; //LEXING: MINUS SIGN VERY IMPORTANT
	part1(1) = -res[DY];
	part1(2) = -res[DZ];
      }
      //Method II: Calculate the force using the derivative on the
      //electrostatic potential to improve accuracy.
      if(1){
	DblNumMat vhartdev(_ntot, 3); // three derivatives of vhart
	double res[3];
	double wgt = _vol/double(_ntot);
	// Calculate the gradient of vhart
	fftw_complex *cpxFTvhart, *cpxtmp1, *cpxtmp2;
	cpxFTvhart = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*_ntot);
	cpxtmp1    = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*_ntot);
	cpxtmp2    = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*_ntot);

	for(int i = 0; i < _ntot; i++){
	  cpxtmp1[i][0] = _vhart[i];
	  cpxtmp1[i][1] = 0.0;
	}
	fftw_execute_dft(_planpsiforward, 
			 (&cpxtmp1[0]), 
			 (&cpxFTvhart[0]) );
	for(int g = 0; g < 3; g++){
	  cpx *vik = _ik.clmdata(g);
	  for(int i = 0; i < _ntot; i++){
	    cpxtmp1[i][0] = cpxFTvhart[i][0] * vik[i].real() - cpxFTvhart[i][1] * vik[i].imag();
	    cpxtmp1[i][1] = cpxFTvhart[i][0] * vik[i].imag() + cpxFTvhart[i][1] * vik[i].real();
	  }
	  fftw_execute_dft(_planpsibackward, 
			   (&cpxtmp1[0]), 
			   (&cpxtmp2[0]) );
	  double *vtmp = vhartdev.clmdata(g);
	  for(int i = 0; i < _ntot; i++){
	    vtmp[i] = cpxtmp2[i][0]/(double)_ntot; //LL: VERY IMPORTANT
	  }
	}

	fftw_free(cpxFTvhart);
	fftw_free(cpxtmp1);
	fftw_free(cpxtmp2);


	iC( force_innerprod_val(_rho0s[a], vhartdev.clmdata(0), wgt, res[0]) );
	iC( force_innerprod_val(_rho0s[a], vhartdev.clmdata(1), wgt, res[1]) );
	iC( force_innerprod_val(_rho0s[a], vhartdev.clmdata(2), wgt, res[2]) );
	//LL: NOTE There is no MINUS sign here due to integration by
	//parts.
	part1(0) = res[0];
	part1(1) = res[1];
	part1(2) = res[2];
      }
      //2.
      Point3 part2(0,0,0);
      vector< pair< SparseVec,double > >& vnls = _vnlss[a];
      for(int e=0; e<vnls.size(); e++) {
	SparseVec& be = vnls[e].first;
	double gamma = vnls[e].second;
	for(int i=0; i<_npsi; i++) {
	  double res[4];
	  double wgt = _vol/double(_ntot);
	  iC( force_innerprod(be, &(_psi[i*_ntot]), sqrt(wgt), res) ); //LEXING: IMPORTANT _psi already contains half of the weight
	  part2(0) += 4*_occ[i]*(gamma* res[VL] * res[DX]); //LEXING: COEFFICIENT IS 4 here due to spin degeneracy
	  part2(1) += 4*_occ[i]*(gamma* res[VL] * res[DY]); //LEXING: COEFFICIENT IS 4 here due to spin degeneracy
	  part2(2) += 4*_occ[i]*(gamma* res[VL] * res[DZ]); //LEXING: COEFFICIENT IS 4 here due to spin degeneracy
	}
      }
      //    //cerr<<"atom "<<a<<" "<<part1<<" "<<part2<<endl;
      force[a] = part1+part2;
    }

    for(int a=0; a<natoms; a++){
      Point3& fs = _atomvec[a].force();
      fs = force[a];
    }

    return 0;
  }

}
