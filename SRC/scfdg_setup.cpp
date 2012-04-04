#include "scfdg.hpp"
#include "eigdg.hpp"
#include "interp.hpp"
#include "eigpw.hpp"
#include "vecmatop.hpp"

extern FILE* fhstat;

//------------------------------------------------------------------------
//------------------------------------------------------------------------
//------------------------------------------------------------------------
//------------------------------------------------------------------------
//--------------------------------------
Buff::Buff()
{
  //------------------------
  _planpsibackward = NULL;
  _planpsiforward  = NULL;
}

//--------------------------------------
Buff::~Buff()
{
  if( _planpsibackward ){    fftw_destroy_plan(_planpsibackward);  }
  if( _planpsiforward ){    fftw_destroy_plan(_planpsiforward);  }
}

//--------------------------------------
int Buff::setup()
{
  _Ls = _dm.Ls();
  _Ns = _dm.Ns();
  _pos = _dm.pos();
  _vol = _Ls[0] * _Ls[1] * _Ls[2];
  _ntot = _Ns[0] * _Ns[1] * _Ns[2];
  //----------------------------------------
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

  return 0;
}

//------------------------------------------------------------------------
//------------------------------------------------------------------------
//------------------------------------------------------------------------
//------------------------------------------------------------------------
int Elem::setup()
{
  _Ls = _dm.Ls();
  _Ns = _dm.Ns();
  _pos = _dm.pos();
  _vol = _Ls[0] * _Ls[1] * _Ls[2];
  _ntot = _Ns[0] * _Ns[1] * _Ns[2];
  return 0;
}
//------------
int Elem::CalTransMatGlb(Domain g)
{
  _TransGlblx.resize(3);
  
  double pi = 4.0 * atan(1.0);
  double EPS = 1e-13;
  int ndim = 3;
  Index3 Ns1 = _Ns;
  Point3 Ls1 = _Ls;
  Point3 pos1 = _pos;
  Index3 Ns2 = g._Ns;
  Point3 Ls2 = g._Ls;
  Point3 pos2 = g._pos;
  double h2;

  /* Calculate Trans matrices for element to global interpolation */
  vector<double> lglmesh;
  vector<double> glbmesh;
  vector<double> denom;
  vector<double> lambda;

  for(int d = 0; d < ndim; d++){
    /* Calculate lgl mesh at element level*/
    h2 = Ls2[d] / double(Ns2[d]);  // Mesh size for the global level along direction d.
    _Nsglb[d] = iround( Ls1[d] / double(h2) );
    
    lglmesh.resize(Ns1[d]);
    lglnodes(lglmesh, Ns1[d]-1);
    for(int i = 0; i < Ns1[d]; i++){
      lglmesh[i] = pos1[d] - pos2[d] + (lglmesh[i]+1.0) * Ls1[d] * 0.5;
    }
    
    glbmesh.resize(_Nsglb[d]);
    for(int i = 0; i < _Nsglb[d]; i++){
      glbmesh[i] = g._pos[d] + (i + _posidx[d]) * h2;
    }
    /* Barycentric method for calculating Lagrange polynomials */
    _TransGlblx[d].resize(Ns1[d]*_Nsglb[d]);
    lambda.clear();
    denom.clear();
    lambda.resize(Ns1[d]);
    denom.resize(_Nsglb[d]);
    for(int i = 0; i < Ns1[d]; i++){
      lambda[i] = 1.0;
      for(int j = 0; j < Ns1[d]; j++){
	if( j != i ) lambda[i] *= (lglmesh[i] - lglmesh[j]);
      }
      lambda[i] = 1.0 / lambda[i];
      for(int j = 0; j < _Nsglb[d]; j++){
	denom[j] += lambda[i] / (glbmesh[j] - lglmesh[i] + EPS);
      }
    }
    for(int i = 0; i < Ns1[d]; i++){
      for(int j = 0; j < _Nsglb[d]; j++){
	_TransGlblx[d][j+i*_Nsglb[d]] = 
	  lambda[i] / (denom[j] * (glbmesh[j] - lglmesh[i] + EPS));
      }
    }
  }
  return 0;
}

//------------
int Elem::CalTransMatBuf(Domain b)
{
  _TransBufkl.resize(3);
  
  double pi = 4.0 * atan(1.0);
  int ndim = 3;
  Index3 Ns1 = _Ns;
  Point3 Ls1 = _Ls;
  Point3 pos1 = _pos;
  Index3 Ns2 = b._Ns;
  Point3 Ls2 = b._Ls;
  Point3 pos2 = b._pos;

  /* Calculate Trans matrices for Buffer to element kl-interpolation */
  vector<double> lglmesh;
  vector<double> kmesh;
  for(int d = 0; d < ndim; d++){
    _TransBufkl[d].resize(Ns1[d]*Ns2[d]);
    
    /* Calculate lgl mesh at element level*/
    lglmesh.resize(Ns1[d]);
    lglnodes(lglmesh, Ns1[d]-1);
    for(int i = 0; i < Ns1[d]; i++){
      lglmesh[i] = pos1[d] - pos2[d] + (lglmesh[i]+1.0) * Ls1[d] * 0.5;
    }

    /* Calculate k mesh at buffer level*/
    kmesh.resize(Ns2[d]);
    for (int i=0;i<=Ns2[d]/2;i++){
      kmesh[i] = i * 2*pi / Ls2[d];
    }
    for (int i=Ns2[d]/2 + 1; i<Ns2[d]; i++){
      kmesh[i] = (i - Ns2[d]) * 2*pi / Ls2[d];
    }

    /* Calculate transfer matrix */
    for(int j = 0; j < Ns2[d]; j++){
      for(int i = 0; i < Ns1[d]; i++){
	_TransBufkl[d][i+j*Ns1[d]] = cpx(cos(lglmesh[i]*kmesh[j]), 
					 sin(lglmesh[i]*kmesh[j]));
      }
    }
  }   
  return 0;
}



//--------------------------------------
ScfDG::ScfDG()
{
  //---------------------------------------------------------------
  _mixtypeset.insert("anderson"); //LY: USED
  _mixtypeset.insert("kerker");
  //---------------------------------------------------------------
  _planpsibackward = NULL;
  _planpsiforward  = NULL;
}

//--------------------------------------
ScfDG::~ScfDG()
{
  if( _planpsibackward ){    fftw_destroy_plan(_planpsibackward);  }
  if( _planpsiforward ){    fftw_destroy_plan(_planpsiforward);  }
}

//--------------------------------------
int ScfDG::setup()
{
  int myid = mpirank();
  int mpirank = this->mpirank();
  
  //---------------------------------------------------------
  //FIXME: LLIN: _nbufextra is set as a constant here.  Turns out to be
  //NOT useful for stablizing SCF.
  _nbufextra = 0;


  
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
  
  _rho.resize(_ntot);
  _rho0.resize(_ntot);
  
  _vtot.resize(_ntot);
  _vhart.resize(_ntot);
  _vxc.resize(_ntot);

  for(int i = 0; i < _ntot; i++){
    _rho[i]  = 0.0;
    _rho0[i] = 0.0;
    _vtot[i] = 0.0;
    _vhart[i] = 0.0;
    _vxc[i] = 0.0;
  }
  
  /* Ewald and Ealphat, CURRENTLY set to ZERO */
  //_Ewald = 0.0;
  //_Ealphat = 0.0;
  
  int nelec = 0;
  int indtyp;
  for(int i=0; i<_atomvec.size(); i++) {
    int type = _atomvec[i].type();
    Point3 coord = _atomvec[i].coord();
    nelec = nelec + _ptable.ptemap()[type].params()(PeriodTable::i_Zion);
  }
  if( nelec % 2 == 1 ){
    fprintf(stderr,"odd number of electrons is not supported in the current spin-unpolarized code!\n");
    exit(1);
  }
  _nOccStates = ceil((double)nelec/2.0);
  _npsi = _nOccStates + _nExtraStates;
  
  /* Initialize eigenvalues and wavefunctions */
  _ev.resize(_npsi);
  _occ.resize(_npsi);
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpirank==0) { fprintf(stderr, "scf setup prelim done\n"); }
  
  //-----------------------------
  _elemptn.ownerinfo() = _elemptninfo;
  
  //-----------------------------
  if(mpirank==0) { fprintf(stderr, "NElems %d %d %d\n",_NElems[0],_NElems[1],_NElems[2]); }
  _buffvec.resize(_NElems[0],_NElems[1],_NElems[2]);
  for(int k=0; k<_NElems[2]; k++)
    for(int j=0; j<_NElems[1]; j++)
      for(int i=0; i<_NElems[0]; i++) {
	if(_elemptn.owner(Index3(i,j,k))==myid) {
	  //-------------
	  Domain cbd = _bufftns(i,j,k); //current buffer domain
	  //
	  Buff& buff = _buffvec(i,j,k); //LEXING: VERY IMPORTANT
	  //
	  buff._dm = cbd;	  iC( buff.setup() );
	  //
	  vector<Atom> atomvec;
	  for(int a=0; a<_atomvec.size(); a++) {
	    int type = _atomvec[a].type();
	    Point3 coord = _atomvec[a].coord();
	    bool is_atm_in = CheckInterval(coord, cbd._pos, cbd._Ls, this->_Ls);
	    if( is_atm_in == true ){
	      for(int d=0; d<3; d++){		//LEXING: VERY IMPORTANT, UPDATE COORD
		coord[d] -= floor( (coord[d]-cbd._pos[d]) / _Ls[d] ) * _Ls[d];
	      }
	      atomvec.push_back( Atom(type, _atomvec[a].mass(), coord, 
				      _atomvec[a].vel(), _atomvec[a].force()) );
	    }
	  }
	  buff._atomvec = atomvec;
	  //
	  int ndim = 3;
	  double EPS = 1e-5;  /* Position difference beyond this will not be concerned */
	  Index3& Ns1 = cbd._Ns;
	  Point3& Ls1 = cbd._Ls;
	  Point3& pos1 = cbd._pos;
	  Index3& Ns2 = this->_Ns;
	  Point3& Ls2 = this->_Ls;
	  Point3& pos2 = this->_pos;
	  vector<DblNumVec> gridpos;	  gridpos.resize(ndim);
	  for(int d = 0; d < ndim; d++){
	    gridpos[d].resize(Ns1[d]);
	    double *posdata = gridpos[d].data();
	    double h1 = Ls1[d] / double(Ns1[d]);
	    for(int i = 0; i < Ns1[d]; i++){
	      posdata[i] = pos1[d] + i*h1;
	    }
	  }
	  buff._gridpos = gridpos;
	  //
	  Index3 posidx;
	  for(int d = 0; d < ndim; d++){
	    double h2 = Ls2[d] / Ns2[d];
	    double dtmp = (pos1[d] - pos2[d]) / h2;
	    if( fabs(dtmp - iround(dtmp)) > EPS ){
	      ABORT("The buffer starting position must be a grid point in the global level.  Reset Position_Start", 1);
	    }
	    posidx[d] = iround(dtmp);
	  }
	  buff._posidx = posidx;
	  //
	  int buffntot = cbd._Ns(0) * cbd._Ns(1) * cbd._Ns(2);
	  buff._vtot.resize(buffntot);
	  //
	  buff._npsi = _nenrich + _nbufextra;
	  buff._ev.resize(buff._npsi);
	  
	  // LLIN: setup the initial values of active indices
	  buff._nactive = buff._npsi;
	  buff._active_indices.resize(buff._nactive);
	  for(int i = 0; i < buff._nactive; i++){
	    buff._active_indices[i] = i;
	  }
	 
	
	  vector<double> psi;	  psi.resize(buff._npsi * buffntot);
	  for(vector<double>::iterator vi = psi.begin(); vi != psi.end(); vi++) {
	    (*vi) = dunirand();
	  }
	  buff._psi = psi;
	  // get its non-local pseudopotential for buffers
	  for(int a=0; a<buff._atomvec.size(); a++) {
	    vector< pair<SparseVec,double> > tmpvnls;
	    iC( _ptable.pseudoNL(buff._atomvec[a], buff._Ls, buff._pos, buff._gridpos, tmpvnls) );
	    buff._vnls.insert(buff._vnls.end(), tmpvnls.begin(), tmpvnls.end() ); //LEXING: CHECK
	  }
	  //---------------------------------------------------
	  //FINALLY WRITE IT IN
	  //LEXING: THE FOLLOWING LINE IS WRONG SINCE BUFF HAS FFTWPLANS IN IT
	  //_buffvec(i,j,k) = buff; 	  //_buffvec.lclmap()[Index3(i,j,k)] = buff;
	}
      }
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpirank==0) { fprintf(stderr, "scf setup buffvec done\n"); }
  
  //-----------------------------
  _elemvec.resize(_NElems[0],_NElems[1],_NElems[2]); //LEXING: elemvec is local now
  for(int k=0; k<_NElems[2]; k++)
    for(int j=0; j<_NElems[1]; j++)
      for(int i=0; i<_NElems[0]; i++) {
	if(_elemptn.owner(Index3(i,j,k))==myid) {
	  //-------------
	  Domain cbd = _elemtns(i,j,k); //current element domain
	  //
	  Elem& elem = _elemvec(i,j,k);
	  //
	  elem._dm = cbd;	  iC( elem.setup() );
	  //
	  //
	  int ndim = 3;
	  double EPS = 1e-5;  /* Position difference beyond this will not be concerned */
	  Index3& Ns1 = cbd._Ns;
	  Point3& Ls1 = cbd._Ls;
	  Point3& pos1 = cbd._pos;
	  Index3& Ns2 = this->_Ns;
	  Point3& Ls2 = this->_Ls;
	  Point3& pos2 = this->_pos;
	  vector<DblNumVec> gridpos;	  gridpos.resize(ndim);
	  for(int d = 0; d < ndim; d++){
	    gridpos[d].resize(Ns1[d]);
	    double *posdata = gridpos[d].data();
	    lglnodes(posdata, Ns1[d]-1);
	    for(int g=0; g<Ns1[d]; g++){
	      posdata[g] = pos1[d] + (posdata[g]+1.0)*Ls1[d]*0.5;
	    }
	  }
	  elem._gridpos = gridpos;
	  //
	  Index3 posidx, Nsglb;
	  for(int d = 0; d < ndim; d++){
	    double h2 = Ls2[d] / Ns2[d];
	    double dtmp = (pos1[d] - pos2[d]) / h2;
	    if( fabs(dtmp - iround(dtmp)) > EPS ){
	      ABORT("The element starting position must be a grid point in the global level.  Reset Position_Start", 1);
	    }
	    posidx[d] = iround(dtmp);
	    Nsglb[d] = iround( Ls1[d] / double(h2) );
	  }
	  elem._posidx = posidx;
	  elem._Nsglb = Nsglb;
	  //
	  //int elemntot = cbd._Ns(0) * cbd._Ns(1) * cbd._Ns(2);
	  //elem._vtot.resize(elemntot); //TO BE USED LATER
	  //
	  //vector< vector<double> > TransGlblx;
	  iC( elem.CalTransMatGlb(_dm) );	  //elem._TransGlblx = TransGlblx;
	  //fprintf(stderr, "scf setup elemvec calmat1 proc=%d\n", mpirank);
	  //
	  //vector< vector<cpx> >    TransBufkl; //LY: 3 times matrix element by buffer (k)
	  iC( elem.CalTransMatBuf(_bufftns(i,j,k)) );	  //elem._TransBufkl = TransBufkl;
	  //fprintf(stderr, "scf setup elemvec calmat2 proc=%d\n", mpirank);
	  //
	  //do nothing for the bases
	  //---------------------------------------------------
	  //FINALLY WRITE IT IN
	  //_elemvec.lclmap()[Index3(i,j,k)] = elem;
	  //fprintf(stderr, "scf setup elemvec addin proc=%d\n", mpirank);
	}
      }
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpirank==0) { fprintf(stderr, "scf setup elemvec done\n"); fflush(stderr); }
  
  //-----------------------------
  //LL: NOT initialized yet, when used they should be resized first!
  _vtotvec.prtn() = _elemptn;
  for(int k=0; k<_NElems[2]; k++)
    for(int j=0; j<_NElems[1]; j++)
      for(int i=0; i<_NElems[0]; i++) {
	if(_elemptn.owner(Index3(i,j,k))==myid) {
	  vector<double> dummy;
	  _vtotvec.lclmap()[Index3(i,j,k)] = dummy;
	}
      }
  //-----------------------------
  _basesvec.prtn() = _elemptn;
  for(int k=0; k<_NElems[2]; k++)
    for(int j=0; j<_NElems[1]; j++)
      for(int i=0; i<_NElems[0]; i++) {
	if(_elemptn.owner(Index3(i,j,k))==myid) {
	  vector<DblNumTns> dummy;
	  _basesvec.lclmap()[Index3(i,j,k)] = dummy;
	}
      }
  //-----------------------------
  _eigvecsvec.prtn() = _elemptn;
  for(int k=0; k<_NElems[2]; k++)
    for(int j=0; j<_NElems[1]; j++)
      for(int i=0; i<_NElems[0]; i++) {
	if(_elemptn.owner(Index3(i,j,k))==myid) {
	  DblNumMat dummy;
	  _eigvecsvec.lclmap()[Index3(i,j,k)] = dummy;
	}
      }
  //-----------------------------
  //PsdoPtn _psdoptn;
  vector<int> aux(_atomvec.size()); //owner of each atom
  for(int a=0; a<_atomvec.size(); a++) {
    int type = _atomvec[a].type();
    Point3 coord = _atomvec[a].coord();
    //find owner
    bool is_atm_in = false;
    Index3 ijk(-1,-1,-1);
    for(int k=0; k<_NElems[2]; k++)
      for(int j=0; j<_NElems[1]; j++)
	for(int i=0; i<_NElems[0]; i++) {
	  Domain cbd = _elemtns(i,j,k);
	  is_atm_in = CheckInterval(coord, cbd._pos, cbd._Ls, this->_Ls);
	  if(is_atm_in==true) {
	    ijk = Index3(i,j,k);
	    break;
	  }
	}
    iA(ijk(0)>=0 && ijk(1)>=0 && ijk(2)>=0);    //iA(is_atm_in==true);
    aux[a] = _elemptn.owner(ijk);
  }
  _psdoptn.ownerinfo() = aux;
  //-----------------------------
  _psdovec.prtn() = _psdoptn;
  //
  NumTns< vector<DblNumVec>  > gridpostns;
  gridpostns.resize(_NElems[0],_NElems[1],_NElems[2]);
  for(int k=0; k<_NElems[2]; k++)
    for(int j=0; j<_NElems[1]; j++)
      for(int i=0; i<_NElems[0]; i++) {
	//compute the gridpos
	Domain cbd = _elemtns(i,j,k);
	Index3& Ns1 = cbd._Ns;
	Point3& Ls1 = cbd._Ls;
	Point3& pos1 = cbd._pos;
	vector<DblNumVec> gridpos;	  gridpos.resize(ndim);
	for(int d = 0; d < ndim; d++){
	  gridpos[d].resize(Ns1[d]);
	  double *posdata = gridpos[d].data();
	  lglnodes(posdata, Ns1[d]-1);
	  for(int g=0; g<Ns1[d]; g++){
	    posdata[g] = pos1[d] + (posdata[g]+1.0) * Ls1[d] * 0.5;
	  }
	}
	gridpostns(i,j,k) = gridpos;
      }
  for(int a=0; a<_atomvec.size(); a++) {
    if(_psdoptn.owner(a)==myid) {      //(aux[a]== myid)
      Psdo psdo;
      //call rho0 evaluation
      iC( _ptable.pseudoRho0(_atomvec[a], _Ls, _pos, _Ns, psdo._rho0) );
      //call nonlocal pseudopot evaluation
      iC( _ptable.pseudoNL(_atomvec[a], _Ls, _pos, gridpostns, psdo._vnls) );
      _psdovec.lclmap()[a] = psdo;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpirank==0) { fprintf(stderr, "scf setup psdovec done\n"); }
  
  //--------------------------------------
  //form _rho0
  vector<double> rho0tmp(_ntot, 0.0);
  for(int a=0; a<_atomvec.size(); a++) {
    if(_psdoptn.owner(a)==myid) {
      SparseVec& rho0now = _psdovec.lclmap()[a].rho0();
      IntNumVec& iv = rho0now.first;
      DblNumMat& dv = rho0now.second;
      int VL=0, DX=1, DY=2, DZ=3;
      for(int k=0; k<iv.m(); k++) {
	rho0tmp[iv(k)] += dv(VL,k);
      }
    }
  }
  //SUM TOGETHER
  iC( MPI_Allreduce(&(rho0tmp[0]), &(_rho0[0]), _ntot, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );
  double sumrho = 0;
  for(int i = 0; i < _ntot; i++)    sumrho += _rho0[i];
  sumrho *= vol()/_ntot;
  //
  if(mpirank==0) { fprintf(stderr, "sumrho %e %e:\n", sumrho, _nOccStates*2.0); }
  double diff = (_nOccStates*2-sumrho)/(vol());
  for(int i=0; i<_ntot; i++)    _rho0[i] += diff;
  //
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpirank==0) { fprintf(stderr, "scf setup diagpseudo done\n"); }
  
  //--------------------------------------
  //PLANS
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
//LL: Update the information of the local pseudopotentials
//due to the change of atomic configurations
int ScfDG::update()
{
  int myid = mpirank();
  int mpirank = this->mpirank();
  
  _rho0.clear();
  _rho0.resize(_ntot);
  
  double pi = 4.0 * atan(1.0);
  int ndim = 3;
  
  for(int k=0; k<_NElems[2]; k++)
    for(int j=0; j<_NElems[1]; j++)
      for(int i=0; i<_NElems[0]; i++) {
	if(_elemptn.owner(Index3(i,j,k))==myid) {
	  //-------------
	  Domain cbd = _bufftns(i,j,k); //current buffer domain
	  //
	  Buff& buff = _buffvec(i,j,k); //LEXING: VERY IMPORTANT
	  // Update the atom localtions in the buffer
	  vector<Atom> atomvec;
	  atomvec.clear();
	  for(int a=0; a<_atomvec.size(); a++) {
	    int type = _atomvec[a].type();
	    Point3 coord = _atomvec[a].coord();
	    bool is_atm_in = CheckInterval(coord, cbd._pos, cbd._Ls, this->_Ls);
	    if( is_atm_in == true ){
	      for(int d=0; d<3; d++){		//LEXING: VERY IMPORTANT, UPDATE COORD
		coord[d] -= floor( (coord[d]-cbd._pos[d]) / _Ls[d] ) * _Ls[d];
	      }
	      atomvec.push_back( Atom(type, _atomvec[a].mass(), coord, 
				      _atomvec[a].vel(), _atomvec[a].force()) );
	    }
	  }
	  buff._atomvec = atomvec;
	  // get its non-local pseudopotential for buffers
	  buff._vnls.clear();   //LL: VERY IMPORTANT
	  for(int a=0; a<buff._atomvec.size(); a++) {
	    vector< pair<SparseVec,double> > tmpvnls;
	    iC( _ptable.pseudoNL(buff._atomvec[a], buff._Ls, buff._pos, buff._gridpos, tmpvnls) );
	    buff._vnls.insert(buff._vnls.end(), tmpvnls.begin(), tmpvnls.end() ); //LEXING: CHECK
	  }
	}
      }
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpirank==0) { fprintf(stderr, "scf setup buffvec done\n"); }
  
  //-----------------------------
  //PsdoPtn _psdoptn;
  //LL: NOTE: The pseudopotential structure 
  //should also be updated on-the-fly.
  vector<int> aux(_atomvec.size()); //owner of each atom
  for(int a=0; a<_atomvec.size(); a++) {
    int type = _atomvec[a].type();
    Point3 coord = _atomvec[a].coord();
    //find owner
    bool is_atm_in = false;
    Index3 ijk(-1,-1,-1);
    for(int k=0; k<_NElems[2]; k++)
      for(int j=0; j<_NElems[1]; j++)
	for(int i=0; i<_NElems[0]; i++) {
	  Domain cbd = _elemtns(i,j,k);
	  is_atm_in = CheckInterval(coord, cbd._pos, cbd._Ls, this->_Ls);
	  if(is_atm_in==true) {
	    ijk = Index3(i,j,k);
	    break;
	  }
	}
    iA(ijk(0)>=0 && ijk(1)>=0 && ijk(2)>=0);    //iA(is_atm_in==true);
    aux[a] = _elemptn.owner(ijk);
  }
  _psdoptn.ownerinfo() = aux;
  //-----------------------------
  _psdovec.prtn() = _psdoptn;
  //
  NumTns< vector<DblNumVec>  > gridpostns;
  gridpostns.resize(_NElems[0],_NElems[1],_NElems[2]);
  for(int k=0; k<_NElems[2]; k++)
    for(int j=0; j<_NElems[1]; j++)
      for(int i=0; i<_NElems[0]; i++) {
	//compute the gridpos
	Domain cbd = _elemtns(i,j,k);
	Index3& Ns1 = cbd._Ns;
	Point3& Ls1 = cbd._Ls;
	Point3& pos1 = cbd._pos;
	vector<DblNumVec> gridpos;	  gridpos.resize(ndim);
	for(int d = 0; d < ndim; d++){
	  gridpos[d].resize(Ns1[d]);
	  double *posdata = gridpos[d].data();
	  lglnodes(posdata, Ns1[d]-1);
	  for(int g=0; g<Ns1[d]; g++){
	    posdata[g] = pos1[d] + (posdata[g]+1.0) * Ls1[d] * 0.5;
	  }
	}
	gridpostns(i,j,k) = gridpos;
      }
  for(int a=0; a<_atomvec.size(); a++) {
    if(_psdoptn.owner(a)==myid) {      //(aux[a]== myid)
      Psdo psdo;
      //call rho0 evaluation
      iC( _ptable.pseudoRho0(_atomvec[a], _Ls, _pos, _Ns, psdo._rho0) );
      //call nonlocal pseudopot evaluation
      iC( _ptable.pseudoNL(_atomvec[a], _Ls, _pos, gridpostns, psdo._vnls) );
      _psdovec.lclmap()[a] = psdo;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpirank==0) { fprintf(stderr, "scf setup psdovec done\n"); }
  
  //--------------------------------------
  //form _rho0
  vector<double> rho0tmp(_ntot, 0.0);
  for(int a=0; a<_atomvec.size(); a++) {
    if(_psdoptn.owner(a)==myid) {
      SparseVec& rho0now = _psdovec.lclmap()[a].rho0();
      IntNumVec& iv = rho0now.first;
      DblNumMat& dv = rho0now.second;
      int VL=0, DX=1, DY=2, DZ=3;
      for(int k=0; k<iv.m(); k++) {
	rho0tmp[iv(k)] += dv(VL,k);
      }
    }
  }
  //SUM TOGETHER
  iC( MPI_Allreduce(&(rho0tmp[0]), &(_rho0[0]), _ntot, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );
  double sumrho = 0;
  for(int i = 0; i < _ntot; i++)    sumrho += _rho0[i];
  sumrho *= vol()/_ntot;
  //
  if(mpirank==0) { fprintf(stderr, "sumrho %e %e:\n", sumrho, _nOccStates*2.0); }
  double diff = (_nOccStates*2-sumrho)/(vol());
  for(int i=0; i<_ntot; i++)    _rho0[i] += diff;
  //
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpirank==0) { fprintf(stderr, "scf setup diagpseudo done\n"); }
  
  return 0;
}

//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
int serialize(const Psdo& val, ostream& os, const vector<int>& mask)
{
  int i = 0;
  if(mask[i]==1) serialize(val._rho0, os, mask);  i++;  //if(mask[i]==1) serialize(val._atomvec, os, mask);  i++;
  if(mask[i]==1) serialize(val._vnls, os, mask);  i++;
  iA(i==Psdo_Number);
  return 0;
}


int deserialize(Psdo& val, istream& is, const vector<int>& mask)
{
  int i = 0;
  if(mask[i]==1) deserialize(val._rho0, is, mask);  i++;  //if(mask[i]==1) deserialize(val._atomvec, is, mask);  i++;
  if(mask[i]==1) deserialize(val._vnls, is, mask);  i++;
  iA(i==Psdo_Number);
  return 0;
}

int combine(Psdo& val, Psdo& ext)
{
  iA(0);
  return 0;
}

/*
//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
int serialize(const Elem& val, ostream& os, const vector<int>& mask)
{
  int i = 0;
  if(mask[i]==1) serialize(val._dm, os, mask);  i++;  //if(mask[i]==1) serialize(val._atomvec, os, mask);  i++;
  if(mask[i]==1) serialize(val._gridpos, os, mask);  i++;
  if(mask[i]==1) serialize(val._posidx, os, mask);  i++;
  if(mask[i]==1) serialize(val._Nsglb, os, mask);  i++;
  if(mask[i]==1) serialize(val._vtot, os, mask);  i++;
  if(mask[i]==1) serialize(val._TransGlblx, os, mask);  i++;
  if(mask[i]==1) serialize(val._TransBufkl, os, mask);  i++;
  if(mask[i]==1) serialize(val._bases, os, mask);  i++;
  if(mask[i]==1) serialize(val._index, os, mask);  i++;
  if(mask[i]==1) serialize(val._eigvecs, os, mask);  i++;
  if(mask[i]==1) serialize(val._Ls, os, mask);  i++;
  if(mask[i]==1) serialize(val._Ns, os, mask);  i++;
  if(mask[i]==1) serialize(val._pos, os, mask);  i++;
  if(mask[i]==1) serialize(val._vol, os, mask);  i++;
  if(mask[i]==1) serialize(val._ntot, os, mask);  i++;
  iA(i==Elem_Number);
  return 0;
}


int deserialize(Elem& val, istream& is, const vector<int>& mask)
{
  int i = 0;
  if(mask[i]==1) deserialize(val._dm, is, mask);  i++;  //if(mask[i]==1) deserialize(val._atomvec, is, mask);  i++;
  if(mask[i]==1) deserialize(val._gridpos, is, mask);  i++;
  if(mask[i]==1) deserialize(val._posidx, is, mask);  i++;
  if(mask[i]==1) deserialize(val._Nsglb, is, mask);  i++;
  if(mask[i]==1) deserialize(val._vtot, is, mask);  i++;
  if(mask[i]==1) deserialize(val._TransGlblx, is, mask);  i++;
  if(mask[i]==1) deserialize(val._TransBufkl, is, mask);  i++;
  if(mask[i]==1) deserialize(val._bases, is, mask);  i++;
  if(mask[i]==1) deserialize(val._index, is, mask);  i++;
  if(mask[i]==1) deserialize(val._eigvecs, is, mask);  i++;
  if(mask[i]==1) deserialize(val._Ls, is, mask);  i++;
  if(mask[i]==1) deserialize(val._Ns, is, mask);  i++;
  if(mask[i]==1) deserialize(val._pos, is, mask);  i++;
  if(mask[i]==1) deserialize(val._vol, is, mask);  i++;
  if(mask[i]==1) deserialize(val._ntot, is, mask);  i++;
  iA(i==Elem_Number);
  return 0;
}

int combine(Elem& val, Elem& ext)
{
  iA(0);
  return 0;
}
*/
