#include "eigpw.hpp"

extern FILE* fhstat;
extern int   CONTXT;

//--------------------------------------
EigPW::EigPW()
{
  _planpsibackward = NULL;
  _planpsiforward  = NULL;
  _planpsic2r = NULL;
  _planpsir2c = NULL;
}

//--------------------------------------
EigPW::~EigPW()
{
  if( _planpsic2r ){
    fftw_destroy_plan(_planpsic2r);
  }
  if( _planpsir2c ){
    fftw_destroy_plan(_planpsir2c);
  }
  if( _planpsibackward ){
    fftw_destroy_plan(_planpsibackward);
  }
  if( _planpsiforward ){
    fftw_destroy_plan(_planpsiforward);
  }
}

//--------------------------------------
int EigPW::setup()
{
  //-------------
  _vol = _Ls[0] * _Ls[1] * _Ls[2];
  _ntot = _Ns[0] * _Ns[1] * _Ns[2];
  
  //-------------
  double pi = 4.0 * atan(1.0);
  int ndim = 3;
  
  //PLANS
  if(1){
    fftw_complex *psitemp1, *psitemp2;
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
    psitemp1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(_Ns[0]/2+1)*(_Ns[1])*(_Ns[2]));

    double* psitempreal;
    psitempreal = (double*)fftw_malloc(sizeof(double)*_ntot);


    _planpsic2r = 
      fftw_plan_dft_c2r_3d(_Ns[2], _Ns[1], _Ns[0], \
			   (&psitemp1[0]), \
			   (&psitempreal[0]), \
			   FFTW_MEASURE);

    _planpsir2c = 
      fftw_plan_dft_r2c_3d(_Ns[2], _Ns[1], _Ns[0],	\
			   (&psitempreal[0]), \
			   (&psitemp1[0]), \
			   FFTW_MEASURE);

    fftw_free(psitempreal);
    fftw_free(psitemp1);
    fftw_free(psitemp2);
  }
  
  // Old plan without guarantee of aligning
  if(0){
    vector<cpx> psitemp1, psitemp2;
    psitemp1.resize(_ntot);
    psitemp2.resize(_ntot);

    _planpsibackward = 
      fftw_plan_dft_3d(_Ns[2], _Ns[1], _Ns[0], \
		       reinterpret_cast<fftw_complex*>(&psitemp1[0]), \
		       reinterpret_cast<fftw_complex*>(&psitemp2[0]), \
		       FFTW_BACKWARD, FFTW_MEASURE);
    _planpsiforward = 
      fftw_plan_dft_3d(_Ns[2], _Ns[1], _Ns[0], \
		       reinterpret_cast<fftw_complex*>(&psitemp2[0]), \
		       reinterpret_cast<fftw_complex*>(&psitemp1[0]), \
		       FFTW_FORWARD, FFTW_MEASURE);
    
    vector<double> psitempreal;
    psitemp1.resize((_Ns[0]/2+1)*(_Ns[1])*(_Ns[2]));
    psitempreal.resize(_ntot);
    _planpsic2r = 
      fftw_plan_dft_c2r_3d(_Ns[2], _Ns[1], _Ns[0], \
			   reinterpret_cast<fftw_complex*>(&psitemp1[0]), \
			   (&psitempreal[0]), \
			   FFTW_MEASURE);

    _planpsir2c = 
      fftw_plan_dft_r2c_3d(_Ns[2], _Ns[1], _Ns[0],	\
			   (&psitempreal[0]), \
			   reinterpret_cast<fftw_complex*>(&psitemp1[0]), \
			   FFTW_MEASURE);


  }

  
  //MUL PRECOND
  {
    _gkk.resize(_ntot);
    vector<double> k1, k2, k3;
    k1.resize(_Ns[0]); k2.resize(_Ns[1]); k3.resize(_Ns[2]); 
    for (int i=0;i<=_Ns[0]/2;i++){
      k1[i] = i * 2*pi / _Ls[0];
    }
    for (int i=_Ns[0]/2 + 1; i<_Ns[0]; i++){
      k1[i] = (i - _Ns[0]) * 2*pi / _Ls[0];
    }
    for (int i=0;i<=_Ns[1]/2;i++){
      k2[i] = i * 2*pi / _Ls[1];
    }
    for (int i=_Ns[1]/2 + 1; i<_Ns[1]; i++){
      k2[i] = (i - _Ns[1]) * 2*pi / _Ls[1];
    }
    for (int i=0;i<=_Ns[2]/2;i++){
      k3[i] = i * 2*pi / _Ls[2];
    }
    for (int i=_Ns[2]/2 + 1; i<_Ns[2]; i++){
      k3[i] = (i - _Ns[2]) * 2*pi / _Ls[2];
    }

    int cnt = 0;
    for (int k=0; k<_Ns[2]; k++){
      for (int j=0; j<_Ns[1]; j++){
	for (int i=0; i<_Ns[0]; i++){
	  _gkk[cnt++] = (k1[i]*k1[i] + k2[j]*k2[j] + k3[k]*k3[k])/2.;
	}
      }
    }

    _gkkhalf.resize((_Ns[0]/2+1)*(_Ns[1])*(_Ns[2]));
    cnt = 0;
    for (int k=0; k<_Ns[2]; k++){
      for (int j=0; j<_Ns[1]; j++){
	for (int i=0; i<_Ns[0]/2+1; i++){
	  _gkkhalf[cnt++] = (k1[i]*k1[i] + k2[j]*k2[j] + k3[k]*k3[k])/2.;
	}
      }
    }
  }
  
  {
    _prec.resize(_ntot);
    double X, Y;
    for(int i = 0; i < _ntot; i++)
    {
      X = _gkk[i] * 2.0;
      Y  = 27.0 + X * (18.0 + X*(12.0 + 8.0*X));
      _prec[i] = Y/(Y + 16.0*pow(X,4.0));
    }
  }
  
  {
    int ntothalf = (_Ns[0]/2+1) * (_Ns[1]) * (_Ns[2]);
    _prechalf.resize(ntothalf);
    double X, Y;
    for(int i = 0; i < ntothalf; i++)
    {
      X = _gkkhalf[i] * 2.0;
      Y  = 27.0 + X * (18.0 + X*(12.0 + 8.0*X));
      _prechalf[i] = Y/(Y + 16.0*pow(X,4.0));
    }
  }
  
  return 0;
}

//--------------------------------------
int EigPW::solve(vector<double>& _vtot, vector< pair<SparseVec,double> >& _vnl,
		 int npsi, vector<double>& _psi, vector<double>& _ev, int& nactive, 
		 vector<int>& active_indices)
{
  _vtotptr = &_vtot;
  _vnlptr = &_vnl;
  
  //lobpcgsolvereal(molglb, pm);
  serial_Multi_Vector * x;
  serial_Multi_Vector * prec;
  mv_MultiVectorPtr xx;
  double * eigs;
  double * resid;
  int iterations;
  lobpcg_Tolerance lobpcg_tol;
  mv_InterfaceInterpreter ii;
  lobpcg_BLASLAPACKFunctions blap_fn;
  
  BlopexInt MV_HEIGHT, MV_WIDTH;

  /* ====== argv = L Aname Xname   */
  MV_HEIGHT = _ntot;
  MV_WIDTH = npsi; //LEXING
  
  /* Initialization */
  x = (serial_Multi_Vector*) malloc(sizeof(serial_Multi_Vector));
  x->data = reinterpret_cast<double*>(&_psi[0]);
  x->owns_data = 0;
  x->size = MV_HEIGHT;
  x->num_vectors = MV_WIDTH;

  // LLIN: Treat all indices as active indices at the beginning
  if(1){
    x->num_active_vectors = MV_WIDTH;
    x->active_indices = (BlopexInt*) malloc(sizeof(BlopexInt)*x->num_active_vectors);
    for(int i = 0; i < x->num_active_vectors; i++)
      x->active_indices[i] = i;
  }
  

  // Use the active indices from the residue last time. LLIN: DOES NOT
  // WORK since lobpcg.c defines activeMask to be 1 at the beginning of
  // lobpcg_solve.  Currently it does not take any active_indices as
  // input vector.  
  if(0){
    x->num_active_vectors = nactive;
    x->active_indices = (BlopexInt*) malloc(sizeof(BlopexInt)*x->num_active_vectors);
    for(int i = 0; i < x->num_active_vectors; i++)
      x->active_indices[i] = active_indices[i];
    
    if(1){
      fprintf(fhstat, "Initial number of active indices = %5d\n", nactive);
      for(int i = 0; i < nactive; i++){
	fprintf(fhstat, "active_indices[%5d] = %5d\n", i, active_indices[i]);
      }
    }
  }
//  
  
  /* get memory for eigenvalues, eigenvalue history, residual norms, residual norms history */
  eigs = (double *)malloc(sizeof(double)*MV_WIDTH);
  resid = (double *)malloc(sizeof(double)*MV_WIDTH);
  
  /* LLIN: set tolerances, use absolute tolerance */
  lobpcg_tol.absolute = _eigtol;
  lobpcg_tol.relative = 1e-16;

  /* setup interface interpreter and wrap around "x" another structure */
  SerialSetupInterpreter( &ii );
  xx = mv_MultiVectorWrap( &ii, x, 0);

  /* set pointers to lapack functions */
  blap_fn.dpotrf = dpotrf_;
  blap_fn.dsygv  = dsygv_;

  /* execute lobpcg to solve for Ax=(lambda)Bx
     with initial guess of e-vectors xx
     and preconditioner T
     number of vectors in xx determines number of e-values eig to solve for
     solving for the smallest e-values under constraints Y
     execution stops when e-values with tolerances or after max iterations */

  fprintf(fhstat, "Call lobpcg solve double \n");

  clock_t start,end;
  double cpu_time_used;
  
  start = clock();
  
  lobpcg_solve_double( xx,           /*input-initial guess of e-vectors */
		       this,  //  (void *) operatorA,      /*input-matrix A */
		       solve_MatMultiVecWrapper,      /*input-operator A */
		       NULL,             /*input-matrix B */
		       NULL,             /*input-operator B */
		       this,             /*input-matrix T */
		       solve_ApplyPrecWrapper,        /*input-operator T */
		       NULL,             /*input-matrix Y */
		       blap_fn,      /*input-lapack functions */
		       lobpcg_tol,   /*input-tolerances */
		       _eigmaxiter,        /*input-max iterations */
		       1,            /*input-verbosity level */
		       
		       &iterations,  /*output-actual iterations */
		       eigs,         /*output-eigenvalues */
		       NULL,             /*output-eigenvalues history */
		       0,            /*output-history global height */
		       resid,        /*output-residual norms */
		       NULL,            /*output-residual norms history */
		       0             /*output-history global height  */
		     );
  
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  
  /* print eigenvector */
  /*
    serial_Multi_VectorPrint(x,"eigenvectors",0);
  */
  for(int i = 0; i < MV_WIDTH; i++){
    _ev[i] = eigs[i];
    fprintf(fhstat, "Eig[%5d] = %25.15e,  Res[%5d] = %25.15e\n", i, eigs[i], i, resid[i]);
  } 

  // Use the active indices from the residue last time.
  if(1){
    active_indices.clear();
    nactive = 0;
    for(int i = 0; i < MV_WIDTH; i++){
      if(resid[i] >= _eigtol){
	active_indices.push_back(i);
	nactive++;
      }
    }
    fprintf(fhstat, "\n Number of active indices remained = %5d \n", nactive);
  }

  fprintf(fhstat, "\n LOBPCG CPU time used = %22.16e \n\n",cpu_time_used);
 

  /* destroy multivector and other objects */
  serial_Multi_VectorDestroy(x);
  mv_MultiVectorDestroy(xx);
  free(eigs);
  free(resid);
  
  _vtotptr = NULL;
  _vnlptr = NULL;
  
  return 0;
}

//--------------------------------------
void EigPW::solve_MatMultiVecWrapper(void * A, void * X, void * AX)
{
  iC( ((EigPW*)A)->solve_MatMultiVec((serial_Multi_Vector*) X, (serial_Multi_Vector *) AX) );
  return;
}

//--------------------------------------
void EigPW::solve_ApplyPrecWrapper(  void * A, void * X, void * AX) {
  iC( ((EigPW*)A)->solve_ApplyPrec(  (serial_Multi_Vector*) X, (serial_Multi_Vector *) AX) );
  return;
}

//--------------------------------------
BlopexInt EigPW::solve_MatMultiVec(serial_Multi_Vector* x, serial_Multi_Vector* y)
{
  // Old version without forcing 16 byte alignment
  if(0){
    //LEXING:
    vector<double>& _vtot = *(_vtotptr);
    vector< pair<SparseVec,double> >& _vnl = *(_vnlptr);
    int nvnl = _vnl.size();
    //
    double *x_data;
    double *y_data;
    double * psiptr;
    double * Hpsiptr;
    BlopexInt * x_active_ind;
    BlopexInt * y_active_ind;
    BlopexInt i, j;
    BlopexInt size;
    BlopexInt num_active_vectors;
    double h;
    int ntot = this->_ntot;
    int INTONE = 1;

    assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);
    assert (x->size>1);
    x_data = x->data;
    y_data = y->data;
    size = x->size;
    num_active_vectors = x->num_active_vectors;
    x_active_ind = x->active_indices;
    y_active_ind = y->active_indices;

    double pi = 4.0 * atan(1.0);
    vector<cpx> psitemp;
    vector<cpx> psicpx;
    vector<cpx> Hpsicpx;
    vector<double> tmpvec;

    psitemp.resize(ntot);
    psicpx.resize(ntot);
    Hpsicpx.resize(ntot);
    tmpvec.resize(ntot);

    int ntothalf = (this->_Ns[0]/2+1) * (this->_Ns[1]) * (this->_Ns[2]);

    double weight;
    //vector<SparseVec>::iterator vnliter;
    vector< pair<SparseVec,double> >::iterator vnliter;
    for(i=0; i<num_active_vectors; i++)
    {
      psiptr = x_data + x_active_ind[i]*size;
      Hpsiptr = y_data + y_active_ind[i]*size;

      // Hpsi = H * psi
      for(int k = 0; k < ntot; k++){
	Hpsiptr[k] = 0.0;
      }

      // Apply vtot
      for(int k = 0; k < ntot; k++){
	Hpsiptr[k] += psiptr[k] * _vtot[k];
      }

      // Nonlocal pseudopotential
      if(1){
	/*LEXING REWRITTEN
	  for (int l=0; l<nvnl; l++) {
	  weight = 0;
	  vnliter = _vnl.begin() + l;
	  SparseVec& vnlvec = (*vnliter).first;
	  double vnlsgn = (*vnliter).second;
	  int numval = vnlvec.first.m();
	  for(int k = 0; k < numval; k++){
	  tmpvec[k] = psiptr[vnlvec.first[k] ];
	  }
	  weight = ddot_(&numval, &tmpvec[0], &INTONE, &vnlvec.second[0], &INTONE);
	  weight *= this->_vol / double(this->_ntot) * vnlsgn;

	  for(int k = 0; k < numval; k++){
	  Hpsiptr[vnlvec.first[k] ] += vnlvec.second[k] * weight;
	  }
	  }
	  */
	int VL = 0;
	for(int l=0; l<nvnl; l++) {
	  SparseVec& vnlvec = _vnl[l].first;
	  double vnlsgn = _vnl[l].second;
	  IntNumVec& iv = vnlvec.first;
	  DblNumMat& dv = vnlvec.second;
	  double weight = 0;

	  /*
	     for(int k=0; k<iv.m(); k++) {
	     weight+= dv(VL,k)*psiptr[iv(k)];
	     }
	     */
	  int dvm = dv.m();
	  int* ivptr = iv.data();
	  double* dvptr = dv.data();
	  for(int k=0; k<iv.m(); k++) {
	    weight += (*dvptr) * psiptr[*ivptr];
	    ivptr ++;	  dvptr += dvm;
	  }

	  weight *= _vol/double(_ntot) * vnlsgn;

	  /*
	     for(int k=0; k<iv.m(); k++) {
	     Hpsiptr[iv[k]] += dv(VL,k) * weight;
	     }
	     */
	  ivptr = iv.data();
	  dvptr = dv.data();
	  for(int k=0; k<iv.m(); k++) {
	    Hpsiptr[*ivptr] += (*dvptr) * weight;
	    ivptr ++;	  dvptr += dvm;
	  }
	}
      }

      // Complex version of applying Laplacian
      /*
	 if(0){
	 for(int k = 0; k < ntot; k++)
	 psitemp[k] = cpx(psiptr[k], 0.0);

	 fftw_execute_dft(this->_planpsiforward, 
	 reinterpret_cast<fftw_complex*>(&psitemp[0]), 
	 reinterpret_cast<fftw_complex*>(&psicpx[0]));

	 for(int k = 0; k < ntot; k++)
	 Hpsicpx[k] = psicpx[k] * this->_gkk[k];

	 fftw_execute_dft(this->_planpsibackward, 
	 reinterpret_cast<fftw_complex*>(&Hpsicpx[0]), 
	 reinterpret_cast<fftw_complex*>(&psitemp[0]) );

	 for(int k = 0; k < ntot; k++)
	 Hpsiptr[k] += psitemp[k].real() / ntot;
	 }
	 */

      // real version of applying Laplacian
      if(1){

	fftw_execute_dft_r2c(this->_planpsir2c, 
			     (&psiptr[0]), 
			     reinterpret_cast<fftw_complex*>(&psicpx[0]));

	/* Apply Laplacian */

	for(int k = 0; k < ntothalf; k++)
	  Hpsicpx[k] = psicpx[k] * this->_gkkhalf[k];

	fftw_execute_dft_c2r(this->_planpsic2r, 
			     reinterpret_cast<fftw_complex*>(&Hpsicpx[0]), 
			     (&tmpvec[0]) );

	for(int k = 0; k < ntot; k++)
	  Hpsiptr[k] += tmpvec[k] / ntot;
      }
    }

    return 0;
  }

  // New version forcing 16 byte alignment
  if(1){
    //LEXING:
    vector<double>& _vtot = *(_vtotptr);
    vector< pair<SparseVec,double> >& _vnl = *(_vnlptr);
    int nvnl = _vnl.size();
    //
    double *x_data;
    double *y_data;
    double * psiptr;
    double * Hpsiptr;
    BlopexInt * x_active_ind;
    BlopexInt * y_active_ind;
    BlopexInt i, j;
    BlopexInt size;
    BlopexInt num_active_vectors;
    double h;
    int ntot = this->_ntot;
    int INTONE = 1;

    assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);
    assert (x->size>1);
    x_data = x->data;
    y_data = y->data;
    size = x->size;
    num_active_vectors = x->num_active_vectors;
    x_active_ind = x->active_indices;
    y_active_ind = y->active_indices;

    double pi = 4.0 * atan(1.0);
    double* psitemp;
    fftw_complex *psicpx, *Hpsicpx;

    psitemp = (double*)fftw_malloc(sizeof(double)*size);
    psicpx  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*size);
    Hpsicpx = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*size);

    int ntothalf = (this->_Ns[0]/2+1) * (this->_Ns[1]) * (this->_Ns[2]);

    double weight;
    //vector<SparseVec>::iterator vnliter;
    vector< pair<SparseVec,double> >::iterator vnliter;
    for(i=0; i<num_active_vectors; i++)
    {
      psiptr = x_data + x_active_ind[i]*size;
      Hpsiptr = y_data + y_active_ind[i]*size;

      // Hpsi = H * psi
      for(int k = 0; k < ntot; k++){
	Hpsiptr[k] = 0.0;
      }

      // Apply vtot
      for(int k = 0; k < ntot; k++){
	Hpsiptr[k] += psiptr[k] * _vtot[k];
      }

      // Nonlocal pseudopotential
      if(1){
	int VL = 0;
	for(int l=0; l<nvnl; l++) {
	  SparseVec& vnlvec = _vnl[l].first;
	  double vnlsgn = _vnl[l].second;
	  IntNumVec& iv = vnlvec.first;
	  DblNumMat& dv = vnlvec.second;
	  double weight = 0;

	  /*
	     for(int k=0; k<iv.m(); k++) {
	     weight+= dv(VL,k)*psiptr[iv(k)];
	     }
	     */
	  int dvm = dv.m();
	  int* ivptr = iv.data();
	  double* dvptr = dv.data();
	  for(int k=0; k<iv.m(); k++) {
	    weight += (*dvptr) * psiptr[*ivptr];
	    ivptr ++;	  dvptr += dvm;
	  }

	  weight *= _vol/double(_ntot) * vnlsgn;

	  /*
	     for(int k=0; k<iv.m(); k++) {
	     Hpsiptr[iv[k]] += dv(VL,k) * weight;
	     }
	     */
	  ivptr = iv.data();
	  dvptr = dv.data();
	  for(int k=0; k<iv.m(); k++) {
	    Hpsiptr[*ivptr] += (*dvptr) * weight;
	    ivptr ++;	  dvptr += dvm;
	  }
	}
      }

      // real version of applying Laplacian
      if(1){

	for(int k = 0; k < ntot; k++){
	  psitemp[k] = psiptr[k];
	}

	fftw_execute_dft_r2c(this->_planpsir2c, 
			     (&psitemp[0]), 
			     (&psicpx[0]));

	/* Apply Laplacian */

	for(int k = 0; k < ntothalf; k++){
	  Hpsicpx[k][0] = psicpx[k][0] * this->_gkkhalf[k];
	  Hpsicpx[k][1] = psicpx[k][1] * this->_gkkhalf[k];
	}

	fftw_execute_dft_c2r(this->_planpsic2r, 
			     (&Hpsicpx[0]), 
			     (&psitemp[0]) );

	for(int k = 0; k < ntot; k++)
	  Hpsiptr[k] += psitemp[k] / ntot;
      }
    }

    fftw_free(psitemp);
    fftw_free(psicpx);
    fftw_free(Hpsicpx);
    return 0;
  }

}

//--------------------------------------
BlopexInt EigPW::solve_ApplyPrec(  serial_Multi_Vector* x, serial_Multi_Vector * y)
{
  // Old version without forcing 16 byte alignment
  if(0){
    double *x_data;
    double *y_data;
    double * prec_data;
    double * psiptr;
    double * precpsiptr;

    BlopexInt * x_active_ind;
    BlopexInt * y_active_ind;
    BlopexInt i, j;
    BlopexInt size;
    BlopexInt num_active_vectors;
    double h;

    assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);
    assert (x->size>1);
    x_data = x->data;
    y_data = y->data;
    prec_data = &(this->_prec[0]);

    size = x->size;
    num_active_vectors = x->num_active_vectors;
    x_active_ind = x->active_indices;
    y_active_ind = y->active_indices;

    vector<cpx> psitemp, psicpx, precpsicpx;
    psitemp.resize(size);
    psicpx.resize(size);
    precpsicpx.resize(size);
    vector<double> tmpvec;
    tmpvec.resize(size);
    int ntothalf = (this->_Ns[0]/2+1) * (this->_Ns[1]) * (this->_Ns[2]);

    for(i=0; i<num_active_vectors; i++)
    {
      psiptr = x_data + x_active_ind[i]*size;
      precpsiptr = y_data + y_active_ind[i]*size;

      // Complex version of applying preconditioner
      if(0){
	for(int k = 0; k < size; k++)
	  psitemp[k] = cpx(psiptr[k], 0.0);

	fftw_execute_dft(this->_planpsiforward, 
			 reinterpret_cast<fftw_complex*>(&psitemp[0]), 
			 reinterpret_cast<fftw_complex*>(&psicpx[0]));

	for(int k = 0; k < size; k++){
	  precpsicpx[k] = prec_data[k] * psicpx[k];
	}

	fftw_execute_dft(this->_planpsibackward, 
			 reinterpret_cast<fftw_complex*>(&precpsicpx[0]), 
			 reinterpret_cast<fftw_complex*>(&psitemp[0]) );

	for(int k = 0; k < size; k++)
	  precpsiptr[k] = psitemp[k].real() / double(size);
      }

      // Real version of applying preconditioner
      if(1){

	fftw_execute_dft_r2c(this->_planpsir2c, 
			     (&psiptr[0]), 
			     reinterpret_cast<fftw_complex*>(&psicpx[0]));

	for(int k = 0; k < ntothalf; k++){
	  precpsicpx[k] = this->_prechalf[k] * psicpx[k];
	}

	fftw_execute_dft_c2r(this->_planpsic2r, 
			     reinterpret_cast<fftw_complex*>(&precpsicpx[0]),
			     (&tmpvec[0]));

	for(int k = 0; k < size; k++)
	  precpsiptr[k] = tmpvec[k] / double(size);
      }
    }
    return 0;
  }

  // New version forcing 16 byte alignment
  if(1){
    double *x_data;
    double *y_data;
    double * prec_data;
    double * psiptr;
    double * precpsiptr;

    BlopexInt * x_active_ind;
    BlopexInt * y_active_ind;
    BlopexInt i, j;
    BlopexInt size;
    BlopexInt num_active_vectors;
    double h;

    assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);
    assert (x->size>1);
    x_data = x->data;
    y_data = y->data;
    prec_data = &(this->_prec[0]);

    size = x->size;
    num_active_vectors = x->num_active_vectors;
    x_active_ind = x->active_indices;
    y_active_ind = y->active_indices;

    double* psitemp; 
    fftw_complex* psicpx, *precpsicpx;
    psitemp = (double*)fftw_malloc(sizeof(double)*size);
    psicpx  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*size);
    precpsicpx = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*size);

    int ntothalf = (this->_Ns[0]/2+1) * (this->_Ns[1]) * (this->_Ns[2]);

    for(i=0; i<num_active_vectors; i++)
    {
      psiptr = x_data + x_active_ind[i]*size;
      precpsiptr = y_data + y_active_ind[i]*size;

      // Real version of applying preconditioner
      if(1){

	for(int k = 0; k < size; k++){
	  psitemp[k] = psiptr[k];
	}

	fftw_execute_dft_r2c(this->_planpsir2c, 
			     (&psitemp[0]), 
			     (&psicpx[0]));

	for(int k = 0; k < ntothalf; k++){
	  precpsicpx[k][0] = this->_prechalf[k] * psicpx[k][0];
	  precpsicpx[k][1] = this->_prechalf[k] * psicpx[k][1];
	}

	fftw_execute_dft_c2r(this->_planpsic2r, 
			     (&precpsicpx[0]),
			     (&psitemp[0]));

	for(int k = 0; k < size; k++)
	  precpsiptr[k] = psitemp[k] / double(size);
      }
    }
    
    fftw_free(psitemp); 
    fftw_free(psicpx);
    fftw_free(precpsicpx);

    return 0;
  }

}



