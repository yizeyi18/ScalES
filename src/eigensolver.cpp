/*
	 Copyright (c) 2012 The Regents of the University of California,
	 through Lawrence Berkeley National Laboratory.  

   Author: Lin Lin
	 
   This file is part of DGDFT. All rights reserved.

	 Redistribution and use in source and binary forms, with or without
	 modification, are permitted provided that the following conditions are met:

	 (1) Redistributions of source code must retain the above copyright notice, this
	 list of conditions and the following disclaimer.
	 (2) Redistributions in binary form must reproduce the above copyright notice,
	 this list of conditions and the following disclaimer in the documentation
	 and/or other materials provided with the distribution.
	 (3) Neither the name of the University of California, Lawrence Berkeley
	 National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
	 be used to endorse or promote products derived from this software without
	 specific prior written permission.

	 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
	 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
	 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	 You are under no obligation whatsoever to provide any bug fixes, patches, or
	 upgrades to the features, functionality or performance of the source code
	 ("Enhancements") to anyone; however, if you choose to make your Enhancements
	 available either publicly, or directly to Lawrence Berkeley National
	 Laboratory, without imposing a separate written license agreement for such
	 Enhancements, then you hereby grant the following license: a non-exclusive,
	 royalty-free perpetual license to install, use, modify, prepare derivative
	 works, incorporate into other computer software, distribute, and sublicense
	 such enhancements or derivative works thereof, in binary and source code form.
*/
/// @file eigensolver.cpp
/// @brief Eigensolver in the global domain or extended element.
/// @date 2014-04-25 Paralle version
#include	"eigensolver.hpp"
#include  "utility.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"

#define _DEBUGlevel_ 0

namespace dgdft{

extern "C"{

  void Cblacs_get(const Int contxt, const Int what, Int* val);

  void Cblacs_gridinit(Int* contxt, const char* order, const Int nprow, const Int npcol);

  void Cblacs_gridmap(Int* contxt, Int* pmap, const Int ldpmap, const Int nprow, const Int npcol);

  void Cblacs_gridinfo(const Int contxt,  Int* nprow, Int* npcol, 
      Int* myprow, Int* mypcol);

  void Cblacs_gridexit	(	int contxt );	

  void SCALAPACK(descinit)(Int* desc, const Int* m, const Int * n, const Int* mb,
      const Int* nb, const Int* irsrc, const Int* icsrc,
      const Int* contxt, const Int* lld, Int* info);

  void SCALAPACK(pdsyev)(const char *jobz, const char *uplo, const Int *n, double *a, 
      const Int *ia, const Int *ja, const Int *desca, double *w, 
      double *z, const Int *iz, const Int *jz, const Int *descz, 
      double *work, const Int *lwork, Int *info);

  void SCALAPACK(pdsyevd)(const char *jobz, const char *uplo, const Int *n, double *a, 
      const Int *ia, const Int *ja, const Int *desca, double *w, 
      const double *z, const Int *iz, const Int *jz, const Int *descz, 
      double *work, const Int *lwork, Int* iwork, const Int* liwork, 
      Int *info);

  void SCALAPACK(pdsyevr)(const char *jobz, const char *range, const char *uplo,
      const Int *n, double* a, const Int *ia, const Int *ja,
      const Int *desca, const double* vl, const double *vu,
      const Int *il, const Int* iu, Int *m, Int *nz, 
      double *w, double *z, const Int *iz, const Int *jz, 
      const Int *descz, double *work, const Int *lwork, 
      Int *iwork, const Int *liwork, Int *info);

  void SCALAPACK(pdlacpy)(const char* uplo,
      const Int* m, const Int* n,
      const double* A, const Int* ia, const Int* ja, const Int* desca, 
      const double* B, const Int* ib, const Int* jb, const Int* descb );

  void SCALAPACK(pdgemm)(const char* transA, const char* transB,
      const Int* m, const Int* n, const Int* k,
      const double* alpha,
      const double* A, const Int* ia, const Int* ja, const Int* desca, 
      const double* B, const Int* ib, const Int* jb, const Int* descb,
      const double* beta,
      double* C, const Int* ic, const Int* jc, const Int* descc,
      const Int* contxt);

  void SCALAPACK(pdgemr2d)(const Int* m, const Int* n, const double* A, const Int* ia, 
      const Int* ja, const Int* desca, double* B,
      const Int* ib, const Int* jb, const Int* descb,
      const Int* contxt);

  void SCALAPACK(pdpotrf)( const char* uplo, const Int* n, 
      double* A, const Int* ia, const Int* ja, const Int* desca, 
      Int* info );

  void SCALAPACK(pdsygst)( const Int* ibtype, const char* uplo, 
      const Int* n, double* A, const Int* ia, const Int* ja, 
      const Int* desca, const double* b, const Int* ib, const Int* jb,
      const Int* descb, double* scale, Int* info );

  void SCALAPACK(pdtrsm)( const char* side, const char* uplo, 
      const char* trans, const char* diag,
      const int* m, const int* n, const double* alpha,
      const double* a, const int* ia, const int* ja, const int* desca, 
      double* b, const int* ib, const int* jb, const int* descb );

}

EigenSolver::EigenSolver() {}

EigenSolver::~EigenSolver() {}

void EigenSolver::Setup(
			const esdf::ESDFInputParam& esdfParam,
			Hamiltonian& ham,
			Spinor& psi,
			Fourier& fft ) {
#ifndef _RELEASE_
	PushCallStack("EigenSolver::Setup");
#endif  // ifndef _RELEASE_
	hamPtr_ = &ham;
	psiPtr_ = &psi;
	fftPtr_ = &fft;

	eigMaxIter_    = esdfParam.eigMaxIter;
	eigToleranceSave_  = esdfParam.eigTolerance;
  eigTolerance_      = esdfParam.eigTolerance;


	eigVal_.Resize(psiPtr_->NumStateTotal());  SetValue(eigVal_, 0.0);
	resVal_.Resize(psiPtr_->NumStateTotal());  SetValue(resVal_, 0.0);
#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_
	return;
} 		// -----  end of method EigenSolver::Setup ----- 

BlopexInt EigenSolver::HamiltonianMult
(serial_Multi_Vector *x, serial_Multi_Vector *y) {
#ifndef _RELEASE_
	PushCallStack("EigenSolver::HamiltonianMult");
#endif
  Int ntot = psiPtr_->NumGridTotal();
  Int ncom = psiPtr_->NumComponent();
  Int noccTotal = psiPtr_->NumStateTotal();
  Int noccLocal = psiPtr_->NumState();
  Int nocc = noccLocal;

	if( (x->size * x->num_vectors) != ntot*ncom*noccLocal ) {
		throw std::logic_error("Vector size does not match.");
	}

  Spinor psitemp(fftPtr_->domain, ncom, noccTotal, noccLocal, false, x->data);
  NumTns<Scalar> a3(ntot, ncom, nocc, false, y->data);

	SetValue( a3, SCALAR_ZERO ); // IMPORTANT
  hamPtr_->MultSpinor(psitemp, a3, *fftPtr_);
	
#ifndef _RELEASE_
	PopCallStack();
#endif  
  return 0;
}

BlopexInt EigenSolver::PrecondMult
(serial_Multi_Vector *x, serial_Multi_Vector *y) {
#ifndef _RELEASE_
	PushCallStack("EigenSolver::PrecondMult");
#endif
  if( !fftPtr_->isInitialized ){
    throw std::runtime_error("Fourier is not prepared.");
  }
  Int ntot = psiPtr_->NumGridTotal();
  Int ncom = psiPtr_->NumComponent();
  Int nocc = psiPtr_->NumState();

  if( fftPtr_->domain.NumGridTotal() != ntot ){
    throw std::logic_error("Domain size does not match.");
  }

  NumTns<Scalar> a3i(ntot, ncom, nocc, false, x->data);
  NumTns<Scalar> a3o(ntot, ncom, nocc, false, y->data);

  // Important to set the values of y to be 0
  SetValue( a3o, SCALAR_ZERO );

#ifdef _USE_OPENMP_
#pragma omp parallel
  {
#endif
#ifndef _USE_COMPLEX_ // Real case
    Int ntothalf = fftPtr_->numGridTotalR2C;
    // These two are private variables in the OpenMP context
    DblNumVec realInVec(ntot);
		CpxNumVec cpxOutVec(ntothalf);

#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) nowait
#endif
    for (Int k=0; k<nocc; k++) {
      for (Int j=0; j<ncom; j++) {
        // For c2r and r2c transforms, the default is to DESTROY the
        // input, therefore a copy of the original matrix is necessary. 
        blas::Copy( ntot, a3i.VecData(j,k), 1, 
            realInVec.Data(), 1 );

				fftw_execute_dft_r2c(
						fftPtr_->forwardPlanR2C, 
						realInVec.Data(),
						reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ));

        Real*    ptr1d   = fftPtr_->TeterPrecondR2C.Data();
				Complex* ptr2    = cpxOutVec.Data();
        for (Int i=0; i<ntothalf; i++) 
					*(ptr2++) *= *(ptr1d++);

				fftw_execute_dft_c2r(
						fftPtr_->backwardPlanR2C,
						reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ),
						realInVec.Data() );

        blas::Axpy( ntot, 1.0 / Real(ntot), realInVec.Data(), 1, 
            a3o.VecData(j, k), 1 );
      }
    }
#else // Complex case
  // TODO OpenMP implementation
    for (Int k=0; k<nocc; k++) {
      for (Int j=0; j<ncom; j++) {
        Complex *ptr0  = a3i.VecData(j,k);

        fftw_execute_dft(fftPtr_>forwardPlan, reinterpret_cast<fftw_complex*>(ptr0), 
            reinterpret_cast<fftw_complex*>(fftPtr_->outputComplexVec.Data() ));
        Real *ptr1d = fftPtr_->TeterPrecond.Data();
        ptr0 = fftPtr_->outputComplexVec.Data();
        for (Int i=0; i<ntot; i++) 
          *(ptr0++) *= *(ptr1d++);

        fftw_execute(fftPtr_->backwardPlan);
        ptr0 = fftPtr_->inputComplexVec.Data();

        Complex *ptra3o = a3o.VecData(j, k); 
        for (Int i=0; i<ntot; i++) 
          *(ptra3o++) = *(ptr0++) / Real(ntot);
      }
    }
#endif
#ifdef _USE_OPENMP_
  }
#endif

#ifndef _RELEASE_
  PopCallStack();
#endif  
  return 0;
}

void EigenSolver::LOBPCGHamiltonianMult(void *A, void *X, void *AX) {
  ((EigenSolver*)A)->HamiltonianMult((serial_Multi_Vector*)X,
      (serial_Multi_Vector*)AX);
  return;
};

void EigenSolver::LOBPCGPrecondMult(void *A, void *X, void *AX) {
  ((EigenSolver*)A)->PrecondMult((serial_Multi_Vector*)X,
      (serial_Multi_Vector*)AX);
  return;
};

void
EigenSolver::Solve	()
{
#ifndef _RELEASE_
	PushCallStack("EigenSolver::Solve");
#endif 
  
  serial_Multi_Vector *x;
  x = (serial_Multi_Vector*)malloc(sizeof(serial_Multi_Vector));

  x->data = psiPtr_->Wavefun().Data();
  x->owns_data = 0;
  x->size = psiPtr_->NumGridTotal() * psiPtr_->NumComponent();
  x->num_vectors = psiPtr_->NumState();
  x->num_active_vectors = x->num_vectors;
  x->active_indices = (BlopexInt*)malloc(sizeof(BlopexInt)*x->num_active_vectors);
  for (Int i=0; i<x->num_active_vectors; i++) x->active_indices[i] = i;

  mv_MultiVectorPtr xx;
  mv_InterfaceInterpreter ii;
  lobpcg_Tolerance lobpcg_tol;
  lobpcg_BLASLAPACKFunctions blap_fn;

  lobpcg_tol.absolute = eigTolerance_;
  lobpcg_tol.relative = eigTolerance_;

  serialSetupInterpreter ( &ii );
  xx = mv_MultiVectorWrap( &ii, x, 0);

  Int iterations;

#ifndef _USE_COMPLEX_ // Real case
  blap_fn.dpotrf = LAPACK(dpotrf);
  blap_fn.dsygv  = LAPACK(dsygv);
	std::cout<<"Call lobpcg_double"<<std::endl;
	
	lobpcg_solve_double ( 
			xx,
			this,
			LOBPCGHamiltonianMult,
			NULL,
			NULL,
			this,
			LOBPCGPrecondMult,
			NULL,
			blap_fn,
			lobpcg_tol,
			eigMaxIter_,
			1,
			&iterations,
			eigVal_.Data(),
			NULL,
			0,
			resVal_.Data(),
			NULL,
			0);

#else // Complex case
  blap_fn.zpotrf = LAPACK(zpotrf);
  blap_fn.zhegv  = LAPACK(zhegv);
	std::cout<<"Call lobpcg_complex"<<std::endl;
  CpxNumVec cpxEigVal;
  cpxEigVal.Resize(eigVal_.m());  
  SetValue(cpxEigVal, Z_ZERO);
	lobpcg_solve_complex( 
			xx,
			this,
			LOBPCGHamiltonianMult,
			NULL,
			NULL,
			this,
			LOBPCGPrecondMult,
			NULL,
			blap_fn,
			lobpcg_tol,
			eigMaxIter_,
			0,
			&iterations,
			(komplex*)(cpxEigVal.Data()),
			NULL,
			0,
			resVal_.Data(),
			NULL,
			0);

  for(Int i = 0; i < eigVal_.m(); i++)
    eigVal_(i) = cpxEigVal(i).real();
#endif

  

	// Assign the eigenvalues to the Hamiltonian
	hamPtr_->EigVal() = eigVal_;

  serial_Multi_VectorDestroy(x);
  mv_MultiVectorDestroy(xx);

#ifndef _RELEASE_
	PopCallStack();
#endif  

	return ;
} 		// -----  end of method EigenSolver::Solve  ----- 


void
EigenSolver::LOBPCGSolveReal	( 
      Int          numEig,
      Int          eigMaxIter,
      Real         eigTolerance)
{
#ifndef _RELEASE_
	PushCallStack("EigenSolver::LOBPCGSolveReal");
#endif

  // *********************************************************************
  // Initialization
  // *********************************************************************
  Int ntot = psiPtr_->NumGridTotal();
  Int ncom = psiPtr_->NumComponent();
  Int noccLocal = psiPtr_->NumState();
  Int noccTotal = psiPtr_->NumStateTotal();

  Int height = ntot * ncom;
  Int widthTotal = noccTotal;
  Int widthLocal = noccLocal;
  Int width = noccTotal;
  Int lda = 3 * width;

  Real timeSta, timeEnd;
  Real timeGemmT = 0.0;
  Real timeGemmN = 0.0;
  Real timeTrsm = 0.0;
  Real timeSpinor = 0.0;
  Real timeMpirank0 = 0.0;
  Int  iterGemmT = 0;
  Int  iterGemmN = 0;
  Int  iterTrsm = 0;
  Int  iterSpinor = 0;
  Int  iterMpirank0 = 0;

//  GetTime( timeSta );
//  GetTime( timeEnd );
//  statusOFS << "Time for Gemm is " << timeEnd - timeSta << << std::endl;


  if( numEig > width ){
    std::ostringstream msg;
    msg 
      << "Number of eigenvalues requested  = " << numEig << std::endl
      << "which is larger than the number of columns in psi = " << width << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }


  Int contxt;
  Int nprow, npcol, myrow, mycol, info;

  MPI_Comm mpi_comm = fftPtr_->domain.comm;

  MPI_Barrier(mpi_comm);
  Int mpirank;  MPI_Comm_rank(mpi_comm, &mpirank);
  Int mpisize;  MPI_Comm_size(mpi_comm, &mpisize);

  Cblacs_get(0, 0, &contxt);

  nprow = 1;
  npcol = mpisize;

  Cblacs_gridinit(&contxt, "C", nprow, npcol);
  Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);

  Int desc[9];
  Int desc_width[9];
  Int desc_width3[9];

  Int irsrc = 0;
  Int icsrc = 0;
  Int mb = height;
  Int nb = 1;
  Int width3 = 3 * noccTotal; 
  Int ld_width = (myrow == 0) ? width : 1;
  Int ld_width3 = (myrow == 0) ? width3 : 1;
  SCALAPACK(descinit)(&desc[0], &height, &width, &mb, &nb, &irsrc, &icsrc, &contxt, &height, &info);
  SCALAPACK(descinit)(&desc_width[0], &width, &width, &width, &width, &irsrc, &icsrc, &contxt, &ld_width, &info);
  SCALAPACK(descinit)(&desc_width3[0], &width3, &width3, &width3, &width3, &irsrc, &icsrc, &contxt, &ld_width3, &info);

  // S = ( X | W | P ) is a triplet used for LOBPCG.  
  // W is the preconditioned residual
  DblNumMat  S( height, 3*widthLocal ), AS( height, 3*widthLocal ); 
  // AMat = S' * (AS),  BMat = S' * S
  // 
  // AMat = (X'*AX   X'*AW   X'*AP)
  //      = (  *     W'*AW   W'*AP)
  //      = (  *       *     P'*AP)
  //
  // BMat = (X'*X   X'*W   X'*P)
  //      = (  *    W'*W   W'*P)
  //      = (  *      *    P'*P)
  //
  DblNumMat  AMat( 3*width, 3*width ), BMat( 3*width, 3*width );
  DblNumMat  AMatT1( 3*width, 3*width );
  // AMatSave and BMatSave are used for restart
  // Temporary buffer array.
  // The unpreconditioned residual will also be saved in Xtemp
  DblNumMat  XTX( width, width ), Xtemp( height, widthLocal );

  // rexNorm Grobal matrix  similar to numEig 
  DblNumVec  resNormLocal ( widthLocal ); 
  SetValue( resNormLocal, 0.0 );
  DblNumVec  resNorm( width );
  SetValue( resNorm, 0.0 );
  Real       resMaxLocal, resMinLocal;
  Real       resMax, resMin;

  // For convenience
  DblNumMat  X( height, widthLocal, false, S.VecData(0) );
  DblNumMat  W( height, widthLocal, false, S.VecData(widthLocal) );
  DblNumMat  P( height, widthLocal, false, S.VecData(2*widthLocal) );
  DblNumMat AX( height, widthLocal, false, AS.VecData(0) );
  DblNumMat AW( height, widthLocal, false, AS.VecData(widthLocal) );
  DblNumMat AP( height, widthLocal, false, AS.VecData(2*widthLocal) );

  //Int info;
  bool isRestart = false;
  // numSet = 2    : Steepest descent (Davidson), only use (X | W)
  //        = 3    : Conjugate gradient, use all the triplet (X | W | P)
  Int numSet = 2;

  // numLocked is the number of converged vectors
  Int numLockedLocal = 0, numLockedSaveLocal = 0;
  Int numLockedTotal = 0, numLockedSaveTotal = 0; 
  Int numLockedSave = 0;
  // numActive = width - numLocked
  Int numActiveLocal = 0;
  Int numActiveTotal = 0;

  // Real lockTolerance = std::min( eigTolerance_, 1e-2 );
  
  const Int numLocked = 0;  // Never perform locking in this version
  const Int numActive = width;

  bool isConverged = false;

  // Initialization
  SetValue( S, 0.0 );
  SetValue( AS, 0.0 );

  DblNumVec  eigValS(lda);
  SetValue( eigValS, 0.0 );
  DblNumVec  sigma2(lda);
  DblNumVec  invsigma(lda);
  SetValue( sigma2, 0.0 );
  SetValue( invsigma, 0.0 );

  // Initialize X by the data in psi
  // lapack::Lacpy( 'A', height, width, psiPtr_->Wavefun().Data(), height, 
  //    X.Data(), height );
 
  char AA = 'A';
  SCALAPACK(pdlacpy)(&AA, &height, &width, 
      psiPtr_->Wavefun().Data(), &I_ONE, &I_ONE, &desc[0],
      X.Data(), &I_ONE, &I_ONE, &desc[0]);

  // *********************************************************************
  // Main loop
  // *********************************************************************

  // Orthogonalization through Cholesky factorization
  // blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(), 
  // height, X.Data(), height, 0.0, XTX.Data(), width );
  char TT = 'T';
  char NN = 'N';
  double D_ONE = 1.0;
  double D_ZERO = 0.0;
  int I_ONE = 1;
  GetTime( timeSta );
  SCALAPACK(pdgemm)(&TT, &NN, &width, &width, &height, 
      &D_ONE,
      X.Data(), &I_ONE, &I_ONE, &desc[0],
      X.Data(), &I_ONE, &I_ONE, &desc[0], 
      &D_ZERO,
      XTX.Data(), &I_ONE, &I_ONE, &desc_width[0], 
      &contxt );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );

  // X'*X=U'*U
  // lapack::Potrf( 'U', width, XTX.Data(), width );
  char UU = 'U';
  
  // SCALAPACK(pdpotrf)(&UU, &width, XTX.Data(), &I_ONE,
  //    &I_ONE, &desc_width[0], &info);

  if ( mpirank == 0) {
  GetTime( timeSta );
    lapack::Potrf( 'U', width, XTX.Data(), width );
  GetTime( timeEnd );
  iterMpirank0 = iterMpirank0 + 1;
  timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
  }
  MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);

  // X <- X * U^{-1} is orthogonal
  // blas::Trsm( 'R', 'U', 'N', 'N', height, width, 1.0, XTX.Data(), width, 
  //    X.Data(), height );
  char RR = 'R';
  GetTime( timeSta );
  SCALAPACK(pdtrsm)( &RR, &UU, &NN, &NN,
      &height, &width, 
      &D_ONE,
      XTX.Data(), &I_ONE, &I_ONE, &desc_width[0],
      X.Data(), &I_ONE, &I_ONE, &desc[0]);
  GetTime( timeEnd );
  iterTrsm = iterTrsm + 1;
  timeTrsm = timeTrsm + ( timeEnd - timeSta );


  // Applying the Hamiltonian matrix
  {
  GetTime( timeSta );
    Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, noccLocal, false, X.Data());
    NumTns<Scalar> tnsTemp(ntot, ncom, noccLocal, false, AX.Data());

    hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
  GetTime( timeEnd );
  iterSpinor = iterSpinor + 1;
  timeSpinor = timeSpinor + ( timeEnd - timeSta );
  }

  // Start the main loop
  Int iter;
  for( iter = 1; iter < eigMaxIter_; iter++ ){
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "iter = " << iter << std::endl;
#endif

    if( iter == 1 || isRestart == true )
      numSet = 2;
    else
      numSet = 3;

    SetValue( AMat, 0.0 );
    SetValue( BMat, 0.0 );

    // Rayleigh Ritz in Q = span( X, gradient )

    // XTX <- X' * (AX)
    // blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(),
    //    height, AX.Data(), height, 0.0, XTX.Data(), width );
   
  GetTime( timeSta );
    SCALAPACK(pdgemm)(&TT, &NN, &width, &width, &height, 
        &D_ONE,
        X.Data(), &I_ONE, &I_ONE, &desc[0],
        AX.Data(), &I_ONE, &I_ONE, &desc[0], 
        &D_ZERO,
        XTX.Data(), &I_ONE, &I_ONE, &desc_width[0], 
        &contxt );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );


    // lapack::Lacpy( 'A', width, width, XTX.Data(), width, AMat.Data(), lda );
    SCALAPACK(pdlacpy)(&AA, &width, &width, 
        XTX.Data(), &I_ONE, &I_ONE, &desc_width[0],
        AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0]);

    // Compute the residual.
    // R <- AX - X*(X'*AX)
    // lapack::Lacpy( 'A', height, width, AX.Data(), height, Xtemp.Data(), height );
    SCALAPACK(pdlacpy)(&AA, &height, &width, 
        AX.Data(), &I_ONE, &I_ONE, &desc[0],
        Xtemp.Data(), &I_ONE, &I_ONE, &desc[0]);

    // blas::Gemm( 'N', 'N', height, width, width, -1.0, 
    //    X.Data(), height, AMat.Data(), lda, 1.0, Xtemp.Data(), height );
    double D_MinusONE = -1.0;
  GetTime( timeSta );
    SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &width, 
        &D_MinusONE,
        X.Data(), &I_ONE, &I_ONE, &desc[0],
        AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0], 
        &D_ONE,
        Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
        &contxt );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

    // Compute the norm of the residual
    SetValue( resNorm, 0.0 );
    for( Int k = 0; k < widthLocal; k++ ){
      resNormLocal(k) = Energy(DblNumVec(height, false, Xtemp.VecData(k))); 
      resNorm( psiPtr_->WavefunIdx(k) ) = resNormLocal(k);
    }

    DblNumVec  resNormTemp( width );
    for( Int k = 0; k < width; k++ ){
      resNormTemp(k) = resNorm(k) ;
    }

    SetValue( resNorm, 0.0 );
    
    MPI_Barrier( mpi_comm );

    MPI_Allreduce( resNormTemp.Data(), resNorm.Data(), width, MPI_DOUBLE, 
        MPI_SUM, mpi_comm );
    


    if ( mpirank == 0 ){
  GetTime( timeSta );
      for( Int k = 0; k < width; k++ ){
        resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( XTX(k,k) ) );
      }
  GetTime( timeEnd );
  iterMpirank0 = iterMpirank0 + 1;
  timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
    }
    MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm);


    resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
    resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "resNorm = " << resNorm << std::endl;
    statusOFS << "maxRes  = " << resMax  << std::endl;
    statusOFS << "minRes  = " << resMin  << std::endl;
#endif

    if( resMax < eigTolerance_ ){
      isConverged = true;;
      break;
    }

    //numActive = width - numLocked;
    numActiveLocal = widthLocal - numLockedLocal;
    numActiveTotal = widthTotal - numLockedTotal;

    // If the number of locked vectors goes down, perform steppest
    // descent rather than conjugate gradient
    // if( numLockedTotal < numLockedSaveTotal )
    //  numSet = 2;

    // Compute the preconditioned residual W = T*R.
    // The residual is saved in Xtemp
    {
  GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, Xtemp.VecData(numLockedLocal));
      NumTns<Scalar> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, W.VecData(numLockedLocal));

      SetValue( tnsTemp, 0.0 );
      spnTemp.AddTeterPrecond( fftPtr_, tnsTemp );
  GetTime( timeEnd );
  iterSpinor = iterSpinor + 1;
  timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

//    Real normLocal; 
    Real norm; 
   
    norm = 0.0; 
    // Normalize the preconditioned residual
    for( Int k = numLockedLocal; k < widthLocal; k++ ){
      norm = Energy(DblNumVec(height, false, W.VecData(k)));
      norm = std::sqrt( norm );
      blas::Scal( height, 1.0 / norm, W.VecData(k), 1 );
    }
   
    norm = 0.0; 
    // Normalize the conjugate direction
    if( numSet == 3 ){
      for( Int k = numLockedLocal; k < widthLocal; k++ ){
        norm = Energy(DblNumVec(height, false, P.VecData(k)));
        norm = std::sqrt( norm );
        blas::Scal( height, 1.0 / norm, P.VecData(k), 1 );
        blas::Scal( height, 1.0 / norm, AP.VecData(k), 1 );
      }
    }

    // Compute AMat

    // Compute AW = A*W
    {
  GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, W.VecData(numLockedLocal));
      NumTns<Scalar> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, AW.VecData(numLockedLocal));

      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
  GetTime( timeEnd );
  iterSpinor = iterSpinor + 1;
  timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

    // Compute X' * (AW)
    // Instead of saving the block at &AMat(0,width+numLocked), the data
    // is saved at &AMat(0,width) to guarantee a continuous data
    // arrangement of AMat.  The same treatment applies to the blocks
    // below in both AMat and BMat.
    // blas::Gemm( 'T', 'N', width, numActive, height, 1.0, X.Data(),
    //    height, AW.VecData(numLocked), height, 
    //    0.0, &AMat(0,width), lda );
    int ia = 1;
    int ja = 1;
    int ib = 1;
    int jb = numLockedTotal + 1;
    int ic = 1;
    int jc = width + 1;
  GetTime( timeSta );
    SCALAPACK(pdgemm)(&TT, &NN, &width, &numActiveTotal, &height, 
        &D_ONE,
        X.Data(), &I_ONE, &I_ONE, &desc[0],
        AW.Data(), &I_ONE, &jb, &desc[0], 
        &D_ZERO,
        AMat.Data(), &I_ONE, &jc, &desc_width3[0], 
        &contxt );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );

    // Compute W' * (AW)
    // blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
    //    W.VecData(numLocked), height, AW.VecData(numLocked), height, 
    //    0.0, &AMat(width, width), lda );
    
    ia = 1;
    ja = numLockedTotal + 1;
    ib = 1;
    jb = numLockedTotal + 1;
    ic = width + 1;
    jc = width + 1;
  GetTime( timeSta );
    SCALAPACK(pdgemm)(&TT, &NN, &numActiveTotal, &numActiveTotal, &height, 
        &D_ONE,
        W.Data(), &I_ONE, &ja, &desc[0],
        AW.Data(), &I_ONE, &jb, &desc[0], 
        &D_ZERO,
        AMat.Data(), &ic, &jc, &desc_width3[0], 
        &contxt );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );

    if( numSet == 3 ){
      // Compute X' * (AP)
      // blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
      //    X.Data(), height, AP.VecData(numLocked), height, 
      //    0.0, &AMat(0, width+numActive), lda );

    ia = 1;
    ja = 1;
    ib = 1;
    jb = numLockedTotal + 1;
    ic = 1;
    jc = width + numActiveTotal + 1;
  GetTime( timeSta );
    SCALAPACK(pdgemm)(&TT, &NN, &width, &numActiveTotal, &height, 
        &D_ONE,
        X.Data(), &I_ONE, &I_ONE, &desc[0],
        AP.Data(), &I_ONE, &jb, &desc[0], 
        &D_ZERO,
        AMat.Data(), &I_ONE, &jc, &desc_width3[0], 
        &contxt );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );

      // Compute W' * (AP)
      // blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
      //    W.VecData(numLocked), height, AP.VecData(numLocked), height, 
      //    0.0, &AMat(width, width+numActive), lda );

    ia = 1;
    ja = numLockedTotal + 1;
    ib = 1;
    jb = numLockedTotal + 1;
    ic = width + 1 ;
    jc = width + numActiveTotal + 1;
  GetTime( timeSta );
    SCALAPACK(pdgemm)(&TT, &NN, &numActiveTotal, &numActiveTotal, &height, 
        &D_ONE,
        W.Data(), &I_ONE, &ja, &desc[0],
        AP.Data(), &I_ONE, &jb, &desc[0], 
        &D_ZERO,
        AMat.Data(), &ic, &jc, &desc_width3[0], 
        &contxt );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );

      // Compute P' * (AP)
      // blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
      //    P.VecData(numLocked), height, AP.VecData(numLocked), height, 
      //    0.0, &AMat(width+numActive, width+numActive), lda );
    
    ia = 1;
    ja = numLockedTotal + 1;
    ib = 1;
    jb = numLockedTotal + 1;
    ic = width + numActiveTotal + 1;
    jc = width + numActiveTotal + 1;
  GetTime( timeSta );
    SCALAPACK(pdgemm)(&TT, &NN, &numActiveTotal, &numActiveTotal, &height, 
        &D_ONE,
        P.Data(), &I_ONE, &ja, &desc[0],
        AP.Data(), &I_ONE, &jb, &desc[0], 
        &D_ZERO,
        AMat.Data(), &ic, &jc, &desc_width3[0], 
        &contxt );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
    
    }

    // Compute BMat (overlap matrix)

    // Compute X'*X
    // blas::Gemm( 'T', 'N', width, width, height, 1.0, 
    //    X.Data(), height, X.Data(), height, 
    //     0.0, &BMat(0,0), lda );


  GetTime( timeSta );
    SCALAPACK(pdgemm)(&TT, &NN, &width, &width, &height, 
        &D_ONE,
        X.Data(), &I_ONE, &I_ONE, &desc[0],
        X.Data(), &I_ONE, &I_ONE, &desc[0], 
        &D_ZERO,
        BMat.Data(), &I_ONE, &I_ONE, &desc_width3[0], 
        &contxt );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
    

    // Compute X'*W
    // blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
    //    X.Data(), height, W.VecData(numLocked), height,
    //    0.0, &BMat(0,width), lda );


    ia = 1;
    ja = 1;
    ib = 1;
    jb = numLockedTotal + 1;
    ic = 1;
    jc = width + 1;
  GetTime( timeSta );
    SCALAPACK(pdgemm)(&TT, &NN, &width, &numActiveTotal, &height, 
        &D_ONE,
        X.Data(), &I_ONE, &I_ONE, &desc[0],
        W.Data(), &I_ONE, &jb, &desc[0], 
        &D_ZERO,
        BMat.Data(), &I_ONE, &jc, &desc_width3[0], 
        &contxt );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
    

    // Compute W'*W
    // blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
    //    W.VecData(numLocked), height, W.VecData(numLocked), height,
    //    0.0, &BMat(width, width), lda );


    ia = 1;
    ja = numLockedTotal + 1;
    ib = 1;
    jb = numLockedTotal + 1;
    ic = width + 1;
    jc = width + 1;
  GetTime( timeSta );
    SCALAPACK(pdgemm)(&TT, &NN, &numActiveTotal, &numActiveTotal, &height, 
        &D_ONE,
        W.Data(), &I_ONE, &ja, &desc[0],
        W.Data(), &I_ONE, &jb, &desc[0], 
        &D_ZERO,
        BMat.Data(), &ic, &jc, &desc_width3[0], 
        &contxt );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
    

    if( numSet == 3 ){
      // Compute X'*P
     //  blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
     //     X.Data(), height, P.VecData(numLocked), height, 
     //     0.0, &BMat(0, width+numActive), lda );
    

    ia = 1;
    ja = 1;
    ib = 1;
    jb = numLockedTotal + 1;
    ic = 1;
    jc = width + numActiveTotal + 1;
  GetTime( timeSta );
    SCALAPACK(pdgemm)(&TT, &NN, &width, &numActiveTotal, &height, 
        &D_ONE,
        X.Data(), &I_ONE, &I_ONE, &desc[0],
        P.Data(), &I_ONE, &jb, &desc[0], 
        &D_ZERO,
        BMat.Data(), &I_ONE, &jc, &desc_width3[0], 
        &contxt );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
    

      // Compute W'*P
      // blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
      //    W.VecData(numLocked), height, P.VecData(numLocked), height,
      //    0.0, &BMat(width, width+numActive), lda );


    ia = 1;
    ja = numLockedTotal + 1;
    ib = 1;
    jb = numLockedTotal + 1;
    ic = width + 1;
    jc = width + numActiveTotal + 1;
  GetTime( timeSta );
    SCALAPACK(pdgemm)(&TT, &NN, &numActiveTotal, &numActiveTotal, &height, 
        &D_ONE,
        W.Data(), &I_ONE, &ja, &desc[0],
        P.Data(), &I_ONE, &jb, &desc[0], 
        &D_ZERO,
        BMat.Data(), &ic, &jc, &desc_width3[0], 
        &contxt );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
    

      // Compute P'*P
      // blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
      //    P.VecData(numLocked), height, P.VecData(numLocked), height,
      //    0.0, &BMat(width+numActive, width+numActive), lda );
    
   
    ia = 1;
    ja = numLockedTotal + 1;
    ib = 1;
    jb = numLockedTotal + 1;
    ic = width + numActiveTotal + 1;
    jc = width + numActiveTotal + 1;
  GetTime( timeSta );
    SCALAPACK(pdgemm)(&TT, &NN, &numActiveTotal, &numActiveTotal, &height, 
        &D_ONE,
        P.Data(), &I_ONE, &ja, &desc[0],
        P.Data(), &I_ONE, &jb, &desc[0], 
        &D_ZERO,
        BMat.Data(), &ic, &jc, &desc_width3[0], 
        &contxt );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
  
    }

#if ( _DEBUGlevel_ >= 2 )
    {
      DblNumMat WTW( width, width );
      // lapack::Lacpy( 'A', width, width, &BMat(width, width), lda,
      //    WTW.Data(), width );

      ia = width + 1;
      ja = wdith + 1;
      ib = 1;
      jb = 1;
      SCALAPACK(pdlacpy)(&AA, &width, &width, 
        BMat.Data(), &ia, &ib, &desc_width3[0],
        WTW.Data(), &I_ONE, &I_ONE, &desc_width[0]);

      statusOFS << "W'*W = " << WTW << std::endl;
      if( numSet == 3 )
      {
        DblNumMat PTP( width, width );
        // lapack::Lacpy( 'A', width, width, &BMat(width+numActive, width+numActive), 
        //    lda, PTP.Data(), width );
        
        ia = width + numActiveTotal + 1;
        ja = wdith + numActiveTotal + 1;
        ib = 1;
        jb = 1;
        SCALAPACK(pdlacpy)(&AA, &width, &width, 
          BMat.Data(), &ia, &ib, &desc_width3[0],
          PTP.Data(), &I_ONE, &I_ONE, &desc_width[0]);
        
        statusOFS << "P'*P = " << PTP << std::endl;
      }
    }
#endif


   
    // Rayleigh-Ritz procedure
    // AMat * C = BMat * C * Lambda
    // Assuming the dimension (needed) for C is width * width, then
    //     ( C_X )
    //     ( --- )
    // C = ( C_W )
    //     ( --- )
    //     ( C_P )
    //
    Int numCol;
    if( numSet == 3 ){
      // Conjugate gradient
      numCol = width + 2 * numActiveTotal;
    }
    else{
      numCol = width + numActiveTotal;
    }

    // Solve the generalized eigenvalue problem with thresholding
    if(1){

      if ( mpirank == 0 ) {
       
      // Symmetrize A and B first.  This is important.
      for( Int j = 0; j < numCol; j++ ){
        for( Int i = j+1; i < numCol; i++ ){
          AMat(i,j) = AMat(j,i);
          BMat(i,j) = BMat(j,i);
        }
      }

  GetTime( timeSta );
      lapack::Syevd( 'V', 'U', numCol, BMat.Data(), lda, sigma2.Data() );
  GetTime( timeEnd );
  iterMpirank0 = iterMpirank0 + 1;
  timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
      
      Int numKeep = 0;
      for( Int i = numCol-1; i>=0; i-- ){
        if( sigma2(i) / sigma2(numCol-1) >  1e-12 )
          numKeep++;
        else
          break;
      }

#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "sigma2 = " << sigma2 << std::endl;
#endif

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "sigma2(0)        = " << sigma2(0) << std::endl;
      statusOFS << "sigma2(numCol-1) = " << sigma2(numCol-1) << std::endl;
      statusOFS << "numKeep          = " << numKeep << std::endl;
#endif

      for( Int i = 0; i < numKeep; i++ ){
        invsigma(i) = 1.0 / std::sqrt( sigma2(i+numCol-numKeep) );
      }

      if( numKeep < width ){
        std::ostringstream msg;
        msg 
          << "width   = " << width << std::endl
          << "numKeep =  " << numKeep << std::endl
          << "there are not enough number of columns." << std::endl;
        throw std::runtime_error( msg.str().c_str() );
      }

      SetValue( AMatT1, 0.0 );
      // Evaluate S^{-1/2} (U^T A U) S^{-1/2}
  GetTime( timeSta );
      blas::Gemm( 'N', 'N', numCol, numKeep, numCol, 1.0,
          AMat.Data(), lda, BMat.VecData(numCol-numKeep), lda,
          0.0, AMatT1.Data(), lda );
  GetTime( timeEnd );
  iterMpirank0 = iterMpirank0 + 1;
  timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

  GetTime( timeSta );
      blas::Gemm( 'T', 'N', numKeep, numKeep, numCol, 1.0,
          BMat.VecData(numCol-numKeep), lda, AMatT1.Data(), lda, 
          0.0, AMat.Data(), lda );
  GetTime( timeEnd );
  iterMpirank0 = iterMpirank0 + 1;
  timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

  GetTime( timeSta );
      for( Int j = 0; j < numKeep; j++ ){
        for( Int i = 0; i < numKeep; i++ ){
          AMat(i,j) *= invsigma(i)*invsigma(j);
        }
      }
  GetTime( timeEnd );
  iterMpirank0 = iterMpirank0 + 1;
  timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

      // Solve the standard eigenvalue problem
  GetTime( timeSta );
      lapack::Syevd( 'V', 'U', numKeep, AMat.Data(), lda,
          eigValS.Data() );
  GetTime( timeEnd );
  iterMpirank0 = iterMpirank0 + 1;
  timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

      // Compute the correct eigenvectors and save them in AMat
      for( Int j = 0; j < numKeep; j++ ){
        for( Int i = 0; i < numKeep; i++ ){
          AMat(i,j) *= invsigma(i);
        }
      }

  GetTime( timeSta );
      blas::Gemm( 'N', 'N', numCol, numKeep, numKeep, 1.0,
          BMat.VecData(numCol-numKeep), lda, AMat.Data(), lda,
          0.0, AMatT1.Data(), lda );
  GetTime( timeEnd );
  iterMpirank0 = iterMpirank0 + 1;
  timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

      lapack::Lacpy( 'A', numCol, numKeep, AMatT1.Data(), lda, 
          AMat.Data(), lda );
      
      } // mpirank ==0


        MPI_Bcast(AMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm);
        MPI_Bcast(eigValS.Data(), lda, MPI_DOUBLE, 0, mpi_comm);


    } // if(1)
   
  
    else{

      // BMat = U' * U
      // lapack::Potrf( 'U', numCol, BMat.Data(), lda );

      // SCALAPACK(pdpotrf)(&UU, &numCol, BMat.Data(), &I_ONE, &I_ONE, &desc_width3[0], &info);

      if ( mpirank == 0 ) {
  GetTime( timeSta );
        lapack::Potrf( 'U', numCol, BMat.Data(), lda );
  GetTime( timeEnd );
  iterMpirank0 = iterMpirank0 + 1;
  timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
      }
        MPI_Bcast(BMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm);
      
      // TODO Add Pocon and restart strategy
      // AMat <- U^{-T} * AMat * U^{-1}
      // lapack::Hegst( 1, 'U', numCol, AMat.Data(), lda,
      //    BMat.Data(), lda );

  GetTime( timeSta );
      SCALAPACK(pdsygst)( &I_ONE, &UU, &numCol,
         AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0],
         BMat.Data(), &I_ONE, &I_ONE, &desc_width3[0],
         &D_ONE, &info); 
  GetTime( timeEnd );
  iterTrsm = iterTrsm + 1;
  timeTrsm = timeTrsm + ( timeEnd - timeSta );
  
  // Eigenvalue problem, the eigenvectors are saved in AMat
      // lapack::Syevd( 'V', 'U', numCol, AMat.Data(), lda, 
      //    eigValS.Data() );
       
      if ( mpirank == 0 ){
  GetTime( timeSta );
        lapack::Syevd( 'V', 'U', numCol, AMat.Data(), lda, eigValS.Data() );
  GetTime( timeEnd );
  iterMpirank0 = iterMpirank0 + 1;
  timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
      }
        MPI_Bcast(AMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm);
        MPI_Bcast(eigValS.Data(), numCol, MPI_DOUBLE, 0, mpi_comm);

  // Compute the correct eigenvectors C (but saved in AMat)
      // blas::Trsm( 'L', 'U', 'N', 'N', numCol, numCol, 1.0, 
      //    BMat.Data(), lda, AMat.Data(), lda );

      char LL = 'L';
  GetTime( timeSta );
      SCALAPACK(pdtrsm)( &LL, &UU, &NN, &NN,
          &numCol, &numCol, 
          &D_ONE,
          BMat.Data(), &I_ONE, &I_ONE, &desc_width3[0],
          AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0]);
  GetTime( timeEnd );
  iterTrsm = iterTrsm + 1;
  timeTrsm = timeTrsm + ( timeEnd - timeSta );

      // TODO Add Pocon and restart strategy, and try steepest descent first
    }


      if( numSet == 2 ){
        // Update the eigenvectors 
        // X <- X * C_X + W * C_W
        // blas::Gemm( 'N', 'N', height, width, width, 1.0,
        //    X.Data(), height, &AMat(0,0), lda,
        //    0.0, Xtemp.Data(), height );

  GetTime( timeSta );
        SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &width, 
            &D_ONE,
            X.Data(), &I_ONE, &I_ONE, &desc[0],
            AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0], 
            &D_ZERO,
            Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
            &contxt );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );
       


        // blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
        //    W.VecData(numLocked), height, &AMat(width,0), lda,
        //    1.0, Xtemp.Data(), height );

        ia = 1;
        ja = numLockedTotal + 1;
        ib = width + 1;
        jb = 1;
        ic = 1;
        jc = 1;
  GetTime( timeSta );
        SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &numActiveTotal, 
            &D_ONE,
            W.Data(), &I_ONE, &ja, &desc[0],
            AMat.Data(), &ib, &I_ONE, &desc_width3[0], 
            &D_ONE,
            Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
            &contxt );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

        // Save the result into X
        // lapack::Lacpy( 'A', height, width, Xtemp.Data(), height, 
        //    X.Data(), height );

        SCALAPACK(pdlacpy)(&AA, &height, &width, 
            Xtemp.Data(), &I_ONE, &I_ONE, &desc[0],
            X.Data(), &I_ONE, &I_ONE, &desc[0]);

        // P <- W
        // lapack::Lacpy( 'A', height, numActive, W.VecData(numLocked), 
        //    height, P.VecData(numLocked), height );
      
        ia = 1;
        ja = numLockedTotal + 1;
        ib = 1;
        jb = numLockedTotal + 1;
        SCALAPACK(pdlacpy)(&AA, &height, &numActiveTotal, 
            W.Data(), &I_ONE, &ja, &desc[0],
            P.Data(), &I_ONE, &jb, &desc[0]);
     
      }
      else{ //numSet == 3
        // Compute the conjugate direction
        // P <- W * C_W + P * C_P
        // blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
        //    W.VecData(numLocked), height, &AMat(width, 0), lda, 
        //    0.0, Xtemp.Data(), height );

        ia = 1;
        ja = numLockedTotal + 1;
        ib = width + 1;
        jb = 1;
        ic = 1;
        jc = 1;
  GetTime( timeSta );
        SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &numActiveTotal, 
            &D_ONE,
            W.Data(), &I_ONE, &ja, &desc[0],
            AMat.Data(), &ib, &I_ONE, &desc_width3[0], 
            &D_ZERO,
            Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
            &contxt );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );
        
       
        // blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
        //    P.VecData(numLocked), height, &AMat(width+numActive,0), lda,
        //    1.0, Xtemp.Data(), height );

        ia = 1;
        ja = numLockedTotal + 1;
        ib = width + numActiveTotal + 1;
        jb = 1;
        ic = 1;
        jc = 1;
  GetTime( timeSta );
        SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &numActiveTotal, 
            &D_ONE,
            P.Data(), &I_ONE, &ja, &desc[0],
            AMat.Data(), &ib, &I_ONE, &desc_width3[0], 
            &D_ONE,
            Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
            &contxt );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );



        // lapack::Lacpy( 'A', height, numActive, Xtemp.VecData(numLocked), 
        //    height, P.VecData(numLocked), height );

        ia = 1;
        ja = numLockedTotal + 1;
        ib = 1;
        jb = numLockedTotal + 1;
        SCALAPACK(pdlacpy)(&AA, &height, &numActiveTotal, 
            Xtemp.Data(), &I_ONE, &ja, &desc[0],
            P.Data(), &I_ONE, &jb, &desc[0]);
        
        // Update the eigenvectors
        // X <- X * C_X + P
        // blas::Gemm( 'N', 'N', height, width, width, 1.0, 
        //    X.Data(), height, &AMat(0,0), lda, 
        //    1.0, Xtemp.Data(), height );


  GetTime( timeSta );
        SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &width, 
            &D_ONE,
            X.Data(), &I_ONE, &I_ONE, &desc[0],
            AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0], 
            &D_ONE,
            Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
            &contxt );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

        // lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
        //    X.Data(), height );
      


        SCALAPACK(pdlacpy)(&AA, &height, &width, 
            Xtemp.Data(), &I_ONE, &I_ONE, &desc[0],
            X.Data(), &I_ONE, &I_ONE, &desc[0] );
        
        } // if ( numSet == 2 )

    
    // Update AX and AP
    if( numSet == 2 ){
      // AX <- AX * C_X + AW * C_W
      // blas::Gemm( 'N', 'N', height, width, width, 1.0,
      //    AX.Data(), height, &AMat(0,0), lda,
      //     0.0, Xtemp.Data(), height );

  GetTime( timeSta );
      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &width, 
          &D_ONE,
          AX.Data(), &I_ONE, &I_ONE, &desc[0],
          AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0], 
          &D_ZERO,
          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
          &contxt );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );


      // blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
      //    AW.VecData(numLocked), height, &AMat(width,0), lda,
      //     1.0, Xtemp.Data(), height );

      ia = 1;
      ja = numLockedTotal + 1;
      ib = width + 1;
      jb = 1;
      ic = 1;
      jc = 1;
  GetTime( timeSta );
      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &numActiveTotal, 
          &D_ONE,
          AW.Data(), &I_ONE, &ja, &desc[0],
          AMat.Data(), &ib, &I_ONE, &desc_width3[0], 
          &D_ONE,
          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
          &contxt );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

      // lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
      //    AX.Data(), height );

      SCALAPACK(pdlacpy)(&AA, &height, &width, 
          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0],
          AX.Data(), &I_ONE, &I_ONE, &desc[0]);
      
      // AP <- AW
      // lapack::Lacpy( 'A', height, numActive, AW.VecData(numLocked), height,
      //    AP.VecData(numLocked), height );
    
      ia = 1;
      ja = numLockedTotal + 1;
      ib = 1;
      jb = numLockedTotal + 1;
      SCALAPACK(pdlacpy)(&AA, &height, &numActiveTotal, 
          AW.Data(), &I_ONE, &ja, &desc[0],
          AP.Data(), &I_ONE, &jb, &desc[0]);
  
    }
    else{
      // AP <- AW * C_W + A_P * C_P
      // blas::Gemm( 'N', 'N', height, width, numActive, 1.0, 
      //    AW.VecData(numLocked), height, &AMat(width,0), lda,
      //    0.0, Xtemp.Data(), height );
    
      ia = 1;
      ja = numLockedTotal + 1;
      ib = width + 1;
      jb = 1;
      ic = 1;
      jc = 1;
  GetTime( timeSta );
      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &numActiveTotal, 
          &D_ONE,
          AW.Data(), &I_ONE, &ja, &desc[0],
          AMat.Data(), &ib, &I_ONE, &desc_width3[0], 
          &D_ZERO,
          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
          &contxt );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

      // blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
      //    AP.VecData(numLocked), height, &AMat(width+numActive, 0), lda,
      //    1.0, Xtemp.Data(), height );

      ia = 1;
      ja = numLockedTotal + 1;
      ib = width + numActiveTotal + 1;
      jb = 1;
      ic = 1;
      jc = 1;
  GetTime( timeSta );
      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &numActiveTotal, 
          &D_ONE,
          AP.Data(), &I_ONE, &ja, &desc[0],
          AMat.Data(), &ib, &I_ONE, &desc_width3[0], 
          &D_ONE,
          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
          &contxt );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );


      // lapack::Lacpy( 'A', height, numActive, Xtemp.VecData(numLocked), 
      //    height, AP.VecData(numLocked), height );

      ia = 1;
      ja = numLockedTotal + 1;
      ib = 1;
      jb = numLockedTotal + 1;
      SCALAPACK(pdlacpy)(&AA, &height, &numActiveTotal, 
          Xtemp.Data(), &I_ONE, &ja, &desc[0],
          AP.Data(), &I_ONE, &jb, &desc[0]);

      // AX <- AX * C_X + AP
      // blas::Gemm( 'N', 'N', height, width, width, 1.0,
      //    AX.Data(), height, &AMat(0,0), lda,
      //    1.0, Xtemp.Data(), height );

  GetTime( timeSta );
      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &width, 
          &D_ONE,
          AX.Data(), &I_ONE, &I_ONE, &desc[0],
          AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0], 
          &D_ONE,
          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
          &contxt );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

      // lapack::Lacpy( 'A', height, width, Xtemp.Data(), height, 
      //    AX.Data(), height );
   
      SCALAPACK(pdlacpy)(&AA, &height, &width, 
          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0],
          AX.Data(), &I_ONE, &I_ONE, &desc[0]);
  
    } // if ( numSet == 2 )



#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "numLocked = " << numLocked << std::endl;
    statusOFS << "eigValS   = " << eigValS << std::endl;
#endif



  } // for (iter) end for main loop




  // *********************************************************************
  // Post processing
  // *********************************************************************

  // Obtain the eigenvalues and eigenvectors
  // XTX should now contain the matrix X' * (AX), and X is an
  // orthonormal set
  // FIXME
  // lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );

  char VV = 'V';
  
  if ( mpirank == 0 ){
  GetTime( timeSta );
    lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );
  GetTime( timeEnd );
  iterMpirank0 = iterMpirank0 + 1;
  timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
  }

    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
    MPI_Bcast(eigValS.Data(), width, MPI_DOUBLE, 0, mpi_comm);
  
  // X <- X*C
  // blas::Gemm( 'N', 'N', height, width, width, 1.0, X.Data(),
  //    height, XTX.Data(), width, 0.0, Xtemp.Data(), height );

  GetTime( timeSta );
  SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &width, 
      &D_ONE,
      X.Data(), &I_ONE, &I_ONE, &desc[0],
      XTX.Data(), &I_ONE, &I_ONE, &desc_width[0], 
      &D_ZERO,
      Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
      &contxt );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

  // lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
  //    X.Data(), height );

  SCALAPACK(pdlacpy)(&AA, &height, &width, 
      Xtemp.Data(), &I_ONE, &I_ONE, &desc[0],
      X.Data(), &I_ONE, &I_ONE, &desc[0]);

#if ( _DEBUGlevel_ >= 2 )

  // blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(), 
  //    height, X.Data(), height, 0.0, XTX.Data(), width );

  GetTime( timeSta );
  SCALAPACK(pdgemm)(&TT, &NN, &width, &width, &height, 
      &D_ONE,
      X.Data(), &I_ONE, &I_ONE, &desc[0],
      X.Data(), &I_ONE, &I_ONE, &desc[0], 
      &D_ZERO,
      XTX.Data(), &I_ONE, &I_ONE, &desc_width[0], 
      &contxt );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );

  statusOFS << "After the LOBPCG, XTX = " << XTX << std::endl;
#endif

  // Save the eigenvalues and eigenvectors back to the eigensolver data
  // structure

  eigVal_ = DblNumVec( width, true, eigValS.Data() );
  resVal_ = resNorm;

  //lapack::Lacpy( 'A', height, width, X.Data(), height, 
  //    psiPtr_->Wavefun().Data(), height );

  SCALAPACK(pdlacpy)(&AA, &height, &width, 
      X.Data(), &I_ONE, &I_ONE, &desc[0],
      psiPtr_->Wavefun().Data(), &I_ONE, &I_ONE, &desc[0]);
  
  if( isConverged ){
    statusOFS << std::endl << "After " << iter 
      << " iterations, LOBPCG has converged."  << std::endl
      << "The maximum norm of the residual is " 
      << resMax << std::endl << std::endl;
  }
  else{
    statusOFS << std::endl << "After " << iter 
      << " iterations, LOBPCG did not converge. " << std::endl
      << "The maximum norm of the residual is " 
      << resMax << std::endl << std::endl;
  }

  statusOFS << "Time for iterGemmT = " << iterGemmT << "  timeGemmT = " << timeGemmT << std::endl;
  statusOFS << "Time for iterGemmN = " << iterGemmN << "  timeGemmN = " << timeGemmN << std::endl;
  statusOFS << "Time for iterTrsm = " << iterTrsm << "  timeTrsm = " << timeTrsm << std::endl;
  statusOFS << "Time for iterSpinor = " << iterSpinor << "  timeSpinor = " << timeSpinor << std::endl;
  statusOFS << "Time for iterMpirank0 = " << iterMpirank0 << "  timeMpirank0 = " << timeMpirank0 << std::endl;

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method EigenSolver::LOBPCGSolveReal  ----- 








// huwei

void
EigenSolver::LOBPCGSolveReal2	( 
      Int          numEig,
      Int          eigMaxIter,
      Real         eigTolerance)
{
#ifndef _RELEASE_
  PushCallStack("EigenSolver::LOBPCGSolveReal2");
#endif

  // *********************************************************************
  // Initialization
  // *********************************************************************
  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  MPI_Barrier(mpi_comm);
  Int mpirank;  MPI_Comm_rank(mpi_comm, &mpirank);
  Int mpisize;  MPI_Comm_size(mpi_comm, &mpisize);

  Int ntot = psiPtr_->NumGridTotal();
  Int ncom = psiPtr_->NumComponent();
  Int noccLocal = psiPtr_->NumState();
  Int noccTotal = psiPtr_->NumStateTotal();

  Int height = ntot * ncom;
  Int width = noccTotal;
  Int lda = 3 * width;

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }

  if(mpirank == (mpisize - 1)){
    heightLocal = heightBlocksize + height % mpisize;
  }

  if( widthLocal != noccLocal ){
    throw std::logic_error("widthLocal != noccLocal.");
  }

  Real timeSta, timeEnd;
  Real timeGemmT = 0.0;
  Real timeGemmN = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeSpinor = 0.0;
  Real timeTrsm = 0.0;
  Real timeMpirank0 = 0.0;
  Int  iterGemmT = 0;
  Int  iterGemmN = 0;
  Int  iterAlltoallv = 0;
  Int  iterSpinor = 0;
  Int  iterTrsm = 0;
  Int  iterMpirank0 = 0;

  if( numEig > width ){
    std::ostringstream msg;
    msg 
      << "Number of eigenvalues requested  = " << numEig << std::endl
      << "which is larger than the number of columns in psi = " << width << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }

  // For Alltoall
  double sendbuf[height*widthLocal]; 
  double recvbuf[heightLocal*width];
  int sendcounts[mpisize];
  int recvcounts[mpisize];
  int senddispls[mpisize];
  int recvdispls[mpisize];
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    if( k < (mpisize - 1)){
      sendcounts[k] = heightBlocksize * widthLocal;
    }
    else {
      sendcounts[mpisize - 1] = (heightBlocksize + (height % mpisize)) * widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  if((height % heightBlocksize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if((i / heightBlocksize) < (mpisize - 1)){
          sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
        }
        else {
          sendk(i, j) = senddispls[mpisize -1] + j * (heightBlocksize + height % heightBlocksize) 
            + (i - (mpisize - 1) * heightBlocksize) % (heightBlocksize + height % heightBlocksize);
        }
      }
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvk(i, j) = recvdispls[j % mpisize] + (j / mpisize) * heightLocal + i;
    }
  }
  // end For Alltoall

  // S = ( X | W | P ) is a triplet used for LOBPCG.  
  // W is the preconditioned residual
  // DblNumMat  S( height, 3*widthLocal ), AS( height, 3*widthLocal ); 
  DblNumMat  S( heightLocal, 3*width ), AS( heightLocal, 3*width ); 
  // AMat = S' * (AS),  BMat = S' * S
  // 
  // AMat = (X'*AX   X'*AW   X'*AP)
  //      = (  *     W'*AW   W'*AP)
  //      = (  *       *     P'*AP)
  //
  // BMat = (X'*X   X'*W   X'*P)
  //      = (  *    W'*W   W'*P)
  //      = (  *      *    P'*P)
  //
  DblNumMat  AMat( 3*width, 3*width ), BMat( 3*width, 3*width );
  DblNumMat  AMatT1( 3*width, 3*width );
  // AMatSave and BMatSave are used for restart
  // Temporary buffer array.
  // The unpreconditioned residual will also be saved in Xtemp
  DblNumMat  XTX( width, width );
  DblNumMat  XTXtemp( width, width );
  DblNumMat  XTXtemp1( width, width );

  DblNumMat  Xtemp( heightLocal, width );

  // rexNorm Grobal matrix  similar to numEig 
  DblNumVec  resNormLocal ( width ); 
  SetValue( resNormLocal, 0.0 );
  DblNumVec  resNorm( width );
  SetValue( resNorm, 0.0 );
  Real       resMaxLocal, resMinLocal;
  Real       resMax, resMin;

  // For convenience
  DblNumMat  X( heightLocal, width, false, S.VecData(0) );
  DblNumMat  W( heightLocal, width, false, S.VecData(width) );
  DblNumMat  P( heightLocal, width, false, S.VecData(2*width) );
  DblNumMat AX( heightLocal, width, false, AS.VecData(0) );
  DblNumMat AW( heightLocal, width, false, AS.VecData(width) );
  DblNumMat AP( heightLocal, width, false, AS.VecData(2*width) );

  DblNumMat  Xcol( height, widthLocal );
  DblNumMat  Wcol( height, widthLocal );
  DblNumMat  Pcol( height, widthLocal );
  DblNumMat AXcol( height, widthLocal );
  DblNumMat AWcol( height, widthLocal );
  DblNumMat APcol( height, widthLocal );

  //Int info;
  bool isRestart = false;
  // numSet = 2    : Steepest descent (Davidson), only use (X | W)
  //        = 3    : Conjugate gradient, use all the triplet (X | W | P)
  Int numSet = 2;

  // numLocked is the number of converged vectors
  Int numLockedLocal = 0, numLockedSaveLocal = 0;
  Int numLockedTotal = 0, numLockedSaveTotal = 0; 
  Int numLockedSave = 0;
  Int numActiveLocal = 0;
  Int numActiveTotal = 0;
  // Real lockTolerance = std::min( eigTolerance_, 1e-2 );

  const Int numLocked = 0;  // Never perform locking in this version
  const Int numActive = width;

  bool isConverged = false;

  // Initialization
  SetValue( S, 0.0 );
  SetValue( AS, 0.0 );

  DblNumVec  eigValS(lda);
  SetValue( eigValS, 0.0 );
  DblNumVec  sigma2(lda);
  DblNumVec  invsigma(lda);
  SetValue( sigma2, 0.0 );
  SetValue( invsigma, 0.0 );

  // Initialize X by the data in psi
  lapack::Lacpy( 'A', height, widthLocal, psiPtr_->Wavefun().Data(), height, 
      Xcol.Data(), height );

  GetTime( timeSta );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      sendbuf[sendk(i, j)] = Xcol(i, j); 
    }
  }
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      X(i, j) = recvbuf[recvk(i, j)];
    }
  }
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  // *********************************************************************
  // Main loop
  // *********************************************************************

  // Orthogonalization through Cholesky factorization
  GetTime( timeSta );
  blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, X.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
  SetValue( XTX, 0.0 );
  MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );

  if ( mpirank == 0) {
    GetTime( timeSta );
    lapack::Potrf( 'U', width, XTX.Data(), width );
    GetTime( timeEnd );
    iterMpirank0 = iterMpirank0 + 1;
    timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
  }
  MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);

  // X <- X * U^{-1} is orthogonal
  GetTime( timeSta );
  blas::Trsm( 'R', 'U', 'N', 'N', heightLocal, width, 1.0, XTX.Data(), width, 
      X.Data(), heightLocal );
  GetTime( timeEnd );
  iterTrsm = iterTrsm + 1;
  timeTrsm = timeTrsm + ( timeEnd - timeSta );

  GetTime( timeSta );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvbuf[recvk(i, j)] = X(i, j);
    }
  }
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      Xcol(i, j) = sendbuf[sendk(i, j)]; 
    }
  }
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  // Applying the Hamiltonian matrix
  {
    GetTime( timeSta );
    Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, noccLocal, false, Xcol.Data());
    NumTns<Scalar> tnsTemp(ntot, ncom, noccLocal, false, AXcol.Data());

    hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
    GetTime( timeEnd );
    iterSpinor = iterSpinor + 1;
    timeSpinor = timeSpinor + ( timeEnd - timeSta );
  }

  GetTime( timeSta );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      sendbuf[sendk(i, j)] = Xcol(i, j); 
    }
  }
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      X(i, j) = recvbuf[recvk(i, j)];
    }
  }
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  GetTime( timeSta );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      sendbuf[sendk(i, j)] = AXcol(i, j); 
    }
  }
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      AX(i, j) = recvbuf[recvk(i, j)];
    }
  }
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );


  // Start the main loop
  Int iter;
  for( iter = 1; iter < eigMaxIter_; iter++ ){
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "iter = " << iter << std::endl;
#endif

    if( iter == 1 || isRestart == true )
      numSet = 2;
    else
      numSet = 3;

    SetValue( AMat, 0.0 );
    SetValue( BMat, 0.0 );

    // XTX <- X' * (AX)
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
        heightLocal, AX.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    lapack::Lacpy( 'A', width, width, XTX.Data(), width, AMat.Data(), lda );

    // Compute the residual.
    // R <- AX - X*(X'*AX)
    lapack::Lacpy( 'A', heightLocal, width, AX.Data(), heightLocal, Xtemp.Data(), heightLocal );

    GetTime( timeSta );
    blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
        X.Data(), heightLocal, AMat.Data(), lda, 1.0, Xtemp.Data(), heightLocal );
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 1;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );

    // Compute the norm of the residual
    SetValue( resNormLocal, 0.0 );
    for( Int k = 0; k < width; k++ ){
      resNormLocal(k) = Energy(DblNumVec(heightLocal, false, Xtemp.VecData(k)));
    }

    SetValue( resNorm, 0.0 );
    MPI_Allreduce( resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE, 
        MPI_SUM, mpi_comm );

    if ( mpirank == 0 ){
      GetTime( timeSta );
      for( Int k = 0; k < width; k++ ){
        resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( XTX(k,k) ) );
      }
      GetTime( timeEnd );
      iterMpirank0 = iterMpirank0 + 1;
      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
    }
    MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm);

    resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
    resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "resNorm = " << resNorm << std::endl;
    statusOFS << "maxRes  = " << resMax  << std::endl;
    statusOFS << "minRes  = " << resMin  << std::endl;
#endif

    if( resMax < eigTolerance_ ){
      isConverged = true;;
      break;
    }

    numActiveTotal = width - numLockedTotal;
    numActiveLocal = widthLocal - numLockedLocal;

    // If the number of locked vectors goes down, perform steppest
    // descent rather than conjugate gradient
    // if( numLockedTotal < numLockedSaveTotal )
    //  numSet = 2;

    // Compute the preconditioned residual W = T*R.
    // The residual is saved in Xtemp

    // Convert from row format to column format.
    // MPI_Alltoallv
    // Only convert Xtemp here

    GetTime( timeSta );
    for( Int j = 0; j < width; j++ ){ 
      for( Int i = 0; i < heightLocal; i++ ){
        recvbuf[recvk(i, j)] = Xtemp(i, j);
      }
    }
    MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
        &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        Xcol(i, j) = sendbuf[sendk(i, j)]; 
      }
    }
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

    {
      GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, Xcol.VecData(numLockedLocal));
      NumTns<Scalar> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));

      SetValue( tnsTemp, 0.0 );
      spnTemp.AddTeterPrecond( fftPtr_, tnsTemp );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

    Real norm = 0.0; 
    // Normalize the preconditioned residual
    for( Int k = numLockedLocal; k < widthLocal; k++ ){
      norm = Energy(DblNumVec(height, false, Wcol.VecData(k)));
      norm = std::sqrt( norm );
      blas::Scal( height, 1.0 / norm, Wcol.VecData(k), 1 );
    }

    // Normalize the conjugate direction
    //Real normLocal = 0.0; 
    //if( numSet == 3 ){
    //  for( Int k = numLockedLocal; k < width; k++ ){
    //    normLocal = Energy(DblNumVec(heightLocal, false, P.VecData(k)));
    //    norm = 0.0; 
    //    MPI_Allreduce( &normLocal, &norm, 1, MPI_DOUBLE, MPI_SUM, mpi_comm );
    //    norm = std::sqrt( norm );
    //    blas::Scal( heightLocal, 1.0 / norm, P.VecData(k), 1 );
    //    blas::Scal( heightLocal, 1.0 / norm, AP.VecData(k), 1 );
    //  }
    //} 
    
    // Normalize the conjugate direction
    Real normPLocal[width]; 
    Real normP[width]; 
    if( numSet == 3 ){
      for( Int k = numLockedLocal; k < width; k++ ){
        normPLocal[k] = Energy(DblNumVec(heightLocal, false, P.VecData(k)));
        normP[k] = 0.0;
      }
      MPI_Allreduce( &normPLocal[0], &normP[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      for( Int k = numLockedLocal; k < width; k++ ){
        norm = std::sqrt( normP[k] );
        blas::Scal( heightLocal, 1.0 / norm, P.VecData(k), 1 );
        blas::Scal( heightLocal, 1.0 / norm, AP.VecData(k), 1 );
      }
    } 

    // Compute AMat
    // Compute AW = A*W
    {
      GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));
      NumTns<Scalar> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, AWcol.VecData(numLockedLocal));

      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

    // huwei
    // Convert from column format to row format
    // MPI_Alltoallv
    // Only convert W and AW

    GetTime( timeSta );
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendbuf[sendk(i, j)] = Wcol(i, j); 
      }
      }
    MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
    for( Int j = 0; j < width; j++ ){ 
      for( Int i = 0; i < heightLocal; i++ ){
        W(i, j) = recvbuf[recvk(i, j)];
      }
    }
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

    GetTime( timeSta );
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendbuf[sendk(i, j)] = AWcol(i, j); 
      }
    }
    MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
    for( Int j = 0; j < width; j++ ){ 
      for( Int i = 0; i < heightLocal; i++ ){
        AW(i, j) = recvbuf[recvk(i, j)];
      }
    }
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

    // Compute X' * (AW)
    // Instead of saving the block at &AMat(0,width+numLocked), the data
    // is saved at &AMat(0,width) to guarantee a continuous data
    // arrangement of AMat.  The same treatment applies to the blocks
    // below in both AMat and BMat.
    GetTime( timeSta );
    // blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0, X.Data(),
    //    heightLocal, AW.VecData(numLocked), heightLocal, 
    //    0.0, &AMat(0,width), lda );
    blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0, X.Data(),
        heightLocal, AW.VecData(numLocked), heightLocal, 
        0.0, XTXtemp1.Data(), width );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(0,width), lda );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    // Compute W' * (AW)
    GetTime( timeSta );
    //blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
    //    W.VecData(numLocked), heightLocal, AW.VecData(numLocked), heightLocal, 
    //    0.0, &AMat(width, width), lda );
    blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
        W.VecData(numLocked), heightLocal, AW.VecData(numLocked), heightLocal, 
        0.0, XTXtemp1.Data(), width );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width,width), lda );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    if( numSet == 3 ){

      // Compute X' * (AP)
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
      //    X.Data(), heightLocal, AP.VecData(numLocked), heightLocal, 
      //   0.0, &AMat(0, width+numActive), lda );
      blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
          X.Data(), heightLocal, AP.VecData(numLocked), heightLocal, 
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(0, width+numActive), lda );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      // Compute W' * (AP)
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
      //    W.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal, 
      //    0.0, &AMat(width, width+numActive), lda );
      blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
          W.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal, 
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width, width+numActive), lda );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      // Compute P' * (AP)
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
      //    P.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal, 
      //    0.0, &AMat(width+numActive, width+numActive), lda );
      blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
          P.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal, 
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width+numActive, width+numActive), lda );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

    }


    // Compute BMat (overlap matrix)

    // Compute X'*X
    GetTime( timeSta );
    //blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, 
    //    X.Data(), heightLocal, X.Data(), heightLocal, 
    //    0.0, &BMat(0,0), lda );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, 
        X.Data(), heightLocal, X.Data(), heightLocal, 
        0.0, XTXtemp1.Data(), width );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0,0), lda );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    // Compute X'*W
    GetTime( timeSta );
    //blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
    //    X.Data(), height, W.VecData(numLocked), height,
    //    0.0, &BMat(0,width), lda );
    blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
        X.Data(), heightLocal, W.VecData(numLocked), heightLocal,
        0.0, XTXtemp1.Data(), width );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0,width), lda );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    // Compute W'*W
    GetTime( timeSta );
    //blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
    //    W.VecData(numLocked), height, W.VecData(numLocked), height,
    //    0.0, &BMat(width, width), lda );
    blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
        W.VecData(numLocked), heightLocal, W.VecData(numLocked), heightLocal,
        0.0, XTXtemp1.Data(), width );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width, width), lda );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );


    if( numSet == 3 ){
      // Compute X'*P
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
      //    X.Data(), heightLocal, P.VecData(numLocked), heightLocal, 
      //    0.0, &BMat(0, width+numActive), lda );
      blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
          X.Data(), heightLocal, P.VecData(numLocked), heightLocal, 
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0, width+numActive), lda );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      // Compute W'*P
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
      //    W.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal,
      //    0.0, &BMat(width, width+numActive), lda );
      blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
          W.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal,
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width, width+numActive), lda );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      // Compute P'*P
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
      //    P.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal,
      //    0.0, &BMat(width+numActive, width+numActive), lda );
      blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
          P.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal,
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width+numActive, width+numActive), lda );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

    } // if( numSet == 3 )

#if ( _DEBUGlevel_ >= 2 )
    {
      DblNumMat WTW( width, width );
      lapack::Lacpy( 'A', width, width, &BMat(width, width), lda,
          WTW.Data(), width );
      statusOFS << "W'*W = " << WTW << std::endl;
      if( numSet == 3 )
      {
        DblNumMat PTP( width, width );
        lapack::Lacpy( 'A', width, width, &BMat(width+numActive, width+numActive), 
            lda, PTP.Data(), width );
        statusOFS << "P'*P = " << PTP << std::endl;
      }
    }
#endif

    // Rayleigh-Ritz procedure
    // AMat * C = BMat * C * Lambda
    // Assuming the dimension (needed) for C is width * width, then
    //     ( C_X )
    //     ( --- )
    // C = ( C_W )
    //     ( --- )
    //     ( C_P )
    //
    Int numCol;
    if( numSet == 3 ){
      // Conjugate gradient
      numCol = width + 2 * numActiveTotal;
    }
    else{
      numCol = width + numActiveTotal;
    }

    // Solve the generalized eigenvalue problem with thresholding
    if(1){  

      if ( mpirank == 0 ) {

        // Symmetrize A and B first.  This is important.
        for( Int j = 0; j < numCol; j++ ){
          for( Int i = j+1; i < numCol; i++ ){
            AMat(i,j) = AMat(j,i);
            BMat(i,j) = BMat(j,i);
          }
        }

        GetTime( timeSta );
        lapack::Syevd( 'V', 'U', numCol, BMat.Data(), lda, sigma2.Data() );
        GetTime( timeEnd );
        iterMpirank0 = iterMpirank0 + 1;
        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

        Int numKeep = 0;
        for( Int i = numCol-1; i>=0; i-- ){
          if( sigma2(i) / sigma2(numCol-1) >  1e-12 )
            numKeep++;
          else
            break;
        }

#if ( _DEBUGlevel_ >= 1 )
        statusOFS << "sigma2 = " << sigma2 << std::endl;
#endif

#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "sigma2(0)        = " << sigma2(0) << std::endl;
        statusOFS << "sigma2(numCol-1) = " << sigma2(numCol-1) << std::endl;
        statusOFS << "numKeep          = " << numKeep << std::endl;
#endif

        for( Int i = 0; i < numKeep; i++ ){
          invsigma(i) = 1.0 / std::sqrt( sigma2(i+numCol-numKeep) );
        }

        if( numKeep < width ){
          std::ostringstream msg;
          msg 
            << "width   = " << width << std::endl
            << "numKeep =  " << numKeep << std::endl
            << "there are not enough number of columns." << std::endl;
          throw std::runtime_error( msg.str().c_str() );
        }

        SetValue( AMatT1, 0.0 );
        // Evaluate S^{-1/2} (U^T A U) S^{-1/2}
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', numCol, numKeep, numCol, 1.0,
            AMat.Data(), lda, BMat.VecData(numCol-numKeep), lda,
            0.0, AMatT1.Data(), lda );
        GetTime( timeEnd );
        iterMpirank0 = iterMpirank0 + 1;
        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

        GetTime( timeSta );
        blas::Gemm( 'T', 'N', numKeep, numKeep, numCol, 1.0,
            BMat.VecData(numCol-numKeep), lda, AMatT1.Data(), lda, 
            0.0, AMat.Data(), lda );
        GetTime( timeEnd );
        iterMpirank0 = iterMpirank0 + 1;
        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

        GetTime( timeSta );
        for( Int j = 0; j < numKeep; j++ ){
          for( Int i = 0; i < numKeep; i++ ){
            AMat(i,j) *= invsigma(i)*invsigma(j);
          }
        }
        GetTime( timeEnd );
        iterMpirank0 = iterMpirank0 + 1;
        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

        // Solve the standard eigenvalue problem
        GetTime( timeSta );
        lapack::Syevd( 'V', 'U', numKeep, AMat.Data(), lda,
            eigValS.Data() );
        GetTime( timeEnd );
        iterMpirank0 = iterMpirank0 + 1;
        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

        // Compute the correct eigenvectors and save them in AMat
        for( Int j = 0; j < numKeep; j++ ){
          for( Int i = 0; i < numKeep; i++ ){
            AMat(i,j) *= invsigma(i);
          }
        }

        GetTime( timeSta );
        blas::Gemm( 'N', 'N', numCol, numKeep, numKeep, 1.0,
            BMat.VecData(numCol-numKeep), lda, AMat.Data(), lda,
            0.0, AMatT1.Data(), lda );
        GetTime( timeEnd );
        iterMpirank0 = iterMpirank0 + 1;
        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

        lapack::Lacpy( 'A', numCol, numKeep, AMatT1.Data(), lda, 
            AMat.Data(), lda );

      } // mpirank ==0

      MPI_Bcast(AMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm);
      MPI_Bcast(BMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm);
      MPI_Bcast(eigValS.Data(), lda, MPI_DOUBLE, 0, mpi_comm);


    } // if(1)

    if( numSet == 2 ){

      // Update the eigenvectors 
      // X <- X * C_X + W * C_W
      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0,
          X.Data(), heightLocal, &AMat(0,0), lda,
          0.0, Xtemp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, numActive, 1.0,
          W.VecData(numLocked), heightLocal, &AMat(width,0), lda,
          1.0, Xtemp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      // Save the result into X
      lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal, 
          X.Data(), heightLocal );

      // P <- W
      lapack::Lacpy( 'A', heightLocal, numActive, W.VecData(numLocked), 
          heightLocal, P.VecData(numLocked), heightLocal );
    } 
    else{ //numSet == 3
      // Compute the conjugate direction
      // P <- W * C_W + P * C_P
      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, numActive, 1.0,
          W.VecData(numLocked), heightLocal, &AMat(width, 0), lda, 
          0.0, Xtemp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, numActive, 1.0,
          P.VecData(numLocked), heightLocal, &AMat(width+numActive,0), lda,
          1.0, Xtemp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      lapack::Lacpy( 'A', heightLocal, numActive, Xtemp.VecData(numLocked), 
          heightLocal, P.VecData(numLocked), heightLocal );

      // Update the eigenvectors
      // X <- X * C_X + P
      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, 
          X.Data(), heightLocal, &AMat(0,0), lda, 
          1.0, Xtemp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,
          X.Data(), heightLocal );

    } // if ( numSet == 2 )


    // Update AX and AP
    if( numSet == 2 ){
      // AX <- AX * C_X + AW * C_W
      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0,
          AX.Data(), heightLocal, &AMat(0,0), lda,
          0.0, Xtemp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, numActive, 1.0,
          AW.VecData(numLocked), heightLocal, &AMat(width,0), lda,
          1.0, Xtemp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,
          AX.Data(), heightLocal );

      // AP <- AW
      lapack::Lacpy( 'A', heightLocal, numActive, AW.VecData(numLocked), heightLocal,
          AP.VecData(numLocked), heightLocal );

    }
    else{
      // AP <- AW * C_W + A_P * C_P
      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, numActive, 1.0, 
          AW.VecData(numLocked), heightLocal, &AMat(width,0), lda,
          0.0, Xtemp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, numActive, 1.0,
          AP.VecData(numLocked), heightLocal, &AMat(width+numActive, 0), lda,
          1.0, Xtemp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      lapack::Lacpy( 'A', heightLocal, numActive, Xtemp.VecData(numLocked),
          heightLocal, AP.VecData(numLocked), heightLocal );

      // AX <- AX * C_X + AP
      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0,
          AX.Data(), heightLocal, &AMat(0,0), lda,
          1.0, Xtemp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal, 
          AX.Data(), heightLocal );

    } // if ( numSet == 2 )

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "numLocked = " << numLocked << std::endl;
    statusOFS << "eigValS   = " << eigValS << std::endl;
#endif

  } // for (iter) end for main loop


  // *********************************************************************
  // Post processing
  // *********************************************************************

  // Obtain the eigenvalues and eigenvectors
  // XTX should now contain the matrix X' * (AX), and X is an
  // orthonormal set

  if ( mpirank == 0 ){
    GetTime( timeSta );
    lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );
    GetTime( timeEnd );
    iterMpirank0 = iterMpirank0 + 1;
    timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
  }

  MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
  MPI_Bcast(eigValS.Data(), width, MPI_DOUBLE, 0, mpi_comm);

  GetTime( timeSta );
  // X <- X*C
  blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, X.Data(),
      heightLocal, XTX.Data(), width, 0.0, Xtemp.Data(), heightLocal );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

  lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,
      X.Data(), heightLocal );

#if ( _DEBUGlevel_ >= 2 )

  GetTime( timeSta );
  blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, X.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
  SetValue( XTX, 0.0 );
  MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );

  statusOFS << "After the LOBPCG, XTX = " << XTX << std::endl;

#endif

  // Save the eigenvalues and eigenvectors back to the eigensolver data
  // structure

  eigVal_ = DblNumVec( width, true, eigValS.Data() );
  resVal_ = resNorm;

  GetTime( timeSta );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvbuf[recvk(i, j)] = X(i, j);
    }
  }
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      Xcol(i, j) = sendbuf[sendk(i, j)]; 
    }
  }
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  lapack::Lacpy( 'A', height, widthLocal, Xcol.Data(), height, 
      psiPtr_->Wavefun().Data(), height );

  if( isConverged ){
    statusOFS << std::endl << "After " << iter 
      << " iterations, LOBPCG has converged."  << std::endl
      << "The maximum norm of the residual is " 
      << resMax << std::endl << std::endl;
  }
  else{
    statusOFS << std::endl << "After " << iter 
      << " iterations, LOBPCG did not converge. " << std::endl
      << "The maximum norm of the residual is " 
      << resMax << std::endl << std::endl;
  }

  statusOFS << "Time for iterGemmT = " << iterGemmT << "  timeGemmT = " << timeGemmT << std::endl;
  statusOFS << "Time for iterGemmN = " << iterGemmN << "  timeGemmN = " << timeGemmN << std::endl;
  statusOFS << "Time for iterAlltoallv = " << iterAlltoallv << "  timeAlltoallv = " << timeAlltoallv << std::endl;
  statusOFS << "Time for iterSpinor = " << iterSpinor << "  timeSpinor = " << timeSpinor << std::endl;
  statusOFS << "Time for iterTrsm = " << iterTrsm << "  timeTrsm = " << timeTrsm << std::endl;
  statusOFS << "Time for iterMpirank0 = " << iterMpirank0 << "  timeMpirank0 = " << timeMpirank0 << std::endl;

#ifndef _RELEASE_
  PopCallStack();
#endif

  return ;
} 		// -----  end of method EigenSolver::LOBPCGSolveReal2  ----- 

} // namespace dgdft
