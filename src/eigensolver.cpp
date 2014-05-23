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
/// @date 2012-11-20
#include	"eigensolver.hpp"
#include  "utility.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"

#define _DEBUGlevel_ 0

namespace dgdft{

EigenSolver::EigenSolver() {}

EigenSolver::~EigenSolver() {}

void EigenSolver::Setup(
			const esdf::ESDFInputParam& esdfParam,
			Hamiltonian& ham,
			Spinor& psi,
			Fourier& fft ) {
#ifndef _RELEASE_
	PushCallStack("EigenSolver::SEtup");
#endif  // ifndef _RELEASE_
	hamPtr_ = &ham;
	psiPtr_ = &psi;
	fftPtr_ = &fft;

	eigMaxIter_    = esdfParam.eigMaxIter;
	eigToleranceSave_  = esdfParam.eigTolerance;
  eigTolerance_      = esdfParam.eigTolerance;


	eigVal_.Resize(psiPtr_->NumState());  SetValue(eigVal_, 0.0);
	resVal_.Resize(psiPtr_->NumState());  SetValue(resVal_, 0.0);
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
  Int nocc = psiPtr_->NumState();

	if( (x->size * x->num_vectors) != ntot*ncom*nocc ) {
		throw std::logic_error("Vector size does not match.");
	}

  Spinor psitemp(fftPtr_->domain, ncom, nocc, false, x->data);
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

  SerialSetupInterpreter ( &ii );
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
  Int nocc = psiPtr_->NumState();

  Int height = ntot * ncom, width = nocc;
  Int lda = 3 * width;

  if( numEig > width ){
    std::ostringstream msg;
    msg 
      << "Number of eigenvalues requested  = " << numEig << std::endl
      << "which is larger than the number of columns in psi = " << width << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }



  // S = ( X | W | P ) is a triplet used for LOBPCG.  
  // W is the preconditioned residual
  DblNumMat  S( height, 3*width ), AS( height, 3*width ); 
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
  // AMatSave and BMatSave are used for restart
  DblNumMat  AMatSave( 3*width, 3*width ), BMatSave( 3*width, 3*width );
  // Temporary buffer array.
  // The unpreconditioned residual will also be saved in Xtemp
  DblNumMat  XTX( width, width ), Xtemp( height, width );
  DblNumVec  resNorm( width );
  Real       resMax, resMin;

  // For convenience
  DblNumMat  X( height, width, false, S.VecData(0) );
  DblNumMat  W( height, width, false, S.VecData(width) );
  DblNumMat  P( height, width, false, S.VecData(2*width) );
  DblNumMat AX( height, width, false, AS.VecData(0) );
  DblNumMat AW( height, width, false, AS.VecData(width) );
  DblNumMat AP( height, width, false, AS.VecData(2*width) );

  
  Int info;
  bool isRestart = false;
  // numSet = 2    : Steepest descent (Davidson), only use (X | W)
  //        = 3    : Conjugate gradient, use all the triplet (X | W | P)
  Int numSet = 2;

  // numLocked is the number of converged vectors
  Int numLocked = 0, numLockedSave = 0; 
  // numActive = width - numLocked
  Int numActive;

  Real lockTolerance = std::min( eigTolerance, 1e-2 );
  bool isConverged = false;

  // Initialization
  SetValue( S, 0.0 );
  SetValue( AS, 0.0 );

  DblNumVec  eigValS(lda);
  SetValue( eigValS, 0.0 );

  // Initialize X by the data in psi
  lapack::Lacpy( 'A', height, width, psiPtr_->Wavefun().Data(), height, 
      X.Data(), height );

  // *********************************************************************
  // Main loop
  // *********************************************************************

  // Orthogonalization through Cholesky factorization
  blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(), 
     height, X.Data(), height, 0.0, XTX.Data(), width );

  // X'*X=U'*U
  lapack::Potrf( 'U', width, XTX.Data(), width );

  // X <- X * U^{-1} is orthogonal
  blas::Trsm( 'R', 'U', 'N', 'N', height, width, 1.0, XTX.Data(), width, 
      X.Data(), height );


  // Applying the Hamiltonian matrix
  {
    Spinor spnTemp(fftPtr_->domain, ncom, width, false, X.Data());
    NumTns<Scalar> tnsTemp(ntot, ncom, width, false, AX.Data());

    hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
  }

  // Start the main loop
  Int iter;
  for( iter = 1; iter < eigMaxIter; iter++ ){
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

    blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(),
        height, AX.Data(), height, 0.0, XTX.Data(), width );

    lapack::Lacpy( 'A', width, width, XTX.Data(), width, AMat.Data(), lda );

    // Compute the residual.
    // R <- AX - X*(X'*AX)
    lapack::Lacpy( 'A', height, width, AX.Data(), height, Xtemp.Data(), height );

    blas::Gemm( 'N', 'N', height, width, width, -1.0, 
        X.Data(), height, AMat.Data(), lda, 1.0, Xtemp.Data(), height );

    // Compute the norm of the residual
    for( Int k = 0; k < width; k++ ){
      resNorm(k) = std::sqrt( Energy(DblNumVec(height, false, Xtemp.VecData(k))) ) / 
        std::max( 1.0, std::abs( XTX(k,k) ) );
    }

    // Only take into account the residual of the first numEig eigenvalues.
    resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
    resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "resNorm = " << resNorm << std::endl;
    statusOFS << "maxRes  = " << resMax  << std::endl;
    statusOFS << "minRes  = " << resMin  << std::endl;
#endif

    if( resMax < eigTolerance ){
      isConverged = true;;
      break;
    }

    // Locking according to the residual value
    numLockedSave = numLocked;
    numLocked = 0;
    for( Int k = 0; k < numEig; k++ ){
      if( resNorm(k) < lockTolerance )
        numLocked++;
      else
        break;
    }
    numActive = width - numLocked;

    // If the number of locked vectors goes down, perform steppest
    // descent rather than conjugate gradient
    if( numLocked < numLockedSave )
      numSet = 2;

    // Compute the preconditioned residual W = T*R.
    // The residual is saved in Xtemp
    {
      Spinor spnTemp(fftPtr_->domain, ncom, width-numLocked, false, Xtemp.VecData(numLocked));
      NumTns<Scalar> tnsTemp(ntot, ncom, width-numLocked, false, W.VecData(numLocked));

      SetValue( tnsTemp, 0.0 );
      spnTemp.AddTeterPrecond( fftPtr_, tnsTemp );
    }

    // Normalize the preconditioned residual
    for( Int k = numLocked; k < width; k++ ){
      Real norm = std::sqrt( Energy(DblNumVec(height, false, W.VecData(k))) );
      blas::Scal( height, 1.0 / norm, W.VecData(k), 1 );
    }
   
    // Normalize the conjugate direction
    if( numSet == 3 ){
      for( Int k = numLocked; k < width; k++ ){
        Real norm = std::sqrt( Energy(DblNumVec(height, false, P.VecData(k))) );
        blas::Scal( height, 1.0 / norm, P.VecData(k), 1 );
        blas::Scal( height, 1.0 / norm, AP.VecData(k), 1 );
      }
    }

    // Compute AMat

    // Compute AW = A*W
    {
      Spinor spnTemp(fftPtr_->domain, ncom, width-numLocked, false, W.VecData(numLocked));
      NumTns<Scalar> tnsTemp(ntot, ncom, width-numLocked, false, AW.VecData(numLocked));

      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
    }

    // Compute X' * (AW)
    // Instead of saving the block at &AMat(0,width+numLocked), the data
    // is saved at &AMat(0,width) to guarantee a continuous data
    // arrangement of AMat.  The same treatment applies to the blocks
    // below in both AMat and BMat.
    blas::Gemm( 'T', 'N', width, numActive, height, 1.0, X.Data(),
        height, AW.VecData(numLocked), height, 
        0.0, &AMat(0,width), lda );

    // Compute W' * (AW)
    blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
        W.VecData(numLocked), height, AW.VecData(numLocked), height, 
        0.0, &AMat(width, width), lda );

    if( numSet == 3 ){
      // Compute X' * (AP)
      blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
          X.Data(), height, AP.VecData(numLocked), height, 
          0.0, &AMat(0, width+numActive), lda );

      // Compute W' * (AP)
      blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
          W.VecData(numLocked), height, AP.VecData(numLocked), height, 
          0.0, &AMat(width, width+numActive), lda );

      // Compute P' * (AP)
      blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
          P.VecData(numLocked), height, AP.VecData(numLocked), height, 
          0.0, &AMat(width+numActive, width+numActive), lda );
    }

    // Compute BMat (overlap matrix)

    // Compute X'*X
    blas::Gemm( 'T', 'N', width, width, height, 1.0, 
        X.Data(), height, X.Data(), height, 
        0.0, &BMat(0,0), lda );

    // Compute X'*W
    blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
        X.Data(), height, W.VecData(numLocked), height,
        0.0, &BMat(0,width), lda );

    // Compute W'*W
    blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
        W.VecData(numLocked), height, W.VecData(numLocked), height,
        0.0, &BMat(width, width), lda );


    if( numSet == 3 ){
      // Compute X'*P
      blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
          X.Data(), height, P.VecData(numLocked), height, 
          0.0, &BMat(0, width+numActive), lda );
      
      // Compute W'*P
      blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
          W.VecData(numLocked), height, P.VecData(numLocked), height,
          0.0, &BMat(width, width+numActive), lda );

      // Compute P'*P
      blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
          P.VecData(numLocked), height, P.VecData(numLocked), height,
          0.0, &BMat(width+numActive, width+numActive), lda );
    }

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

    // Keep a copy of the A and B matrices for restarting purpose
    lapack::Lacpy( 'A', lda, lda, AMat.Data(), lda, AMatSave.Data(), lda );
    lapack::Lacpy( 'A', lda, lda, BMat.Data(), lda, BMatSave.Data(), lda );

    isRestart = false;

    // Rayleigh-Ritz procedure
    // AMat * C = BMat * C * Lambda
    // Assuming the dimension (needed) for C is width * width, then
    //     ( C_X )
    //     ( --- )
    // C = ( C_W )
    //     ( --- )
    //     ( C_P )
    //
    if( numSet == 3 ){
      // Conjugate gradient
      Int numCol = width + 2 * numActive;

      // BMat = U' * U
      lapack::Potrf( 'U', numCol, BMat.Data(), lda );

      // TODO Add Pocon and restart strategy
      // AMat <- U^{-T} * AMat * U^{-1}
      lapack::Hegst( 1, 'U', numCol, AMat.Data(), lda,
          BMat.Data(), lda );

      // Eigenvalue problem, the eigenvectors are saved in AMat
      lapack::Syevd( 'V', 'U', numCol, AMat.Data(), lda, 
          eigValS.Data() );

      // Compute the correct eigenvectors C (but saved in AMat)
      blas::Trsm( 'L', 'U', 'N', 'N', numCol, numCol, 1.0, 
          BMat.Data(), lda, AMat.Data(), lda );

      // TODO Add Pocon and restart strategy, and try steepest descent first
    }
    
    if( numSet == 2 ){
      // Steepest descent
      Int numCol = width + numActive;

      lapack::Potrf( 'U', numCol, BMat.Data(), lda );

      // TODO Add Pocon and restart strategy
      lapack::Hegst( 1, 'U', numCol, AMat.Data(), lda,
          BMat.Data(), lda );

      lapack::Syevd( 'V', 'U', numCol, AMat.Data(), lda, 
          eigValS.Data() );

      blas::Trsm( 'L', 'U', 'N', 'N', numCol, numCol, 1.0, 
          BMat.Data(), lda, AMat.Data(), lda );


      // TODO Add Pocon and restart strategy
    }


    if( isRestart ){
      // TODO Add restart strategy
    }
    else{
      // Update X and P
      if( numSet == 2 ){
        // Update the eigenvectors 
        // X <- X * C_X + W * C_W
        blas::Gemm( 'N', 'N', height, width, width, 1.0,
            X.Data(), height, &AMat(0,0), lda,
            0.0, Xtemp.Data(), height );

        blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
            W.VecData(numLocked), height, &AMat(width,0), lda,
            1.0, Xtemp.Data(), height );

        // Save the result into X
        lapack::Lacpy( 'A', height, width, Xtemp.Data(), height, 
            X.Data(), height );

        // P <- W
        lapack::Lacpy( 'A', height, numActive, W.VecData(numLocked), 
            height, P.VecData(numLocked), height );
      }
      else{
        // Compute the conjugate direction
        // P <- W * C_W + P * C_P
        blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
            W.VecData(numLocked), height, &AMat(width, 0), lda, 
            0.0, Xtemp.Data(), height );

        blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
            P.VecData(numLocked), height, &AMat(width+numActive,0), lda,
            1.0, Xtemp.Data(), height );

        lapack::Lacpy( 'A', height, numActive, Xtemp.VecData(numLocked), 
            height, P.VecData(numLocked), height );

        // Update the eigenvectors
        // X <- X * C_X + P
        blas::Gemm( 'N', 'N', height, width, width, 1.0, 
            X.Data(), height, &AMat(0,0), lda, 
            1.0, Xtemp.Data(), height );

        lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
            X.Data(), height );
      } // if ( numSet == 2 )

    } // if ( isRestart )

    // Update AX and AP
    if( numSet == 2 ){
      // AX <- AX * C_X + AW * C_W
      blas::Gemm( 'N', 'N', height, width, width, 1.0,
          AX.Data(), height, &AMat(0,0), lda,
          0.0, Xtemp.Data(), height );

      blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
          AW.VecData(numLocked), height, &AMat(width,0), lda,
          1.0, Xtemp.Data(), height );

      lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
          AX.Data(), height );

      // AP <- AW
      lapack::Lacpy( 'A', height, numActive, AW.VecData(numLocked), height,
          AP.VecData(numLocked), height );
    }
    else{
      // AP <- AW * C_W + A_P * C_P
      blas::Gemm( 'N', 'N', height, width, numActive, 1.0, 
          AW.VecData(numLocked), height, &AMat(width,0), lda,
          0.0, Xtemp.Data(), height );

      blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
          AP.VecData(numLocked), height, &AMat(width+numActive, 0), lda,
          1.0, Xtemp.Data(), height );

      lapack::Lacpy( 'A', height, numActive, Xtemp.VecData(numLocked), 
          height, AP.VecData(numLocked), height );

      // AX <- AX * C_X + AP
      blas::Gemm( 'N', 'N', height, width, width, 1.0,
          AX.Data(), height, &AMat(0,0), lda,
          1.0, Xtemp.Data(), height );

      lapack::Lacpy( 'A', height, width, Xtemp.Data(), height, 
          AX.Data(), height );
    } // if ( numSet == 2 )



#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "numLocked = " << numLocked << std::endl;
    statusOFS << "eigValS   = " << eigValS << std::endl;
#endif


  } // for (iter)



  // *********************************************************************
  // Post processing
  // *********************************************************************

  // Obtain the eigenvalues and eigenvectors
  // XTX should now contain the matrix X' * (AX), and X is an
  // orthonormal set
  lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );

  // X <- X*C
  blas::Gemm( 'N', 'N', height, width, width, 1.0, X.Data(),
      height, XTX.Data(), width, 0.0, Xtemp.Data(), height );

  lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
      X.Data(), height );

#if ( _DEBUGlevel_ >= 2 )

  blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(), 
      height, X.Data(), height, 0.0, XTX.Data(), width );
  
  statusOFS << "After the LOBPCG, XTX = " << XTX << std::endl;
#endif

  // Save the eigenvalues and eigenvectors back to the eigensolver data
  // structure

  eigVal_ = DblNumVec( width, true, eigValS.Data() );
  resVal_ = DblNumVec( width, true, resNorm.Data() );

  lapack::Lacpy( 'A', height, width, X.Data(), height, 
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


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method EigenSolver::LOBPCGSolveReal  ----- 


} // namespace dgdft
