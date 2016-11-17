/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin, Wei Hu and Amartya Banerjee

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
/// @date 2014-04-25 First version of parallelized version. This does
/// not scale well.
/// @date 2014-08-07 Intra-element parallelization.  This has much
/// improved scalability.
/// @date 2016-04-04 Adjust some parameters for controlling the number
/// of iterations dynamically.
/// @date 2016-04-07 Add Chebyshev filtering.
#include  "eigensolver.hpp"
#include  "utility.hpp"
#include  "blas.hpp"
#ifdef GPU
#include  "cublas.hpp"
#include  "cu_nummat_impl.hpp"
#endif
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"

using namespace dgdft::scalapack;


namespace dgdft{

EigenSolver::EigenSolver() {
  // IMPORTANT: 
  // Set contxt_ here. Otherwise if an empty Eigensolver realization
  // is used, there could be error in the exit
  contxt_ = -1;
}

EigenSolver::~EigenSolver() {
  // Finish Cblacs
  if(contxt_ >= 0) {
    Cblacs_gridexit( contxt_ );
  }
}

void EigenSolver::Setup(
    const esdf::ESDFInputParam& esdfParam,
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft ) {
  hamPtr_ = &ham;
  psiPtr_ = &psi;
  fftPtr_ = &fft;


  eigVal_.Resize(psiPtr_->NumStateTotal());  SetValue(eigVal_, 0.0);
  resVal_.Resize(psiPtr_->NumStateTotal());  SetValue(resVal_, 0.0);

  PWSolver_          = esdfParam.PWSolver;
  scaBlockSize_      = esdfParam.scaBlockSize;
  numProcScaLAPACK_  = esdfParam.numProcScaLAPACKPW; 

  PWDFT_Cheby_use_scala_ = esdfParam.PWDFT_Cheby_use_scala;

  // Setup BLACS
  if( PWSolver_ == "LOBPCGScaLAPACK" || PWSolver_ == "PPCGScaLAPACK" || 
      (PWSolver_ == "CheFSI" && PWDFT_Cheby_use_scala_) ){
    for( Int i = IRound(sqrt(double(numProcScaLAPACK_))); 
        i <= numProcScaLAPACK_; i++){
      nprow_ = i; npcol_ = numProcScaLAPACK_ / nprow_;
      if( nprow_ * npcol_ == numProcScaLAPACK_ ) break;
    }

    IntNumVec pmap(numProcScaLAPACK_);
    // Take the first numProcScaLAPACK processors for diagonalization
    for ( Int i = 0; i < numProcScaLAPACK_; i++ ){
      pmap[i] = i;
    }

    Cblacs_get(0, 0, &contxt_);

    Cblacs_gridmap(&contxt_, &pmap[0], nprow_, nprow_, npcol_);
  }

  return;
}         // -----  end of method EigenSolver::Setup ----- 


// NOTE: This version uses ScaLAPACK and is not used anymore
//void
//EigenSolver::LOBPCGSolveReal    ( 
//      Int          numEig,
//      Int          eigMaxIter,
//      Real         eigTolerance)
//{
//
//  // *********************************************************************
//  // Initialization
//  // *********************************************************************
//  Int ntot = psiPtr_->NumGridTotal();
//  Int ncom = psiPtr_->NumComponent();
//  Int noccLocal = psiPtr_->NumState();
//  Int noccTotal = psiPtr_->NumStateTotal();
//
//  Int height = ntot * ncom;
//  Int widthTotal = noccTotal;
//  Int widthLocal = noccLocal;
//  Int width = noccTotal;
//  Int lda = 3 * width;
//
//  Real timeSta, timeEnd;
//  Real timeGemmT = 0.0;
//  Real timeGemmN = 0.0;
//  Real timeTrsm = 0.0;
//  Real timeSpinor = 0.0;
//  Real timeMpirank0 = 0.0;
//  Int  iterGemmT = 0;
//  Int  iterGemmN = 0;
//  Int  iterTrsm = 0;
//  Int  iterSpinor = 0;
//  Int  iterMpirank0 = 0;
//
//  //  GetTime( timeSta );
//  //  GetTime( timeEnd );
//  //  statusOFS << "Time for Gemm is " << timeEnd - timeSta << << std::endl;
//
//
//  if( numEig > width ){
//    std::ostringstream msg;
//    msg 
//      << "Number of eigenvalues requested  = " << numEig << std::endl
//      << "which is larger than the number of columns in psi = " << width << std::endl;
//    ErrorHandling( msg.str().c_str() );
//  }
//
//
//  Int contxt;
//  Int nprow, npcol, myrow, mycol, info;
//
//  MPI_Comm mpi_comm = fftPtr_->domain.comm;
//
//  MPI_Barrier(mpi_comm);
//  Int mpirank;  MPI_Comm_rank(mpi_comm, &mpirank);
//  Int mpisize;  MPI_Comm_size(mpi_comm, &mpisize);
//
//  Cblacs_get(0, 0, &contxt);
//
//  nprow = 1;
//  npcol = mpisize;
//
//  Cblacs_gridinit(&contxt, "C", nprow, npcol);
//  Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);
//
//  Int desc[9];
//  Int desc_width[9];
//  Int desc_width3[9];
//
//  Int irsrc = 0;
//  Int icsrc = 0;
//  Int mb = height;
//  Int nb = 1;
//  Int width3 = 3 * noccTotal; 
//  Int ld_width = (myrow == 0) ? width : 1;
//  Int ld_width3 = (myrow == 0) ? width3 : 1;
//  SCALAPACK(descinit)(&desc[0], &height, &width, &mb, &nb, &irsrc, &icsrc, &contxt, &height, &info);
//  SCALAPACK(descinit)(&desc_width[0], &width, &width, &width, &width, &irsrc, &icsrc, &contxt, &ld_width, &info);
//  SCALAPACK(descinit)(&desc_width3[0], &width3, &width3, &width3, &width3, &irsrc, &icsrc, &contxt, &ld_width3, &info);
//
//  // S = ( X | W | P ) is a triplet used for LOBPCG.  
//  // W is the preconditioned residual
//  DblNumMat  S( height, 3*widthLocal ), AS( height, 3*widthLocal ); 
//  // AMat = S' * (AS),  BMat = S' * S
//  // 
//  // AMat = (X'*AX   X'*AW   X'*AP)
//  //      = (  *     W'*AW   W'*AP)
//  //      = (  *       *     P'*AP)
//  //
//  // BMat = (X'*X   X'*W   X'*P)
//  //      = (  *    W'*W   W'*P)
//  //      = (  *      *    P'*P)
//  //
//  DblNumMat  AMat( 3*width, 3*width ), BMat( 3*width, 3*width );
//  DblNumMat  AMatT1( 3*width, 3*width );
//  // AMatSave and BMatSave are used for restart
//  // Temporary buffer array.
//  // The unpreconditioned residual will also be saved in Xtemp
//  DblNumMat  XTX( width, width ), Xtemp( height, widthLocal );
//
//  // rexNorm Grobal matrix  similar to numEig 
//  DblNumVec  resNormLocal ( widthLocal ); 
//  SetValue( resNormLocal, 0.0 );
//  DblNumVec  resNorm( width );
//  SetValue( resNorm, 0.0 );
//  Real       resMax, resMin;
//
//  // For convenience
//  DblNumMat  X( height, widthLocal, false, S.VecData(0) );
//  DblNumMat  W( height, widthLocal, false, S.VecData(widthLocal) );
//  DblNumMat  P( height, widthLocal, false, S.VecData(2*widthLocal) );
//  DblNumMat AX( height, widthLocal, false, AS.VecData(0) );
//  DblNumMat AW( height, widthLocal, false, AS.VecData(widthLocal) );
//  DblNumMat AP( height, widthLocal, false, AS.VecData(2*widthLocal) );
//
//  //Int info;
//  bool isRestart = false;
//  // numSet = 2    : Steepest descent (Davidson), only use (X | W)
//  //        = 3    : Conjugate gradient, use all the triplet (X | W | P)
//  Int numSet = 2;
//
//  // numLocked is the number of converged vectors
//  Int numLockedLocal = 0, numLockedSaveLocal = 0;
//  Int numLockedTotal = 0, numLockedSaveTotal = 0; 
//  Int numLockedSave = 0;
//  // numActive = width - numLocked
//  Int numActiveLocal = 0;
//  Int numActiveTotal = 0;
//
//  // Real lockTolerance = std::min( eigTolerance_, 1e-2 );
//
//  const Int numLocked = 0;  // Never perform locking in this version
//  const Int numActive = width;
//
//  bool isConverged = false;
//
//  // Initialization
//  SetValue( S, 0.0 );
//  SetValue( AS, 0.0 );
//
//  DblNumVec  eigValS(lda);
//  SetValue( eigValS, 0.0 );
//  DblNumVec  sigma2(lda);
//  DblNumVec  invsigma(lda);
//  SetValue( sigma2, 0.0 );
//  SetValue( invsigma, 0.0 );
//
//  // Initialize X by the data in psi
//  // lapack::Lacpy( 'A', height, width, psiPtr_->Wavefun().Data(), height, 
//  //    X.Data(), height );
//
//  char AA = 'A';
//  SCALAPACK(pdlacpy)(&AA, &height, &width, 
//      psiPtr_->Wavefun().Data(), &I_ONE, &I_ONE, &desc[0],
//      X.Data(), &I_ONE, &I_ONE, &desc[0]);
//
//  // *********************************************************************
//  // Main loop
//  // *********************************************************************
//
//  // Orthogonalization through Cholesky factorization
//  // blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(), 
//  // height, X.Data(), height, 0.0, XTX.Data(), width );
//  char TT = 'T';
//  char NN = 'N';
//  double D_ONE = 1.0;
//  double D_ZERO = 0.0;
//  int I_ONE = 1;
//  GetTime( timeSta );
//  SCALAPACK(pdgemm)(&TT, &NN, &width, &width, &height, 
//      &D_ONE,
//      X.Data(), &I_ONE, &I_ONE, &desc[0],
//      X.Data(), &I_ONE, &I_ONE, &desc[0], 
//      &D_ZERO,
//      XTX.Data(), &I_ONE, &I_ONE, &desc_width[0], 
//      &contxt );
//  GetTime( timeEnd );
//  iterGemmT = iterGemmT + 1;
//  timeGemmT = timeGemmT + ( timeEnd - timeSta );
//
//  // X'*X=U'*U
//  // lapack::Potrf( 'U', width, XTX.Data(), width );
//  char UU = 'U';
//
//  // SCALAPACK(pdpotrf)(&UU, &width, XTX.Data(), &I_ONE,
//  //    &I_ONE, &desc_width[0], &info);
//
//  if ( mpirank == 0) {
//    GetTime( timeSta );
//    lapack::Potrf( 'U', width, XTX.Data(), width );
//    GetTime( timeEnd );
//    iterMpirank0 = iterMpirank0 + 1;
//    timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
//  }
//  MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
//
//  // X <- X * U^{-1} is orthogonal
//  // blas::Trsm( 'R', 'U', 'N', 'N', height, width, 1.0, XTX.Data(), width, 
//  //    X.Data(), height );
//  char RR = 'R';
//  GetTime( timeSta );
//  SCALAPACK(pdtrsm)( &RR, &UU, &NN, &NN,
//      &height, &width, 
//      &D_ONE,
//      XTX.Data(), &I_ONE, &I_ONE, &desc_width[0],
//      X.Data(), &I_ONE, &I_ONE, &desc[0]);
//  GetTime( timeEnd );
//  iterTrsm = iterTrsm + 1;
//  timeTrsm = timeTrsm + ( timeEnd - timeSta );
//
//
//  // Applying the Hamiltonian matrix
//  {
//    GetTime( timeSta );
//    Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, noccLocal, false, X.Data());
//    NumTns<Real> tnsTemp(ntot, ncom, noccLocal, false, AX.Data());
//
//    hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
//    GetTime( timeEnd );
//    iterSpinor = iterSpinor + 1;
//    timeSpinor = timeSpinor + ( timeEnd - timeSta );
//  }
//
//  // Start the main loop
//  Int iter;
//  for( iter = 1; iter < eigMaxIter; iter++ ){
//#if ( _DEBUGlevel_ >= 1 )
//    statusOFS << "iter = " << iter << std::endl;
//#endif
//
//    if( iter == 1 || isRestart == true )
//      numSet = 2;
//    else
//      numSet = 3;
//
//    SetValue( AMat, 0.0 );
//    SetValue( BMat, 0.0 );
//
//    // Rayleigh Ritz in Q = span( X, gradient )
//
//    // XTX <- X' * (AX)
//    // blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(),
//    //    height, AX.Data(), height, 0.0, XTX.Data(), width );
//
//    GetTime( timeSta );
//    SCALAPACK(pdgemm)(&TT, &NN, &width, &width, &height, 
//        &D_ONE,
//        X.Data(), &I_ONE, &I_ONE, &desc[0],
//        AX.Data(), &I_ONE, &I_ONE, &desc[0], 
//        &D_ZERO,
//        XTX.Data(), &I_ONE, &I_ONE, &desc_width[0], 
//        &contxt );
//    GetTime( timeEnd );
//    iterGemmT = iterGemmT + 1;
//    timeGemmT = timeGemmT + ( timeEnd - timeSta );
//
//
//    // lapack::Lacpy( 'A', width, width, XTX.Data(), width, AMat.Data(), lda );
//    SCALAPACK(pdlacpy)(&AA, &width, &width, 
//        XTX.Data(), &I_ONE, &I_ONE, &desc_width[0],
//        AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0]);
//
//    // Compute the residual.
//    // R <- AX - X*(X'*AX)
//    // lapack::Lacpy( 'A', height, width, AX.Data(), height, Xtemp.Data(), height );
//    SCALAPACK(pdlacpy)(&AA, &height, &width, 
//        AX.Data(), &I_ONE, &I_ONE, &desc[0],
//        Xtemp.Data(), &I_ONE, &I_ONE, &desc[0]);
//
//    // blas::Gemm( 'N', 'N', height, width, width, -1.0, 
//    //    X.Data(), height, AMat.Data(), lda, 1.0, Xtemp.Data(), height );
//    double D_MinusONE = -1.0;
//    GetTime( timeSta );
//    SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &width, 
//        &D_MinusONE,
//        X.Data(), &I_ONE, &I_ONE, &desc[0],
//        AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0], 
//        &D_ONE,
//        Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
//        &contxt );
//    GetTime( timeEnd );
//    iterGemmN = iterGemmN + 1;
//    timeGemmN = timeGemmN + ( timeEnd - timeSta );
//
//    // Compute the norm of the residual
//    SetValue( resNorm, 0.0 );
//    for( Int k = 0; k < widthLocal; k++ ){
//      resNormLocal(k) = Energy(DblNumVec(height, false, Xtemp.VecData(k))); 
//      resNorm( psiPtr_->WavefunIdx(k) ) = resNormLocal(k);
//    }
//
//    DblNumVec  resNormTemp( width );
//    for( Int k = 0; k < width; k++ ){
//      resNormTemp(k) = resNorm(k) ;
//    }
//
//    SetValue( resNorm, 0.0 );
//
//    MPI_Barrier( mpi_comm );
//
//    MPI_Allreduce( resNormTemp.Data(), resNorm.Data(), width, MPI_DOUBLE, 
//        MPI_SUM, mpi_comm );
//
//
//
//    if ( mpirank == 0 ){
//      GetTime( timeSta );
//      for( Int k = 0; k < width; k++ ){
//        resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( XTX(k,k) ) );
//      }
//      GetTime( timeEnd );
//      iterMpirank0 = iterMpirank0 + 1;
//      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
//    }
//    MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm);
//
//
//    resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
//    resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );
//
//#if ( _DEBUGlevel_ >= 1 )
//    statusOFS << "resNorm = " << resNorm << std::endl;
//    statusOFS << "maxRes  = " << resMax  << std::endl;
//    statusOFS << "minRes  = " << resMin  << std::endl;
//#endif
//
//    if( resMax < eigTolerance ){
//      isConverged = true;;
//      break;
//    }
//
//    //numActive = width - numLocked;
//    numActiveLocal = widthLocal - numLockedLocal;
//    numActiveTotal = widthTotal - numLockedTotal;
//
//    // If the number of locked vectors goes down, perform steppest
//    // descent rather than conjugate gradient
//    // if( numLockedTotal < numLockedSaveTotal )
//    //  numSet = 2;
//
//    // Compute the preconditioned residual W = T*R.
//    // The residual is saved in Xtemp
//    {
//      GetTime( timeSta );
//      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, Xtemp.VecData(numLockedLocal));
//      NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, W.VecData(numLockedLocal));
//
//      SetValue( tnsTemp, 0.0 );
//      spnTemp.AddTeterPrecond( fftPtr_, tnsTemp );
//      GetTime( timeEnd );
//      iterSpinor = iterSpinor + 1;
//      timeSpinor = timeSpinor + ( timeEnd - timeSta );
//    }
//
//    //    Real normLocal; 
//    Real norm; 
//
//    norm = 0.0; 
//    // Normalize the preconditioned residual
//    for( Int k = numLockedLocal; k < widthLocal; k++ ){
//      norm = Energy(DblNumVec(height, false, W.VecData(k)));
//      norm = std::sqrt( norm );
//      blas::Scal( height, 1.0 / norm, W.VecData(k), 1 );
//    }
//
//    norm = 0.0; 
//    // Normalize the conjugate direction
//    if( numSet == 3 ){
//      for( Int k = numLockedLocal; k < widthLocal; k++ ){
//        norm = Energy(DblNumVec(height, false, P.VecData(k)));
//        norm = std::sqrt( norm );
//        blas::Scal( height, 1.0 / norm, P.VecData(k), 1 );
//        blas::Scal( height, 1.0 / norm, AP.VecData(k), 1 );
//      }
//    }
//
//    // Compute AMat
//
//    // Compute AW = A*W
//    {
//      GetTime( timeSta );
//      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, W.VecData(numLockedLocal));
//      NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, AW.VecData(numLockedLocal));
//
//      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
//      GetTime( timeEnd );
//      iterSpinor = iterSpinor + 1;
//      timeSpinor = timeSpinor + ( timeEnd - timeSta );
//    }
//
//    // Compute X' * (AW)
//    // Instead of saving the block at &AMat(0,width+numLocked), the data
//    // is saved at &AMat(0,width) to guarantee a continuous data
//    // arrangement of AMat.  The same treatment applies to the blocks
//    // below in both AMat and BMat.
//    // blas::Gemm( 'T', 'N', width, numActive, height, 1.0, X.Data(),
//    //    height, AW.VecData(numLocked), height, 
//    //    0.0, &AMat(0,width), lda );
//    int ia = 1;
//    int ja = 1;
//    int ib = 1;
//    int jb = numLockedTotal + 1;
//    int ic = 1;
//    int jc = width + 1;
//    GetTime( timeSta );
//    SCALAPACK(pdgemm)(&TT, &NN, &width, &numActiveTotal, &height, 
//        &D_ONE,
//        X.Data(), &I_ONE, &I_ONE, &desc[0],
//        AW.Data(), &I_ONE, &jb, &desc[0], 
//        &D_ZERO,
//        AMat.Data(), &I_ONE, &jc, &desc_width3[0], 
//        &contxt );
//    GetTime( timeEnd );
//    iterGemmT = iterGemmT + 1;
//    timeGemmT = timeGemmT + ( timeEnd - timeSta );
//
//    // Compute W' * (AW)
//    // blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
//    //    W.VecData(numLocked), height, AW.VecData(numLocked), height, 
//    //    0.0, &AMat(width, width), lda );
//
//    ia = 1;
//    ja = numLockedTotal + 1;
//    ib = 1;
//    jb = numLockedTotal + 1;
//    ic = width + 1;
//    jc = width + 1;
//    GetTime( timeSta );
//    SCALAPACK(pdgemm)(&TT, &NN, &numActiveTotal, &numActiveTotal, &height, 
//        &D_ONE,
//        W.Data(), &I_ONE, &ja, &desc[0],
//        AW.Data(), &I_ONE, &jb, &desc[0], 
//        &D_ZERO,
//        AMat.Data(), &ic, &jc, &desc_width3[0], 
//        &contxt );
//    GetTime( timeEnd );
//    iterGemmT = iterGemmT + 1;
//    timeGemmT = timeGemmT + ( timeEnd - timeSta );
//
//    if( numSet == 3 ){
//      // Compute X' * (AP)
//      // blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
//      //    X.Data(), height, AP.VecData(numLocked), height, 
//      //    0.0, &AMat(0, width+numActive), lda );
//
//      ia = 1;
//      ja = 1;
//      ib = 1;
//      jb = numLockedTotal + 1;
//      ic = 1;
//      jc = width + numActiveTotal + 1;
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&TT, &NN, &width, &numActiveTotal, &height, 
//          &D_ONE,
//          X.Data(), &I_ONE, &I_ONE, &desc[0],
//          AP.Data(), &I_ONE, &jb, &desc[0], 
//          &D_ZERO,
//          AMat.Data(), &I_ONE, &jc, &desc_width3[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmT = iterGemmT + 1;
//      timeGemmT = timeGemmT + ( timeEnd - timeSta );
//
//      // Compute W' * (AP)
//      // blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
//      //    W.VecData(numLocked), height, AP.VecData(numLocked), height, 
//      //    0.0, &AMat(width, width+numActive), lda );
//
//      ia = 1;
//      ja = numLockedTotal + 1;
//      ib = 1;
//      jb = numLockedTotal + 1;
//      ic = width + 1 ;
//      jc = width + numActiveTotal + 1;
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&TT, &NN, &numActiveTotal, &numActiveTotal, &height, 
//          &D_ONE,
//          W.Data(), &I_ONE, &ja, &desc[0],
//          AP.Data(), &I_ONE, &jb, &desc[0], 
//          &D_ZERO,
//          AMat.Data(), &ic, &jc, &desc_width3[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmT = iterGemmT + 1;
//      timeGemmT = timeGemmT + ( timeEnd - timeSta );
//
//      // Compute P' * (AP)
//      // blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
//      //    P.VecData(numLocked), height, AP.VecData(numLocked), height, 
//      //    0.0, &AMat(width+numActive, width+numActive), lda );
//
//      ia = 1;
//      ja = numLockedTotal + 1;
//      ib = 1;
//      jb = numLockedTotal + 1;
//      ic = width + numActiveTotal + 1;
//      jc = width + numActiveTotal + 1;
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&TT, &NN, &numActiveTotal, &numActiveTotal, &height, 
//          &D_ONE,
//          P.Data(), &I_ONE, &ja, &desc[0],
//          AP.Data(), &I_ONE, &jb, &desc[0], 
//          &D_ZERO,
//          AMat.Data(), &ic, &jc, &desc_width3[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmT = iterGemmT + 1;
//      timeGemmT = timeGemmT + ( timeEnd - timeSta );
//
//    }
//
//    // Compute BMat (overlap matrix)
//
//    // Compute X'*X
//    // blas::Gemm( 'T', 'N', width, width, height, 1.0, 
//    //    X.Data(), height, X.Data(), height, 
//    //     0.0, &BMat(0,0), lda );
//
//
//    GetTime( timeSta );
//    SCALAPACK(pdgemm)(&TT, &NN, &width, &width, &height, 
//        &D_ONE,
//        X.Data(), &I_ONE, &I_ONE, &desc[0],
//        X.Data(), &I_ONE, &I_ONE, &desc[0], 
//        &D_ZERO,
//        BMat.Data(), &I_ONE, &I_ONE, &desc_width3[0], 
//        &contxt );
//    GetTime( timeEnd );
//    iterGemmT = iterGemmT + 1;
//    timeGemmT = timeGemmT + ( timeEnd - timeSta );
//
//
//    // Compute X'*W
//    // blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
//    //    X.Data(), height, W.VecData(numLocked), height,
//    //    0.0, &BMat(0,width), lda );
//
//
//    ia = 1;
//    ja = 1;
//    ib = 1;
//    jb = numLockedTotal + 1;
//    ic = 1;
//    jc = width + 1;
//    GetTime( timeSta );
//    SCALAPACK(pdgemm)(&TT, &NN, &width, &numActiveTotal, &height, 
//        &D_ONE,
//        X.Data(), &I_ONE, &I_ONE, &desc[0],
//        W.Data(), &I_ONE, &jb, &desc[0], 
//        &D_ZERO,
//        BMat.Data(), &I_ONE, &jc, &desc_width3[0], 
//        &contxt );
//    GetTime( timeEnd );
//    iterGemmT = iterGemmT + 1;
//    timeGemmT = timeGemmT + ( timeEnd - timeSta );
//
//
//    // Compute W'*W
//    // blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
//    //    W.VecData(numLocked), height, W.VecData(numLocked), height,
//    //    0.0, &BMat(width, width), lda );
//
//
//    ia = 1;
//    ja = numLockedTotal + 1;
//    ib = 1;
//    jb = numLockedTotal + 1;
//    ic = width + 1;
//    jc = width + 1;
//    GetTime( timeSta );
//    SCALAPACK(pdgemm)(&TT, &NN, &numActiveTotal, &numActiveTotal, &height, 
//        &D_ONE,
//        W.Data(), &I_ONE, &ja, &desc[0],
//        W.Data(), &I_ONE, &jb, &desc[0], 
//        &D_ZERO,
//        BMat.Data(), &ic, &jc, &desc_width3[0], 
//        &contxt );
//    GetTime( timeEnd );
//    iterGemmT = iterGemmT + 1;
//    timeGemmT = timeGemmT + ( timeEnd - timeSta );
//
//
//    if( numSet == 3 ){
//      // Compute X'*P
//      //  blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
//      //     X.Data(), height, P.VecData(numLocked), height, 
//      //     0.0, &BMat(0, width+numActive), lda );
//
//
//      ia = 1;
//      ja = 1;
//      ib = 1;
//      jb = numLockedTotal + 1;
//      ic = 1;
//      jc = width + numActiveTotal + 1;
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&TT, &NN, &width, &numActiveTotal, &height, 
//          &D_ONE,
//          X.Data(), &I_ONE, &I_ONE, &desc[0],
//          P.Data(), &I_ONE, &jb, &desc[0], 
//          &D_ZERO,
//          BMat.Data(), &I_ONE, &jc, &desc_width3[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmT = iterGemmT + 1;
//      timeGemmT = timeGemmT + ( timeEnd - timeSta );
//
//
//      // Compute W'*P
//      // blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
//      //    W.VecData(numLocked), height, P.VecData(numLocked), height,
//      //    0.0, &BMat(width, width+numActive), lda );
//
//
//      ia = 1;
//      ja = numLockedTotal + 1;
//      ib = 1;
//      jb = numLockedTotal + 1;
//      ic = width + 1;
//      jc = width + numActiveTotal + 1;
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&TT, &NN, &numActiveTotal, &numActiveTotal, &height, 
//          &D_ONE,
//          W.Data(), &I_ONE, &ja, &desc[0],
//          P.Data(), &I_ONE, &jb, &desc[0], 
//          &D_ZERO,
//          BMat.Data(), &ic, &jc, &desc_width3[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmT = iterGemmT + 1;
//      timeGemmT = timeGemmT + ( timeEnd - timeSta );
//
//
//      // Compute P'*P
//      // blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
//      //    P.VecData(numLocked), height, P.VecData(numLocked), height,
//      //    0.0, &BMat(width+numActive, width+numActive), lda );
//
//
//      ia = 1;
//      ja = numLockedTotal + 1;
//      ib = 1;
//      jb = numLockedTotal + 1;
//      ic = width + numActiveTotal + 1;
//      jc = width + numActiveTotal + 1;
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&TT, &NN, &numActiveTotal, &numActiveTotal, &height, 
//          &D_ONE,
//          P.Data(), &I_ONE, &ja, &desc[0],
//          P.Data(), &I_ONE, &jb, &desc[0], 
//          &D_ZERO,
//          BMat.Data(), &ic, &jc, &desc_width3[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmT = iterGemmT + 1;
//      timeGemmT = timeGemmT + ( timeEnd - timeSta );
//
//    }
//
//#if ( _DEBUGlevel_ >= 2 )
//    {
//      DblNumMat WTW( width, width );
//      // lapack::Lacpy( 'A', width, width, &BMat(width, width), lda,
//      //    WTW.Data(), width );
//
//      ia = width + 1;
//      ja = wdith + 1;
//      ib = 1;
//      jb = 1;
//      SCALAPACK(pdlacpy)(&AA, &width, &width, 
//          BMat.Data(), &ia, &ib, &desc_width3[0],
//          WTW.Data(), &I_ONE, &I_ONE, &desc_width[0]);
//
//      statusOFS << "W'*W = " << WTW << std::endl;
//      if( numSet == 3 )
//      {
//        DblNumMat PTP( width, width );
//        // lapack::Lacpy( 'A', width, width, &BMat(width+numActive, width+numActive), 
//        //    lda, PTP.Data(), width );
//
//        ia = width + numActiveTotal + 1;
//        ja = wdith + numActiveTotal + 1;
//        ib = 1;
//        jb = 1;
//        SCALAPACK(pdlacpy)(&AA, &width, &width, 
//            BMat.Data(), &ia, &ib, &desc_width3[0],
//            PTP.Data(), &I_ONE, &I_ONE, &desc_width[0]);
//
//        statusOFS << "P'*P = " << PTP << std::endl;
//      }
//    }
//#endif
//
//
//
//    // Rayleigh-Ritz procedure
//    // AMat * C = BMat * C * Lambda
//    // Assuming the dimension (needed) for C is width * width, then
//    //     ( C_X )
//    //     ( --- )
//    // C = ( C_W )
//    //     ( --- )
//    //     ( C_P )
//    //
//    Int numCol;
//    if( numSet == 3 ){
//      // Conjugate gradient
//      numCol = width + 2 * numActiveTotal;
//    }
//    else{
//      numCol = width + numActiveTotal;
//    }
//
//    // Solve the generalized eigenvalue problem with thresholding
//    if(1){
//
//      if ( mpirank == 0 ) {
//
//        // Symmetrize A and B first.  This is important.
//        for( Int j = 0; j < numCol; j++ ){
//          for( Int i = j+1; i < numCol; i++ ){
//            AMat(i,j) = AMat(j,i);
//            BMat(i,j) = BMat(j,i);
//          }
//        }
//
//        GetTime( timeSta );
//        lapack::Syevd( 'V', 'U', numCol, BMat.Data(), lda, sigma2.Data() );
//        GetTime( timeEnd );
//        iterMpirank0 = iterMpirank0 + 1;
//        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
//
//        Int numKeep = 0;
//        for( Int i = numCol-1; i>=0; i-- ){
//          if( sigma2(i) / sigma2(numCol-1) >  1e-12 )
//            numKeep++;
//          else
//            break;
//        }
//
//#if ( _DEBUGlevel_ >= 1 )
//        statusOFS << "sigma2 = " << sigma2 << std::endl;
//#endif
//
//#if ( _DEBUGlevel_ >= 1 )
//        statusOFS << "sigma2(0)        = " << sigma2(0) << std::endl;
//        statusOFS << "sigma2(numCol-1) = " << sigma2(numCol-1) << std::endl;
//        statusOFS << "numKeep          = " << numKeep << std::endl;
//#endif
//
//        for( Int i = 0; i < numKeep; i++ ){
//          invsigma(i) = 1.0 / std::sqrt( sigma2(i+numCol-numKeep) );
//        }
//
//        if( numKeep < width ){
//          std::ostringstream msg;
//          msg 
//            << "width   = " << width << std::endl
//            << "numKeep =  " << numKeep << std::endl
//            << "there are not enough number of columns." << std::endl;
//          ErrorHandling( msg.str().c_str() );
//        }
//
//        SetValue( AMatT1, 0.0 );
//        // Evaluate S^{-1/2} (U^T A U) S^{-1/2}
//        GetTime( timeSta );
//        blas::Gemm( 'N', 'N', numCol, numKeep, numCol, 1.0,
//            AMat.Data(), lda, BMat.VecData(numCol-numKeep), lda,
//            0.0, AMatT1.Data(), lda );
//        GetTime( timeEnd );
//        iterMpirank0 = iterMpirank0 + 1;
//        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
//
//        GetTime( timeSta );
//        blas::Gemm( 'T', 'N', numKeep, numKeep, numCol, 1.0,
//            BMat.VecData(numCol-numKeep), lda, AMatT1.Data(), lda, 
//            0.0, AMat.Data(), lda );
//        GetTime( timeEnd );
//        iterMpirank0 = iterMpirank0 + 1;
//        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
//
//        GetTime( timeSta );
//        for( Int j = 0; j < numKeep; j++ ){
//          for( Int i = 0; i < numKeep; i++ ){
//            AMat(i,j) *= invsigma(i)*invsigma(j);
//          }
//        }
//        GetTime( timeEnd );
//        iterMpirank0 = iterMpirank0 + 1;
//        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
//
//        // Solve the standard eigenvalue problem
//        GetTime( timeSta );
//        lapack::Syevd( 'V', 'U', numKeep, AMat.Data(), lda,
//            eigValS.Data() );
//        GetTime( timeEnd );
//        iterMpirank0 = iterMpirank0 + 1;
//        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
//
//        // Compute the correct eigenvectors and save them in AMat
//        for( Int j = 0; j < numKeep; j++ ){
//          for( Int i = 0; i < numKeep; i++ ){
//            AMat(i,j) *= invsigma(i);
//          }
//        }
//
//        GetTime( timeSta );
//        blas::Gemm( 'N', 'N', numCol, numKeep, numKeep, 1.0,
//            BMat.VecData(numCol-numKeep), lda, AMat.Data(), lda,
//            0.0, AMatT1.Data(), lda );
//        GetTime( timeEnd );
//        iterMpirank0 = iterMpirank0 + 1;
//        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
//
//        lapack::Lacpy( 'A', numCol, numKeep, AMatT1.Data(), lda, 
//            AMat.Data(), lda );
//
//      } // mpirank ==0
//
//
//      MPI_Bcast(AMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm);
//      MPI_Bcast(eigValS.Data(), lda, MPI_DOUBLE, 0, mpi_comm);
//
//
//    } // if(1)
//
//
//    else{
//
//      // BMat = U' * U
//      // lapack::Potrf( 'U', numCol, BMat.Data(), lda );
//
//      // SCALAPACK(pdpotrf)(&UU, &numCol, BMat.Data(), &I_ONE, &I_ONE, &desc_width3[0], &info);
//
//      if ( mpirank == 0 ) {
//        GetTime( timeSta );
//        lapack::Potrf( 'U', numCol, BMat.Data(), lda );
//        GetTime( timeEnd );
//        iterMpirank0 = iterMpirank0 + 1;
//        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
//      }
//      MPI_Bcast(BMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm);
//
//      // TODO Add Pocon and restart strategy
//      // AMat <- U^{-T} * AMat * U^{-1}
//      // lapack::Hegst( 1, 'U', numCol, AMat.Data(), lda,
//      //    BMat.Data(), lda );
//
//      GetTime( timeSta );
//      SCALAPACK(pdsygst)( &I_ONE, &UU, &numCol,
//          AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0],
//          BMat.Data(), &I_ONE, &I_ONE, &desc_width3[0],
//          &D_ONE, &info); 
//      GetTime( timeEnd );
//      iterTrsm = iterTrsm + 1;
//      timeTrsm = timeTrsm + ( timeEnd - timeSta );
//
//      // Eigenvalue problem, the eigenvectors are saved in AMat
//      // lapack::Syevd( 'V', 'U', numCol, AMat.Data(), lda, 
//      //    eigValS.Data() );
//
//      if ( mpirank == 0 ){
//        GetTime( timeSta );
//        lapack::Syevd( 'V', 'U', numCol, AMat.Data(), lda, eigValS.Data() );
//        GetTime( timeEnd );
//        iterMpirank0 = iterMpirank0 + 1;
//        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
//      }
//      MPI_Bcast(AMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm);
//      MPI_Bcast(eigValS.Data(), numCol, MPI_DOUBLE, 0, mpi_comm);
//
//      // Compute the correct eigenvectors C (but saved in AMat)
//      // blas::Trsm( 'L', 'U', 'N', 'N', numCol, numCol, 1.0, 
//      //    BMat.Data(), lda, AMat.Data(), lda );
//
//      char LL = 'L';
//      GetTime( timeSta );
//      SCALAPACK(pdtrsm)( &LL, &UU, &NN, &NN,
//          &numCol, &numCol, 
//          &D_ONE,
//          BMat.Data(), &I_ONE, &I_ONE, &desc_width3[0],
//          AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0]);
//      GetTime( timeEnd );
//      iterTrsm = iterTrsm + 1;
//      timeTrsm = timeTrsm + ( timeEnd - timeSta );
//
//      // TODO Add Pocon and restart strategy, and try steepest descent first
//    }
//
//
//    if( numSet == 2 ){
//      // Update the eigenvectors 
//      // X <- X * C_X + W * C_W
//      // blas::Gemm( 'N', 'N', height, width, width, 1.0,
//      //    X.Data(), height, &AMat(0,0), lda,
//      //    0.0, Xtemp.Data(), height );
//
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &width, 
//          &D_ONE,
//          X.Data(), &I_ONE, &I_ONE, &desc[0],
//          AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0], 
//          &D_ZERO,
//          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmN = iterGemmN + 1;
//      timeGemmN = timeGemmN + ( timeEnd - timeSta );
//
//
//
//      // blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
//      //    W.VecData(numLocked), height, &AMat(width,0), lda,
//      //    1.0, Xtemp.Data(), height );
//
//      ia = 1;
//      ja = numLockedTotal + 1;
//      ib = width + 1;
//      jb = 1;
//      ic = 1;
//      jc = 1;
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &numActiveTotal, 
//          &D_ONE,
//          W.Data(), &I_ONE, &ja, &desc[0],
//          AMat.Data(), &ib, &I_ONE, &desc_width3[0], 
//          &D_ONE,
//          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmN = iterGemmN + 1;
//      timeGemmN = timeGemmN + ( timeEnd - timeSta );
//
//      // Save the result into X
//      // lapack::Lacpy( 'A', height, width, Xtemp.Data(), height, 
//      //    X.Data(), height );
//
//      SCALAPACK(pdlacpy)(&AA, &height, &width, 
//          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0],
//          X.Data(), &I_ONE, &I_ONE, &desc[0]);
//
//      // P <- W
//      // lapack::Lacpy( 'A', height, numActive, W.VecData(numLocked), 
//      //    height, P.VecData(numLocked), height );
//
//      ia = 1;
//      ja = numLockedTotal + 1;
//      ib = 1;
//      jb = numLockedTotal + 1;
//      SCALAPACK(pdlacpy)(&AA, &height, &numActiveTotal, 
//          W.Data(), &I_ONE, &ja, &desc[0],
//          P.Data(), &I_ONE, &jb, &desc[0]);
//
//    }
//    else{ //numSet == 3
//      // Compute the conjugate direction
//      // P <- W * C_W + P * C_P
//      // blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
//      //    W.VecData(numLocked), height, &AMat(width, 0), lda, 
//      //    0.0, Xtemp.Data(), height );
//
//      ia = 1;
//      ja = numLockedTotal + 1;
//      ib = width + 1;
//      jb = 1;
//      ic = 1;
//      jc = 1;
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &numActiveTotal, 
//          &D_ONE,
//          W.Data(), &I_ONE, &ja, &desc[0],
//          AMat.Data(), &ib, &I_ONE, &desc_width3[0], 
//          &D_ZERO,
//          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmN = iterGemmN + 1;
//      timeGemmN = timeGemmN + ( timeEnd - timeSta );
//
//
//      // blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
//      //    P.VecData(numLocked), height, &AMat(width+numActive,0), lda,
//      //    1.0, Xtemp.Data(), height );
//
//      ia = 1;
//      ja = numLockedTotal + 1;
//      ib = width + numActiveTotal + 1;
//      jb = 1;
//      ic = 1;
//      jc = 1;
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &numActiveTotal, 
//          &D_ONE,
//          P.Data(), &I_ONE, &ja, &desc[0],
//          AMat.Data(), &ib, &I_ONE, &desc_width3[0], 
//          &D_ONE,
//          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmN = iterGemmN + 1;
//      timeGemmN = timeGemmN + ( timeEnd - timeSta );
//
//
//
//      // lapack::Lacpy( 'A', height, numActive, Xtemp.VecData(numLocked), 
//      //    height, P.VecData(numLocked), height );
//
//      ia = 1;
//      ja = numLockedTotal + 1;
//      ib = 1;
//      jb = numLockedTotal + 1;
//      SCALAPACK(pdlacpy)(&AA, &height, &numActiveTotal, 
//          Xtemp.Data(), &I_ONE, &ja, &desc[0],
//          P.Data(), &I_ONE, &jb, &desc[0]);
//
//      // Update the eigenvectors
//      // X <- X * C_X + P
//      // blas::Gemm( 'N', 'N', height, width, width, 1.0, 
//      //    X.Data(), height, &AMat(0,0), lda, 
//      //    1.0, Xtemp.Data(), height );
//
//
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &width, 
//          &D_ONE,
//          X.Data(), &I_ONE, &I_ONE, &desc[0],
//          AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0], 
//          &D_ONE,
//          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmN = iterGemmN + 1;
//      timeGemmN = timeGemmN + ( timeEnd - timeSta );
//
//      // lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
//      //    X.Data(), height );
//
//
//
//      SCALAPACK(pdlacpy)(&AA, &height, &width, 
//          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0],
//          X.Data(), &I_ONE, &I_ONE, &desc[0] );
//
//    } // if ( numSet == 2 )
//
//
//    // Update AX and AP
//    if( numSet == 2 ){
//      // AX <- AX * C_X + AW * C_W
//      // blas::Gemm( 'N', 'N', height, width, width, 1.0,
//      //    AX.Data(), height, &AMat(0,0), lda,
//      //     0.0, Xtemp.Data(), height );
//
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &width, 
//          &D_ONE,
//          AX.Data(), &I_ONE, &I_ONE, &desc[0],
//          AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0], 
//          &D_ZERO,
//          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmN = iterGemmN + 1;
//      timeGemmN = timeGemmN + ( timeEnd - timeSta );
//
//
//      // blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
//      //    AW.VecData(numLocked), height, &AMat(width,0), lda,
//      //     1.0, Xtemp.Data(), height );
//
//      ia = 1;
//      ja = numLockedTotal + 1;
//      ib = width + 1;
//      jb = 1;
//      ic = 1;
//      jc = 1;
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &numActiveTotal, 
//          &D_ONE,
//          AW.Data(), &I_ONE, &ja, &desc[0],
//          AMat.Data(), &ib, &I_ONE, &desc_width3[0], 
//          &D_ONE,
//          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmN = iterGemmN + 1;
//      timeGemmN = timeGemmN + ( timeEnd - timeSta );
//
//      // lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
//      //    AX.Data(), height );
//
//      SCALAPACK(pdlacpy)(&AA, &height, &width, 
//          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0],
//          AX.Data(), &I_ONE, &I_ONE, &desc[0]);
//
//      // AP <- AW
//      // lapack::Lacpy( 'A', height, numActive, AW.VecData(numLocked), height,
//      //    AP.VecData(numLocked), height );
//
//      ia = 1;
//      ja = numLockedTotal + 1;
//      ib = 1;
//      jb = numLockedTotal + 1;
//      SCALAPACK(pdlacpy)(&AA, &height, &numActiveTotal, 
//          AW.Data(), &I_ONE, &ja, &desc[0],
//          AP.Data(), &I_ONE, &jb, &desc[0]);
//
//    }
//    else{
//      // AP <- AW * C_W + A_P * C_P
//      // blas::Gemm( 'N', 'N', height, width, numActive, 1.0, 
//      //    AW.VecData(numLocked), height, &AMat(width,0), lda,
//      //    0.0, Xtemp.Data(), height );
//
//      ia = 1;
//      ja = numLockedTotal + 1;
//      ib = width + 1;
//      jb = 1;
//      ic = 1;
//      jc = 1;
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &numActiveTotal, 
//          &D_ONE,
//          AW.Data(), &I_ONE, &ja, &desc[0],
//          AMat.Data(), &ib, &I_ONE, &desc_width3[0], 
//          &D_ZERO,
//          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmN = iterGemmN + 1;
//      timeGemmN = timeGemmN + ( timeEnd - timeSta );
//
//      // blas::Gemm( 'N', 'N', height, width, numActive, 1.0,
//      //    AP.VecData(numLocked), height, &AMat(width+numActive, 0), lda,
//      //    1.0, Xtemp.Data(), height );
//
//      ia = 1;
//      ja = numLockedTotal + 1;
//      ib = width + numActiveTotal + 1;
//      jb = 1;
//      ic = 1;
//      jc = 1;
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &numActiveTotal, 
//          &D_ONE,
//          AP.Data(), &I_ONE, &ja, &desc[0],
//          AMat.Data(), &ib, &I_ONE, &desc_width3[0], 
//          &D_ONE,
//          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmN = iterGemmN + 1;
//      timeGemmN = timeGemmN + ( timeEnd - timeSta );
//
//
//      // lapack::Lacpy( 'A', height, numActive, Xtemp.VecData(numLocked), 
//      //    height, AP.VecData(numLocked), height );
//
//      ia = 1;
//      ja = numLockedTotal + 1;
//      ib = 1;
//      jb = numLockedTotal + 1;
//      SCALAPACK(pdlacpy)(&AA, &height, &numActiveTotal, 
//          Xtemp.Data(), &I_ONE, &ja, &desc[0],
//          AP.Data(), &I_ONE, &jb, &desc[0]);
//
//      // AX <- AX * C_X + AP
//      // blas::Gemm( 'N', 'N', height, width, width, 1.0,
//      //    AX.Data(), height, &AMat(0,0), lda,
//      //    1.0, Xtemp.Data(), height );
//
//      GetTime( timeSta );
//      SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &width, 
//          &D_ONE,
//          AX.Data(), &I_ONE, &I_ONE, &desc[0],
//          AMat.Data(), &I_ONE, &I_ONE, &desc_width3[0], 
//          &D_ONE,
//          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
//          &contxt );
//      GetTime( timeEnd );
//      iterGemmN = iterGemmN + 1;
//      timeGemmN = timeGemmN + ( timeEnd - timeSta );
//
//      // lapack::Lacpy( 'A', height, width, Xtemp.Data(), height, 
//      //    AX.Data(), height );
//
//      SCALAPACK(pdlacpy)(&AA, &height, &width, 
//          Xtemp.Data(), &I_ONE, &I_ONE, &desc[0],
//          AX.Data(), &I_ONE, &I_ONE, &desc[0]);
//
//    } // if ( numSet == 2 )
//
//
//
//#if ( _DEBUGlevel_ >= 1 )
//    statusOFS << "numLocked = " << numLocked << std::endl;
//    statusOFS << "eigValS   = " << eigValS << std::endl;
//#endif
//
//
//
//  } // for (iter) end for main loop
//
//
//
//
//  // *********************************************************************
//  // Post processing
//  // *********************************************************************
//
//  // Obtain the eigenvalues and eigenvectors
//  // XTX should now contain the matrix X' * (AX), and X is an
//  // orthonormal set
//  // lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );
//
//  char VV = 'V';
//
//  if ( mpirank == 0 ){
//    GetTime( timeSta );
//    lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );
//    GetTime( timeEnd );
//    iterMpirank0 = iterMpirank0 + 1;
//    timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
//  }
//
//  MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
//  MPI_Bcast(eigValS.Data(), width, MPI_DOUBLE, 0, mpi_comm);
//
//  // X <- X*C
//  // blas::Gemm( 'N', 'N', height, width, width, 1.0, X.Data(),
//  //    height, XTX.Data(), width, 0.0, Xtemp.Data(), height );
//
//  GetTime( timeSta );
//  SCALAPACK(pdgemm)(&NN, &NN, &height, &width, &width, 
//      &D_ONE,
//      X.Data(), &I_ONE, &I_ONE, &desc[0],
//      XTX.Data(), &I_ONE, &I_ONE, &desc_width[0], 
//      &D_ZERO,
//      Xtemp.Data(), &I_ONE, &I_ONE, &desc[0], 
//      &contxt );
//  GetTime( timeEnd );
//  iterGemmN = iterGemmN + 1;
//  timeGemmN = timeGemmN + ( timeEnd - timeSta );
//
//  // lapack::Lacpy( 'A', height, width, Xtemp.Data(), height,
//  //    X.Data(), height );
//
//  SCALAPACK(pdlacpy)(&AA, &height, &width, 
//      Xtemp.Data(), &I_ONE, &I_ONE, &desc[0],
//      X.Data(), &I_ONE, &I_ONE, &desc[0]);
//
//#if ( _DEBUGlevel_ >= 2 )
//
//  // blas::Gemm( 'T', 'N', width, width, height, 1.0, X.Data(), 
//  //    height, X.Data(), height, 0.0, XTX.Data(), width );
//
//  GetTime( timeSta );
//  SCALAPACK(pdgemm)(&TT, &NN, &width, &width, &height, 
//      &D_ONE,
//      X.Data(), &I_ONE, &I_ONE, &desc[0],
//      X.Data(), &I_ONE, &I_ONE, &desc[0], 
//      &D_ZERO,
//      XTX.Data(), &I_ONE, &I_ONE, &desc_width[0], 
//      &contxt );
//  GetTime( timeEnd );
//  iterGemmT = iterGemmT + 1;
//  timeGemmT = timeGemmT + ( timeEnd - timeSta );
//
//  statusOFS << "After the LOBPCG, XTX = " << XTX << std::endl;
//#endif
//
//  // Save the eigenvalues and eigenvectors back to the eigensolver data
//  // structure
//
//  eigVal_ = DblNumVec( width, true, eigValS.Data() );
//  resVal_ = resNorm;
//
//  //lapack::Lacpy( 'A', height, width, X.Data(), height, 
//  //    psiPtr_->Wavefun().Data(), height );
//
//  SCALAPACK(pdlacpy)(&AA, &height, &width, 
//      X.Data(), &I_ONE, &I_ONE, &desc[0],
//      psiPtr_->Wavefun().Data(), &I_ONE, &I_ONE, &desc[0]);
//
//  if( isConverged ){
//    statusOFS << std::endl << "After " << iter 
//      << " iterations, LOBPCG has converged."  << std::endl
//      << "The maximum norm of the residual is " 
//      << resMax << std::endl << std::endl;
//  }
//  else{
//    statusOFS << std::endl << "After " << iter 
//      << " iterations, LOBPCG did not converge. " << std::endl
//      << "The maximum norm of the residual is " 
//      << resMax << std::endl << std::endl;
//  }
//
//#if ( _DEBUGlevel_ >= 0 ) 
//  statusOFS << "Time for iterGemmT = " << iterGemmT << "  timeGemmT = " << timeGemmT << std::endl;
//  statusOFS << "Time for iterGemmN = " << iterGemmN << "  timeGemmN = " << timeGemmN << std::endl;
//  statusOFS << "Time for iterTrsm = " << iterTrsm << "  timeTrsm = " << timeTrsm << std::endl;
//  statusOFS << "Time for iterSpinor = " << iterSpinor << "  timeSpinor = " << timeSpinor << std::endl;
//  statusOFS << "Time for iterMpirank0 = " << iterMpirank0 << "  timeMpirank0 = " << timeMpirank0 << std::endl;
//#endif
//
//
//
//  return ;
//}         // -----  end of method EigenSolver::LOBPCGSolveReal  ----- 



// NOTE: This is the scalable version.
void
EigenSolver::LOBPCGSolveReal2    (
    Int          numEig,
    Int          eigMaxIter,
    Real         eigMinTolerance,
    Real         eigTolerance)
{

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

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  if( widthLocal != noccLocal ){
    ErrorHandling("widthLocal != noccLocal.");
  }

  // Time for GemmT, GemmN, Alltoallv, Spinor, Mpirank0 
  // GemmT: blas::Gemm( 'T', 'N')
  // GemmN: blas::Gemm( 'N', 'N')
  // Alltoallv: row-partition to column partition via MPI_Alltoallv 
  // Spinor: Applying the Hamiltonian matrix 
  // Mpirank0: Serial calculation part

  Real timeSta, timeEnd;
  Real timeGemmT = 0.0;
  Real timeGemmN = 0.0;
  Real timeAllreduce = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeSpinor = 0.0;
  Real timeTrsm = 0.0;
  Real timeMpirank0 = 0.0;
  Int  iterGemmT = 0;
  Int  iterGemmN = 0;
  Int  iterAllreduce = 0;
  Int  iterAlltoallv = 0;
  Int  iterSpinor = 0;
  Int  iterTrsm = 0;
  Int  iterMpirank0 = 0;

  if( numEig > width ){
    std::ostringstream msg;
    msg 
      << "Number of eigenvalues requested  = " << numEig << std::endl
      << "which is larger than the number of columns in psi = " << width << std::endl;
    ErrorHandling( msg.str().c_str() );
  }


  // The following codes are not replaced by AlltoallForward /
  // AlltoallBackward since they are repetitively used in the
  // eigensolver.
  //
  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
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

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
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
  DblNumMat AXcol( height, widthLocal );
  DblNumMat AWcol( height, widthLocal );

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
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
  GetTime( timeSta );
  SetValue( XTX, 0.0 );
  MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterAllreduce = iterAllreduce + 1;
  timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

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
    NumTns<Real> tnsTemp(ntot, ncom, noccLocal, false, AXcol.Data());

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
  Int iter = 0;
  statusOFS << "Minimum tolerance is " << eigMinTolerance << std::endl;

  do{
    iter++;
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
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

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
    statusOFS << "eigValS = " << eigValS << std::endl;
    statusOFS << "maxRes  = " << resMax  << std::endl;
    statusOFS << "minRes  = " << resMin  << std::endl;
#endif

    if( resMax < eigTolerance ){
      isConverged = true;
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
      NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));

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
      NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, AWcol.VecData(numLockedLocal));

      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

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
    GetTime( timeSta );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
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
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width,width), lda );

    if( numSet == 3 ){

      // Compute X' * (AP)
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
      //    X.Data(), heightLocal, AP.VecData(numLocked), heightLocal, 
      //   0.0, &AMat(0, width+numActive), lda );
      blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
          X.Data(), heightLocal, AP.VecData(numLocked), heightLocal, 
          0.0, XTXtemp1.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(0, width+numActive), lda );

      // Compute W' * (AP)
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
      //    W.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal, 
      //    0.0, &AMat(width, width+numActive), lda );
      blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
          W.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal, 
          0.0, XTXtemp1.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width, width+numActive), lda );

      // Compute P' * (AP)
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
      //    P.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal, 
      //    0.0, &AMat(width+numActive, width+numActive), lda );
      blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
          P.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal, 
          0.0, XTXtemp1.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width+numActive, width+numActive), lda );

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
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0,0), lda );

    // Compute X'*W
    GetTime( timeSta );
    //blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
    //    X.Data(), height, W.VecData(numLocked), height,
    //    0.0, &BMat(0,width), lda );
    blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
        X.Data(), heightLocal, W.VecData(numLocked), heightLocal,
        0.0, XTXtemp1.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0,width), lda );

    // Compute W'*W
    GetTime( timeSta );
    //blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
    //    W.VecData(numLocked), height, W.VecData(numLocked), height,
    //    0.0, &BMat(width, width), lda );
    blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
        W.VecData(numLocked), heightLocal, W.VecData(numLocked), heightLocal,
        0.0, XTXtemp1.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width, width), lda );


    if( numSet == 3 ){
      // Compute X'*P
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
      //    X.Data(), heightLocal, P.VecData(numLocked), heightLocal, 
      //    0.0, &BMat(0, width+numActive), lda );
      blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
          X.Data(), heightLocal, P.VecData(numLocked), heightLocal, 
          0.0, XTXtemp1.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0, width+numActive), lda );

      // Compute W'*P
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
      //    W.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal,
      //    0.0, &BMat(width, width+numActive), lda );
      blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
          W.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal,
          0.0, XTXtemp1.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width, width+numActive), lda );

      // Compute P'*P
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
      //    P.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal,
      //    0.0, &BMat(width+numActive, width+numActive), lda );
      blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
          P.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal,
          0.0, XTXtemp1.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width+numActive, width+numActive), lda );

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
          if( sigma2(i) / sigma2(numCol-1) >  1e-8 )
            numKeep++;
          else
            break;
        }

#if ( _DEBUGlevel_ >= 1 )
        statusOFS << "sigma2 = " << sigma2 << std::endl;
#endif

#if ( _DEBUGlevel_ >= 1 )
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
          ErrorHandling( msg.str().c_str() );
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

    //        statusOFS << iter << " " << eigMaxIter << std::endl;
    //        statusOFS << resMin << " " << eigMinTolerance << std::endl;
  } while( (iter < eigMaxIter) || (resMin > eigMinTolerance) );



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
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
  GetTime( timeSta );
  SetValue( XTX, 0.0 );
  MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterAllreduce = iterAllreduce + 1;
  timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

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
      << resMax << std::endl << std::endl
      << "The minimum norm of the residual is " 
      << resMin << std::endl << std::endl;
  }
  else{
    statusOFS << std::endl << "After " << iter 
      << " iterations, LOBPCG did not converge. " << std::endl
      << "The maximum norm of the residual is " 
      << resMax << std::endl << std::endl
      << "The minimum norm of the residual is " 
      << resMin << std::endl << std::endl;
  }

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for iterGemmT     = " << iterGemmT           << "  timeGemmT     = " << timeGemmT << std::endl;
  statusOFS << "Time for iterGemmN     = " << iterGemmN           << "  timeGemmN     = " << timeGemmN << std::endl;
  statusOFS << "Time for iterAllreduce = " << iterAllreduce       << "  timeAllreduce = " << timeAllreduce << std::endl;
  statusOFS << "Time for iterAlltoallv = " << iterAlltoallv       << "  timeAlltoallv = " << timeAlltoallv << std::endl;
  statusOFS << "Time for iterSpinor    = " << iterSpinor          << "  timeSpinor    = " << timeSpinor << std::endl;
  statusOFS << "Time for iterTrsm      = " << iterTrsm            << "  timeTrsm      = " << timeTrsm << std::endl;
  statusOFS << "Time for iterMpirank0  = " << iterMpirank0        << "  timeMpirank0  = " << timeMpirank0 << std::endl;
#endif


  return ;
}         // -----  end of method EigenSolver::LOBPCGSolveReal2  ----- 


// Use SCALAPACK for solving the small problem
void
EigenSolver::LOBPCGSolveReal3    (
    Int          numEig,
    Int          eigMaxIter,
    Real         eigMinTolerance,
    Real         eigTolerance)
{
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

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  if( widthLocal != noccLocal ){
    ErrorHandling("widthLocal != noccLocal.");
  }

  // Time for GemmT, GemmN, Alltoallv, Spinor, Mpirank0 
  // GemmT: blas::Gemm( 'T', 'N')
  // GemmN: blas::Gemm( 'N', 'N')
  // Alltoallv: row-partition to column partition via MPI_Alltoallv 
  // Spinor: Applying the Hamiltonian matrix 
  // Mpirank0: Serial calculation part

  Real timeSta, timeEnd;
  Real timeGemmT = 0.0;
  Real timeGemmN = 0.0;
  Real timeAllreduce = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeSpinor = 0.0;
  Real timeTrsm = 0.0;
  Real timeMpirank0 = 0.0;
  Real timeScaLAPACKFactor = 0.0;
  Real timeScaLAPACK = 0.0;
  Int  iterGemmT = 0;
  Int  iterGemmN = 0;
  Int  iterAllreduce = 0;
  Int  iterAlltoallv = 0;
  Int  iterSpinor = 0;
  Int  iterTrsm = 0;
  Int  iterMpirank0 = 0;
  Int  iterScaLAPACKFactor = 0;
  Int  iterScaLAPACK = 0;

  if( numEig > width ){
    std::ostringstream msg;
    msg 
      << "Number of eigenvalues requested  = " << numEig << std::endl
      << "which is larger than the number of columns in psi = " << width << std::endl;
    ErrorHandling( msg.str().c_str() );
  }


  // The following codes are not replaced by AlltoallForward /
  // AlltoallBackward since they are repetitively used in the
  // eigensolver.
  //
  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
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

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
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
  DblNumMat AXcol( height, widthLocal );
  DblNumMat AWcol( height, widthLocal );

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

  const Int numLocked = 0;  // Never perform locking in this version
  const Int numActive = width;

  bool isConverged = false;

  // Initialization
  SetValue( S, 0.0 );
  SetValue( AS, 0.0 );

  DblNumVec  eigValS(lda);
  SetValue( eigValS, 0.0 );
  std::vector<double> sigma2(lda);
  DblNumVec invsigma(lda);

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
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
  GetTime( timeSta );
  SetValue( XTX, 0.0 );
  MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterAllreduce = iterAllreduce + 1;
  timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

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
    NumTns<Real> tnsTemp(ntot, ncom, noccLocal, false, AXcol.Data());

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
  Int iter = 0;
  statusOFS << "Minimum tolerance is " << eigMinTolerance << std::endl;

  do{
    iter++;
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
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

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
      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
    }
    MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm);

    resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
    resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "resNorm = " << resNorm << std::endl;
    statusOFS << "eigValS = " << eigValS << std::endl;
    statusOFS << "maxRes  = " << resMax  << std::endl;
    statusOFS << "minRes  = " << resMin  << std::endl;
#endif

    if( resMax < eigTolerance ){
      isConverged = true;
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
      NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));

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
      NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, AWcol.VecData(numLockedLocal));

      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

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
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(0,width), lda );

    // Compute W' * (AW)
    GetTime( timeSta );
    //blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
    //    W.VecData(numLocked), heightLocal, AW.VecData(numLocked), heightLocal, 
    //    0.0, &AMat(width, width), lda );
    blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
        W.VecData(numLocked), heightLocal, AW.VecData(numLocked), heightLocal, 
        0.0, XTXtemp1.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width,width), lda );

    if( numSet == 3 ){

      // Compute X' * (AP)
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
      //    X.Data(), heightLocal, AP.VecData(numLocked), heightLocal, 
      //   0.0, &AMat(0, width+numActive), lda );
      blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
          X.Data(), heightLocal, AP.VecData(numLocked), heightLocal, 
          0.0, XTXtemp1.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(0, width+numActive), lda );

      // Compute W' * (AP)
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
      //    W.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal, 
      //    0.0, &AMat(width, width+numActive), lda );
      blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
          W.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal, 
          0.0, XTXtemp1.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width, width+numActive), lda );

      // Compute P' * (AP)
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
      //    P.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal, 
      //    0.0, &AMat(width+numActive, width+numActive), lda );
      blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
          P.VecData(numLocked), heightLocal, AP.VecData(numLocked), heightLocal, 
          0.0, XTXtemp1.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width+numActive, width+numActive), lda );

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
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0,0), lda );

    // Compute X'*W
    GetTime( timeSta );
    //blas::Gemm( 'T', 'N', width, numActive, height, 1.0,
    //    X.Data(), height, W.VecData(numLocked), height,
    //    0.0, &BMat(0,width), lda );
    blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
        X.Data(), heightLocal, W.VecData(numLocked), heightLocal,
        0.0, XTXtemp1.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0,width), lda );

    // Compute W'*W
    GetTime( timeSta );
    //blas::Gemm( 'T', 'N', numActive, numActive, height, 1.0,
    //    W.VecData(numLocked), height, W.VecData(numLocked), height,
    //    0.0, &BMat(width, width), lda );
    blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
        W.VecData(numLocked), heightLocal, W.VecData(numLocked), heightLocal,
        0.0, XTXtemp1.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTXtemp, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width, width), lda );


    if( numSet == 3 ){
      // Compute X'*P
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
      //    X.Data(), heightLocal, P.VecData(numLocked), heightLocal, 
      //    0.0, &BMat(0, width+numActive), lda );
      blas::Gemm( 'T', 'N', width, numActive, heightLocal, 1.0,
          X.Data(), heightLocal, P.VecData(numLocked), heightLocal, 
          0.0, XTXtemp1.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0, width+numActive), lda );

      // Compute W'*P
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
      //    W.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal,
      //    0.0, &BMat(width, width+numActive), lda );
      blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
          W.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal,
          0.0, XTXtemp1.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width, width+numActive), lda );

      // Compute P'*P
      GetTime( timeSta );
      //blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
      //    P.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal,
      //    0.0, &BMat(width+numActive, width+numActive), lda );
      blas::Gemm( 'T', 'N', numActive, numActive, heightLocal, 1.0,
          P.VecData(numLocked), heightLocal, P.VecData(numLocked), heightLocal,
          0.0, XTXtemp1.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width+numActive, width+numActive), lda );

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

    // Solve the generalized eigenvalue problem using ScaLAPACK
    // NOTE: This uses a simplified implementation with Sygst / Syevd / Trsm. 
    // For ill-conditioned matrices this might be unstable. So BE CAREFUL
    if( contxt_ >= 0 ){
      GetTime( timeSta );
      // Note: No need for symmetrization of A, B matrices due to
      // the usage of symmetric version of the algorithm

      // For stability reason, need to find a well conditioned
      // submatrix of B to solve the generalized eigenvalue problem. 
      // This is done by possibly repeatedly doing potrf until
      // info == 0 (no error)
      bool factorizeB = true;
      Int numKeep = numCol;
      scalapack::ScaLAPACKMatrix<Real> BMatSca;
      scalapack::Descriptor descReduceSeq, descReducePar;
      Real timeFactorSta, timeFactorEnd;
      GetTime( timeFactorSta );
      while( factorizeB ){
        // Redistributed the B matrix

        // Provided LDA
        descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt_, lda );
        // Automatically comptued LDA
        descReducePar.Init( numKeep, numKeep, scaBlockSize_, scaBlockSize_, I_ZERO, I_ZERO, contxt_ );
        BMatSca.SetDescriptor( descReducePar );
        // Redistribute the matrix due to the changed size. 
        SCALAPACK(pdgemr2d)(&numKeep, &numKeep, BMat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
            &BMatSca.LocalMatrix()[0], &I_ONE, &I_ONE, BMatSca.Desc().Values(), &contxt_ );

        // Factorize
        Int info;
        char uplo = 'U';
        SCALAPACK(pdpotrf)(&uplo, &numKeep, BMatSca.Data(), &I_ONE,
            &I_ONE, BMatSca.Desc().Values(), &info);
        if( info == 0 ){
          // Finish
          factorizeB = false;
        }
        else if( info > width + 1 ){
          // Reduce numKeep and solve again
          // NOTE: (int) is in fact redundant due to integer operation
          numKeep = (int)((info + width)/2);
          // Need to modify the descriptor
          statusOFS << "pdpotrf returns info = " << info << std::endl;
          statusOFS << "retry with size = " << numKeep << std::endl;
        }
        else if (info > 0 && info <=width + 1){
          std::ostringstream msg;
          msg << "pdpotrf: returns info = " << info << std::endl
            << "Not enough columns. The matrix is very ill conditioned." << std::endl;
          ErrorHandling( msg );
        }
        else if( info < 0 ){
          std::ostringstream msg;
          msg << "pdpotrf: runtime error. Info = " << info << std::endl;
          ErrorHandling( msg );
        }

        iterScaLAPACKFactor ++;
      } // while (factorizeB)

      GetTime( timeFactorEnd );
      timeScaLAPACKFactor += timeFactorEnd - timeFactorSta;
      //            statusOFS << "Factorization has finished in " << timeFactorEnd - timeFactorSta << " [s]" << std::endl;

      scalapack::ScaLAPACKMatrix<Real> AMatSca, ZMatSca;
      AMatSca.SetDescriptor( descReducePar );
      ZMatSca.SetDescriptor( descReducePar );

      SCALAPACK(pdgemr2d)(&numKeep, &numKeep, AMat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
          &AMatSca.LocalMatrix()[0], &I_ONE, &I_ONE, AMatSca.Desc().Values(), &contxt_ );

#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "pass pdgemr2d" << std::endl;
#endif

      // Solve the generalized eigenvalue problem
      std::vector<Real> eigs(lda);
      // Keep track of whether Potrf is stable or not.
      scalapack::Sygst( 1, 'U', AMatSca, BMatSca );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "pass Sygst" << std::endl;
#endif
      scalapack::Syevd('U', AMatSca, eigs, ZMatSca );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "pass Syevd" << std::endl;
#endif
      scalapack::Trsm('L', 'U', 'N', 'N', 1.0, BMatSca, ZMatSca);
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "pass Trsm" << std::endl;
#endif

      // Copy the eigenvalues
      SetValue( eigValS, 0.0 );
      for( Int i = 0; i < numKeep; i++ ){
        eigValS[i] = eigs[i];
      }

      // Copy the eigenvectors back to the 0-th processor
      SetValue( AMat, 0.0 );
      SCALAPACK(pdgemr2d)( &numKeep, &numKeep, ZMatSca.Data(), &I_ONE, &I_ONE, ZMatSca.Desc().Values(),
          AMat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt_ );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "pass pdgemr2d" << std::endl;
#endif


      GetTime( timeEnd );
      timeScaLAPACK += timeEnd - timeSta;
      iterScaLAPACK += 1;
    } // solve generalized eigenvalue problem

    // All processors synchronize the information
    MPI_Bcast(AMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm);
    MPI_Bcast(eigValS.Data(), lda, MPI_DOUBLE, 0, mpi_comm);


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

  } while( (iter < eigMaxIter) || (resMin > eigMinTolerance) );



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
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
  GetTime( timeSta );
  SetValue( XTX, 0.0 );
  MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterAllreduce = iterAllreduce + 1;
  timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

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
      << resMax << std::endl << std::endl
      << "The minimum norm of the residual is " 
      << resMin << std::endl << std::endl;
  }
  else{
    statusOFS << std::endl << "After " << iter 
      << " iterations, LOBPCG did not converge. " << std::endl
      << "The maximum norm of the residual is " 
      << resMax << std::endl << std::endl
      << "The minimum norm of the residual is " 
      << resMin << std::endl << std::endl;
  }

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for iterGemmT     = " << iterGemmT           << "  timeGemmT     = " << timeGemmT << std::endl;
  statusOFS << "Time for iterGemmN     = " << iterGemmN           << "  timeGemmN     = " << timeGemmN << std::endl;
  statusOFS << "Time for iterAllreduce = " << iterAllreduce       << "  timeAllreduce = " << timeAllreduce << std::endl;
  statusOFS << "Time for iterAlltoallv = " << iterAlltoallv       << "  timeAlltoallv = " << timeAlltoallv << std::endl;
  statusOFS << "Time for iterSpinor    = " << iterSpinor          << "  timeSpinor    = " << timeSpinor << std::endl;
  statusOFS << "Time for iterTrsm      = " << iterTrsm            << "  timeTrsm      = " << timeTrsm << std::endl;
  statusOFS << "Time for iterMpirank0  = " << iterMpirank0        << "  timeMpirank0  = " << timeMpirank0 << std::endl;
  statusOFS << "Time for ScaLAPACK     = " << iterScaLAPACK       << "  timeScaLAPACK = " << timeScaLAPACK << std::endl;
  statusOFS << "Time for pdpotrf       = " << iterScaLAPACKFactor << "  timepdpotrf   = " << timeScaLAPACKFactor << std::endl;
#endif

  return ;
}         // -----  end of method EigenSolver::LOBPCGSolveReal3  ----- 



double EigenSolver::Cheby_Upper_bound_estimator(DblNumVec& ritz_values, int Num_Lanczos_Steps)
{

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

  // Timestamping ...
  Real timeSta, timeEnd;
  Real timeSpinor = 0.0;
  Int  iterSpinor = 0;



  // a) Setup a temporary spinor with random numbers
  // One band on each process
  Spinor  temp_spinor_v; 
  temp_spinor_v.Setup( fftPtr_->domain, 1, mpisize, 1, 0.0 );
  UniformRandom( temp_spinor_v.Wavefun() );

  // b) Normalize this vector : Current data distribution is height * widthLocal ( = 1)
  double temp_spinor_v_norm = 0.0, *v_access_ptr =  temp_spinor_v.Wavefun().Data();

  temp_spinor_v_norm = blas::Nrm2(height, temp_spinor_v.Wavefun().Data(), 1);
  blas::Scal( height , ( 1.0 /  temp_spinor_v_norm), temp_spinor_v.Wavefun().Data(), 1);

  // c) Compute the Hamiltonian * vector product
  // Applying the Hamiltonian matrix
  GetTime( timeSta );

  Spinor  temp_spinor_f; 
  temp_spinor_f.Setup( fftPtr_->domain, 1, mpisize, 1, 0.0 );

  Spinor  temp_spinor_v0; 
  temp_spinor_v0.Setup( fftPtr_->domain, 1, mpisize, 1, 0.0 );

  // This is required for the  Hamiltonian * vector product
  NumTns<double> tnsTemp_spinor_f(ntot, ncom, 1, false, temp_spinor_f.Wavefun().Data());

  // Set the first wavefunction filter since we are starting from a random guess
  hamPtr_->set_wfn_filter(1);


  SetValue( temp_spinor_f.Wavefun(), 0.0);
  hamPtr_->MultSpinor( temp_spinor_v, tnsTemp_spinor_f, *fftPtr_ ); // f = H * v
  GetTime( timeEnd );
  iterSpinor = iterSpinor + 1;
  timeSpinor = timeSpinor + ( timeEnd - timeSta );


  double alpha, beta;

  alpha = blas::Dot( height, temp_spinor_f.Wavefun().Data(), 1, temp_spinor_v.Wavefun().Data(), 1 );
  blas::Axpy(height, (-alpha), temp_spinor_v.Wavefun().Data(), 1, temp_spinor_f.Wavefun().Data(), 1);

  DblNumMat matT( Num_Lanczos_Steps, Num_Lanczos_Steps );
  SetValue(matT, 0.0);

  matT(0,0) = alpha;


  for(Int j = 1; j < Num_Lanczos_Steps; j ++)
  {
    beta = blas::Nrm2(height, temp_spinor_f.Wavefun().Data(), 1);

    // v0 = v
    blas::Copy( height, temp_spinor_v.Wavefun().Data(), 1, temp_spinor_v0.Wavefun().Data(), 1 );

    // v = f / beta
    blas::Copy( height, temp_spinor_f.Wavefun().Data(), 1, temp_spinor_v.Wavefun().Data(), 1 );
    blas::Scal( height , ( 1.0 /  beta), temp_spinor_v.Wavefun().Data(), 1);

    // f = H * v
    SetValue( temp_spinor_f.Wavefun(), 0.0);
    hamPtr_->MultSpinor( temp_spinor_v, tnsTemp_spinor_f, *fftPtr_ );

    // f = f - beta * v0
    blas::Axpy(height, (-beta), temp_spinor_v0.Wavefun().Data(), 1, temp_spinor_f.Wavefun().Data(), 1);

    // alpha = f' * v
    alpha = blas::Dot( height, temp_spinor_f.Wavefun().Data(), 1, temp_spinor_v.Wavefun().Data(), 1 );

    // f = f - alpha * v
    blas::Axpy(height, (-alpha), temp_spinor_v.Wavefun().Data(), 1, temp_spinor_f.Wavefun().Data(), 1);

    matT(j, j - 1) = beta;
    matT(j - 1, j) = beta;
    matT(j, j) = alpha;
  } // End of loop over Lanczos steps 

  ritz_values.Resize(Num_Lanczos_Steps);
  SetValue( ritz_values, 0.0 );


  // Solve the eigenvalue problem for the Ritz values
  lapack::Syevd( 'N', 'U', Num_Lanczos_Steps, matT.Data(), Num_Lanczos_Steps, ritz_values.Data() );

  // Compute the upper bound on each process
  double b_up= ritz_values(Num_Lanczos_Steps - 1) + blas::Nrm2(height, temp_spinor_f.Wavefun().Data(), 1);


  //statusOFS << std::endl << std::endl << " In estimator here : " << ritz_values(Num_Lanczos_Steps - 1) << '\t' << blas::Nrm2(height, //temp_spinor_f.Wavefun().Data(), 1) << std::endl;
  //statusOFS << std::endl << " Ritz values in estimator here : " << ritz_values ;

  // Need to synchronize the Ritz values and the upper bound across the processes
  MPI_Bcast(&b_up, 1, MPI_DOUBLE, 0, mpi_comm);
  MPI_Bcast(ritz_values.Data(), Num_Lanczos_Steps, MPI_DOUBLE, 0, mpi_comm);



  return b_up;
} // -----  end of method EigenSolver::Cheby_Upper_bound_estimator -----


void EigenSolver::Chebyshev_filter_scaled(int m, double a, double b, double a_L)
{


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

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  if( widthLocal != noccLocal ){
    ErrorHandling("widthLocal != noccLocal.");
  }

  double e, c, sigma, tau, sigma_new;

  // Declare some storage space
  // We will do the filtering band by band on each process.
  // So very little extra space is required
  Spinor spinor_X;
  spinor_X.Setup( fftPtr_->domain, 1, mpisize, 1, 0.0 );

  Spinor  spinor_Y; 
  spinor_Y.Setup( fftPtr_->domain, 1, mpisize, 1, 0.0 );

  Spinor  spinor_Yt; 
  spinor_Yt.Setup( fftPtr_->domain, 1, mpisize, 1, 0.0 );   

  NumTns<double> tns_spinor_X(ntot, ncom, 1, false, spinor_X.Wavefun().Data());
  NumTns<double> tns_spinor_Y(ntot, ncom, 1, false, spinor_Y.Wavefun().Data());
  NumTns<double> tns_spinor_Yt(ntot, ncom, 1, false, spinor_Yt.Wavefun().Data());

  statusOFS << std::endl << std::endl << " Applying the scaled Chebyshev Filter: Order = " << m << " , on " << widthLocal << " local bands."; 
  // Begin iteration over local bands
  for (Int local_band_iter = 0; local_band_iter < widthLocal; local_band_iter ++)
  {
    statusOFS << std::endl << " Band " << local_band_iter << " of " << widthLocal;
    statusOFS << std::endl << " Filter step: 1";
    // Step 1: Set up a few scalars
    e = (b - a) / 2.0; 
    c = (a + b) / 2.0;
    sigma = e / (c - a_L);
    tau = 2.0 / sigma;

    // Step 2: Copy the required band into X
    blas::Copy( height, psiPtr_->Wavefun().Data() + local_band_iter * height, 1, spinor_X.Wavefun().Data(), 1 );

    // Step 3: Compute Y = (H * X - c * X) * (sigma / e)

    // Set the first wavefunction filter since we are starting from a random guess
    hamPtr_->set_wfn_filter(1);

    SetValue( spinor_Y.Wavefun(), 0.0); // Y = 0
    hamPtr_->MultSpinor( spinor_X, tns_spinor_Y, *fftPtr_ ); // Y = H * X

    blas::Axpy(height, (-c), spinor_X.Wavefun().Data(), 1, spinor_Y.Wavefun().Data(), 1); // Y = Y - c * X

    blas::Scal( height , ( sigma / e), spinor_Y.Wavefun().Data(), 1); // Y = Y * (sigma/e)

    // Begin filtering
    for(Int filter_iter = 2; filter_iter <= m; filter_iter ++)
    {
      statusOFS << " " << filter_iter;
      sigma_new = 1.0 / (tau - sigma);

      // Step 4: Compute Yt = (H * Y - c* Y) * (2.0 * sigma_new / e) - (sigma * sigma_new) * X
      SetValue( spinor_Yt.Wavefun(), 0.0); // Yt = 0
      hamPtr_->MultSpinor( spinor_Y, tns_spinor_Yt, *fftPtr_ ); // Yt = H * Y

      blas::Axpy(height, (-c), spinor_Y.Wavefun().Data(), 1, spinor_Yt.Wavefun().Data(), 1); // Yt = Yt - c * Y

      blas::Scal(height , ( 2.0 * sigma_new / e), spinor_Yt.Wavefun().Data(), 1); // Yt = Yt * (2.0 * sigma_new / e)

      // Yt = Yt - (sigma * sigma_new) * X
      blas::Axpy(height, (-sigma * sigma_new), spinor_X.Wavefun().Data(), 1, spinor_Yt.Wavefun().Data(), 1); 

      // Step 5: Re-assignments
      blas::Copy( height, spinor_Y.Wavefun().Data(), 1, spinor_X.Wavefun().Data(), 1 ); // X = Y
      blas::Copy( height, spinor_Yt.Wavefun().Data(), 1, spinor_Y.Wavefun().Data(), 1 );// Y = Yt
      sigma = sigma_new;

    }


    // Step : Copy back the processed band into X
    blas::Copy( height, spinor_X.Wavefun().Data(), 1, psiPtr_->Wavefun().Data() + local_band_iter * height, 1 );
    statusOFS << std::endl << " Band " << local_band_iter << " completed.";
  }

  statusOFS << std::endl << " Filtering Completed !"; 


} // -----  end of method EigenSolver::Chebyshev_filter_scaled -----


// Unscaled filter  
void EigenSolver::Chebyshev_filter(int m, double a, double b)
{


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

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  if( widthLocal != noccLocal ){
    ErrorHandling("widthLocal != noccLocal.");
  }

  double e, c;

  // Declare some storage space
  // We will do the filtering band by band on each process.
  // So very little extra space is required
  Spinor spinor_X;
  spinor_X.Setup( fftPtr_->domain, 1, mpisize, 1, 0.0 );

  Spinor  spinor_Y; 
  spinor_Y.Setup( fftPtr_->domain, 1, mpisize, 1, 0.0 );

  Spinor  spinor_Yt; 
  spinor_Yt.Setup( fftPtr_->domain, 1, mpisize, 1, 0.0 );   

  NumTns<double> tns_spinor_X(ntot, ncom, 1, false, spinor_X.Wavefun().Data());
  NumTns<double> tns_spinor_Y(ntot, ncom, 1, false, spinor_Y.Wavefun().Data());
  NumTns<double> tns_spinor_Yt(ntot, ncom, 1, false, spinor_Yt.Wavefun().Data());

  statusOFS << std::endl << std::endl << " Applying the Chebyshev Filter of order " << m << " ... "; 
  // Begin iteration over local bands
  for (Int local_band_iter = 0; local_band_iter < widthLocal; local_band_iter ++)
  {
    //statusOFS << std::endl << " Band " << local_band_iter << " of " << widthLocal;
    //statusOFS << std::endl << " Filter step: 1";
    // Step 1: Set up a few doubles
    e = (b - a) / 2.0; 
    c = (a + b) / 2.0;

    // Step 2: Copy the required band into X
    blas::Copy( height, psiPtr_->Wavefun().Data() + local_band_iter * height, 1, spinor_X.Wavefun().Data(), 1 );

    // Step 3: Compute Y = (H * X - c * X) * (1.0 / e)
    // Set the first wavefunction filter since we are starting from a random guess
    hamPtr_->set_wfn_filter(1);

    SetValue( spinor_Y.Wavefun(), 0.0); // Y = 0
    hamPtr_->MultSpinor( spinor_X, tns_spinor_Y, *fftPtr_ ); // Y = H * X

    blas::Axpy(height, (-c), spinor_X.Wavefun().Data(), 1, spinor_Y.Wavefun().Data(), 1); // Y = Y - c * X

    blas::Scal( height , ( 1.0 / e), spinor_Y.Wavefun().Data(), 1); // Y = Y * (sigma/e)

    // Begin filtering
    for(Int filter_iter = 2; filter_iter <= m; filter_iter ++)
    {
      // statusOFS << " " << filter_iter;

      // Step 4: Compute Yt = (H * Y - c* Y) * (2.0  / e) -  X
      SetValue( spinor_Yt.Wavefun(), 0.0); // Yt = 0
      hamPtr_->MultSpinor( spinor_Y, tns_spinor_Yt, *fftPtr_ ); // Yt = H * Y

      blas::Axpy(height, (-c), spinor_Y.Wavefun().Data(), 1, spinor_Yt.Wavefun().Data(), 1); // Yt = Yt - c * Y

      blas::Scal(height , ( 2.0  / e), spinor_Yt.Wavefun().Data(), 1); // Yt = Yt * (2.0 * sigma_new / e)

      // Yt = Yt -  X
      blas::Axpy(height, (-1.0), spinor_X.Wavefun().Data(), 1, spinor_Yt.Wavefun().Data(), 1); 

      // Step 5: Re-assignments
      blas::Copy( height, spinor_Y.Wavefun().Data(), 1, spinor_X.Wavefun().Data(), 1 ); // X = Y
      blas::Copy( height, spinor_Yt.Wavefun().Data(), 1, spinor_Y.Wavefun().Data(), 1 );// Y = Yt

    }


    // Step 6 : Copy back the processed band into X
    blas::Copy( height, spinor_X.Wavefun().Data(), 1, psiPtr_->Wavefun().Data() + local_band_iter * height, 1 );
    //statusOFS << std::endl << " Band " << local_band_iter << " completed.";
  }

  statusOFS << std::endl << " Filtering Completed !"; 


} // -----  end of method EigenSolver::Chebyshev_filter -----



void
EigenSolver::FirstChebyStep    (
    Int          numEig,
    Int          eigMaxIter,
    Int           filter_order) {

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

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  if( widthLocal != noccLocal ){
    ErrorHandling("widthLocal != noccLocal.");
  }


  // Temporary safeguard 
  if(noccTotal % mpisize != 0)
  {
    MPI_Barrier(mpi_comm);  
    statusOFS << std::endl << std::endl 
      <<" Input Error ! Currently CheFSI within PWDFT requires total number of bands to be divisble by mpisize. " << std::endl << " Total No. of states = " 
      << noccTotal << " , mpisize = " << mpisize << " ." 
      << std::endl <<  " Use different parameters." << std::endl << " Aborting ..." << std::endl << std::endl;
    MPI_Barrier(mpi_comm);
    exit(-1);  
  }    

  // Time for GemmT, GemmN, Alltoallv, Spinor, Mpirank0 
  // GemmT: blas::Gemm( 'T', 'N')
  // GemmN: blas::Gemm( 'N', 'N')
  // Alltoallv: row-partition to column partition via MPI_Alltoallv 
  // Spinor: Applying the Hamiltonian matrix 
  // Mpirank0: Serial calculation part

  Real timeSta, timeEnd;
  Real extra_timeSta, extra_timeEnd;
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
    ErrorHandling( msg.str().c_str() );
  }

  // The following codes are not replaced by AlltoallForward /
  // AlltoallBackward since they are repetitively used in the
  // eigensolver.
  //
  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
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

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
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



  // TODO: In what follows, we do not care about multiple spinor components
  // For now, just a spin unpolarized setup is considered, so ntot = height
  // TODO: In what follows, we assume real valued wavefunctions

  // Step 1: Obtain the upper bound and the Ritz values (for the lower bound)
  // using the Lanczos estimator
  DblNumVec Lanczos_Ritz_values;

  statusOFS << std::endl << " Estimating the spectral bounds ... "; 
  GetTime( extra_timeSta );
  const int Num_Lanczos_Steps = 6;
  double b_up = Cheby_Upper_bound_estimator(Lanczos_Ritz_values, Num_Lanczos_Steps);  
  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";
  statusOFS << std::endl << " Lanczos Ritz values = " << Lanczos_Ritz_values; 

  // Step 2: Set up the lower bound and the filter scale
  double a_L = Lanczos_Ritz_values(0);
  double beta = 0.5; // 0.5 <= beta < 1.0
  double b_low = beta * Lanczos_Ritz_values(0) + (1.0 - beta) * Lanczos_Ritz_values(Num_Lanczos_Steps - 1);


  // Step 3: Main loop
  const Int Iter_Max = eigMaxIter;
  const Int Filter_Order = filter_order ;

  // Space for linear algebra operations
  DblNumMat  X( heightLocal, width);
  DblNumMat  HX( heightLocal, width);

  DblNumMat  Xcol( height, widthLocal );
  DblNumMat  HXcol( height, widthLocal );

  DblNumMat  square_mat( width, width);
  DblNumMat  square_mat_temp( width, width);

  DblNumVec eig_vals_Raleigh_Ritz;
  eig_vals_Raleigh_Ritz.Resize(width);


  for(Int iter = 1; iter <= Iter_Max; iter ++){

    statusOFS << std::endl << " First CheFSI for PWDFT cycle " << iter << " of " << Iter_Max << " ." << std::endl;
    if(PWDFT_Cheby_use_scala_ == 1)
      statusOFS << " Cholesky factorlization and eigen-decomposition to be done using ScaLAPACK." << std::endl;
    else
      statusOFS << " Cholesky factorlization and eigen-decomposition to be done serially. " << std::endl;



    statusOFS << std::endl << " Upper bound = (to be mapped to +1) " << b_up;
    statusOFS << std::endl << " Lower bound (to be mapped to -1) = " << b_low;
    statusOFS << std::endl << " Lowest eigenvalue = " << a_L;




    // Step 3a : Compute the filtered block of vectors
    // This always works on the vectors in psiPtr_
    GetTime( extra_timeSta );
    Chebyshev_filter_scaled(Filter_Order, b_low, b_up, a_L);
    GetTime( extra_timeEnd );
    statusOFS << std::endl << " Chebyshev filter applied in " << (extra_timeEnd - extra_timeSta ) << " s."; 


    // Step 3b : Orthonormalize
    // ~~ First copy psi (now filtered) to Xcol --- can do away with this to save memory

    GetTime( extra_timeSta );
    statusOFS << std::endl << " Orthonormalizing  ... "; 
    lapack::Lacpy( 'A', height, widthLocal, psiPtr_->Wavefun().Data(), height, 
        Xcol.Data(), height );

    GetTime( timeSta );

    // ~~ Flip into the alternate distribution in prep for Orthogonalization
    // So data goes from Xcol to X
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

    // ~~ Orthogonalization through Cholesky factorization
    // Compute square_mat = X^T * X for Cholesky
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(), 
        heightLocal, X.Data(), heightLocal, 0.0, square_mat_temp.Data(), width );
    SetValue( square_mat, 0.0 );
    MPI_Allreduce( square_mat_temp.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );


    if(PWDFT_Cheby_use_scala_ == 1)
    {
      if( contxt_ >= 0 )
      {
        Int numKeep = width; 
        Int lda = width;

        scalapack::ScaLAPACKMatrix<Real> square_mat_scala;
        scalapack::Descriptor descReduceSeq, descReducePar;
        Real timeCholScala_sta, timeCholScala_end;

        GetTime( timeCholScala_sta );

        // Leading dimension provided
        descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt_, lda );

        // Automatically comptued Leading Dimension
        descReducePar.Init( numKeep, numKeep, scaBlockSize_, scaBlockSize_, I_ZERO, I_ZERO, contxt_ );

        square_mat_scala.SetDescriptor( descReducePar );

        // Redistribute the matrix over the process grid
        SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
            &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt_ );

        // Make the ScaLAPACK call
        Int info;
        char uplo = 'U';
        SCALAPACK(pdpotrf)(&uplo, &numKeep, square_mat_scala.Data(), &I_ONE,
            &I_ONE, square_mat_scala.Desc().Values(), &info);

        // Redistribute back
        SetValue(square_mat , 0.0 );
        SCALAPACK(pdgemr2d)( &numKeep, &numKeep, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
            square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt_ );

        GetTime( timeCholScala_end );

        //statusOFS << std::endl << " ScaLAPACK Cholesky time = " << (timeCholScala_end - timeCholScala_sta) << " s." << std::endl;

      }
      else
      {
        statusOFS << std::endl << " Error in ScaLAPACK context for CheFSI in PWDFT ." << std::endl;
        exit(1);
      }
    }
    else
    {
      // Make the Cholesky factorization call on proc 0
      // This is the non-scalable part and should be fixed later
      if ( mpirank == 0) {
        GetTime( timeSta );
        lapack::Potrf( 'U', width, square_mat.Data(), width );
        GetTime( timeEnd );
        iterMpirank0 = iterMpirank0 + 1;
        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
      }

    }

    // Regardless of how this was solved (ScaLAPACK or locally), send the Cholesky factor to every process
    MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);


    // Do a solve with the Cholesky factor
    // X = X * U^{-1} is orthogonal, where U is the Cholesky factor
    GetTime( timeSta );
    blas::Trsm( 'R', 'U', 'N', 'N', heightLocal, width, 1.0, square_mat.Data(), width, 
        X.Data(), heightLocal );
    GetTime( timeEnd );
    iterTrsm = iterTrsm + 1;
    timeTrsm = timeTrsm + ( timeEnd - timeSta );

    // ~~ So X is now orthonormalized AND in row-distributed format
    // ~~ Switch back to column-distributed format in preparation 
    // for applying Hamiltonian (Raleigh-Ritz step)
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

    GetTime( extra_timeEnd );
    statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";


    // Step 3c : Raleigh-Ritz step
    GetTime( extra_timeSta );
    statusOFS << std::endl << " Raleigh-Ritz step  ... "; 

    // ~~ Applying the Hamiltonian matrix
    // HXcol = H * Xcol : Both are in height * local_width dimensioned format
    {
      GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, noccLocal, false, Xcol.Data());
      NumTns<double> tnsTemp(ntot, ncom, noccLocal, false, HXcol.Data());

      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

    // ~~ Flip into the alternate distribution for linear algebra
    // So data goes from Xcol to X
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

    // As well as HXcol to HX
    GetTime( timeSta );
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendbuf[sendk(i, j)] = HXcol(i, j); 
      }
    }
    MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
    for( Int j = 0; j < width; j++ ){ 
      for( Int i = 0; i < heightLocal; i++ ){
        HX(i, j) = recvbuf[recvk(i, j)];
      }
    }
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

    // ~~ Now compute the matrix for the projected problem
    //square_mat = X' * (HX)
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
        heightLocal, HX.Data(), heightLocal, 0.0, square_mat_temp.Data(), width );
    SetValue(square_mat , 0.0 );
    MPI_Allreduce( square_mat_temp.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    // ~~ Now solve the eigenvalue problem
    SetValue(eig_vals_Raleigh_Ritz, 0.0);

    if(PWDFT_Cheby_use_scala_ == 1)
    {
      if( contxt_ >= 0 )
      {
        Int numKeep = width; 
        Int lda = width;

        scalapack::ScaLAPACKMatrix<Real> square_mat_scala;
        scalapack::ScaLAPACKMatrix<Real> eigvecs_scala;

        scalapack::Descriptor descReduceSeq, descReducePar;
        Real timeEigScala_sta, timeEigScala_end;

        // Leading dimension provided
        descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt_, lda );

        // Automatically comptued Leading Dimension
        descReducePar.Init( numKeep, numKeep, scaBlockSize_, scaBlockSize_, I_ZERO, I_ZERO, contxt_ );

        square_mat_scala.SetDescriptor( descReducePar );
        eigvecs_scala.SetDescriptor( descReducePar );


        // Redistribute the input matrix over the process grid
        SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
            &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt_ );


        // Make the ScaLAPACK call
        char uplo = 'U';
        std::vector<Real> temp_eigs(lda);

        scalapack::Syevd(uplo, square_mat_scala, temp_eigs, eigvecs_scala );

        for(Int copy_iter = 0; copy_iter < lda; copy_iter ++)
          eig_vals_Raleigh_Ritz[copy_iter] = temp_eigs[copy_iter];

        // Redistribute back eigenvectors
        SetValue(square_mat , 0.0 );
        SCALAPACK(pdgemr2d)( &numKeep, &numKeep, eigvecs_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
            square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt_ );

        GetTime( timeEigScala_end );

        //statusOFS << std::endl << " ScaLAPACK EigenSolve time = " << (timeEigScala_end - timeEigScala_sta) << " s." << std::endl;


      }
      else
      {
        statusOFS << std::endl << " Error in ScaLAPACK context for CheFSI in PWDFT ." << std::endl;
        exit(1);
      }




    }   
    else
    {  

      // Make the eigen decomposition call on proc 0
      // This is the non-scalable part and should be fixed later
      if ( mpirank == 0 ) {

        GetTime( timeSta );
        lapack::Syevd( 'V', 'U', width, square_mat.Data(), width, eig_vals_Raleigh_Ritz.Data() );
        GetTime( timeEnd );

        iterMpirank0 = iterMpirank0 + 1;
        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

      }

    }

    // ~~ Send the results to every process regardless of whether ScaLAPACK or local solve was used
    MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, mpi_comm); // Eigen-vectors
    MPI_Bcast(eig_vals_Raleigh_Ritz.Data(), width, MPI_DOUBLE, 0, mpi_comm); // Eigen-values


    // Step 3d : Subspace rotation step : psi <-- psi * Q, where Q are the eigen-vectors
    // from the Raleigh-Ritz step.
    // Do this on the X matrix :  X is heightLocal * width and Q is width * width, 
    // So, this can be done independently on each process


    // We copy X to HX for saving space before multiplying
    // Results are finally stored in X

    // ~~ So copy X to HX 
    lapack::Lacpy( 'A', heightLocal, width, X.Data(),  heightLocal, HX.Data(), heightLocal );

    // ~~ Gemm: X <-- HX (= X) * Q
    GetTime( timeSta );
    blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, HX.Data(),
        heightLocal, square_mat.Data(), width, 0.0, X.Data(), heightLocal );    
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    // ~~ Flip X to Xcol, i.e. switch back to column-distributed format 
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

    // ~~ Copy Xcol to psiPtr_
    lapack::Lacpy( 'A', height, widthLocal, Xcol.Data(), height, 
        psiPtr_->Wavefun().Data(), height );

    // Step 3e : Reset the upper and lower bounds using the results of
    // the Raleigh-Ritz step
    b_low = eig_vals_Raleigh_Ritz(width - 1);
    a_L = eig_vals_Raleigh_Ritz(0);

    GetTime( extra_timeEnd );
    statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

    //statusOFS << std::endl << " Intermediate eig vals " << std::endl<< eig_vals_Raleigh_Ritz;


    // ----------------------------------------------------------
    // As a postprocessing step, compute the residual of the eigenvectors
    // Do this by R = HX - X * (X^T * H * X)
    {
      double res_time_sta, res_time_end;

      GetTime( res_time_sta );

      // First compute HX
      // ~~ Applying the Hamiltonian matrix
      // HXcol = H * Xcol : Both are in height * local_width dimensioned format
      {
        GetTime( timeSta );
        Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, noccLocal, false, Xcol.Data());
        NumTns<double> tnsTemp(ntot, ncom, noccLocal, false, HXcol.Data());

        hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
        GetTime( timeEnd );
        iterSpinor = iterSpinor + 1;
        timeSpinor = timeSpinor + ( timeEnd - timeSta );
      }

      // ~~ Flip into the alternate distribution for linear algebra
      // So data goes from Xcol to X
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

      // As well as HXcol to HX
      GetTime( timeSta );
      for( Int j = 0; j < widthLocal; j++ ){ 
        for( Int i = 0; i < height; i++ ){
          sendbuf[sendk(i, j)] = HXcol(i, j); 
        }
      }
      MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
          &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
      for( Int j = 0; j < width; j++ ){ 
        for( Int i = 0; i < heightLocal; i++ ){
          HX(i, j) = recvbuf[recvk(i, j)];
        }
      }
      GetTime( timeEnd );
      iterAlltoallv = iterAlltoallv + 1;
      timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

      //square_mat = X' * (HX)
      GetTime( timeSta );
      blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
          heightLocal, HX.Data(), heightLocal, 0.0, square_mat_temp.Data(), width );
      SetValue(square_mat , 0.0 );
      MPI_Allreduce( square_mat_temp.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      // So square_mat = X^T * (H * X)
      // Now compute HX - X* square_mat (i.e., the second part is like a subspace rotation)

      // Storage for residuals
      DblNumMat  Res( heightLocal, width);

      // Set Res <-- X
      lapack::Lacpy( 'A', heightLocal, width, HX.Data(),  heightLocal, Res.Data(), heightLocal );


      // ~~ Gemm: Res <-- X * Q - Res (= X)
      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, X.Data(),
          heightLocal, square_mat.Data(), width, -1.0, Res.Data(), heightLocal );    
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );


      // Compute the norm of the residuals
      DblNumVec  resNorm( width );
      SetValue( resNorm, 0.0 );
      DblNumVec  resNormLocal ( width ); 
      SetValue( resNormLocal, 0.0 );

      for( Int k = 0; k < width; k++ ){
        resNormLocal(k) = Energy(DblNumVec(heightLocal, false, Res.VecData(k)));
      }

      MPI_Allreduce( resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE, 
          MPI_SUM, mpi_comm );

      if ( mpirank == 0 ){
        GetTime( timeSta );
        for( Int k = 0; k < width; k++ ){
          resNorm(k) = std::sqrt( resNorm(k) );
        }
        GetTime( timeEnd );
        timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
      }
      MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm);

      double resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
      double resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

      statusOFS << std::endl << " Maximum residual = " << resMax;
      statusOFS << std::endl << " Minimum residual = " << resMin;
      statusOFS << std::endl;

#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "resNorm = " << resNorm << std::endl;
#endif

      resVal_ = resNorm;


      GetTime( res_time_end );
      statusOFS << std::endl << " Time for computing residual = " << (res_time_end - res_time_sta) << " s.";
      statusOFS << std::endl;


    } // End of residual computation


  } // Loop for(Int iter = 1; iter <= Iter_Max; iter ++)



  // Save the eigenvalues to the eigensolver data structure  
  eigVal_ = DblNumVec( width, true, eig_vals_Raleigh_Ritz.Data() );    


  return;
} // -----  end of method EigenSolver::FirstChebyStep -----

void
EigenSolver::GeneralChebyStep    (
    Int          numEig,
    Int        filter_order )
{

  statusOFS << std::endl << std::endl << " Subsequent CheFSI for PWDFT ... " << std::endl;
  if(PWDFT_Cheby_use_scala_ == 1)
    statusOFS << " Cholesky factorlization and eigen-decomposition to be done using ScaLAPACK." << std::endl;
  else
    statusOFS << " Cholesky factorlization and eigen-decomposition to be done serially. " << std::endl;


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

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  if( widthLocal != noccLocal ){
    ErrorHandling("widthLocal != noccLocal.");
  }

  // Temporary safeguard 
  if(noccTotal % mpisize != 0)
  {
    MPI_Barrier(mpi_comm);  
    statusOFS << std::endl << std::endl 
      <<" Input Error ! Currently CheFSI within PWDFT requires total number of bands to be divisble by mpisize. " << std::endl << " Total No. of states = " 
      << noccTotal << " , mpisize = " << mpisize << " ." 
      << std::endl <<  " Use different parameters." << std::endl << " Aborting ..." << std::endl << std::endl;
    MPI_Barrier(mpi_comm);
    exit(-1);  
  }    

  // Time for GemmT, GemmN, Alltoallv, Spinor, Mpirank0 
  // GemmT: blas::Gemm( 'T', 'N')
  // GemmN: blas::Gemm( 'N', 'N')
  // Alltoallv: row-partition to column partition via MPI_Alltoallv 
  // Spinor: Applying the Hamiltonian matrix 
  // Mpirank0: Serial calculation part

  Real timeSta, timeEnd;
  Real extra_timeSta, extra_timeEnd;
  Real timeGemmT = 0.0;
  Real timeGemmN = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeSpinor = 0.0;
  Real timeTrsm = 0.0;
  Real timeSyevd = 0.0;
  Real timeMpirank0 = 0.0;
  Int  iterGemmT = 0;
  Int  iterGemmN = 0;
  Int  iterAlltoallv = 0;
  Int  iterSpinor = 0;
  Int  iterTrsm = 0;
  Int  iterSyevd = 0;
  Int  iterMpirank0 = 0;

  if( numEig > width ){
    std::ostringstream msg;
    msg 
      << "Number of eigenvalues requested  = " << numEig << std::endl
      << "which is larger than the number of columns in psi = " << width << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  // The following codes are not replaced by AlltoallForward /
  // AlltoallBackward since they are repetitively used in the
  // eigensolver.
  //
  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
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

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
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



  // TODO: In what follows, we do not care about multiple spinor components
  // For now, just a spin unpolarized setup is considered, so ntot = height
  // TODO: In what follows, we assume real valued wavefunctions

  // Step 1: Obtain the upper bound and the Ritz values (for the lower bound)
  // using the Lanczos estimator
  DblNumVec Lanczos_Ritz_values;

  statusOFS << std::endl << " Estimating the upper bound ... "; 
  GetTime( extra_timeSta );
  const int Num_Lanczos_Steps = 5;
  double b_up = Cheby_Upper_bound_estimator(Lanczos_Ritz_values, Num_Lanczos_Steps);  
  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";


  // Step 2: Set up the lower bound from previous Raleigh-Ritz / Eigensolver values
  double a_L = eigVal_(0);
  double b_low = eigVal_(width - 1);

  statusOFS << std::endl << " Upper bound = (to be mapped to +1) " << b_up;
  statusOFS << std::endl << " Lower bound (to be mapped to -1) = " << b_low;
  statusOFS << std::endl << " Lowest eigenvalue = " << a_L;



  // Step 3: Main loop
  const Int Filter_Order = filter_order ;

  // Space for linear algebra operations
  DblNumMat  X( heightLocal, width);
  DblNumMat  HX( heightLocal, width);

  DblNumMat  Xcol( height, widthLocal );
  DblNumMat  HXcol( height, widthLocal );

  DblNumMat  square_mat( width, width);
  DblNumMat  square_mat_temp( width, width);

  DblNumVec eig_vals_Raleigh_Ritz;
  eig_vals_Raleigh_Ritz.Resize(width);



  // Step 3a : Compute the filtered block of vectors
  // This always works on the vectors in psiPtr_
  GetTime( extra_timeSta );
  Chebyshev_filter_scaled(Filter_Order, b_low, b_up, a_L);
  GetTime( extra_timeEnd );

  //       GetTime( extra_timeSta );
  //       Chebyshev_filter(Filter_Order, b_low, b_up);
  //       GetTime( extra_timeEnd );

  statusOFS << std::endl << " Chebyshev filter applied in " << (extra_timeEnd - extra_timeSta ) << " s."; 




  // Step 3b : Orthonormalize
  // ~~ First copy psi (now filtered) to Xcol --- can do away with this to save memory

  GetTime( extra_timeSta );
  statusOFS << std::endl << " Orthonormalizing  ... "; 
  lapack::Lacpy( 'A', height, widthLocal, psiPtr_->Wavefun().Data(), height, 
      Xcol.Data(), height );

  GetTime( timeSta );

  // ~~ Flip into the alternate distribution in prep for Orthogonalization
  // So data goes from Xcol to X
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

  // ~~ Orthogonalization through Cholesky factorization
  // Compute square_mat = X^T * X for Cholesky
  GetTime( timeSta );
  blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, X.Data(), heightLocal, 0.0, square_mat_temp.Data(), width );
  SetValue( square_mat, 0.0 );
  MPI_Allreduce( square_mat_temp.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );


  if(PWDFT_Cheby_use_scala_ == 1)
  {
    if( contxt_ >= 0 )
    {
      Int numKeep = width; 
      Int lda = width;

      scalapack::ScaLAPACKMatrix<Real> square_mat_scala;
      scalapack::Descriptor descReduceSeq, descReducePar;
      Real timeCholScala_sta, timeCholScala_end;

      GetTime( timeCholScala_sta );

      // Leading dimension provided
      descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt_, lda );

      // Automatically comptued Leading Dimension
      descReducePar.Init( numKeep, numKeep, scaBlockSize_, scaBlockSize_, I_ZERO, I_ZERO, contxt_ );

      square_mat_scala.SetDescriptor( descReducePar );

      // Redistribute the matrix over the process grid
      SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
          &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt_ );

      // Make the ScaLAPACK call
      Int info;
      char uplo = 'U';
      SCALAPACK(pdpotrf)(&uplo, &numKeep, square_mat_scala.Data(), &I_ONE,
          &I_ONE, square_mat_scala.Desc().Values(), &info);

      // Redistribute back
      SetValue(square_mat , 0.0 );
      SCALAPACK(pdgemr2d)( &numKeep, &numKeep, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
          square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt_ );

      GetTime( timeCholScala_end );

      //statusOFS << std::endl << " ScaLAPACK Cholesky time = " << (timeCholScala_end - timeCholScala_sta) << " s." << std::endl;

    }
    else
    {
      statusOFS << std::endl << " Error in ScaLAPACK context for CheFSI in PWDFT ." << std::endl;
      exit(1);
    }
  }
  else
  {
    // Make the Cholesky factorization call on proc 0
    // This is the non-scalable part and should be fixed later
    if ( mpirank == 0) {
      GetTime( timeSta );
      lapack::Potrf( 'U', width, square_mat.Data(), width );
      GetTime( timeEnd );
      iterMpirank0 = iterMpirank0 + 1;
      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
    }

  }

  // Regardless of how this was solved (ScaLAPACK or locally), send the Cholesky factor to every process
  MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);


  // Do a solve with the Cholesky factor
  // X = X * U^{-1} is orthogonal, where U is the Cholesky factor
  GetTime( timeSta );
  blas::Trsm( 'R', 'U', 'N', 'N', heightLocal, width, 1.0, square_mat.Data(), width, 
      X.Data(), heightLocal );
  GetTime( timeEnd );
  iterTrsm = iterTrsm + 1;
  timeTrsm = timeTrsm + ( timeEnd - timeSta );

  // ~~ So X is now orthonormalized AND in row-distributed format
  // ~~ Switch back to column-distributed format in preparation 
  // for applying Hamiltonian (Raleigh-Ritz step)
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

  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";


  // Step 3c : Raleigh-Ritz step
  GetTime( extra_timeSta );
  statusOFS << std::endl << " Raleigh-Ritz step  ... "; 

  // ~~ Applying the Hamiltonian matrix
  // HXcol = H * Xcol : Both are in height * local_width dimensioned format
  {
    GetTime( timeSta );
    Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, noccLocal, false, Xcol.Data());
    NumTns<double> tnsTemp(ntot, ncom, noccLocal, false, HXcol.Data());

    hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
    GetTime( timeEnd );
    iterSpinor = iterSpinor + 1;
    timeSpinor = timeSpinor + ( timeEnd - timeSta );
  }

  // ~~ Flip into the alternate distribution for linear algebra
  // So data goes from Xcol to X
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

  // As well as HXcol to HX
  GetTime( timeSta );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      sendbuf[sendk(i, j)] = HXcol(i, j); 
    }
  }
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      HX(i, j) = recvbuf[recvk(i, j)];
    }
  }
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  // ~~ Now compute the matrix for the projected problem
  //square_mat = X' * (HX)
  GetTime( timeSta );
  blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
      heightLocal, HX.Data(), heightLocal, 0.0, square_mat_temp.Data(), width );
  SetValue(square_mat , 0.0 );
  MPI_Allreduce( square_mat_temp.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );

  // ~~ Now solve the eigenvalue problem
  SetValue(eig_vals_Raleigh_Ritz, 0.0);

  if(PWDFT_Cheby_use_scala_ == 1)
  {
    if( contxt_ >= 0 )
    {
      Int numKeep = width; 
      Int lda = width;

      scalapack::ScaLAPACKMatrix<Real> square_mat_scala;
      scalapack::ScaLAPACKMatrix<Real> eigvecs_scala;

      scalapack::Descriptor descReduceSeq, descReducePar;
      Real timeEigScala_sta, timeEigScala_end;

      // Leading dimension provided
      descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt_, lda );

      // Automatically comptued Leading Dimension
      descReducePar.Init( numKeep, numKeep, scaBlockSize_, scaBlockSize_, I_ZERO, I_ZERO, contxt_ );

      square_mat_scala.SetDescriptor( descReducePar );
      eigvecs_scala.SetDescriptor( descReducePar );


      // Redistribute the input matrix over the process grid
      SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
          &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt_ );


      // Make the ScaLAPACK call
      char uplo = 'U';
      std::vector<Real> temp_eigs(lda);

      scalapack::Syevd(uplo, square_mat_scala, temp_eigs, eigvecs_scala );

      for(Int copy_iter = 0; copy_iter < lda; copy_iter ++)
        eig_vals_Raleigh_Ritz[copy_iter] = temp_eigs[copy_iter];

      // Redistribute back eigenvectors
      SetValue(square_mat , 0.0 );
      SCALAPACK(pdgemr2d)( &numKeep, &numKeep, eigvecs_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
          square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt_ );

      GetTime( timeEigScala_end );

      //statusOFS << std::endl << " ScaLAPACK EigenSolve time = " << (timeEigScala_end - timeEigScala_sta) << " s." << std::endl;


    }
    else
    {
      statusOFS << std::endl << " Error in ScaLAPACK context for CheFSI in PWDFT ." << std::endl;
      exit(1);
    }




  }   
  else
  {  

    // Make the eigen decomposition call on proc 0
    // This is the non-scalable part and should be fixed later
    if ( mpirank == 0 ) {

      GetTime( timeSta );
      lapack::Syevd( 'V', 'U', width, square_mat.Data(), width, eig_vals_Raleigh_Ritz.Data() );
      GetTime( timeEnd );

      iterMpirank0 = iterMpirank0 + 1;
      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );

    }

  }

  // ~~ Send the results to every process regardless of whether ScaLAPACK or local solve was used
  MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, mpi_comm); // Eigen-vectors
  MPI_Bcast(eig_vals_Raleigh_Ritz.Data(), width, MPI_DOUBLE, 0, mpi_comm); // Eigen-values

  // Step 3d : Subspace rotation step : psi <-- psi * Q, where Q are the eigen-vectors
  // from the Raleigh-Ritz step.
  // Do this on the X matrix :  X is heightLocal * width and Q is width * width, 
  // So, this can be done independently on each process


  // We copy X to HX for saving space before multiplying
  // Results are finally stored in X

  // ~~ So copy X to HX 
  lapack::Lacpy( 'A', heightLocal, width, X.Data(),  heightLocal, HX.Data(), heightLocal );

  // ~~ Gemm: X <-- HX (= X) * Q
  GetTime( timeSta );
  blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, HX.Data(),
      heightLocal, square_mat.Data(), width, 0.0, X.Data(), heightLocal );    
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );

  // ~~ Flip X to Xcol, i.e. switch back to column-distributed format 
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

  // ~~ Copy Xcol to psiPtr_ to save the eigenvectors
  lapack::Lacpy( 'A', height, widthLocal, Xcol.Data(), height, 
      psiPtr_->Wavefun().Data(), height );

  GetTime( extra_timeEnd );
  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)" << std::endl;

  // Save the eigenvalues to the eigensolver data structure     
  eigVal_ = DblNumVec( width, true, eig_vals_Raleigh_Ritz.Data() );

  //statusOFS << std::endl << " Intermediate eig vals " << std::endl<< eig_vals_Raleigh_Ritz;
  // ----------------------------------------------------------
  // Finally, as a postprocessing step, compute the residual of the eigenvectors
  // Do this by R = HX - X * (X^T * H * X)
  {
    double res_time_sta, res_time_end;

    GetTime( res_time_sta );
    // First compute HX
    // ~~ Applying the Hamiltonian matrix
    // HXcol = H * Xcol : Both are in height * local_width dimensioned format
    {
      GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, noccLocal, false, Xcol.Data());
      NumTns<double> tnsTemp(ntot, ncom, noccLocal, false, HXcol.Data());

      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

    // ~~ Flip into the alternate distribution for linear algebra
    // So data goes from Xcol to X
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

    // As well as HXcol to HX
    GetTime( timeSta );
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendbuf[sendk(i, j)] = HXcol(i, j); 
      }
    }
    MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
    for( Int j = 0; j < width; j++ ){ 
      for( Int i = 0; i < heightLocal; i++ ){
        HX(i, j) = recvbuf[recvk(i, j)];
      }
    }
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

    //square_mat = X' * (HX)
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
        heightLocal, HX.Data(), heightLocal, 0.0, square_mat_temp.Data(), width );
    SetValue(square_mat , 0.0 );
    MPI_Allreduce( square_mat_temp.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    // So square_mat = X^T * (H * X)
    // Now compute HX - X* square_mat (i.e., the second part is like a subspace rotation)

    // Storage for residuals
    DblNumMat  Res( heightLocal, width);

    // Set Res <-- X
    lapack::Lacpy( 'A', heightLocal, width, HX.Data(),  heightLocal, Res.Data(), heightLocal );


    // ~~ Gemm: Res <-- X * Q - Res (= X)
    GetTime( timeSta );
    blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, X.Data(),
        heightLocal, square_mat.Data(), width, -1.0, Res.Data(), heightLocal );    
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );


    // Compute the norm of the residuals
    DblNumVec  resNorm( width );
    SetValue( resNorm, 0.0 );
    DblNumVec  resNormLocal ( width ); 
    SetValue( resNormLocal, 0.0 );

    for( Int k = 0; k < width; k++ ){
      resNormLocal(k) = Energy(DblNumVec(heightLocal, false, Res.VecData(k)));
    }

    MPI_Allreduce( resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE, 
        MPI_SUM, mpi_comm );

    if ( mpirank == 0 ){
      GetTime( timeSta );
      for( Int k = 0; k < width; k++ ){
        resNorm(k) = std::sqrt( resNorm(k) );
      }
      GetTime( timeEnd );
      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
    }
    MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm);

    double resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
    double resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

    statusOFS << std::endl << " Maximum residual = " << resMax;
    statusOFS << std::endl << " Minimum residual = " << resMin;
    statusOFS << std::endl;

    resVal_ = resNorm;

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "resNorm = " << resNorm << std::endl;
#endif

    GetTime( res_time_end );
    statusOFS << std::endl << " Time for computing residual = " << (res_time_end - res_time_sta) << " s.";
    statusOFS << std::endl;


  }








  return;
} // -----  end of method EigenSolver::GeneralChebyStep -----



#ifdef GPU
void
EigenSolver::PPCGSolveReal    (
    Int          numEig,
    Int          eigMaxIter,
    Real         eigMinTolerance,
    Real         eigTolerance)
{
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

  /* init the cublas */
  cublasStatus_t status;
  cublasSideMode_t right  = CUBLAS_SIDE_RIGHT;
  cublasFillMode_t up     = CUBLAS_FILL_MODE_UPPER;
  cublasDiagType_t nondiag   = CUBLAS_DIAG_NON_UNIT;
  cublasOperation_t cu_transT = CUBLAS_OP_T;
  cublasOperation_t cu_transN = CUBLAS_OP_N;
  cublasOperation_t cu_transC = CUBLAS_OP_C;
  cublas::Init();
  std::cout << " GPU PPCG ........... " << std::endl;

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

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  if( widthLocal != noccLocal ){
    throw std::logic_error("widthLocal != noccLocal.");
  }


  Real time1, time2;
  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;
  Real timeSta2, timeEnd2;
  Real timeHpsi = 0.0;
  Real timeGemmT = 0.0;
  Real timeGemmN = 0.0;
  Real timeBcast = 0.0;
  Real timeAllreduce = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeMapping = 0.0;
  Real timeAlltoallvMap = 0.0;
  Real timeSpinor = 0.0;
  Real timeTrsm = 0.0;
  Real timePotrf = 0.0;
  Real timeSyevd = 0.0;
  Real timeSygvd = 0.0;
  Real timeMpirank0 = 0.0;
  Real timeScaLAPACKFactor = 0.0;
  Real timeScaLAPACK = 0.0;
  Real timeSweepT = 0.0;
  Real timeCopy = 0.0;
  Real timeOther = 0.0;
  Int  iterGemmT = 0;
  Int  iterGemmN = 0;
  Int  iterBcast = 0;
  Int  iterAllreduce = 0;
  Int  iterAlltoallv = 0;
  Int  iterAlltoallvMap = 0;
  Int  iterSpinor = 0;
  Int  iterTrsm = 0;
  Int  iterPotrf = 0;
  Int  iterSyevd = 0;
  Int  iterSygvd = 0;
  Int  iterMpirank0 = 0;
  Int  iterScaLAPACKFactor = 0;
  Int  iterScaLAPACK = 0;
  Int  iterSweepT = 0;
  Int  iterCopy = 0;
  Int  iterOther = 0;

  if( numEig > width ){
    std::ostringstream msg;
    msg 
      << "Number of eigenvalues requested  = " << numEig << std::endl
      << "which is larger than the number of columns in psi = " << width << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }

  GetTime( timeSta2 );

  // The following codes are not replaced by AlltoallForward /
  // AlltoallBackward since they are repetitively used in the
  // eigensolver.
  //
  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );
  cuIntNumMat cu_sendk( height, widthLocal );
  cuIntNumMat cu_recvk( heightLocal, width );
  cuDblNumVec cu_sendbuf(height*widthLocal); 
  cuDblNumVec cu_recvbuf(heightLocal*width);

  GetTime( timeSta );
  
  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
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

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
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

  cu_sendk.CopyFrom(sendk);
  cu_recvk.CopyFrom(recvk);
 
  GetTime( timeEnd );
  iterAlltoallvMap = iterAlltoallvMap + 1;
  timeAlltoallvMap = timeAlltoallvMap + ( timeEnd - timeSta );

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


  //    DblNumMat  AMat( 3*width, 3*width ), BMat( 3*width, 3*width );
  //    DblNumMat  AMatT1( 3*width, 3*width );

  // Temporary buffer array.
  // The unpreconditioned residual will also be saved in Xtemp
  DblNumMat  XTX( width, width );
  //? DblNumMat  XTXtemp( width, width );
  DblNumMat  XTXtemp1( width, width );

  DblNumMat  Xtemp( heightLocal, width );

  Real  resBlockNormLocal, resBlockNorm; // Frobenius norm of the residual block  
  Real  resMax, resMin;

  // For convenience
  DblNumMat  X( heightLocal, width, false, S.VecData(0) );
  DblNumMat  W( heightLocal, width, false, S.VecData(width) );
  DblNumMat  P( heightLocal, width, false, S.VecData(2*width) );
  DblNumMat AX( heightLocal, width, false, AS.VecData(0) );
  DblNumMat AW( heightLocal, width, false, AS.VecData(width) );
  DblNumMat AP( heightLocal, width, false, AS.VecData(2*width) );

  DblNumMat  Xcol( height, widthLocal );
  DblNumMat  Wcol( height, widthLocal );
  DblNumMat AXcol( height, widthLocal );
  DblNumMat AWcol( height, widthLocal );

  // for GPU. please note we need to use copyTo adn copyFrom in the GPU matrix 
  cuDblNumMat cu_XTX(width, width);
  cuDblNumMat cu_XTXtemp1(width, width);
  cuDblNumMat cu_X(heightLocal, width);
  cuDblNumMat cu_Xtemp(heightLocal, width);
  cuDblNumMat cu_W(heightLocal, width);
  cuDblNumMat cu_AW(heightLocal, width);
  cuDblNumMat cu_P(heightLocal, width);
  cuDblNumMat cu_AP(heightLocal, width);
  cuDblNumMat cu_AX(heightLocal, width);
  
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

  const Int numLocked = 0;  // Never perform locking in this version
  const Int numActive = width;

  bool isConverged = false;

  // Initialization
#if 0
  SetValue( S, 0.0 );
  SetValue( AS, 0.0 );
#endif

  DblNumVec  eigValS(lda);
  SetValue( eigValS, 0.0 );

  // Initialize X by the data in psi
  GetTime( timeSta );
#if 0
  lapack::Lacpy( 'A', height, widthLocal, psiPtr_->Wavefun().Data(), height, 
      Xcol.Data(), height );
#endif
  GetTime( timeEnd );
  iterCopy = iterCopy + 1;
  timeCopy = timeCopy + ( timeEnd - timeSta );

  GetTime( timeSta );
  
  cuda_memcpy_CPU2GPU(cu_X.Data(), psiPtr_->Wavefun().Data(), sizeof(Real)*heightLocal*width);
  cuda_mapping_to_buf( cu_sendbuf.Data(), cu_X.Data(), cu_sendk.Data(), heightLocal*width);
  cuda_memcpy_GPU2CPU( sendbuf.Data(), cu_sendbuf.Data(), sizeof(Real)*heightLocal*width);

  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;

  GetTime( timeSta );
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
  GetTime( timeEnd );

  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );
  GetTime( timeSta );
  
  cuda_memcpy_CPU2GPU(cu_recvbuf.Data(), recvbuf.Data(), sizeof(Real)*heightLocal*width);
  cuda_mapping_from_buf(cu_X.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);
  cu_X.CopyTo(X);

  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;

  // *********************************************************************
  // Main loop
  // *********************************************************************

  // Orthogonalization through Cholesky factorization
  GetTime( timeSta );
 
  Real one = 1.0;
  Real minus_one = -1.0;
  Real zero = 0.0;

  cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(), 
      heightLocal, cu_X.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
  cu_XTXtemp1.CopyTo(XTXtemp1);

  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
  GetTime( timeSta );

  MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterAllreduce = iterAllreduce + 1;
  timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

//  if ( mpirank == 0) {  
// each node do the Potrf, without the MPI_Bcast.
  GetTime( timeSta );
  lapack::Potrf( 'U', width, XTX.Data(), width );
  GetTime( timeEnd );
  iterMpirank0 = iterMpirank0 + 1;
  timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
//  }
  GetTime( timeSta );
//  MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
  GetTime( timeEnd );
  iterBcast = iterBcast + 1;
  timeBcast = timeBcast + ( timeEnd - timeSta );

  // X <- X * U^{-1} is orthogonal
  GetTime( timeSta );

  cu_X.CopyFrom(X);
  cu_XTX.CopyFrom( XTX );
  cublas::Trsm( right, up, cu_transN, nondiag, heightLocal, width, &one, cu_XTX.Data(), width, cu_X.Data(), heightLocal );
  cu_XTX.CopyTo( XTX );
  
  GetTime( timeEnd );
  iterTrsm = iterTrsm + 1;
  timeTrsm = timeTrsm + ( timeEnd - timeSta );

  GetTime( timeSta );

  cuda_mapping_to_buf( cu_recvbuf.Data(), cu_X.Data(), cu_recvk.Data(), heightLocal*width);
  cuda_memcpy_GPU2CPU( recvbuf.Data(), cu_recvbuf.Data(), sizeof(Real)*heightLocal*width);
  
  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;

  GetTime( timeSta );
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  GetTime( timeSta );
  cuda_memcpy_CPU2GPU(cu_sendbuf.Data(), sendbuf.Data(), sizeof(Real)*heightLocal*width);
  cuda_mapping_from_buf(cu_X.Data(), cu_sendbuf.Data(), cu_sendk.Data(), heightLocal*width);
  cu_X.CopyTo( Xcol );

  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;

  // Applying the Hamiltonian matrix
  {
    GetTime( timeSta );
    Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, noccLocal, false, Xcol.Data());
    NumTns<Real> tnsTemp(ntot, ncom, noccLocal, false, AXcol.Data());

    hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
    GetTime( timeEnd );
    iterSpinor = iterSpinor + 1;
    timeSpinor = timeSpinor + ( timeEnd - timeSta );
  }

  GetTime( timeSta );
  cu_X.CopyFrom(Xcol);
  cuda_mapping_to_buf( cu_sendbuf.Data(), cu_X.Data(), cu_sendk.Data(), heightLocal*width);
  cuda_memcpy_GPU2CPU( sendbuf.Data(), cu_sendbuf.Data(), sizeof(Real)*heightLocal*width);
  
  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;
  GetTime( timeSta );
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  GetTime( timeSta );
  
  cuda_memcpy_CPU2GPU(cu_recvbuf.Data(), recvbuf.Data(), sizeof(Real)*heightLocal*width);
  cuda_mapping_from_buf(cu_X.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);
  cu_X.CopyTo(X);

  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;

  GetTime( timeSta );

  cu_X.CopyFrom(AXcol);
  cuda_mapping_to_buf( cu_sendbuf.Data(), cu_X.Data(), cu_sendk.Data(), heightLocal*width);
  cuda_memcpy_GPU2CPU( sendbuf.Data(), cu_sendbuf.Data(), sizeof(Real)*heightLocal*width);
  
  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;
  GetTime( timeSta );
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  GetTime( timeSta );
  cuda_memcpy_CPU2GPU(cu_recvbuf.Data(), recvbuf.Data(), sizeof(Real)*heightLocal*width);
  cuda_mapping_from_buf(cu_X.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);
  cu_X.CopyTo(AX);

  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;


  // Start the main loop
  Int iter = 0;
  statusOFS << "Minimum tolerance is " << eigMinTolerance << std::endl;

  // GPU arrays and init....
  
  do{
    iter++;
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "iter = " << iter << std::endl;
#endif

    if( iter == 1 || isRestart == true )
      numSet = 2;
    else
      numSet = 3;

    // XTX <- X' * (AX)
    GetTime( timeSta );
    cu_X.CopyFrom(X);
    cu_AX.CopyFrom(AX);
    cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
                  heightLocal, cu_AX.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
    cu_XTXtemp1.CopyTo(XTXtemp1);

    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    // Compute the residual.
    // R <- AX - X*(X'*AX)
    GetTime( timeSta );
    lapack::Lacpy( 'A', heightLocal, width, AX.Data(), heightLocal, Xtemp.Data(), heightLocal );
    GetTime( timeEnd );
    iterCopy = iterCopy + 1;
    timeCopy = timeCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    cu_XTX.CopyFrom(XTX);
    cu_Xtemp.CopyFrom(Xtemp);
    cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one, cu_X.Data(),
                  heightLocal, cu_XTX.Data(), width, &one, cu_Xtemp.Data(), heightLocal );
    cu_Xtemp.CopyTo(Xtemp);
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 1;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );

    // Compute the Frobenius norm of the residual block

    // LOCKING not supported, PPCG needs Rayleigh--Ritz to lock         
    //        numActiveTotal = width - numLockedTotal;
    //        numActiveLocal = widthLocal - numLockedLocal;

    // Compute the preconditioned residual W = T*R.
    // The residual is saved in Xtemp

    // Convert from row format to column format.
    // MPI_Alltoallv
    // Only convert Xtemp here

    GetTime( time1);
    GetTime( timeSta );
    
    cuda_mapping_to_buf( cu_recvbuf.Data(), cu_Xtemp.Data(), cu_recvk.Data(), heightLocal*width);
    cuda_memcpy_GPU2CPU( recvbuf.Data(), cu_recvbuf.Data(), sizeof(Real)*heightLocal*width);

    GetTime( timeEnd );
    timeMapping += timeEnd - timeSta;
    GetTime( timeSta );
    MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
        &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );
    GetTime( timeSta );
    
    cuda_memcpy_CPU2GPU(cu_recvbuf.Data(), sendbuf.Data(), sizeof(Real)*heightLocal*width);
    cuda_mapping_from_buf(cu_Xtemp.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);
    cu_Xtemp.CopyTo(Xcol);

    GetTime( timeEnd );
    timeMapping += timeEnd - timeSta;

    // Compute W = TW
    {
      GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, Xcol.VecData(numLockedLocal));
      NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));

      spnTemp.AddTeterPrecond( fftPtr_, tnsTemp );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }


    // Compute AW = A*W
    {
      GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));
      NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, AWcol.VecData(numLockedLocal));

      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

    // Convert from column format to row format
    // MPI_Alltoallv
    // Only convert W and AW

    GetTime( timeSta );
    
    cu_W.CopyFrom(Wcol);
    cuda_mapping_to_buf( cu_sendbuf.Data(), cu_W.Data(), cu_sendk.Data(), heightLocal*width);
    cuda_memcpy_GPU2CPU( sendbuf.Data(), cu_sendbuf.Data(), sizeof(Real)*heightLocal*width);

    GetTime( timeEnd );
    timeMapping += timeEnd - timeSta;
    GetTime( timeSta );
    MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );
    GetTime( timeSta );
    
    cuda_memcpy_CPU2GPU(cu_recvbuf.Data(), recvbuf.Data(), sizeof(Real)*heightLocal*width);
    cuda_mapping_from_buf(cu_W.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);

    GetTime( timeEnd );
    timeMapping += timeEnd - timeSta;

    GetTime( timeSta );

    cu_AW.CopyFrom(AWcol);
    cuda_mapping_to_buf( cu_sendbuf.Data(), cu_AW.Data(), cu_sendk.Data(), heightLocal*width);
    cuda_memcpy_GPU2CPU( sendbuf.Data(), cu_sendbuf.Data(), sizeof(Real)*heightLocal*width);
    
    GetTime( timeEnd );
    timeMapping += timeEnd - timeSta;
    GetTime( timeSta );
    MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );
    GetTime( timeSta );
    
    cuda_memcpy_CPU2GPU(cu_recvbuf.Data(), recvbuf.Data(), sizeof(Real)*heightLocal*width);
    cuda_mapping_from_buf(cu_AW.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);

    GetTime( timeEnd );
    GetTime( time2);
    timeHpsi += time2 - time1;
    timeMapping += timeEnd - timeSta;


    // W = W - X(X'W), AW = AW - AX(X'W)
    GetTime( timeSta );
    cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
                  heightLocal, cu_W.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
    cu_XTXtemp1.CopyTo( XTXtemp1);

    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );


    GetTime( timeSta );
    cu_XTX.CopyFrom(XTX);
    cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one, cu_X.Data(),
                  heightLocal, cu_XTX.Data(), width, &one, cu_W.Data(), heightLocal );
    cu_W.CopyTo(W);
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 1;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );


    GetTime( timeSta );
    cu_AX.CopyFrom(AX);
    cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one,
                  cu_AX.Data(), heightLocal, cu_XTX.Data(), width, &one, cu_AW.Data(), heightLocal );
    cu_AW.CopyTo(AW);
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 1;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );

    // Normalize columns of W
    Real normLocal[width]; 
    Real normGlobal[width];

    GetTime( timeSta );
    for( Int k = numLockedLocal; k < width; k++ ){
      normLocal[k] = Energy(DblNumVec(heightLocal, false, W.VecData(k)));
      normGlobal[k] = 0.0;
    }
    MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    for( Int k = numLockedLocal; k < width; k++ ){
      Real norm = std::sqrt( normGlobal[k] );
      blas::Scal( heightLocal, 1.0 / norm, W.VecData(k), 1 );
      blas::Scal( heightLocal, 1.0 / norm, AW.VecData(k), 1 );
    }
    GetTime( timeEnd );
    iterOther = iterOther + 2;
    timeOther = timeOther + ( timeEnd - timeSta );

    statusOFS << "Time for norm1 in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;


    // P = P - X(X'P), AP = AP - AX(X'P)
    if( numSet == 3 ){
      
      GetTime( timeSta );
      cu_P.CopyFrom( P );
      cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
                  heightLocal, cu_P.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
      cu_XTXtemp1.CopyTo( XTXtemp1 );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTX, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

      GetTime( timeSta );
      cu_XTX.CopyFrom( XTX );
      cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one,
                    cu_X.Data(), heightLocal, cu_XTX.Data(), width, &one, cu_P.Data(), heightLocal );
      cu_P.CopyTo( P );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );

      cu_AX.CopyFrom( AX );
      cu_AP.CopyFrom( AP );
      cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one,
                    cu_AX.Data(), heightLocal, cu_XTX.Data(), width, &one, cu_AP.Data(), heightLocal );
      cu_AP.CopyTo( AP );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      // Normalize the conjugate direction
      GetTime( timeSta );
      for( Int k = numLockedLocal; k < width; k++ ){
        normLocal[k] = Energy(DblNumVec(heightLocal, false, P.VecData(k)));
        normGlobal[k] = 0.0;
      }
      MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      for( Int k = numLockedLocal; k < width; k++ ){
        Real norm = std::sqrt( normGlobal[k] );
        blas::Scal( heightLocal, 1.0 / norm, P.VecData(k), 1 );
        blas::Scal( heightLocal, 1.0 / norm, AP.VecData(k), 1 );
      }
      GetTime( timeEnd );
      iterOther = iterOther + 2;
      timeOther = timeOther + ( timeEnd - timeSta );
    
      statusOFS << "Time for norm2 in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;
   
    }

    // Perform the sweep
    GetTime( timeSta );
    Int sbSize = 1, nsb = width; // this should be generalized to subblocks 
    DblNumMat  AMat( 3*sbSize, 3*sbSize ), BMat( 3*sbSize, 3*sbSize );
    DblNumMat  AMatAll( 3*sbSize, 3*sbSize*nsb ), BMatAll( 3*sbSize, 3*sbSize*nsb ); // contains all nsb 3-by-3 matrices
    DblNumMat  AMatAllLocal( 3*sbSize, 3*sbSize*nsb ), BMatAllLocal( 3*sbSize, 3*sbSize*nsb ); // contains local parts of all nsb 3-by-3 matrices

    // gpu
    cuDblNumMat cu_AMatAllLocal( 3*sbSize, 3*sbSize*nsb );
    cuDblNumMat cu_BMatAllLocal( 3*sbSize, 3*sbSize*nsb );

    SetValue( AMat, 0.0 ); SetValue( BMat, 0.0 );
    SetValue( AMatAll, 0.0 ); SetValue( BMatAll, 0.0 );
    SetValue( AMatAllLocal, 0.0 ); SetValue( BMatAllLocal, 0.0 );

    // LOCKING NOT SUPPORTED, loop over all columns 
    for( Int k = 0; k < nsb; k++ ){

      // fetch indiviual columns
      DblNumMat  x( heightLocal, sbSize, false, X.VecData(k) );
      DblNumMat  w( heightLocal, sbSize, false, W.VecData(k) );
      DblNumMat ax( heightLocal, sbSize, false, AX.VecData(k) );
      DblNumMat aw( heightLocal, sbSize, false, AW.VecData(k) );

      // gpu data structure. 
      cuDblNumMat cu_ax(heightLocal, sbSize);
      cuDblNumMat cu_x (heightLocal, sbSize);
      cuDblNumMat cu_w (heightLocal, sbSize);
      cuDblNumMat cu_aw(heightLocal, sbSize);
      // Compute AMatAllLoc and BMatAllLoc            
      // AMatAllLoc
      GetTime( timeSta );
      cu_x.CopyFrom( x );
      cu_ax.CopyFrom( ax );
      cu_AMatAllLocal.CopyFrom( AMatAllLocal );  // copy this into GPU to avoid GPU initilization of cu_AmatAllLocal 
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                  heightLocal, cu_ax.Data(), heightLocal, &zero, &cu_AMatAllLocal(0,3*sbSize*k), 3*sbSize );

      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      GetTime( timeSta );
      cu_w.CopyFrom( w );
      cu_aw.CopyFrom( aw );
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
                   heightLocal, cu_aw.Data(), heightLocal, &zero, &cu_AMatAllLocal(sbSize,3*sbSize*k+sbSize), 3*sbSize);

      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      GetTime( timeSta );
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                   heightLocal, cu_aw.Data(), heightLocal, &zero, &cu_AMatAllLocal(0,3*sbSize*k+sbSize), 3*sbSize);
      cu_AMatAllLocal.CopyTo( AMatAllLocal );

      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      // BMatAllLoc            
      GetTime( timeSta );
      cu_BMatAllLocal.CopyFrom( BMatAllLocal );
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                   heightLocal, cu_x.Data(), heightLocal, &zero, &cu_BMatAllLocal(0,3*sbSize*k), 3*sbSize);

      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      GetTime( timeSta );
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
                   heightLocal, cu_w.Data(), heightLocal, &zero, &cu_BMatAllLocal(sbSize,3*sbSize*k+sbSize), 3*sbSize);
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      GetTime( timeSta );
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                   heightLocal, cu_w.Data(), heightLocal, &zero, &cu_BMatAllLocal(0,3*sbSize*k+sbSize), 3*sbSize);
      cu_BMatAllLocal.CopyTo( BMatAllLocal );

      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      if ( numSet == 3 ){

        DblNumMat  p( heightLocal, sbSize, false, P.VecData(k) );
        DblNumMat ap( heightLocal, sbSize, false, AP.VecData(k) );
        
        // GPU numMat
        cuDblNumMat  cu_p (heightLocal, sbSize);
        cuDblNumMat cu_ap (heightLocal, sbSize);

        // AMatAllLoc
        GetTime( timeSta );
        cu_p.CopyFrom( p );
        cu_ap.CopyFrom( ap );
        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_p.Data(),
                     heightLocal, cu_ap.Data(), heightLocal, &zero, &cu_AMatAllLocal(2*sbSize,3*sbSize*k+2*sbSize), 3*sbSize);

        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        GetTime( timeSta );
        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                     heightLocal, cu_ap.Data(), heightLocal, &zero, &cu_AMatAllLocal(0,3*sbSize*k+2*sbSize), 3*sbSize );

        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        GetTime( timeSta );
        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
                     heightLocal, cu_ap.Data(), heightLocal, &zero, &cu_AMatAllLocal(sbSize,3*sbSize*k+2*sbSize), 3*sbSize );
        cu_AMatAllLocal.CopyTo( AMatAllLocal );

        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        // BMatAllLoc
        GetTime( timeSta );
        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_p.Data(),
                     heightLocal, cu_p.Data(), heightLocal, &zero, &cu_BMatAllLocal(2*sbSize,3*sbSize*k+2*sbSize), 3*sbSize );

        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        GetTime( timeSta );
        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                     heightLocal, cu_p.Data(), heightLocal, &zero, &cu_BMatAllLocal(0,3*sbSize*k+2*sbSize), 3*sbSize );

        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        GetTime( timeSta );
        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
                     heightLocal, cu_p.Data(), heightLocal, &zero, &cu_BMatAllLocal(sbSize,3*sbSize*k+2*sbSize), 3*sbSize );
        cu_BMatAllLocal.CopyTo( BMatAllLocal );

        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

      }             

    }

    GetTime( timeSta );
    MPI_Allreduce( AMatAllLocal.Data(), AMatAll.Data(), 9*sbSize*sbSize*nsb, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
    
    GetTime( timeSta );
    MPI_Allreduce( BMatAllLocal.Data(), BMatAll.Data(), 9*sbSize*sbSize*nsb, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    // Solve nsb small eigenproblems and update columns of X 
    for( Int k = 0; k < nsb; k++ ){

      Real eigs[3*sbSize];
      DblNumMat  cx( sbSize, sbSize ), cw( sbSize, sbSize ), cp( sbSize, sbSize);
      DblNumMat tmp( heightLocal, sbSize );            

      // small eigensolve
      GetTime( timeSta );
      lapack::Lacpy( 'A', 3*sbSize, 3*sbSize, &AMatAll(0,3*sbSize*k), 3*sbSize, AMat.Data(), 3*sbSize );
      lapack::Lacpy( 'A', 3*sbSize, 3*sbSize, &BMatAll(0,3*sbSize*k), 3*sbSize, BMat.Data(), 3*sbSize );
      GetTime( timeEnd );
      iterCopy = iterCopy + 2;
      timeCopy = timeCopy + ( timeEnd - timeSta );

      Int dim = (numSet == 3) ? 3*sbSize : 2*sbSize;
      GetTime( timeSta );
      lapack::Sygvd(1, 'V', 'U', dim, AMat.Data(), 3*sbSize, BMat.Data(), 3*sbSize, eigs);
      GetTime( timeEnd );
      iterSygvd = iterSygvd + 1;
      timeSygvd = timeSygvd + ( timeEnd - timeSta );

      // fetch indiviual columns
      DblNumMat  x( heightLocal, sbSize, false, X.VecData(k) );
      DblNumMat  w( heightLocal, sbSize, false, W.VecData(k) );
      DblNumMat  p( heightLocal, sbSize, false, P.VecData(k) );
      DblNumMat ax( heightLocal, sbSize, false, AX.VecData(k) );
      DblNumMat aw( heightLocal, sbSize, false, AW.VecData(k) );
      DblNumMat ap( heightLocal, sbSize, false, AP.VecData(k) );

      GetTime( timeSta );
      lapack::Lacpy( 'A', sbSize, sbSize, &AMat(0,0), 3*sbSize, cx.Data(), sbSize );
      lapack::Lacpy( 'A', sbSize, sbSize, &AMat(sbSize,0), 3*sbSize, cw.Data(), sbSize );
      GetTime( timeEnd );
      iterCopy = iterCopy + 2;
      timeCopy = timeCopy + ( timeEnd - timeSta );

      //  p = w*cw + p*cp; x = x*cx + p; ap = aw*cw + ap*cp; ax = ax*cx + ap;
      if( numSet == 3 ){
        
        GetTime( timeSta );
        lapack::Lacpy( 'A', sbSize, sbSize, &AMat(2*sbSize,0), 3*sbSize, cp.Data(), sbSize );
        GetTime( timeEnd );
        iterCopy = iterCopy + 1;
        timeCopy = timeCopy + ( timeEnd - timeSta );
       
        // tmp <- p*cp 
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            p.Data(), heightLocal, cp.Data(), sbSize,
            0.0, tmp.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );

        // p <- w*cw + tmp
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            w.Data(), heightLocal, cw.Data(), sbSize,
            1.0, tmp.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );

        GetTime( timeSta );
        lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, p.Data(), heightLocal );
        GetTime( timeEnd );
        iterCopy = iterCopy + 1;
        timeCopy = timeCopy + ( timeEnd - timeSta );

        // tmp <- ap*cp 
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            ap.Data(), heightLocal, cp.Data(), sbSize,
            0.0, tmp.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );

        // ap <- aw*cw + tmp
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            aw.Data(), heightLocal, cw.Data(), sbSize,
            1.0, tmp.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
        GetTime( timeSta );
        lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, ap.Data(), heightLocal );
        GetTime( timeEnd );
        iterCopy = iterCopy + 1;
        timeCopy = timeCopy + ( timeEnd - timeSta );

      }else{
        // p <- w*cw
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            w.Data(), heightLocal, cw.Data(), sbSize,
            0.0, p.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
        // ap <- aw*cw
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            aw.Data(), heightLocal, cw.Data(), sbSize,
            0.0, ap.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
      }

      // x <- x*cx + p
      GetTime( timeSta );
      lapack::Lacpy( 'A', heightLocal, sbSize, p.Data(), heightLocal, tmp.Data(), heightLocal );
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );
     
      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
          x.Data(), heightLocal, cx.Data(), sbSize,
          1.0, tmp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );
      
      GetTime( timeSta );
      lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, x.Data(), heightLocal );
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );

      // ax <- ax*cx + ap
      GetTime( timeSta );
      lapack::Lacpy( 'A', heightLocal, sbSize, ap.Data(), heightLocal, tmp.Data(), heightLocal );
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );
      
      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
          ax.Data(), heightLocal, cx.Data(), sbSize,
          1.0, tmp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, ax.Data(), heightLocal );
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );

    }

    // CholeskyQR of the updated block X
    GetTime( timeSta );
    cu_X.CopyFrom(X);
    cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(), 
              heightLocal, cu_X.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
    cu_XTXtemp1.CopyTo(XTXtemp1);

    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    if ( mpirank == 0) {
      GetTime( timeSta );
      GetTime( timeSta1 );
      lapack::Potrf( 'U', width, XTX.Data(), width );
      GetTime( timeEnd1 );
      iterPotrf = iterPotrf + 1;
      timePotrf = timePotrf + ( timeEnd1 - timeSta1 );
      GetTime( timeEnd );
      iterMpirank0 = iterMpirank0 + 1;
      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
    }
    GetTime( timeSta );
    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
    GetTime( timeEnd );
    iterBcast = iterBcast + 1;
    timeBcast = timeBcast + ( timeEnd - timeSta );

    // X <- X * U^{-1} is orthogonal
    GetTime( timeSta );
    cu_XTX.CopyFrom( XTX);
    cublas::Trsm( right, up, cu_transN, nondiag, heightLocal, width, &one, cu_XTX.Data(), width, cu_X.Data(), heightLocal );
    cu_XTX.CopyTo( XTX);
    cu_X.CopyTo( X );

    GetTime( timeEnd );
    iterTrsm = iterTrsm + 1;
    timeTrsm = timeTrsm + ( timeEnd - timeSta );

  } while( (iter < eigMaxIter) || (resMin > eigMinTolerance) );



  // *********************************************************************
  // Post processing
  // *********************************************************************

  // Obtain the eigenvalues and eigenvectors
  // if isConverged==true then XTX should contain the matrix X' * (AX); and X is an
  // orthonormal set

  if (!isConverged){
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
        heightLocal, AX.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
  }

  GetTime( timeSta1 );

  if(PWSolver_ == "PPCGScaLAPACK")
  { 
    if( contxt_ >= 0 )
    {
      Int numKeep = width; 
      Int lda = width;

      scalapack::ScaLAPACKMatrix<Real> square_mat_scala;
      scalapack::ScaLAPACKMatrix<Real> eigvecs_scala;

      scalapack::Descriptor descReduceSeq, descReducePar;
      Real timeEigScala_sta, timeEigScala_end;

      // Leading dimension provided
      descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt_, lda );

      // Automatically comptued Leading Dimension
      descReducePar.Init( numKeep, numKeep, scaBlockSize_, scaBlockSize_, I_ZERO, I_ZERO, contxt_ );

      square_mat_scala.SetDescriptor( descReducePar );
      eigvecs_scala.SetDescriptor( descReducePar );


      DblNumMat&  square_mat = XTX;
      // Redistribute the input matrix over the process grid
      SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
          &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt_ );


      // Make the ScaLAPACK call
      char uplo = 'U';
      std::vector<Real> temp_eigs(lda);

      scalapack::Syevd(uplo, square_mat_scala, temp_eigs, eigvecs_scala );

      // Copy the eigenvalues
      for(Int copy_iter = 0; copy_iter < lda; copy_iter ++){
        eigValS[copy_iter] = temp_eigs[copy_iter];
      }

      // Redistribute back eigenvectors
      SetValue(square_mat, 0.0 );
      SCALAPACK(pdgemr2d)( &numKeep, &numKeep, eigvecs_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
          square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt_ );
    }
  }
  else //PWSolver_ == "PPCG"
  {
    if ( mpirank == 0 ){
      GetTime( timeSta );
      lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );
      GetTime( timeEnd );
      iterMpirank0 = iterMpirank0 + 1;
      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
    }
  }

  GetTime( timeEnd1 );
  iterSyevd = iterSyevd + 1;
  timeSyevd = timeSyevd + ( timeEnd1 - timeSta1 );

  GetTime( timeSta );
  MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
  MPI_Bcast(eigValS.Data(), width, MPI_DOUBLE, 0, mpi_comm);
  GetTime( timeEnd );
  iterBcast = iterBcast + 2;
  timeBcast = timeBcast + ( timeEnd - timeSta );

  GetTime( timeSta );
  // X <- X*C
  cu_X.CopyFrom( X );
  cu_XTX.CopyFrom( XTX );
  cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &one, cu_X.Data(),
                heightLocal, cu_XTX.Data(), width, &zero, cu_Xtemp.Data(), heightLocal);
  cu_Xtemp.CopyTo( Xtemp );
  
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

  GetTime( timeSta );
  lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,
      X.Data(), heightLocal );
  GetTime( timeEnd );
  iterCopy = iterCopy + 1;
  timeCopy = timeCopy + ( timeEnd - timeSta );


  GetTime( timeSta );
  // AX <- AX*C
  cu_AX.CopyFrom( AX );
  cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &one, cu_AX.Data(),
                heightLocal, cu_XTX.Data(), width, &zero, cu_Xtemp.Data(), heightLocal);
  cu_Xtemp.CopyTo( Xtemp );

  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

  GetTime( timeSta );
  lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,
      AX.Data(), heightLocal );
  GetTime( timeEnd );
  iterCopy = iterCopy + 1;
  timeCopy = timeCopy + ( timeEnd - timeSta );

  // Compute norms of individual eigenpairs 
  DblNumVec  resNormLocal ( width ); 
  DblNumVec  resNorm( width );

  GetTime( timeSta );
  for(Int j=0; j < width; j++){
    for(Int i=0; i < heightLocal; i++){
      Xtemp(i,j) = AX(i,j) - X(i,j)*eigValS(j);  
    }
  } 
  GetTime( timeEnd );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd - timeSta );
      
  statusOFS << "Time for Xtemp in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;

  SetValue( resNormLocal, 0.0 );
  GetTime( timeSta );
  for( Int k = 0; k < width; k++ ){
    resNormLocal(k) = Energy(DblNumVec(heightLocal, false, Xtemp.VecData(k)));
  }
  GetTime( timeEnd );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd - timeSta );
  
  statusOFS << "Time for resNorm in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;

  SetValue( resNorm, 0.0 );
  MPI_Allreduce( resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE, 
      MPI_SUM, mpi_comm );

  if ( mpirank == 0 ){
    GetTime( timeSta );
    for( Int k = 0; k < width; k++ ){
      //            resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( XTX(k,k) ) );
      resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( eigValS(k) ) );
    }
    GetTime( timeEnd );
    timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
  }
  GetTime( timeSta );
  MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm);
  GetTime( timeEnd );
  iterBcast = iterBcast + 1;
  timeBcast = timeBcast + ( timeEnd - timeSta );

  GetTime( timeSta );
  resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
  resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );
  GetTime( timeEnd );
  iterOther = iterOther + 2;
  timeOther = timeOther + ( timeEnd - timeSta );
  
  statusOFS << "Time for resMax and resMin in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "resNorm = " << resNorm << std::endl;
  statusOFS << "eigValS = " << eigValS << std::endl;
  statusOFS << "maxRes  = " << resMax  << std::endl;
  statusOFS << "minRes  = " << resMin  << std::endl;
#endif



#if ( _DEBUGlevel_ >= 2 )

  GetTime( timeSta );
  cu_X.CopyFrom( X );
  cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
                heightLocal, cu_X.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
  cu_XTXtemp1.CopyTo( XTXtemp1 );

  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
  GetTime( timeSta );
  SetValue( XTX, 0.0 );
  MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterAllreduce = iterAllreduce + 1;
  timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

  statusOFS << "After the PPCG, XTX = " << XTX << std::endl;

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
  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;
  GetTime( timeSta );
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );
  GetTime( timeSta );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      Xcol(i, j) = sendbuf[sendk(i, j)]; 
    }
  }
  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;

  GetTime( timeSta );
  lapack::Lacpy( 'A', height, widthLocal, Xcol.Data(), height, 
      psiPtr_->Wavefun().Data(), height );
  GetTime( timeEnd );
  iterCopy = iterCopy + 1;
  timeCopy = timeCopy + ( timeEnd - timeSta );

  // REPORT ACTUAL EIGENRESIDUAL NORMS?
  statusOFS << std::endl << "After " << iter 
    << " PPCG iterations the min res norm is " 
    << resMin << ". The max res norm is " << resMax << std::endl << std::endl;

  GetTime( timeEnd2 );
  
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for iterGemmT        = " << iterGemmT           << "  timeGemmT        = " << timeGemmT << std::endl;
    statusOFS << "Time for iterGemmN        = " << iterGemmN           << "  timeGemmN        = " << timeGemmN << std::endl;
    statusOFS << "Time for iterBcast        = " << iterBcast           << "  timeBcast        = " << timeBcast << std::endl;
    statusOFS << "Time for iterAllreduce    = " << iterAllreduce       << "  timeAllreduce    = " << timeAllreduce << std::endl;
    statusOFS << "Time for iterAlltoallv    = " << iterAlltoallv       << "  timeMapping      = " << timeMapping   << std::endl;
    statusOFS << "Time for iterAlltoallv    = " << iterAlltoallv       << "  timeAlltoallv    = " << timeAlltoallv << std::endl;
    statusOFS << "Time for iterAlltoallvMap = " << iterAlltoallvMap    << "  timeAlltoallvMap = " << timeAlltoallvMap << std::endl;
    statusOFS << "Time for iterSpinor       = " << iterSpinor          << "  timeSpinor       = " << timeSpinor << std::endl;
    statusOFS << "Time for iterSpinor       = " << iterSpinor          << "  timeHpsi         = " << timeHpsi   << std::endl;
    statusOFS << "Time for iterTrsm         = " << iterTrsm            << "  timeTrsm         = " << timeTrsm << std::endl;
    statusOFS << "Time for iterPotrf        = " << iterPotrf           << "  timePotrf        = " << timePotrf << std::endl;
    statusOFS << "Time for iterSyevd        = " << iterSyevd           << "  timeSyevd        = " << timeSyevd << std::endl;
    statusOFS << "Time for iterSygvd        = " << iterSygvd           << "  timeSygvd        = " << timeSygvd << std::endl;
    statusOFS << "Time for iterMpirank0     = " << iterMpirank0        << "  timeMpirank0     = " << timeMpirank0 << std::endl;
    statusOFS << "Time for iterSweepT       = " << iterSweepT          << "  timeSweepT       = " << timeSweepT << std::endl;
    statusOFS << "Time for iterCopy         = " << iterCopy            << "  timeCopy         = " << timeCopy << std::endl;
    statusOFS << "Time for iterOther        = " << iterOther           << "  timeOther        = " << timeOther << std::endl;
    statusOFS << "Time for PPCG in PWDFT is " <<  timeEnd2 - timeSta2  << std::endl << std::endl;
#endif

    cublas::Destroy();


    return ;
    }         // -----  end of method EigenSolver::PPCGSolveReal  ----- 


#else

// Basic version of PPCG with columnwise sweep  
void
EigenSolver::PPCGSolveReal    (
    Int          numEig,
    Int          eigMaxIter,
    Real         eigMinTolerance,
    Real         eigTolerance)
{
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

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  if( widthLocal != noccLocal ){
    throw std::logic_error("widthLocal != noccLocal.");
  }

  // Time for GemmT, GemmN, Alltoallv, Spinor, Mpirank0 
  // GemmT: blas::Gemm( 'T', 'N')
  // GemmN: blas::Gemm( 'N', 'N')
  // Alltoallv: row-partition to column partition via MPI_Alltoallv 
  // Spinor: Applying the Hamiltonian matrix 
  // Mpirank0: Serial calculation part

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;
  Real timeSta2, timeEnd2;
  Real timeGemmT = 0.0;
  Real timeGemmN = 0.0;
  Real timeBcast = 0.0;
  Real timeAllreduce = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeAlltoallvMap = 0.0;
  Real timeSpinor = 0.0;
  Real timeTrsm = 0.0;
  Real timePotrf = 0.0;
  Real timeSyevd = 0.0;
  Real timeSygvd = 0.0;
  Real timeMpirank0 = 0.0;
  Real timeScaLAPACKFactor = 0.0;
  Real timeScaLAPACK = 0.0;
  Real timeSweepT = 0.0;
  Real timeCopy = 0.0;
  Real timeOther = 0.0;
  Int  iterGemmT = 0;
  Int  iterGemmN = 0;
  Int  iterBcast = 0;
  Int  iterAllreduce = 0;
  Int  iterAlltoallv = 0;
  Int  iterAlltoallvMap = 0;
  Int  iterSpinor = 0;
  Int  iterTrsm = 0;
  Int  iterPotrf = 0;
  Int  iterSyevd = 0;
  Int  iterSygvd = 0;
  Int  iterMpirank0 = 0;
  Int  iterScaLAPACKFactor = 0;
  Int  iterScaLAPACK = 0;
  Int  iterSweepT = 0;
  Int  iterCopy = 0;
  Int  iterOther = 0;

  if( numEig > width ){
    std::ostringstream msg;
    msg 
      << "Number of eigenvalues requested  = " << numEig << std::endl
      << "which is larger than the number of columns in psi = " << width << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }

  GetTime( timeSta2 );

  // The following codes are not replaced by AlltoallForward /
  // AlltoallBackward since they are repetitively used in the
  // eigensolver.
  //
  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  GetTime( timeSta );
  
  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
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

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
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
 
  GetTime( timeEnd );
  iterAlltoallvMap = iterAlltoallvMap + 1;
  timeAlltoallvMap = timeAlltoallvMap + ( timeEnd - timeSta );

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


  //    DblNumMat  AMat( 3*width, 3*width ), BMat( 3*width, 3*width );
  //    DblNumMat  AMatT1( 3*width, 3*width );

  // Temporary buffer array.
  // The unpreconditioned residual will also be saved in Xtemp
  DblNumMat  XTX( width, width );
  //?    DblNumMat  XTXtemp( width, width );
  DblNumMat  XTXtemp1( width, width );

  DblNumMat  Xtemp( heightLocal, width );

  Real  resBlockNormLocal, resBlockNorm; // Frobenius norm of the residual block  
  Real  resMax, resMin;

  // For convenience
  DblNumMat  X( heightLocal, width, false, S.VecData(0) );
  DblNumMat  W( heightLocal, width, false, S.VecData(width) );
  DblNumMat  P( heightLocal, width, false, S.VecData(2*width) );
  DblNumMat AX( heightLocal, width, false, AS.VecData(0) );
  DblNumMat AW( heightLocal, width, false, AS.VecData(width) );
  DblNumMat AP( heightLocal, width, false, AS.VecData(2*width) );

  DblNumMat  Xcol( height, widthLocal );
  DblNumMat  Wcol( height, widthLocal );
  DblNumMat AXcol( height, widthLocal );
  DblNumMat AWcol( height, widthLocal );

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

  const Int numLocked = 0;  // Never perform locking in this version
  const Int numActive = width;

  bool isConverged = false;

  // Initialization
  SetValue( S, 0.0 );
  SetValue( AS, 0.0 );

  DblNumVec  eigValS(lda);
  SetValue( eigValS, 0.0 );

  // Initialize X by the data in psi
  GetTime( timeSta );
  lapack::Lacpy( 'A', height, widthLocal, psiPtr_->Wavefun().Data(), height, 
      Xcol.Data(), height );
  GetTime( timeEnd );
  iterCopy = iterCopy + 1;
  timeCopy = timeCopy + ( timeEnd - timeSta );

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
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
  GetTime( timeSta );
  SetValue( XTX, 0.0 );
  MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterAllreduce = iterAllreduce + 1;
  timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

  if ( mpirank == 0) {
    GetTime( timeSta );
    lapack::Potrf( 'U', width, XTX.Data(), width );
    GetTime( timeEnd );
    iterMpirank0 = iterMpirank0 + 1;
    timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
  }
  GetTime( timeSta );
  MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
  GetTime( timeEnd );
  iterBcast = iterBcast + 1;
  timeBcast = timeBcast + ( timeEnd - timeSta );

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
    NumTns<Real> tnsTemp(ntot, ncom, noccLocal, false, AXcol.Data());

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
  Int iter = 0;
  statusOFS << "Minimum tolerance is " << eigMinTolerance << std::endl;

  do{
    iter++;
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "iter = " << iter << std::endl;
#endif

    if( iter == 1 || isRestart == true )
      numSet = 2;
    else
      numSet = 3;

    // XTX <- X' * (AX)
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
        heightLocal, AX.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    // Compute the residual.
    // R <- AX - X*(X'*AX)
    GetTime( timeSta );
    lapack::Lacpy( 'A', heightLocal, width, AX.Data(), heightLocal, Xtemp.Data(), heightLocal );
    GetTime( timeEnd );
    iterCopy = iterCopy + 1;
    timeCopy = timeCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
        X.Data(), heightLocal, XTX.Data(), width, 1.0, Xtemp.Data(), heightLocal );
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 1;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );



    // Compute the Frobenius norm of the residual block

    if(0){

      GetTime( timeSta );

      resBlockNormLocal = 0.0; resBlockNorm = 0.0; 
      for (Int i=0; i < heightLocal; i++){
        for (Int j=0; j < width; j++ ){
          resBlockNormLocal += Xtemp(i,j)*Xtemp(i,j); 
        }
      }

      MPI_Allreduce( &resBlockNormLocal, &resBlockNorm, 1, MPI_DOUBLE,
          MPI_SUM, mpi_comm); 
      resBlockNorm = std::sqrt(resBlockNorm);

      GetTime( timeEnd );
      iterOther = iterOther + 1;
      timeOther = timeOther + ( timeEnd - timeSta );

      statusOFS << "Time for resBlockNorm in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;


      /////////////// UNCOMMENT THIS #if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Frob. norm of the residual block = " << resBlockNorm << std::endl;
      //////////////#endif

      // THIS STOPPING CRITERION LIKELY IRRELEVANT
      if( resBlockNorm < eigTolerance ){
        isConverged = true;
        break;
      }

    } // if(0)

    // LOCKING not supported, PPCG needs Rayleigh--Ritz to lock         
    //        numActiveTotal = width - numLockedTotal;
    //        numActiveLocal = widthLocal - numLockedLocal;

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

    // Compute W = TW
    {
      GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, Xcol.VecData(numLockedLocal));
      NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));

      SetValue( tnsTemp, 0.0 );
      spnTemp.AddTeterPrecond( fftPtr_, tnsTemp );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }


    // Compute AW = A*W
    {
      GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));
      NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, AWcol.VecData(numLockedLocal));

      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

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


    // W = W - X(X'W), AW = AW - AX(X'W)
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
        heightLocal, W.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );


    GetTime( timeSta );
    blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
        X.Data(), heightLocal, XTX.Data(), width, 1.0, W.Data(), heightLocal );
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 1;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );


    GetTime( timeSta );
    blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
        AX.Data(), heightLocal, XTX.Data(), width, 1.0, AW.Data(), heightLocal );
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 1;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );

    // Normalize columns of W
    Real normLocal[width]; 
    Real normGlobal[width];

    GetTime( timeSta );
    for( Int k = numLockedLocal; k < width; k++ ){
      normLocal[k] = Energy(DblNumVec(heightLocal, false, W.VecData(k)));
      normGlobal[k] = 0.0;
    }
    MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    for( Int k = numLockedLocal; k < width; k++ ){
      Real norm = std::sqrt( normGlobal[k] );
      blas::Scal( heightLocal, 1.0 / norm, W.VecData(k), 1 );
      blas::Scal( heightLocal, 1.0 / norm, AW.VecData(k), 1 );
    }
    GetTime( timeEnd );
    iterOther = iterOther + 2;
    timeOther = timeOther + ( timeEnd - timeSta );

    statusOFS << "Time for norm1 in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;


    // P = P - X(X'P), AP = AP - AX(X'P)
    if( numSet == 3 ){
      GetTime( timeSta );
      blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
          heightLocal, P.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTX, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
          X.Data(), heightLocal, XTX.Data(), width, 1.0, P.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
          AX.Data(), heightLocal, XTX.Data(), width, 1.0, AP.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      // Normalize the conjugate direction
      GetTime( timeSta );
      for( Int k = numLockedLocal; k < width; k++ ){
        normLocal[k] = Energy(DblNumVec(heightLocal, false, P.VecData(k)));
        normGlobal[k] = 0.0;
      }
      MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      for( Int k = numLockedLocal; k < width; k++ ){
        Real norm = std::sqrt( normGlobal[k] );
        blas::Scal( heightLocal, 1.0 / norm, P.VecData(k), 1 );
        blas::Scal( heightLocal, 1.0 / norm, AP.VecData(k), 1 );
      }
      GetTime( timeEnd );
      iterOther = iterOther + 2;
      timeOther = timeOther + ( timeEnd - timeSta );
    
      statusOFS << "Time for norm2 in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;
   
    }

    // Perform the sweep
    GetTime( timeSta );
    Int sbSize = 1, nsb = width; // this should be generalized to subblocks 
    DblNumMat  AMat( 3*sbSize, 3*sbSize ), BMat( 3*sbSize, 3*sbSize );
    DblNumMat  AMatAll( 3*sbSize, 3*sbSize*nsb ), BMatAll( 3*sbSize, 3*sbSize*nsb ); // contains all nsb 3-by-3 matrices
    DblNumMat  AMatAllLocal( 3*sbSize, 3*sbSize*nsb ), BMatAllLocal( 3*sbSize, 3*sbSize*nsb ); // contains local parts of all nsb 3-by-3 matrices

    SetValue( AMat, 0.0 ); SetValue( BMat, 0.0 );
    SetValue( AMatAll, 0.0 ); SetValue( BMatAll, 0.0 );
    SetValue( AMatAllLocal, 0.0 ); SetValue( BMatAllLocal, 0.0 );

    // LOCKING NOT SUPPORTED, loop over all columns 
    for( Int k = 0; k < nsb; k++ ){

      // fetch indiviual columns
      DblNumMat  x( heightLocal, sbSize, false, X.VecData(k) );
      DblNumMat  w( heightLocal, sbSize, false, W.VecData(k) );
      DblNumMat ax( heightLocal, sbSize, false, AX.VecData(k) );
      DblNumMat aw( heightLocal, sbSize, false, AW.VecData(k) );

      // Compute AMatAllLoc and BMatAllLoc            
      // AMatAllLoc
      GetTime( timeSta );
      blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, x.Data(),
          heightLocal, ax.Data(), heightLocal, 
          0.0, &AMatAllLocal(0,3*sbSize*k), 3*sbSize );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, w.Data(),
          heightLocal, aw.Data(), heightLocal, 
          0.0, &AMatAllLocal(sbSize,3*sbSize*k+sbSize), 3*sbSize );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, x.Data(),
          heightLocal, aw.Data(), heightLocal, 
          0.0, &AMatAllLocal(0,3*sbSize*k+sbSize), 3*sbSize );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      // BMatAllLoc            
      GetTime( timeSta );
      blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, x.Data(),
          heightLocal, x.Data(), heightLocal, 
          0.0, &BMatAllLocal(0,3*sbSize*k), 3*sbSize );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, w.Data(),
          heightLocal, w.Data(), heightLocal, 
          0.0, &BMatAllLocal(sbSize,3*sbSize*k+sbSize), 3*sbSize );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, x.Data(),
          heightLocal, w.Data(), heightLocal, 
          0.0, &BMatAllLocal(0,3*sbSize*k+sbSize), 3*sbSize );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      if ( numSet == 3 ){

        DblNumMat  p( heightLocal, sbSize, false, P.VecData(k) );
        DblNumMat ap( heightLocal, sbSize, false, AP.VecData(k) );

        // AMatAllLoc
        GetTime( timeSta );
        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, p.Data(),
            heightLocal, ap.Data(), heightLocal, 
            0.0, &AMatAllLocal(2*sbSize,3*sbSize*k+2*sbSize), 3*sbSize );
        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        GetTime( timeSta );
        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, x.Data(),
            heightLocal, ap.Data(), heightLocal, 
            0.0, &AMatAllLocal(0, 3*sbSize*k+2*sbSize), 3*sbSize );
        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        GetTime( timeSta );
        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, w.Data(),
            heightLocal, ap.Data(), heightLocal, 
            0.0, &AMatAllLocal(sbSize, 3*sbSize*k+2*sbSize), 3*sbSize );
        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        // BMatAllLoc
        GetTime( timeSta );
        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, p.Data(),
            heightLocal, p.Data(), heightLocal, 
            0.0, &BMatAllLocal(2*sbSize,3*sbSize*k+2*sbSize), 3*sbSize );
        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        GetTime( timeSta );
        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, x.Data(),
            heightLocal, p.Data(), heightLocal, 
            0.0, &BMatAllLocal(0, 3*sbSize*k+2*sbSize), 3*sbSize );
        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        GetTime( timeSta );
        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, w.Data(),
            heightLocal, p.Data(), heightLocal, 
            0.0, &BMatAllLocal(sbSize, 3*sbSize*k+2*sbSize), 3*sbSize );
        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

      }             

    }

    GetTime( timeSta );
    MPI_Allreduce( AMatAllLocal.Data(), AMatAll.Data(), 9*sbSize*sbSize*nsb, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
    
    GetTime( timeSta );
    MPI_Allreduce( BMatAllLocal.Data(), BMatAll.Data(), 9*sbSize*sbSize*nsb, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    // Solve nsb small eigenproblems and update columns of X 
    for( Int k = 0; k < nsb; k++ ){

      Real eigs[3*sbSize];
      DblNumMat  cx( sbSize, sbSize ), cw( sbSize, sbSize ), cp( sbSize, sbSize);
      DblNumMat tmp( heightLocal, sbSize );            

      // small eigensolve
      GetTime( timeSta );
      lapack::Lacpy( 'A', 3*sbSize, 3*sbSize, &AMatAll(0,3*sbSize*k), 3*sbSize, AMat.Data(), 3*sbSize );
      lapack::Lacpy( 'A', 3*sbSize, 3*sbSize, &BMatAll(0,3*sbSize*k), 3*sbSize, BMat.Data(), 3*sbSize );
      GetTime( timeEnd );
      iterCopy = iterCopy + 2;
      timeCopy = timeCopy + ( timeEnd - timeSta );

      //if (mpirank==0){
      //    statusOFS << "sweep num = " << k << std::endl;
      //    statusOFS << "AMat = " << AMat << std::endl;
      //    statusOFS << "BMat = " << BMat << std::endl<<std::endl;
      //}

      Int dim = (numSet == 3) ? 3*sbSize : 2*sbSize;
      GetTime( timeSta );
      lapack::Sygvd(1, 'V', 'U', dim, AMat.Data(), 3*sbSize, BMat.Data(), 3*sbSize, eigs);
      GetTime( timeEnd );
      iterSygvd = iterSygvd + 1;
      timeSygvd = timeSygvd + ( timeEnd - timeSta );

      // fetch indiviual columns
      DblNumMat  x( heightLocal, sbSize, false, X.VecData(k) );
      DblNumMat  w( heightLocal, sbSize, false, W.VecData(k) );
      DblNumMat  p( heightLocal, sbSize, false, P.VecData(k) );
      DblNumMat ax( heightLocal, sbSize, false, AX.VecData(k) );
      DblNumMat aw( heightLocal, sbSize, false, AW.VecData(k) );
      DblNumMat ap( heightLocal, sbSize, false, AP.VecData(k) );

      GetTime( timeSta );
      lapack::Lacpy( 'A', sbSize, sbSize, &AMat(0,0), 3*sbSize, cx.Data(), sbSize );
      lapack::Lacpy( 'A', sbSize, sbSize, &AMat(sbSize,0), 3*sbSize, cw.Data(), sbSize );
      GetTime( timeEnd );
      iterCopy = iterCopy + 2;
      timeCopy = timeCopy + ( timeEnd - timeSta );

      //  p = w*cw + p*cp; x = x*cx + p; ap = aw*cw + ap*cp; ax = ax*cx + ap;
      if( numSet == 3 ){
        
        GetTime( timeSta );
        lapack::Lacpy( 'A', sbSize, sbSize, &AMat(2*sbSize,0), 3*sbSize, cp.Data(), sbSize );
        GetTime( timeEnd );
        iterCopy = iterCopy + 1;
        timeCopy = timeCopy + ( timeEnd - timeSta );
       
        // tmp <- p*cp 
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            p.Data(), heightLocal, cp.Data(), sbSize,
            0.0, tmp.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );

        // p <- w*cw + tmp
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            w.Data(), heightLocal, cw.Data(), sbSize,
            1.0, tmp.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );

        GetTime( timeSta );
        lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, p.Data(), heightLocal );
        GetTime( timeEnd );
        iterCopy = iterCopy + 1;
        timeCopy = timeCopy + ( timeEnd - timeSta );

        // tmp <- ap*cp 
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            ap.Data(), heightLocal, cp.Data(), sbSize,
            0.0, tmp.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );

        // ap <- aw*cw + tmp
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            aw.Data(), heightLocal, cw.Data(), sbSize,
            1.0, tmp.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
        GetTime( timeSta );
        lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, ap.Data(), heightLocal );
        GetTime( timeEnd );
        iterCopy = iterCopy + 1;
        timeCopy = timeCopy + ( timeEnd - timeSta );

      }else{
        // p <- w*cw
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            w.Data(), heightLocal, cw.Data(), sbSize,
            0.0, p.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
        // ap <- aw*cw
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            aw.Data(), heightLocal, cw.Data(), sbSize,
            0.0, ap.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
      }

      // x <- x*cx + p
      GetTime( timeSta );
      lapack::Lacpy( 'A', heightLocal, sbSize, p.Data(), heightLocal, tmp.Data(), heightLocal );
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );
     
      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
          x.Data(), heightLocal, cx.Data(), sbSize,
          1.0, tmp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );
      
      GetTime( timeSta );
      lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, x.Data(), heightLocal );
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );

      // ax <- ax*cx + ap
      GetTime( timeSta );
      lapack::Lacpy( 'A', heightLocal, sbSize, ap.Data(), heightLocal, tmp.Data(), heightLocal );
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );
      
      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
          ax.Data(), heightLocal, cx.Data(), sbSize,
          1.0, tmp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, ax.Data(), heightLocal );
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );

    }

    // CholeskyQR of the updated block X
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(), 
        heightLocal, X.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    if ( mpirank == 0) {
      GetTime( timeSta );
      GetTime( timeSta1 );
      lapack::Potrf( 'U', width, XTX.Data(), width );
      GetTime( timeEnd1 );
      iterPotrf = iterPotrf + 1;
      timePotrf = timePotrf + ( timeEnd1 - timeSta1 );
      GetTime( timeEnd );
      iterMpirank0 = iterMpirank0 + 1;
      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
    }
    GetTime( timeSta );
    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
    GetTime( timeEnd );
    iterBcast = iterBcast + 1;
    timeBcast = timeBcast + ( timeEnd - timeSta );

    // X <- X * U^{-1} is orthogonal
    GetTime( timeSta );
    blas::Trsm( 'R', 'U', 'N', 'N', heightLocal, width, 1.0, XTX.Data(), width, 
        X.Data(), heightLocal );
    GetTime( timeEnd );
    iterTrsm = iterTrsm + 1;
    timeTrsm = timeTrsm + ( timeEnd - timeSta );


    //            // Copy the eigenvalues
    //            SetValue( eigValS, 0.0 );
    //            for( Int i = 0; i < numKeep; i++ ){
    //                eigValS[i] = eigs[i];
    //            }


    //#if ( _DEBUGlevel_ >= 1 )
    //        statusOFS << "numLocked = " << numLocked << std::endl;
    //        statusOFS << "eigValS   = " << eigValS << std::endl;
    //#endif

  } while( (iter < eigMaxIter) || (resMin > eigMinTolerance) );



  // *********************************************************************
  // Post processing
  // *********************************************************************

  // Obtain the eigenvalues and eigenvectors
  // if isConverged==true then XTX should contain the matrix X' * (AX); and X is an
  // orthonormal set

  if (!isConverged){
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
        heightLocal, AX.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
  }

  GetTime( timeSta1 );

  if(PWSolver_ == "PPCGScaLAPACK")
  { 
    if( contxt_ >= 0 )
    {
      Int numKeep = width; 
      Int lda = width;

      scalapack::ScaLAPACKMatrix<Real> square_mat_scala;
      scalapack::ScaLAPACKMatrix<Real> eigvecs_scala;

      scalapack::Descriptor descReduceSeq, descReducePar;
      Real timeEigScala_sta, timeEigScala_end;

      // Leading dimension provided
      descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt_, lda );

      // Automatically comptued Leading Dimension
      descReducePar.Init( numKeep, numKeep, scaBlockSize_, scaBlockSize_, I_ZERO, I_ZERO, contxt_ );

      square_mat_scala.SetDescriptor( descReducePar );
      eigvecs_scala.SetDescriptor( descReducePar );


      DblNumMat&  square_mat = XTX;
      // Redistribute the input matrix over the process grid
      SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
          &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt_ );


      // Make the ScaLAPACK call
      char uplo = 'U';
      std::vector<Real> temp_eigs(lda);

      scalapack::Syevd(uplo, square_mat_scala, temp_eigs, eigvecs_scala );

      // Copy the eigenvalues
      for(Int copy_iter = 0; copy_iter < lda; copy_iter ++){
        eigValS[copy_iter] = temp_eigs[copy_iter];
      }

      // Redistribute back eigenvectors
      SetValue(square_mat, 0.0 );
      SCALAPACK(pdgemr2d)( &numKeep, &numKeep, eigvecs_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
          square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt_ );
    }
  }
  else //PWSolver_ == "PPCG"
  {
    if ( mpirank == 0 ){
      GetTime( timeSta );
      lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );
      GetTime( timeEnd );
      iterMpirank0 = iterMpirank0 + 1;
      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
    }
  }

  GetTime( timeEnd1 );
  iterSyevd = iterSyevd + 1;
  timeSyevd = timeSyevd + ( timeEnd1 - timeSta1 );

  GetTime( timeSta );
  MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
  MPI_Bcast(eigValS.Data(), width, MPI_DOUBLE, 0, mpi_comm);
  GetTime( timeEnd );
  iterBcast = iterBcast + 2;
  timeBcast = timeBcast + ( timeEnd - timeSta );

  GetTime( timeSta );
  // X <- X*C
  blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, X.Data(),
      heightLocal, XTX.Data(), width, 0.0, Xtemp.Data(), heightLocal );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

  GetTime( timeSta );
  lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,
      X.Data(), heightLocal );
  GetTime( timeEnd );
  iterCopy = iterCopy + 1;
  timeCopy = timeCopy + ( timeEnd - timeSta );


  GetTime( timeSta );
  // AX <- AX*C
  blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, AX.Data(),
      heightLocal, XTX.Data(), width, 0.0, Xtemp.Data(), heightLocal );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

  GetTime( timeSta );
  lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,
      AX.Data(), heightLocal );
  GetTime( timeEnd );
  iterCopy = iterCopy + 1;
  timeCopy = timeCopy + ( timeEnd - timeSta );

  // Compute norms of individual eigenpairs 
  DblNumVec  resNormLocal ( width ); 
  DblNumVec  resNorm( width );

  GetTime( timeSta );
  for(Int j=0; j < width; j++){
    for(Int i=0; i < heightLocal; i++){
      Xtemp(i,j) = AX(i,j) - X(i,j)*eigValS(j);  
    }
  } 
  GetTime( timeEnd );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd - timeSta );
      
  statusOFS << "Time for Xtemp in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;

  SetValue( resNormLocal, 0.0 );
  GetTime( timeSta );
  for( Int k = 0; k < width; k++ ){
    resNormLocal(k) = Energy(DblNumVec(heightLocal, false, Xtemp.VecData(k)));
  }
  GetTime( timeEnd );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd - timeSta );
  
  statusOFS << "Time for resNorm in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;

  SetValue( resNorm, 0.0 );
  MPI_Allreduce( resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE, 
      MPI_SUM, mpi_comm );

  if ( mpirank == 0 ){
    GetTime( timeSta );
    for( Int k = 0; k < width; k++ ){
      //            resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( XTX(k,k) ) );
      resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( eigValS(k) ) );
    }
    GetTime( timeEnd );
    timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
  }
  GetTime( timeSta );
  MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm);
  GetTime( timeEnd );
  iterBcast = iterBcast + 1;
  timeBcast = timeBcast + ( timeEnd - timeSta );

  GetTime( timeSta );
  resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
  resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );
  GetTime( timeEnd );
  iterOther = iterOther + 2;
  timeOther = timeOther + ( timeEnd - timeSta );
  
  statusOFS << "Time for resMax and resMin in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "resNorm = " << resNorm << std::endl;
  statusOFS << "eigValS = " << eigValS << std::endl;
  statusOFS << "maxRes  = " << resMax  << std::endl;
  statusOFS << "minRes  = " << resMin  << std::endl;
#endif



#if ( _DEBUGlevel_ >= 2 )

  GetTime( timeSta );
  blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, X.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
  GetTime( timeSta );
  SetValue( XTX, 0.0 );
  MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterAllreduce = iterAllreduce + 1;
  timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

  statusOFS << "After the PPCG, XTX = " << XTX << std::endl;

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

  GetTime( timeSta );
  lapack::Lacpy( 'A', height, widthLocal, Xcol.Data(), height, 
      psiPtr_->Wavefun().Data(), height );
  GetTime( timeEnd );
  iterCopy = iterCopy + 1;
  timeCopy = timeCopy + ( timeEnd - timeSta );

  // REPORT ACTUAL EIGENRESIDUAL NORMS?
  statusOFS << std::endl << "After " << iter 
    << " PPCG iterations the min res norm is " 
    << resMin << ". The max res norm is " << resMax << std::endl << std::endl;

  GetTime( timeEnd2 );
  
  //        statusOFS << std::endl << "After " << iter 
  //            << " iterations, PPCG has converged."  << std::endl
  //            << "The maximum norm of the residual is " 
  //            << resMax << std::endl << std::endl
  //            << "The minimum norm of the residual is " 
  //            << resMin << std::endl << std::endl;
  //    }
  //    else{
  //        statusOFS << std::endl << "After " << iter 
  //            << " iterations, PPCG did not converge. " << std::endl
  //            << "The maximum norm of the residual is " 
  //            << resMax << std::endl << std::endl
  //            << "The minimum norm of the residual is " 
  //            << resMin << std::endl << std::endl;
  //    }

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for iterGemmT        = " << iterGemmT           << "  timeGemmT        = " << timeGemmT << std::endl;
    statusOFS << "Time for iterGemmN        = " << iterGemmN           << "  timeGemmN        = " << timeGemmN << std::endl;
    statusOFS << "Time for iterBcast        = " << iterBcast           << "  timeBcast        = " << timeBcast << std::endl;
    statusOFS << "Time for iterAllreduce    = " << iterAllreduce       << "  timeAllreduce    = " << timeAllreduce << std::endl;
    statusOFS << "Time for iterAlltoallv    = " << iterAlltoallv       << "  timeAlltoallv    = " << timeAlltoallv << std::endl;
    statusOFS << "Time for iterAlltoallvMap = " << iterAlltoallvMap    << "  timeAlltoallvMap = " << timeAlltoallvMap << std::endl;
    statusOFS << "Time for iterSpinor       = " << iterSpinor          << "  timeSpinor       = " << timeSpinor << std::endl;
    statusOFS << "Time for iterTrsm         = " << iterTrsm            << "  timeTrsm         = " << timeTrsm << std::endl;
    statusOFS << "Time for iterPotrf        = " << iterPotrf           << "  timePotrf        = " << timePotrf << std::endl;
    statusOFS << "Time for iterSyevd        = " << iterSyevd           << "  timeSyevd        = " << timeSyevd << std::endl;
    statusOFS << "Time for iterSygvd        = " << iterSygvd           << "  timeSygvd        = " << timeSygvd << std::endl;
    statusOFS << "Time for iterMpirank0     = " << iterMpirank0        << "  timeMpirank0     = " << timeMpirank0 << std::endl;
    statusOFS << "Time for iterSweepT       = " << iterSweepT          << "  timeSweepT       = " << timeSweepT << std::endl;
    statusOFS << "Time for iterCopy         = " << iterCopy            << "  timeCopy         = " << timeCopy << std::endl;
    statusOFS << "Time for iterOther        = " << iterOther           << "  timeOther        = " << timeOther << std::endl;
    statusOFS << "Time for PPCG in PWDFT is " <<  timeEnd2 - timeSta2  << std::endl << std::endl;
#endif

    return ;
    }         // -----  end of method EigenSolver::PPCGSolveReal  ----- 
#endif
} // namespace dgdft
