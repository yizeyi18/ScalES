/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Weile Jia

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
/// @file device_eigensolver.cpp
/// @brief device_eigensolver in the global domain or extended element.
/// @date 2020-08-21
#include  "eigensolver.hpp"
#include  "utility.hpp"
#include  "blas.hpp"
#include  "device_blas.hpp"
#include  "device_utils.h"
#include  "device_nummat_impl.hpp"
#include  "device_numvec_impl.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"
#ifdef USE_MAGMA
#include  "magma.hpp"
#else
#include "device_solver.hpp"
#endif

using namespace dgdft::scalapack;
using namespace dgdft::esdf;


namespace dgdft{


void
EigenSolver::devicePPCGSolveReal (
    Int          numEig,
    Int          eigMaxIter,
    Real         eigMinTolerance,
    Real         eigTolerance, 
    Int          scf_iter)
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

  /* init the CUDA Device */
  cublasStatus_t status;
  /*
  cublasSideMode_t right  = CUBLAS_SIDE_RIGHT;
  cublasFillMode_t up     = CUBLAS_FILL_MODE_UPPER;
  cublasDiagType_t nondiag   = CUBLAS_DIAG_NON_UNIT;
  cublasOperation_t cu_transT = CUBLAS_OP_T;
  cublasOperation_t cu_transN = CUBLAS_OP_N;
  cublasOperation_t cu_transC = CUBLAS_OP_C;
  */
  char right  = 'R';
  char up     = 'U';
  char nondiag   = 'N';
  char cu_transT = 'T';
  char cu_transN = 'N';
  char cu_transC = 'C';
  //device_init_vtot();

#if ( _DEBUGlevel_ >= 1 )
  if(mpirank == 0)
  {
    std::cout << " GPU PPCG ........... " << std::endl;
    device_memory();
  }
#endif
  
  
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
  Real time11, time22;
  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;
  Real timeSta2, timeEnd2;
  Real firstTime = 0.0;
  Real secondTime= 0.0;
  Real thirdTime= 0.0;
  Real calTime = 0.0;
  Real timeHpsi = 0.0;
  Real timeStart = 0.0;
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
  GetTime( time11);

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

  deviceIntNumMat cu_sendk( height, widthLocal );
  deviceIntNumMat cu_recvk( heightLocal, width );
  deviceDblNumVec cu_sendbuf(height*widthLocal); 
  deviceDblNumVec cu_recvbuf(heightLocal*width);
  deviceIntNumVec cu_sendcounts(mpisize);
  deviceIntNumVec cu_recvcounts(mpisize);
  deviceIntNumVec cu_senddispls(mpisize);
  deviceIntNumVec cu_recvdispls(mpisize);

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
  
  device_memcpy_HOST2DEVICE(cu_sendcounts.Data(), sendcounts.Data(), sizeof(Int)*mpisize);
  device_memcpy_HOST2DEVICE(cu_recvcounts.Data(), recvcounts.Data(), sizeof(Int)*mpisize);
  device_memcpy_HOST2DEVICE(cu_senddispls.Data(), senddispls.Data(), sizeof(Int)*mpisize);
  device_memcpy_HOST2DEVICE(cu_recvdispls.Data(), recvdispls.Data(), sizeof(Int)*mpisize);
 
  GetTime( timeEnd );
  iterAlltoallvMap = iterAlltoallvMap + 1;
  timeAlltoallvMap = timeAlltoallvMap + ( timeEnd - timeSta );

  // S = ( X | W | P ) is a triplet used for LOBPCG.  
  // W is the preconditioned residual
  // DblNumMat  S( height, 3*widthLocal ), AS( height, 3*widthLocal ); 
  DblNumMat       S( heightLocal, 3*width ),    AS( heightLocal, 3*width ); 
  deviceDblNumMat  cu_S( heightLocal, 3*width ), cu_AS( heightLocal, 3*width ); 
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
  DblNumMat  XTXtemp1( width, width );

  DblNumMat  Xtemp( heightLocal, width );

  Real  resBlockNormLocal, resBlockNorm; // Frobenius norm of the residual block  
  Real  resMax, resMin;

#if ( _DEBUGlevel_ >= 2 )
  if(mpirank  == 0)  { std::cout << " after malloc cuS, cu_AS " << std::endl; device_memory(); }
#endif
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
  deviceDblNumMat cu_XTX(width, width);
  deviceDblNumMat cu_XTXtemp1(width, width);
  deviceDblNumMat cu_Xtemp(heightLocal, width);

  deviceDblNumMat cu_X ( heightLocal, width, false, cu_S.VecData(0)        );
  deviceDblNumMat cu_W ( heightLocal, width, false, cu_S.VecData(width)    );
  deviceDblNumMat cu_P ( heightLocal, width, false, cu_S.VecData(2*width)  );
  deviceDblNumMat cu_AX( heightLocal, width, false, cu_AS.VecData(0)       );
  deviceDblNumMat cu_AW( heightLocal, width, false, cu_AS.VecData(width)   );
  deviceDblNumMat cu_AP( heightLocal, width, false, cu_AS.VecData(2*width) );
  
  deviceDblNumMat cu_Xcol ( height, widthLocal );
  deviceDblNumMat cu_Wcol ( height, widthLocal );
  deviceDblNumMat cu_AXcol( height, widthLocal );
  deviceDblNumMat cu_AWcol( height, widthLocal );

#if ( _DEBUGlevel_ >= 1 )
  if(mpirank == 0)
  {
    std::cout << " GPU PPCG begins alloc partially done" << std::endl;
    std::cout << " Each G parallel WF takes: " << heightLocal * width/1024/128 << " MB" << std::endl;
    std::cout << " Each band paralelel WF s: " << height* widthLocal/1024/128 << " MB" << std::endl;
    std::cout << " Each S  takes GPU memory: " << width* width/1024/128 << " MB" << std::endl;
    device_memory();
  }
#endif
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

  Real one = 1.0;
  Real minus_one = -1.0;
  Real zero = 0.0;
  
  GetTime( timeSta );

  device_memcpy_HOST2DEVICE(cu_Xcol.Data(), psiPtr_->Wavefun().Data(), sizeof(Real)*height*widthLocal);
  device_mapping_to_buf( cu_sendbuf.Data(), cu_Xcol.Data(), cu_sendk.Data(), height*widthLocal);

  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;

  GetTime( timeSta );
#ifdef GPUDIREC
  MPI_Alltoallv( &cu_sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &cu_recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
#else
  device_memcpy_DEVICE2HOST( sendbuf.Data(), cu_sendbuf.Data(), sizeof(Real)*height*widthLocal);
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
  device_memcpy_HOST2DEVICE(cu_recvbuf.Data(), recvbuf.Data(), sizeof(Real)*heightLocal*width);
#endif
  GetTime( timeEnd );
  
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  GetTime( timeSta );
  device_mapping_from_buf(cu_X.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);
  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;
  
  // *********************************************************************
  // Main loop
  // *********************************************************************
  
  if(scf_iter == 1) 
  {
    // Orthogonalization through Cholesky factorization
    GetTime( timeSta );
   
    DEVICE_BLAS::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(), 
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
    cu_XTX.CopyFrom( XTX );
#ifdef USE_MAGMA
    MAGMA::Potrf( 'U', width, cu_XTX.Data(), width );
#else
    device_solver::Potrf( 'U', width, cu_XTX.Data(), width );
#endif

    cu_XTX.CopyTo( XTX );
#if 0
    lapack::Potrf( 'U', width, XTX.Data(), width );
#endif
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
  
    cu_XTX.CopyFrom( XTX );
    DEVICE_BLAS::Trsm( right, up, cu_transN, nondiag, heightLocal, width, &one, cu_XTX.Data(), width, cu_X.Data(), heightLocal );
    cu_XTX.CopyTo( XTX );
    
    GetTime( timeEnd );
    iterTrsm = iterTrsm + 1;
    timeTrsm = timeTrsm + ( timeEnd - timeSta );
  
    GetTime( timeSta );
  
    device_mapping_to_buf( cu_recvbuf.Data(), cu_X.Data(), cu_recvk.Data(), heightLocal*width);
    
    GetTime( timeEnd );
    timeMapping += timeEnd - timeSta;
  
    GetTime( timeSta );
#ifdef GPUDIRECT
    MPI_Alltoallv( &cu_recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
        &cu_sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
#else
    device_memcpy_DEVICE2HOST( recvbuf.Data(), cu_recvbuf.Data(), sizeof(Real)*heightLocal*width);
    MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
        &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
    device_memcpy_HOST2DEVICE(cu_sendbuf.Data(), sendbuf.Data(), sizeof(Real)*height*widthLocal);
#endif
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );
  
    GetTime( timeSta );
    device_mapping_from_buf(cu_Xcol.Data(), cu_sendbuf.Data(), cu_sendk.Data(), height*widthLocal);
  }
  else
  {   
    // same to comment out the orthogonalization. 
    device_memcpy_HOST2DEVICE(cu_Xcol.Data(), psiPtr_->Wavefun().Data(), sizeof(Real)*height*widthLocal);
  } 

  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;

  // Applying the Hamiltonian matrix
  {
    GetTime( timeSta );
    Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, noccLocal, false, cu_Xcol.Data(), true);
    deviceNumTns<Real> tnsTemp(ntot, ncom, noccLocal, false, cu_AXcol.Data());

    hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
    GetTime( timeEnd );
    iterSpinor = iterSpinor + 1;
    timeSpinor = timeSpinor + ( timeEnd - timeSta );
  }

  GetTime( timeSta );

  device_mapping_to_buf( cu_sendbuf.Data(), cu_AXcol.Data(), cu_sendk.Data(), height*widthLocal);
  
  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;
  GetTime( timeSta );

#ifdef GPUDIRECT
  MPI_Alltoallv( &cu_sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &cu_recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
#else
  device_memcpy_DEVICE2HOST( sendbuf.Data(), cu_sendbuf.Data(), sizeof(Real)*height*widthLocal);
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
  device_memcpy_HOST2DEVICE(cu_recvbuf.Data(), recvbuf.Data(), sizeof(Real)*heightLocal*width);
#endif

  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  GetTime( timeSta );
  device_mapping_from_buf(cu_AX.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);

  // perform the ACE operator 
  hamPtr_->ACEOperator( cu_X, *fftPtr_, cu_AX);

  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;
  GetTime( time22);
  timeStart += time22 - time11;
#if ( _DEBUGlevel_ >= 1 )
  if(mpirank  == 0)  { std:: cout << " before iter "<< std::endl; device_memory(); } 
#endif
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

    GetTime( time11);

    // XTX <- X' * (AX)
    GetTime( timeSta );
    DEVICE_BLAS::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
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
    cu_Xtemp.CopyFrom ( cu_AX );
#if 0
    lapack::Lacpy( 'A', heightLocal, width, AX.Data(), heightLocal, Xtemp.Data(), heightLocal );
#endif
    GetTime( timeEnd );
    iterCopy = iterCopy + 1;
    timeCopy = timeCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    cu_XTX.CopyFrom(XTX);
    DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one, cu_X.Data(),
                  heightLocal, cu_XTX.Data(), width, &one, cu_Xtemp.Data(), heightLocal );
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
    
    device_mapping_to_buf( cu_recvbuf.Data(), cu_Xtemp.Data(), cu_recvk.Data(), heightLocal*width);

    GetTime( timeEnd );
    timeMapping += timeEnd - timeSta;
    GetTime( timeSta );
#ifdef GPUDIRECT
    MPI_Alltoallv( &cu_recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
        &cu_sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
#else
    device_memcpy_DEVICE2HOST( recvbuf.Data(), cu_recvbuf.Data(), sizeof(Real)*heightLocal*width);
    MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
        &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
    device_memcpy_HOST2DEVICE(cu_sendbuf.Data(), sendbuf.Data(), sizeof(Real)*height*widthLocal);
#endif
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );
    GetTime( timeSta );
    
    device_mapping_from_buf(cu_Xcol.Data(), cu_sendbuf.Data(), cu_sendk.Data(), height*widthLocal);

    GetTime( timeEnd );
    timeMapping += timeEnd - timeSta;

    // Compute W = TW
    {
      GetTime( timeSta );
      //Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, Xcol.VecData(numLockedLocal));
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, cu_Xcol.Data(),true);
      deviceNumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, cu_Wcol.Data());
      //NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));

      spnTemp.AddTeterPrecond( fftPtr_, tnsTemp );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

    // Compute AW = A*W
    {
      GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, cu_Wcol.Data(), true);
      deviceNumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, cu_AWcol.Data());
#if 0
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));
      NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, AWcol.VecData(numLockedLocal));
#endif
      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

    // Convert from column format to row format
    // MPI_Alltoallv
    // Only convert W and AW
#if 1
    GetTime( timeSta );
    
    device_mapping_to_buf( cu_sendbuf.Data(), cu_Wcol.Data(), cu_sendk.Data(), height*widthLocal);

    GetTime( timeEnd );
    timeMapping += timeEnd - timeSta;
    GetTime( timeSta );
#ifdef GPUDIRECT
    MPI_Alltoallv( &cu_sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
        &cu_recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
#else
    device_memcpy_DEVICE2HOST( sendbuf.Data(), cu_sendbuf.Data(), sizeof(Real)*height*widthLocal);
    MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
    device_memcpy_HOST2DEVICE(cu_recvbuf.Data(), recvbuf.Data(), sizeof(Real)*heightLocal*width);
#endif
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );
    GetTime( timeSta );
    
    device_mapping_from_buf(cu_W.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);

    GetTime( timeEnd );
    timeMapping += timeEnd - timeSta;
#endif
    GetTime( timeSta );

    device_mapping_to_buf( cu_sendbuf.Data(), cu_AWcol.Data(), cu_sendk.Data(), height*widthLocal);
    
    GetTime( timeEnd );
    timeMapping += timeEnd - timeSta;
    GetTime( timeSta );
#ifdef GPUDIRECT
    MPI_Alltoallv( &cu_sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
        &cu_recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
#else
    device_memcpy_DEVICE2HOST( sendbuf.Data(), cu_sendbuf.Data(), sizeof(Real)*height*widthLocal);
    MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
    device_memcpy_HOST2DEVICE(cu_recvbuf.Data(), recvbuf.Data(), sizeof(Real)*heightLocal*width);
#endif
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );
    GetTime( timeSta );
    
    device_mapping_from_buf(cu_AW.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);

    // perform the ACE operator 
    hamPtr_->ACEOperator( cu_W, *fftPtr_, cu_AW);

    GetTime( timeEnd );
    GetTime( time2);
    timeHpsi += time2 - time1;
    timeMapping += timeEnd - timeSta;


    // W = W - X(X'W), AW = AW - AX(X'W)
    GetTime( timeSta );
    DEVICE_BLAS::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
                  heightLocal, cu_W.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
    cu_XTXtemp1.CopyTo( XTXtemp1);

    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    //SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );


    GetTime( timeSta );
    cu_XTX.CopyFrom(XTX);
    DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one, cu_X.Data(),
                  heightLocal, cu_XTX.Data(), width, &one, cu_W.Data(), heightLocal );
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 1;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );


    GetTime( timeSta );
    DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one,
                  cu_AX.Data(), heightLocal, cu_XTX.Data(), width, &one, cu_AW.Data(), heightLocal );
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 1;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );

    // Normalize columns of W
    Real normLocal[width]; 
    Real normGlobal[width];

    GetTime( timeSta );
    deviceDblNumVec cu_normLocal(width);
    device_calculate_Energy( cu_W.Data(), cu_normLocal.Data(), width-numLockedLocal, heightLocal ); // note, numLockedLocal == 0
    device_memcpy_DEVICE2HOST( normLocal, cu_normLocal.Data(), sizeof(Real)*width);
#if 0
    for( Int k = numLockedLocal; k < width; k++ ){
      normLocal[k] = Energy(DblNumVec(heightLocal, false, W.VecData(k)));
      normGlobal[k] = 0.0;
    }
#endif
    MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    device_memcpy_HOST2DEVICE(cu_normLocal.Data(), normGlobal, sizeof(Real)*width);
    device_batch_Scal( cu_W.Data(),  cu_normLocal.Data(), width, heightLocal);
    device_batch_Scal( cu_AW.Data(), cu_normLocal.Data(), width, heightLocal);
#if 0    
    for( Int k = numLockedLocal; k < width; k++ ){
      Real norm = std::sqrt( normGlobal[k] );
      blas::Scal( heightLocal, 1.0 / norm, W.VecData(k), 1 );
      blas::Scal( heightLocal, 1.0 / norm, AW.VecData(k), 1 );
    }
#endif
    GetTime( timeEnd );
    iterOther = iterOther + 2;
    timeOther = timeOther + ( timeEnd - timeSta );

    statusOFS << "Time for norm1 in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;


    // P = P - X(X'P), AP = AP - AX(X'P)
    if( numSet == 3 ){
      
      GetTime( timeSta );
      DEVICE_BLAS::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
                  heightLocal, cu_P.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
      cu_XTXtemp1.CopyTo( XTXtemp1 );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
#if 0
      SetValue( XTX, 0.0 );
#endif
      MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

      GetTime( timeSta );
      cu_XTX.CopyFrom( XTX );
      DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one,
                    cu_X.Data(), heightLocal, cu_XTX.Data(), width, &one, cu_P.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );

      DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one,
                    cu_AX.Data(), heightLocal, cu_XTX.Data(), width, &one, cu_AP.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      // Normalize the conjugate direction
      device_calculate_Energy( cu_P.Data(), cu_normLocal.Data(), width-numLockedLocal, heightLocal ); // note, numLockedLocal == 0
      device_memcpy_DEVICE2HOST( normLocal, cu_normLocal.Data(), sizeof(Real)*width);
#if 0
      for( Int k = numLockedLocal; k < width; k++ ){
        normLocal[k] = Energy(DblNumVec(heightLocal, false, P.VecData(k)));
        normGlobal[k] = 0.0;
      }
#endif
      MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      device_memcpy_HOST2DEVICE(cu_normLocal.Data(), normGlobal, sizeof(Real)*width);
      device_batch_Scal( cu_P.Data(),  cu_normLocal.Data(), width, heightLocal);
      device_batch_Scal( cu_AP.Data(), cu_normLocal.Data(), width, heightLocal);
#if 0
      for( Int k = numLockedLocal; k < width; k++ ){
        Real norm = std::sqrt( normGlobal[k] );
        blas::Scal( heightLocal, 1.0 / norm, P.VecData(k), 1 );
        blas::Scal( heightLocal, 1.0 / norm, AP.VecData(k), 1 );
      }
#endif
      GetTime( timeEnd );
      iterOther = iterOther + 2;
      timeOther = timeOther + ( timeEnd - timeSta );
    
      statusOFS << "Time for norm2 in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;
   
    }
    GetTime( time22);
    calTime += time22 - time11;

    // Perform the sweep
    GetTime( timeSta );
    Int sbSize = 1, nsb = width; // this should be generalized to subblocks 
    DblNumMat AMat( 3*sbSize, 3*sbSize ), BMat( 3*sbSize, 3*sbSize );
    DblNumMat AMatAll( 3*sbSize, 3*sbSize*nsb ), BMatAll( 3*sbSize, 3*sbSize*nsb ); // contains all nsb 3-by-3 matrices
    DblNumMat AMatAllLocal( 3*sbSize, 3*sbSize*nsb ), BMatAllLocal( 3*sbSize, 3*sbSize*nsb ); // contains local parts of all nsb 3-by-3 matrices

    // gpu
    deviceDblNumMat cu_AMatAllLocal( 3*sbSize, 3*sbSize*nsb );
    deviceDblNumMat cu_BMatAllLocal( 3*sbSize, 3*sbSize*nsb );

    //SetValue( AMat, 0.0 ); SetValue( BMat, 0.0 );
    //SetValue( AMatAll, 0.0 ); SetValue( BMatAll, 0.0 );
    //SetValue( AMatAllLocal, 0.0 ); SetValue( BMatAllLocal, 0.0 );

    // LOCKING NOT SUPPORTED, loop over all columns 
    GetTime( time1);
    GetTime( timeSta );
    device_setValue( cu_AMatAllLocal.Data(), 0.0, 9*sbSize*sbSize*nsb);
    device_setValue( cu_BMatAllLocal.Data(), 0.0, 9*sbSize*sbSize*nsb);
    
    DEVICE_BLAS::batched_Gemm6( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_X.Data(),
                         heightLocal, cu_AX.Data(), heightLocal, &zero, cu_AMatAllLocal.Data(),
                         3*sbSize, nsb, 3*sbSize, 0, 0,
                    cu_W.Data(), cu_AW.Data(), cu_AMatAllLocal.Data(), 3*sbSize, sbSize, sbSize,
                    cu_X.Data(), cu_AW.Data(), cu_AMatAllLocal.Data(), 3*sbSize, sbSize, 0,
                    cu_X.Data(), cu_X.Data(), cu_BMatAllLocal.Data(), 3*sbSize, 0, 0,
                    cu_W.Data(), cu_W.Data(), cu_BMatAllLocal.Data(), 3*sbSize, sbSize, sbSize,
                    cu_X.Data(), cu_W.Data(), cu_BMatAllLocal.Data(), 3*sbSize, sbSize, 0);
#if 0
    DEVICE_BLAS::batched_Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_X.Data(),
                         heightLocal, cu_AX.Data(), heightLocal, &zero, cu_AMatAllLocal.Data(),
                         3*sbSize, nsb, 3*sbSize, 0, 0);

    DEVICE_BLAS::batched_Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_W.Data(),
                         heightLocal, cu_AW.Data(), heightLocal, &zero, cu_AMatAllLocal.Data(),
                         3*sbSize, nsb, 3*sbSize, sbSize, sbSize);

    DEVICE_BLAS::batched_Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_X.Data(),
                         heightLocal, cu_AW.Data(), heightLocal, &zero, cu_AMatAllLocal.Data(),
                         3*sbSize, nsb, 3*sbSize, sbSize, 0);

    DEVICE_BLAS::batched_Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_X.Data(),
                         heightLocal, cu_X.Data(), heightLocal, &zero, cu_BMatAllLocal.Data(),
                         3*sbSize, nsb, 3*sbSize, 0, 0);

    DEVICE_BLAS::batched_Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_W.Data(),
                         heightLocal, cu_W.Data(), heightLocal, &zero, cu_BMatAllLocal.Data(),
                         3*sbSize, nsb, 3*sbSize, sbSize, sbSize);

    DEVICE_BLAS::batched_Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_X.Data(),
                         heightLocal, cu_W.Data(), heightLocal, &zero, cu_BMatAllLocal.Data(),
                         3*sbSize, nsb, 3*sbSize, sbSize, 0);
#endif
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 6;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );

    if ( numSet == 3 ){
      GetTime( timeSta );
      DEVICE_BLAS::batched_Gemm6(cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_P.Data(),
                 heightLocal, cu_AP.Data(), heightLocal, &zero, cu_AMatAllLocal.Data(),
                 3*sbSize, nsb, 3*sbSize, 2*sbSize, 2*sbSize,
                 cu_X.Data(), cu_AP.Data(), cu_AMatAllLocal.Data(), 3*sbSize, 2*sbSize, 0,
                 cu_W.Data(), cu_AP.Data(), cu_AMatAllLocal.Data(), 3*sbSize, 2*sbSize, sbSize,
                 cu_P.Data(), cu_P.Data(), cu_BMatAllLocal.Data(), 3*sbSize, 2*sbSize, 2*sbSize,
                 cu_X.Data(), cu_P.Data(), cu_BMatAllLocal.Data(), 3*sbSize, 2*sbSize, 0,
                 cu_W.Data(), cu_P.Data(), cu_BMatAllLocal.Data(), 3*sbSize, 2*sbSize, sbSize);
#if 0
      DEVICE_BLAS::batched_Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_P.Data(),
                           heightLocal, cu_AP.Data(), heightLocal, &zero, cu_AMatAllLocal.Data(),
                           3*sbSize, nsb, 3*sbSize, 2*sbSize, 2*sbSize);
      DEVICE_BLAS::batched_Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_X.Data(),
                           heightLocal, cu_AP.Data(), heightLocal, &zero, cu_AMatAllLocal.Data(),
                           3*sbSize, nsb, 3*sbSize, 2*sbSize, 0);
      DEVICE_BLAS::batched_Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_W.Data(),
                           heightLocal, cu_AP.Data(), heightLocal, &zero, cu_AMatAllLocal.Data(),
                           3*sbSize, nsb, 3*sbSize, 2*sbSize, sbSize);


      DEVICE_BLAS::batched_Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_P.Data(),
                           heightLocal, cu_P.Data(), heightLocal, &zero, cu_BMatAllLocal.Data(),
                           3*sbSize, nsb, 3*sbSize, 2*sbSize, 2*sbSize);

      DEVICE_BLAS::batched_Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_X.Data(),
                           heightLocal, cu_P.Data(), heightLocal, &zero, cu_BMatAllLocal.Data(),
                           3*sbSize, nsb, 3*sbSize, 2*sbSize, 0);

      DEVICE_BLAS::batched_Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_W.Data(),
                           heightLocal, cu_P.Data(), heightLocal, &zero, cu_BMatAllLocal.Data(),
                           3*sbSize, nsb, 3*sbSize, 2*sbSize, sbSize);
#endif
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 6;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
    }
   
#if 0
    for( Int k = 0; k < nsb; k++ ){

      // fetch indiviual columns
      DblNumMat  x( heightLocal, sbSize, false, X.VecData(k) );
      DblNumMat  w( heightLocal, sbSize, false, W.VecData(k) );
      DblNumMat ax( heightLocal, sbSize, false, AX.VecData(k) );
      DblNumMat aw( heightLocal, sbSize, false, AW.VecData(k) );

      // gpu data structure. 
      deviceDblNumMat cu_ax( heightLocal, sbSize, false, cu_AX.VecData(k)  );
      deviceDblNumMat cu_x ( heightLocal, sbSize, false, cu_X.VecData(k)  );
      deviceDblNumMat cu_w ( heightLocal, sbSize, false, cu_W.VecData(k) );
      deviceDblNumMat cu_aw( heightLocal, sbSize, false, cu_AW.VecData(k) );

      // Compute AMatAllLoc and BMatAllLoc            
      // AMatAllLoc
      GetTime( timeSta );
/*
      DEVICE_BLAS::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                  heightLocal, cu_ax.Data(), heightLocal, &zero, &cu_AMatAllLocal(0,3*sbSize*k), 3*sbSize );
      DEVICE_BLAS::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
                   heightLocal, cu_aw.Data(), heightLocal, &zero, &cu_AMatAllLocal(sbSize,3*sbSize*k+sbSize), 3*sbSize);

      DEVICE_BLAS::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                   heightLocal, cu_aw.Data(), heightLocal, &zero, &cu_AMatAllLocal(0,3*sbSize*k+sbSize), 3*sbSize);

      // BMatAllLoc            
      DEVICE_BLAS::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                   heightLocal, cu_x.Data(), heightLocal, &zero, &cu_BMatAllLocal(0,3*sbSize*k), 3*sbSize);

      DEVICE_BLAS::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
                   heightLocal, cu_w.Data(), heightLocal, &zero, &cu_BMatAllLocal(sbSize,3*sbSize*k+sbSize), 3*sbSize);

      DEVICE_BLAS::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                   heightLocal, cu_w.Data(), heightLocal, &zero, &cu_BMatAllLocal(0,3*sbSize*k+sbSize), 3*sbSize);
*/

      GetTime( timeEnd );
      iterGemmT = iterGemmT + 6;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      if ( numSet == 3 ){

        DblNumMat  p( heightLocal, sbSize, false, P.VecData(k) );
        DblNumMat ap( heightLocal, sbSize, false, AP.VecData(k) );
        
        // GPU numMat
        deviceDblNumMat  cu_p (heightLocal, sbSize, false, cu_P.VecData(k)  );
        deviceDblNumMat cu_ap (heightLocal, sbSize, false, cu_AP.VecData(k) );

        // AMatAllLoc
        GetTime( timeSta );
#if 0
        DEVICE_BLAS::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_p.Data(),
                     heightLocal, cu_ap.Data(), heightLocal, &zero, &cu_AMatAllLocal(2*sbSize,3*sbSize*k+2*sbSize), 3*sbSize);

        DEVICE_BLAS::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                     heightLocal, cu_ap.Data(), heightLocal, &zero, &cu_AMatAllLocal(0,3*sbSize*k+2*sbSize), 3*sbSize );


        DEVICE_BLAS::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
                     heightLocal, cu_ap.Data(), heightLocal, &zero, &cu_AMatAllLocal(sbSize,3*sbSize*k+2*sbSize), 3*sbSize );


        // BMatAllLoc
        DEVICE_BLAS::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_p.Data(),
                     heightLocal, cu_p.Data(), heightLocal, &zero, &cu_BMatAllLocal(2*sbSize,3*sbSize*k+2*sbSize), 3*sbSize );


        DEVICE_BLAS::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
                     heightLocal, cu_p.Data(), heightLocal, &zero, &cu_BMatAllLocal(0,3*sbSize*k+2*sbSize), 3*sbSize );


        DEVICE_BLAS::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
                     heightLocal, cu_p.Data(), heightLocal, &zero, &cu_BMatAllLocal(sbSize,3*sbSize*k+2*sbSize), 3*sbSize );

#endif
        GetTime( timeEnd );
        iterGemmT = iterGemmT + 6;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

      }             

    }
#endif

    cu_AMatAllLocal.CopyTo( AMatAllLocal );
    cu_BMatAllLocal.CopyTo( BMatAllLocal );
    GetTime( time2);
    firstTime += time2 - time1;

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

    GetTime( time1);
    // Solve nsb small eigenproblems and update columns of X 

#if ( _DEBUGlevel_ >= 1 )
    if(mpirank == 0)
      std:: cout << "nsb is : " << nsb << "  num Set " << numSet  << std::endl;
#endif
    for( Int k = 0; k < nsb; k++ ){

      Real eigs[3*sbSize];
      DblNumMat  cx( sbSize, sbSize ), cw( sbSize, sbSize ), cp( sbSize, sbSize);
      DblNumMat tmp( heightLocal, sbSize );      
       
      // gpu     
      deviceDblNumMat  cu_cx( sbSize, sbSize ), cu_cw( sbSize, sbSize ), cu_cp( sbSize, sbSize);
      deviceDblNumMat cu_tmp( heightLocal, sbSize );      

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

      // cuda parts. 
      deviceDblNumMat  cu_x( heightLocal, sbSize, false, cu_X.VecData(k) );
      deviceDblNumMat  cu_w( heightLocal, sbSize, false, cu_W.VecData(k) );
      deviceDblNumMat  cu_p( heightLocal, sbSize, false, cu_P.VecData(k) );
      deviceDblNumMat cu_ax( heightLocal, sbSize, false, cu_AX.VecData(k) );
      deviceDblNumMat cu_aw( heightLocal, sbSize, false, cu_AW.VecData(k) );
      deviceDblNumMat cu_ap( heightLocal, sbSize, false, cu_AP.VecData(k) );
      GetTime( timeSta );

      device_memcpy_HOST2DEVICE( cu_cx.Data(), &AMat(0,0), sbSize *sbSize*sizeof(Real));
      device_memcpy_HOST2DEVICE( cu_cw.Data(), &AMat(sbSize,0), sbSize *sbSize*sizeof(Real));

      lapack::Lacpy( 'A', sbSize, sbSize, &AMat(0,0), 3*sbSize, cx.Data(), sbSize );
      lapack::Lacpy( 'A', sbSize, sbSize, &AMat(sbSize,0), 3*sbSize, cw.Data(), sbSize );
      GetTime( timeEnd );
      iterCopy = iterCopy + 2;
      timeCopy = timeCopy + ( timeEnd - timeSta );

      //  p = w*cw + p*cp; x = x*cx + p; ap = aw*cw + ap*cp; ax = ax*cx + ap;
      if( numSet == 3 ){
        
        GetTime( timeSta );
        device_memcpy_HOST2DEVICE( cu_cp.Data(), &AMat(2*sbSize,0), sbSize *sbSize*sizeof(Real));
#if 0
        lapack::Lacpy( 'A', sbSize, sbSize, &AMat(2*sbSize,0), 3*sbSize, cp.Data(), sbSize );
#endif
        GetTime( timeEnd );
        iterCopy = iterCopy + 1;
        timeCopy = timeCopy + ( timeEnd - timeSta );
       
        // tmp <- p*cp 
        GetTime( timeSta );
        DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
                cu_p.Data(), heightLocal, cu_cp.Data(), sbSize, &zero, cu_tmp.Data(),heightLocal);
#if 0
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            p.Data(), heightLocal, cp.Data(), sbSize,
            0.0, tmp.Data(), heightLocal );
#endif
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );

        // p <- w*cw + tmp
        GetTime( timeSta );
        DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
                cu_w.Data(), heightLocal, cu_cw.Data(), sbSize, &one, cu_tmp.Data(),heightLocal);
#if 0
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            w.Data(), heightLocal, cw.Data(), sbSize,
            1.0, tmp.Data(), heightLocal );
#endif
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );

        GetTime( timeSta );
        device_memcpy_DEVICE2DEVICE( cu_p.Data(), cu_tmp.Data(), heightLocal*sizeof(Real));
#if 0
        lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, p.Data(), heightLocal );
#endif
        GetTime( timeEnd );
        iterCopy = iterCopy + 1;
        timeCopy = timeCopy + ( timeEnd - timeSta );

        // tmp <- ap*cp 
        GetTime( timeSta );
        DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
                cu_ap.Data(), heightLocal, cu_cp.Data(), sbSize, &zero, cu_tmp.Data(),heightLocal);
#if 0
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            ap.Data(), heightLocal, cp.Data(), sbSize,
            0.0, tmp.Data(), heightLocal );
#endif
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );

        // ap <- aw*cw + tmp
        GetTime( timeSta );
        DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
                cu_aw.Data(), heightLocal, cu_cw.Data(), sbSize, &one, cu_tmp.Data(),heightLocal);
#if 0
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            aw.Data(), heightLocal, cw.Data(), sbSize,
            1.0, tmp.Data(), heightLocal );
#endif
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
        GetTime( timeSta );
        device_memcpy_DEVICE2DEVICE( cu_ap.Data(), cu_tmp.Data(), heightLocal*sizeof(Real));
#if 0
        lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, ap.Data(), heightLocal );
#endif
        GetTime( timeEnd );
        iterCopy = iterCopy + 1;
        timeCopy = timeCopy + ( timeEnd - timeSta );

      }else{
        // p <- w*cw
        GetTime( timeSta );
       
        DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
                cu_w.Data(), heightLocal, cu_cw.Data(), sbSize, &zero, cu_p.Data(),heightLocal);
#if 0
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            w.Data(), heightLocal, cw.Data(), sbSize,
            0.0, p.Data(), heightLocal );
#endif
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
        // ap <- aw*cw
        GetTime( timeSta );
        DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
                cu_aw.Data(), heightLocal, cu_cw.Data(), sbSize, &zero, cu_ap.Data(),heightLocal);
#if 0
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            aw.Data(), heightLocal, cw.Data(), sbSize,
            0.0, ap.Data(), heightLocal );
#endif
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
      }

      // x <- x*cx + p
      GetTime( timeSta );
      device_memcpy_DEVICE2DEVICE( cu_tmp.Data(), cu_p.Data(), heightLocal*sizeof(Real));
#if 0
      lapack::Lacpy( 'A', heightLocal, sbSize, p.Data(), heightLocal, tmp.Data(), heightLocal );
#endif
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );
     
      GetTime( timeSta );
      DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
              cu_x.Data(), heightLocal, cu_cx.Data(), sbSize, &one, cu_tmp.Data(),heightLocal);
#if 0
      blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
          x.Data(), heightLocal, cx.Data(), sbSize,
          1.0, tmp.Data(), heightLocal );
#endif
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );
      
      GetTime( timeSta );
      device_memcpy_DEVICE2DEVICE( cu_x.Data(), cu_tmp.Data(), heightLocal*sizeof(Real));
#if 0
      lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, x.Data(), heightLocal );
#endif
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );

      // ax <- ax*cx + ap
      GetTime( timeSta );
      device_memcpy_DEVICE2DEVICE( cu_tmp.Data(), cu_ap.Data(), heightLocal*sizeof(Real));

#if 0
      lapack::Lacpy( 'A', heightLocal, sbSize, ap.Data(), heightLocal, tmp.Data(), heightLocal );
#endif
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );
      
      GetTime( timeSta );
      DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
              cu_ax.Data(), heightLocal, cu_cx.Data(), sbSize, &one, cu_tmp.Data(),heightLocal);
#if 0
      blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
          ax.Data(), heightLocal, cx.Data(), sbSize,
          1.0, tmp.Data(), heightLocal );
#endif
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      device_memcpy_DEVICE2DEVICE( cu_ax.Data(), cu_tmp.Data(), heightLocal*sizeof(Real));
#if 0
      lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, ax.Data(), heightLocal );
#endif
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );

    }

    GetTime( time2);
    secondTime += time2 - time1;

    GetTime( time1);
    // CholeskyQR of the updated block X
    GetTime( timeSta );
    DEVICE_BLAS::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(), 
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

//    if ( mpirank == 0) {
      GetTime( timeSta );
      GetTime( timeSta1 );
#if 0
      lapack::Potrf( 'U', width, XTX.Data(), width );
#endif
      cu_XTX.CopyFrom( XTX );
#ifdef USE_MAGMA
      MAGMA::Potrf( 'U', width, cu_XTX.Data(), width );
#else
      device_solver::Potrf( 'U', width, cu_XTX.Data(), width );
#endif
      //cu_XTX.CopyTo( XTX );

      GetTime( timeEnd1 );
      iterPotrf = iterPotrf + 1;
      timePotrf = timePotrf + ( timeEnd1 - timeSta1 );
      GetTime( timeEnd );
      iterMpirank0 = iterMpirank0 + 1;
      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
//    }
    GetTime( timeSta );
//    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
    GetTime( timeEnd );
    iterBcast = iterBcast + 1;
    timeBcast = timeBcast + ( timeEnd - timeSta );

    // X <- X * U^{-1} is orthogonal
    GetTime( timeSta );
    DEVICE_BLAS::Trsm( right, up, cu_transN, nondiag, heightLocal, width, &one, cu_XTX.Data(), width, cu_X.Data(), heightLocal );
    cu_XTX.CopyTo( XTX);

    GetTime( timeEnd );
    iterTrsm = iterTrsm + 1;
    timeTrsm = timeTrsm + ( timeEnd - timeSta );
    GetTime( time2);
    thirdTime += time2 - time1;

  } while( (iter < eigMaxIter) || (resMin > eigMinTolerance) );



  // *********************************************************************
  // Post processing
  // *********************************************************************

  // Obtain the eigenvalues and eigenvectors
  // if isConverged==true then XTX should contain the matrix X' * (AX); and X is an
  // orthonormal set

  if (!isConverged){
#if ( _DEBUGlevel_ >= 1 )
    if(mpirank == 0) std:: cout <<" not converged.... " << std::endl;
#endif
    GetTime( timeSta );

    DEVICE_BLAS::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
                heightLocal, cu_AX.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width);
    cu_XTXtemp1.CopyTo( XTXtemp1 );
#if 0
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
        heightLocal, AX.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
#endif
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
  }

  GetTime( timeSta1 );

  if(esdfParam.PWSolver == "PPCGScaLAPACK")
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
    
    statusOFS << std::endl;
    statusOFS << std::endl;
    statusOFS << " ********************************************************. " << std::endl;
    statusOFS << " Scalapack PPCG is Error Prone. Double check to make sure. " << std::endl;
    statusOFS << " Scalapack PPCG is Error Prone. Double check to make sure. " << std::endl;
    statusOFS << " Scalapack PPCG is Error Prone. Double check to make sure. " << std::endl;
    statusOFS << " ********************************************************. " << std::endl;
    statusOFS << std::endl;
    statusOFS << std::endl;
  }
  else //PWSolver_ == "PPCG"
  {
#ifdef GPU_NOPTIMIZE
    if ( mpirank == 0 ){
#endif
      GetTime( timeSta );
      //lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );
      cu_XTX.CopyFrom( XTX );
#ifdef USE_MAGMA
      MAGMA::Syevd( 'V', 'U', width, cu_XTX.Data(), width, eigValS.Data() );
#else
      device_solver::Syevd( 'V', 'U', width, cu_XTX.Data(), width, eigValS.Data() );
#endif

#ifdef GPU_NOPTIMIZE
      cu_XTX.CopyTo( XTX );
#endif
      GetTime( timeEnd );
      iterMpirank0 = iterMpirank0 + 1;
      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
#ifdef GPU_NOPTIMIZE
    }
#endif
  }

  GetTime( timeEnd1 );
  iterSyevd = iterSyevd + 1;
  timeSyevd = timeSyevd + ( timeEnd1 - timeSta1 );

  GetTime( timeSta );
#ifdef GPU_NOPTIMIZE
  MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
  MPI_Bcast(eigValS.Data(), width, MPI_DOUBLE, 0, mpi_comm);
  cu_XTX.CopyFrom( XTX );
#endif
  GetTime( timeEnd );
  iterBcast = iterBcast + 2;
  timeBcast = timeBcast + ( timeEnd - timeSta );

  GetTime( timeSta );
  // X <- X*C

  DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, width, width, &one, cu_X.Data(),
                heightLocal, cu_XTX.Data(), width, &zero, cu_Xtemp.Data(), heightLocal);
  cu_Xtemp.CopyTo( cu_X );
#if 0
  cu_Xtemp.CopyTo( Xtemp );
#endif  
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

#if 0
  GetTime( timeSta );
  lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,
      X.Data(), heightLocal );
  GetTime( timeEnd );
  iterCopy = iterCopy + 1;
  timeCopy = timeCopy + ( timeEnd - timeSta );
#endif


  GetTime( timeSta );
  // AX <- AX*C
  DEVICE_BLAS::Gemm( cu_transN, cu_transN, heightLocal, width, width, &one, cu_AX.Data(),
                heightLocal, cu_XTX.Data(), width, &zero, cu_Xtemp.Data(), heightLocal);
  cu_Xtemp.CopyTo( cu_AX );
#if 0
  cu_Xtemp.CopyTo( Xtemp );
#endif
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );
#if 0
  GetTime( timeSta );
  lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,
      AX.Data(), heightLocal );
  GetTime( timeEnd );
  iterCopy = iterCopy + 1;
  timeCopy = timeCopy + ( timeEnd - timeSta );
#endif

  // Compute norms of individual eigenpairs 
  DblNumVec  resNormLocal ( width ); 
  DblNumVec  resNorm( width );

  deviceDblNumVec cu_eigValS(lda);
  cu_eigValS.CopyFrom(eigValS);

  GetTime( timeSta );
  device_X_Equal_AX_minus_X_eigVal(cu_Xtemp.Data(), cu_AX.Data(), cu_X.Data(), 
                               cu_eigValS.Data(), width, heightLocal);
  //cu_Xtemp.CopyTo( Xtemp );
#if 0
  for(Int j=0; j < width; j++){
    for(Int i=0; i < heightLocal; i++){
      Xtemp(i,j) = AX(i,j) - X(i,j)*eigValS(j);  
    }
  } 
#endif
  GetTime( timeEnd );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd - timeSta );
      
  statusOFS << "Time for Xtemp in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;

  SetValue( resNormLocal, 0.0 );
  GetTime( timeSta );

  deviceDblNumVec  cu_resNormLocal ( width ); 
  device_calculate_Energy( cu_Xtemp.Data(), cu_resNormLocal.Data(), width, heightLocal);
  cu_resNormLocal.CopyTo(resNormLocal);

#if 0
  for( Int k = 0; k < width; k++ ){
    resNormLocal(k) = Energy(DblNumVec(heightLocal, false, Xtemp.VecData(k)));
  }
#endif
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
  //cu_X.CopyFrom( X );
  DEVICE_BLAS::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
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
  //cu_X.CopyFrom(X);
  device_mapping_to_buf( cu_recvbuf.Data(), cu_X.Data(), cu_recvk.Data(), heightLocal*width);
#if 0
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvbuf[recvk(i, j)] = X(i, j);
    }
  }
#endif
  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;
  GetTime( timeSta );

#ifdef GPUDIRECT
  MPI_Alltoallv( &cu_recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
      &cu_sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
#else
  device_memcpy_DEVICE2HOST( recvbuf.Data(), cu_recvbuf.Data(), sizeof(Real)*heightLocal*width);
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
  device_memcpy_HOST2DEVICE(cu_sendbuf.Data(), sendbuf.Data(), sizeof(Real)*height*widthLocal);
#endif

  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );
  GetTime( timeSta );

  device_mapping_from_buf(cu_Xcol.Data(), cu_sendbuf.Data(), cu_sendk.Data(), height*widthLocal);
  device_memcpy_DEVICE2HOST( psiPtr_->Wavefun().Data(),cu_Xcol.Data(), sizeof(Real)*height*widthLocal);

#if 0
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      Xcol(i, j) = sendbuf[sendk(i, j)]; 
    }
  }
#endif
  GetTime( timeEnd );
  timeMapping += timeEnd - timeSta;

  GetTime( timeSta );
#if 0
  lapack::Lacpy( 'A', height, widthLocal, Xcol.Data(), height, 
      psiPtr_->Wavefun().Data(), height );
#endif
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
    statusOFS << "Time for start overhead   = " << iterOther           << "  overheadTime     = " << timeStart << std::endl;
    statusOFS << "Time for calTime          = " << iterOther           << "  calTime          = " << calTime   << std::endl;
    statusOFS << "Time for FIRST            = " << iterOther           << "  firstTime        = " << firstTime << std::endl;
    statusOFS << "Time for SECOND           = " << iterOther           << "  secondTime       = " << secondTime<< std::endl;
    statusOFS << "Time for Third            = " << iterOther           << "  thirdTime        = " << thirdTime << std::endl;
    statusOFS << "Time for overhead + first + second + third      = " << timeStart + calTime + firstTime + secondTime + thirdTime << std::endl;

    statusOFS << "Time for PPCG in PWDFT is " <<  timeEnd2 - timeSta2  << std::endl << std::endl;
#endif

    device_reset_vtot_flag();   // set the vtot_flag to false.
    //device_clean_vtot();
    return ;
}         // -----  end of method EigenSolver::PPCGSolveReal  ----- 


} // namespace dgdft
