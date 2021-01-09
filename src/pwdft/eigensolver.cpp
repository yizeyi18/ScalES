/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin, Wei Hu, Amartya Banerjee, Weile Jia

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
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"

#include "block_distributor_decl.hpp"

#include <numeric>

using namespace dgdft::scalapack;
using namespace dgdft::esdf;


namespace dgdft{

namespace detail {

// TODO: make type_traits.hpp
template <typename T>
struct make_real_type {
  using type = T;
};

template <typename T>
struct make_real_type<std::complex<T>> {
  using type = T;
};

template <typename T>
using real_t = typename make_real_type<T>::type;


template <typename T>
void row_dist_col_norm( int64_t NLocal,
                        int64_t K,
                        const T* XLocal,
                        int64_t  LDXLocal,
                        real_t<T>* NRMS,
                        MPI_Comm comm ) {

  // Compute local norm contributions
  for( Int k = 0; k < K; ++k ) {
    NRMS[k] = 0.;
    auto Xk = XLocal + k*LDXLocal;
    // TODO: need to add conj for complex
    for( Int i = 0; i < NLocal; ++i )
      NRMS[k] += Xk[i] * Xk[i];
  }

  // Reduce
  // TODO: Deduce MPI_Type on template param
  MPI_Allreduce( MPI_IN_PLACE, NRMS, K, MPI_DOUBLE, MPI_SUM, comm );

  // Compute norms
  for( Int k = 0; k < K; ++k ) NRMS[k] = std::sqrt( NRMS[k] ); 


}
                        

template <typename T>
void row_dist_inner_replicate( int64_t  NLocal, 
                               int64_t  K,
                               int64_t  L,
                               const T* XLocal,
                               int64_t  LDXLocal,
                               const T* YLocal,   
                               int64_t  LDYLocal,
                               T*       M,
                               MPI_Comm comm ) {

  // Compute Local GEMM
  blas::Gemm( 'C', 'N', K, L, NLocal, T(1.), XLocal, LDXLocal,
              YLocal, LDYLocal, T(0.), M, K );

  // Reduce In Place
  // XXX: Deduce MPI_Type on template param
  MPI_Allreduce( MPI_IN_PLACE, M, K*L, MPI_DOUBLE, MPI_SUM, comm );

}

template <typename T>
void replicated_cholesky_qr_row_dist( int64_t NLocal, 
                                      int64_t K,
                                      T*      XLocal,
                                      int64_t LDXLocal,
                                      T*      R,
                                      MPI_Comm comm ) {

  row_dist_inner_replicate( NLocal, K, K, XLocal, LDXLocal, XLocal, LDXLocal,
                            R, comm );

  Int mpi_rank;
  MPI_Comm_rank( comm, &mpi_rank );

  // Compute Cholesky on root + Bcast
  // XXX: POTRF is replicatable, no reason to waste communication post replicated
  //      inner product
  if( mpi_rank == 0 ) {
    lapack::Potrf( 'U', K, R, K );
  }
  MPI_Bcast( R, K*K, MPI_DOUBLE, 0, comm );

  // X <- X * U**-1
  blas::Trsm( 'R', 'U', 'N', 'N', NLocal, K, T(1.), R, K, XLocal, LDXLocal );

}


}



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
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft ) {
  hamPtr_ = &ham;
  psiPtr_ = &psi;
  fftPtr_ = &fft;


  eigVal_.Resize(psiPtr_->NumStateTotal());  SetValue(eigVal_, 0.0);
  resVal_.Resize(psiPtr_->NumStateTotal());  SetValue(resVal_, 0.0);

  scaBlockSize_      = esdfParam.scaBlockSize;
  numProcScaLAPACK_  = esdfParam.numProcScaLAPACKPW; 

  PWDFT_Cheby_use_scala_ = esdfParam.PWDFT_Cheby_use_scala;

  // Setup BLACS
  if( esdfParam.PWSolver == "LOBPCGScaLAPACK" || esdfParam.PWSolver == "PPCGScaLAPACK" || 
      (esdfParam.PWSolver == "CheFSI" && PWDFT_Cheby_use_scala_) ){
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



void
EigenSolver::LOBPCGSolveReal    (
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
  Real timeTrsm = 0.0;
  Real timeMpirank0 = 0.0;
  Real timeScaLAPACKFactor = 0.0;
  Real timeScaLAPACK = 0.0;
  Int  iterGemmT = 0;
  Int  iterGemmN = 0;
  Int  iterAllreduce = 0;
  Int  iterAlltoallv = 0;
  Int  iterTrsm = 0;
  Int  iterMpirank0 = 0;
  Int  iterScaLAPACKFactor = 0;
  Int  iterScaLAPACK = 0;

  Real timeSpinor     = 0.0;
  Real timePrec       = 0.0;
  Real timeR2C        = 0.0;
  Real timeC2R        = 0.0;
  Real timeCholQR     = 0.0;
  Real timeBlockInner = 0.0;
  Int  iterR2C        = 0;
  Int  iterC2R        = 0;
  Int  iterCholQR     = 0;
  Int  iterSpinor     = 0;
  Int  iterPrec       = 0;
  Int  iterBlockInner = 0;






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
  //statusOFS << "DBWY IN LOBPCG" << std::endl;

  // Set up distributor
  //BlockDistributor<double> bdist( mpi_comm, height, width );
  auto bdist = 
    make_block_distributor<double>( BlockDistAlg::HostGeneric, mpi_comm,
                                    height, width );

  // Setup profiling wrappers
  auto profile_col_to_row = [&]( const DblNumMat& col_data, DblNumMat& row_data ) {

    Real c2rStart, c2rEnd;

    GetTime( c2rStart );
    bdist.redistribute_col_to_row( col_data, row_data );
    GetTime( c2rEnd );

    timeC2R += (c2rEnd - c2rStart);
    iterC2R++;

  };

  auto profile_row_to_col = [&]( const DblNumMat& row_data, DblNumMat& col_data ) {

    Real r2cStart, r2cEnd;

    GetTime( r2cStart );
    bdist.redistribute_row_to_col( row_data, col_data );
    GetTime( r2cEnd );

    timeR2C += (r2cEnd - r2cStart);
    iterR2C++;

  };

  auto profile_chol_qr = [&]( DblNumMat& _X, DblNumMat& _R, MPI_Comm _c ) {

    Real cholQRStart, cholQREnd;

    GetTime( cholQRStart );
    detail::replicated_cholesky_qr_row_dist( _X.m(), _X.n(), _X.Data(), _X.m(),
                                             _R.Data(), _c );
    GetTime( cholQREnd );

    timeCholQR += (cholQREnd - cholQRStart);
    iterCholQR++;

  };

  auto profile_matvec = [&]( int64_t nS_total, int64_t nS_local, 
                             const DblNumMat& _X, DblNumMat& _AX ) {

    Real spinorStart, spinorEnd;

    GetTime( spinorStart );

    // Setup temporary datastructures for Matvec
    Spinor spnTmp( fftPtr_->domain, ncom, nS_total, nS_local, false, _X.Data() );
    NumTns<Real> tnsTmp( ntot, ncom, nS_local, false, _AX.Data() );

    // Perform Matvec
    hamPtr_->MultSpinor( spnTmp, tnsTmp, *fftPtr_ );

    GetTime( spinorEnd );

    timeSpinor += (spinorEnd - spinorStart);
    iterSpinor++;

  };

  auto profile_applyprec = [&]( int64_t nS_total, int64_t nS_local, 
                           const DblNumMat& _X, DblNumMat& _TX ) {

    Real precStart, precEnd;

    GetTime( precStart );

    // Setup temporary datastructures for Preconditioner
    Spinor spnTmp( fftPtr_->domain, ncom, nS_total, nS_local, false, _X.Data() );
    NumTns<Real> tnsTmp( ntot, ncom, nS_local, false, _TX.Data() );

    // Apply preconditioner
    SetValue( tnsTmp, 0. );
    spnTmp.AddTeterPrecond( fftPtr_, tnsTmp );

    GetTime( precEnd );

    timePrec += (precEnd - precStart);
    iterPrec++;

  };



  auto profile_dist_inner = [&]( int64_t _m, int64_t _n, const Real* _X,
                                 const Real* _Y, Real* _RES ) {


    Real innerStart, innerEnd;

    GetTime( innerStart );
    // All matrices in that use this code have a lead dimension / contraction
    // dimenstion of heightLocal
    detail::row_dist_inner_replicate( heightLocal, _m, _n, _X, heightLocal,
                                      _Y, heightLocal, _RES, mpi_comm );

    GetTime( innerEnd );

    timeBlockInner += (innerEnd - innerStart);
    iterBlockInner++;

  };

  // Xp = BETA * Xp + _ALPHA * X * C
  // X is heightLocal x k
  // C is k x l
  // Xp is heightLocal x l
  auto basis_update = 
    [&]( Int _k, Int _l, Real _ALPHA, const Real* _X, const Real* _C, Int _LDC,
         Real _BETA, Real* _XP ) {

    blas::Gemm( 'N', 'N', heightLocal, _l, _k, _ALPHA, _X, heightLocal,
                _C, _LDC, _BETA, _XP, heightLocal );

  };



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

  bool isConverged = false;

  // Initialization
  SetValue( S, 0.0 );
  SetValue( AS, 0.0 );

  DblNumVec  eigValS(lda);
  SetValue( eigValS, 0.0 );

  // Initialize X by the data in psi
  lapack::Lacpy( 'A', height, widthLocal, psiPtr_->Wavefun().Data(), height, 
      Xcol.Data(), height );


  // Redistribute X from Row -> Col format
  profile_col_to_row( Xcol, X );

  // *********************************************************************
  // Main loop
  // *********************************************************************

  // Orthogonalize X via Cholesky QR
  profile_chol_qr( X, XTX, mpi_comm );

  // Redistribute from X Row -> Col format
  profile_row_to_col( X, Xcol );

  // Applying the Hamiltonian matrix
  // AX = H * X (col format)
  profile_matvec( noccTotal, noccLocal, Xcol, AXcol );

  // Redistribute AX from Col -> Row format
  profile_col_to_row( AXcol, AX );



  // Start the main loop
  Int iter = 0;
  Real lockTolerance = eigTolerance;

  bool enableLocking = true;
  if( not enableLocking ) {
    lockTolerance = -1.;
  }

  statusOFS << "Minimum tolerance is " << eigMinTolerance << std::endl;
  statusOFS << "Locking tolerance is " << lockTolerance   << std::endl;


  do {
    iter++;
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "iter = " << iter << std::endl;
#endif

    if( iter == 1 || isRestart == true ) numSet = 2;
    else                                 numSet = 3;

    SetValue( AMat, 0.0 );
    SetValue( BMat, 0.0 );

    // Precompute static pointer offsets
    auto* A_XTAX = &AMat(0,     0);
    auto* A_XTAW = &AMat(0,     width);
    auto* A_WTAW = &AMat(width, width);

    auto* B_XTX = &BMat(0,     0    );
    auto* B_XTW = &BMat(0,     width);
    auto* B_WTW = &BMat(width, width);

    auto* C_X = &AMat(0,     0);
    auto* C_W = &AMat(width, 0);


    // AMat(1:width,1:width) <- X' * (AX)
    // Save a copy in XTX and A_XTAX
    profile_dist_inner( width, width, X.Data(), AX.Data(), XTX.Data() );
    lapack::Lacpy( 'A', width, width, XTX.Data(), width, A_XTAX, lda );

    // Compute the residual  R = AX - X*(X'*AX)

    // R <- AX
    lapack::Lacpy( 'A', heightLocal, width, AX.Data(), heightLocal, 
                   Xtemp.Data(), heightLocal );

    // R <- R - X * (X'*AX)
    // XXX: Is this correct?!
    basis_update( width, width, -1.0, X.Data(), XTX.Data(), width, 1.0, 
                  Xtemp.Data() );

    // Compute the norm of the residual
    detail::row_dist_col_norm( heightLocal, width, Xtemp.Data(), heightLocal,
                               resNorm.Data(), mpi_comm );
    // XXX: ??
    for( Int k = 0; k < width; ++k )
      resNorm(k) /= std::max( 1.0, std::abs( XTX(k,k) ) );

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


    Int nLock = std::count_if( resNorm.Data(), resNorm.Data() + numEig, 
                               [&](const auto x){ return x <= lockTolerance; } );
    Int nActive = width - nLock;


    Int nLockLocal   = (nLock / mpisize) + !!(mpirank < (nLock % mpisize));
    Int nActiveLocal = widthLocal - nLockLocal;


    // Precompute dynamic pointer offsets
#if 0
    // TODO: locking
    auto* A_XTAP = &AMat(0,       2*width);
    auto* A_WTAP = &AMat(width,   2*width);
    auto* A_PTAP = &AMat(2*width, 2*width);
    
    auto* B_XTP = &BMat(0,       2*width);
    auto* B_WTP = &BMat(width,   2*width);
    auto* B_PTP = &BMat(2*width, 2*width);

    auto* C_P = &AMat(2*width, 0);
#else
    auto* A_XTAP = &AMat(0,             width+nActive);
    auto* A_WTAP = &AMat(width,         width+nActive);
    auto* A_PTAP = &AMat(width+nActive, width+nActive);
    
    auto* B_XTP = &BMat(0,       width+nActive);
    auto* B_WTP = &BMat(width,   width+nActive);
    auto* B_PTP = &BMat(width+nActive, width+nActive);

    auto* C_P = &AMat(width+nActive, 0);
#endif
    

#if 1
    statusOFS << "NLOCK         = " << nLock        << std::endl;
    statusOFS << "NACTIVE       = " << nActive      << std::endl;
    statusOFS << std::endl;
#endif

    if( nLockLocal + nActiveLocal != widthLocal )
      ErrorHandling("You Assumed Wrong...");

#if 0
    DblNumMat TMP(5,4);
    std::vector< int > perm = { 2, 1, 0, 3 };
    for( auto j = 0; j < 4; ++j )
    for( auto i = 0; i < 5; ++i )
      TMP(i,j) = j;

    statusOFS << "BEFORE PERM" << std::endl << TMP << std::endl;
    lapack::ColPermute( true, 5, 4, TMP.Data(), 5, perm.data() ); 
    statusOFS << "AFTER PERM" << std::endl << TMP << std::endl;
#endif

    // Permute residual vectors so unconverged ones come first
    {
      std::vector<Int> perm( width );
      std::iota( perm.begin(), perm.end(), 0 );
      std::stable_sort( perm.begin(), perm.end(), [&](Int i, Int j) {
        return resNorm(i) > resNorm(j);
      });

      lapack::ColPermute( true, heightLocal, width, Xtemp.Data(), heightLocal,
                          perm.data() );
    }


    // Compute the preconditioned residual W = T*R.
    // The residual is saved in Xtemp
      
    // Redistribute from Xtemp (Row) -> Xcol (Col)
    profile_row_to_col( Xtemp, Xcol );

    // W <- T * R (Col format) 
    profile_applyprec( noccTotal, noccLocal, Xcol, Wcol );

    // Normalize the preconditioned residual
    ColNormalize( Wcol );

    // Normalize the conjugate direction (P) and update AP
    DblNumVec normP(width);
    detail::row_dist_col_norm( heightLocal, width, P.Data(), heightLocal,
                               normP.Data(), mpi_comm );

    for( Int k = 0; k < width; ++k ) {
      blas::Scal( heightLocal, 1./normP(k), P.VecData(k),  1 );
      blas::Scal( heightLocal, 1./normP(k), AP.VecData(k), 1 );
    }


    /*** Compute AMat ***/

    // Compute AW = A*W
    profile_matvec( noccTotal, noccLocal, Wcol, AWcol );

    // Convert W/AW from Col to Row formats
    profile_col_to_row( Wcol,  W  );
    profile_col_to_row( AWcol, AW );

#if 0
    // Compute X' * (AW)
    // TODO: locking
    profile_dist_inner( width, width, X.Data(), AW.Data(), XTXtemp.Data() );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, A_XTAW, lda );

    // Compute W' * (AW)
    // TODO: locking
    profile_dist_inner( width, width, W.Data(), AW.Data(), XTXtemp.Data() );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, A_WTAW, lda );
#else
    // Compute X' * (AW)
    profile_dist_inner( width, nActive, X.Data(), AW.Data(), XTXtemp.Data() );
    lapack::Lacpy( 'A', width, nActive, XTXtemp.Data(), width, A_XTAW, lda );

    // Compute W' * (AW)
    profile_dist_inner( nActive, nActive, W.Data(), AW.Data(), XTXtemp.Data() );
    lapack::Lacpy( 'A', nActive, nActive, XTXtemp.Data(), nActive, 
                   A_WTAW, lda );
#endif

    if( numSet == 3 ){

#if 0
      // Compute X' * (AP)
      // TODO: locking
      profile_dist_inner( width, width, X.Data(), AP.Data(), XTXtemp.Data() );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, A_XTAP, lda );

      // Compute W' * (AP)
      profile_dist_inner( width, width, W.Data(), AP.Data(), XTXtemp.Data() );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, A_WTAP, lda );

      // Compute P' * (AP)
      profile_dist_inner( width, width, P.Data(), AP.Data(), XTXtemp.Data() );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, A_PTAP, lda );
#else
      // Compute X' * (AP)
      profile_dist_inner( width, nActive, X.Data(), AP.Data(), XTXtemp.Data() );
      lapack::Lacpy( 'A', width, nActive, XTXtemp.Data(), width, A_XTAP, lda );

      // Compute W' * (AP)
      profile_dist_inner( nActive, nActive, W.Data(), AP.Data(), 
                          XTXtemp.Data() );
      lapack::Lacpy( 'A', nActive, nActive, XTXtemp.Data(), nActive, 
                     A_WTAP, lda );

      // Compute P' * (AP)
      profile_dist_inner( nActive, nActive, P.Data(), AP.Data(), 
                          XTXtemp.Data() );
      lapack::Lacpy( 'A', nActive, nActive, XTXtemp.Data(), nActive, 
                     A_PTAP, lda );
#endif

    }


    /*** Compute BMat (overlap matrix) ***/

    // Compute X'*X
    // XXX: Isn't this I?
    profile_dist_inner( width, width, X.Data(), X.Data(), XTXtemp.Data() );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, B_XTX, lda );

#if 0
    // Compute X'*W
    profile_dist_inner( width, width, X.Data(), W.Data(), XTXtemp.Data() );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, B_XTW, lda );

    // Compute W'*W
    profile_dist_inner( width, width, W.Data(), W.Data(), XTXtemp.Data() );
    lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, B_WTW, lda );
#else
    // Compute X'*W
    profile_dist_inner( width, nActive, X.Data(), W.Data(), XTXtemp.Data() );
    lapack::Lacpy( 'A', width, nActive, XTXtemp.Data(), width, B_XTW, lda );

    // Compute W'*W
    profile_dist_inner( nActive, nActive, W.Data(), W.Data(), XTXtemp.Data() );
    lapack::Lacpy( 'A', nActive, nActive, XTXtemp.Data(), nActive, 
                   B_WTW, lda );
#endif


    if( numSet == 3 ) {
#if 0
      // Compute X'*P
      profile_dist_inner( width, width, X.Data(), P.Data(), XTXtemp.Data() );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, B_XTP, lda );

      // Compute W'*P
      profile_dist_inner( width, width, W.Data(), P.Data(), XTXtemp.Data() );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, B_WTP, lda );

      // Compute P'*P
      profile_dist_inner( width, width, P.Data(), P.Data(), XTXtemp.Data() );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, B_PTP, lda );
#else
      // Compute X'*P
      profile_dist_inner( width, nActive, X.Data(), P.Data(), XTXtemp.Data() );
      lapack::Lacpy( 'A', width, nActive, XTXtemp.Data(), width, B_XTP, lda );

      // Compute W'*P
      profile_dist_inner( nActive, nActive, W.Data(), P.Data(), 
                          XTXtemp.Data() );
      lapack::Lacpy( 'A', nActive, nActive, XTXtemp.Data(), nActive, 
                     B_WTP, lda );

      // Compute P'*P
      profile_dist_inner( nActive, nActive, P.Data(), P.Data(), 
                          XTXtemp.Data() );
      lapack::Lacpy( 'A', nActive, nActive, XTXtemp.Data(), nActive, 
                     B_PTP, lda );
#endif

    } // if( numSet == 3 )

#if ( _DEBUGlevel_ >= 2 )
    {
      DblNumMat WTW( nActive, nActive );
      lapack::Lacpy( 'A', nActive, nActive, B_WTW, lda, 
                     WTW.Data(), nActive );
      statusOFS << "W'*W = " << WTW << std::endl;
      if( numSet == 3 )
      {
        DblNumMat PTP( nActive, nActive );
        lapack::Lacpy( 'A', nActive, nActive, B_PTP, lda, 
                       PTP.Data(), nActive );
        statusOFS << "P'*P = " << PTP << std::endl;
      }
    }
#endif

    // Rayleigh-Ritz Procedure
    // AMat * C = BMat * C * Lambda
    // Assuming the dimension (needed) for C is width * width, then
    //     ( C_X )
    //     ( --- )
    // C = ( C_W )
    //     ( --- )
    //     ( C_P )
    //
    Int numCol;
#if 0
    if( numSet == 3 ){
      // Conjugate gradient
      numCol = 3 * width;
    }
    else{
      numCol = 2 * width;
    }
#else
    if( numSet == 3 ){
      // Conjugate gradient
      numCol = width + 2 * nActive;
    }
    else{
      numCol = width + nActive;
    }
#endif

    // Solve Rayleigh-Ritz problem
    // TODO: Handle ScaLAPACK path
    if( mpirank == 0 ) {
      lapack::Sygvd( 1, 'V', 'U', numCol, A_XTAX, lda, B_XTX, lda,
                     eigValS.Data() );
    }


    // All processors synchronize the information
    MPI_Bcast(AMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm);
    MPI_Bcast(BMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm);
    MPI_Bcast(eigValS.Data(), lda, MPI_DOUBLE, 0, mpi_comm);

    if( numSet == 2 ){

      // Update the eigenvectors 
      // X <- X * C_X + W * C_W
      basis_update( width,   width, 1.0, X.Data(), C_X, lda, 0.0, Xtemp.Data() );
      basis_update( nActive, width, 1.0, W.Data(), C_W, lda, 1.0, Xtemp.Data() );

      // Save the result into X
      lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal, 
                     X.Data(), heightLocal );

      // P <- W
      lapack::Lacpy( 'A', heightLocal, nActive, W.Data(), heightLocal, 
                     P.Data(), heightLocal );

    } else { //numSet == 3

      // Compute the conjugate direction
      // P <- W * C_W + P * C_P
      basis_update( nActive, width, 1.0, W.Data(), C_W, lda, 0.0, Xtemp.Data() );
      basis_update( nActive, width, 1.0, P.Data(), C_P, lda, 1.0, Xtemp.Data() );

      lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal, 
                     P.Data(), heightLocal );

      // Update the eigenvectors
      // X <- X * C_X + P
      basis_update( width, width, 1.0, X.Data(), C_X, lda, 1.0, Xtemp.Data() );
      lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,
                     X.Data(), heightLocal );

    } // if ( numSet == 2 )


    // Update AX and AP
    if( numSet == 2 ){

      // AX <- AX * C_X + AW * C_W
      basis_update( width,   width, 1.0, AX.Data(), C_X, lda, 0.0, Xtemp.Data() );
      basis_update( nActive, width, 1.0, AW.Data(), C_W, lda, 1.0, Xtemp.Data() );

      lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,
                     AX.Data(), heightLocal );

      // AP <- AW
      lapack::Lacpy( 'A', heightLocal, nActive, AW.Data(), heightLocal,
                     AP.Data(), heightLocal );

    } else { // numSet == 3
      // AP <- AW * C_W + A_P * C_P
      basis_update( nActive, width, 1.0, AW.Data(), C_W, lda, 0.0, Xtemp.Data() );
      basis_update( nActive, width, 1.0, AP.Data(), C_P, lda, 1.0, Xtemp.Data() );

      lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal, 
                     AP.Data(), heightLocal );

      // AX <- AX * C_X + AP
      basis_update( width, width, 1.0, AX.Data(), C_X, lda, 1.0, Xtemp.Data() );
      lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal, 
                     AX.Data(), heightLocal );

    } // if ( numSet == 2 )

#if ( _DEBUGlevel_ >= 1 )
    //statusOFS << "numLocked = " << numLocked << std::endl;
    statusOFS << "eigValS   = " << eigValS << std::endl;
#endif

  } while( (iter < (10 * eigMaxIter)) && ( (iter < eigMaxIter) || (resMin > eigMinTolerance) ) );



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



  basis_update( width, width, 1.0, X.Data(), XTX.Data(), width, 
                0.0, Xtemp.Data() );

  lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,
                 X.Data(), heightLocal );

#if ( _DEBUGlevel_ >= 2 )

  // Compute basis overlap
  profile_dist_inner( width, width, X.Data(), X.Data(), XTX.Data() );
  statusOFS << "After the LOBPCG, XTX = " << XTX << std::endl;

#endif

  // Save the eigenvalues and eigenvectors back to the eigensolver data
  // structure

  eigVal_ = DblNumVec( width, true, eigValS.Data() );
  resVal_ = resNorm;

  // Redistribute X Row -> Col
  profile_row_to_col( X, Xcol );

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
      << " iterations, LOBPCG reaches the max number of iterations. " << std::endl
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
  if( esdfParam.PWSolver == "LOBPCGScaLAPACK" ){
    statusOFS << "Time for ScaLAPACK     = " << iterScaLAPACK       << "  timeScaLAPACK = " << timeScaLAPACK << std::endl;
    statusOFS << "Time for pdpotrf       = " << iterScaLAPACKFactor << "  timepdpotrf   = " << timeScaLAPACKFactor << std::endl;
  }
#endif

  return ;
}         // -----  end of meehod EigenSolver::LOBPCGSolveReal  ----- 




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
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << std::endl << " Time for computing residual = " << (res_time_end - res_time_sta) << " s.";
      statusOFS << std::endl;
#endif


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
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << std::endl << " Time for computing residual = " << (res_time_end - res_time_sta) << " s.";
    statusOFS << std::endl;
#endif

  }








  return;
} // -----  end of method EigenSolver::GeneralChebyStep -----




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
#if 1
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
#endif


  //statusOFS << "DBWY IN PPCG" << std::endl;
  //BlockDistributor<double> bdist( mpi_comm, height, width );
  auto bdist = 
    make_block_distributor<double>( BlockDistAlg::HostGeneric, mpi_comm,
                                    height, width );

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
  bdist.redistribute_col_to_row( Xcol, X );
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  // *********************************************************************
  // Main loop
  // *********************************************************************

  // Orthogonalization through Cholesky factorization
  detail::replicated_cholesky_qr_row_dist( heightLocal, width, X.Data(), heightLocal,
                                           XTX.Data(), mpi_comm );

  // Redistribute from X Row -> Col format
  bdist.redistribute_row_to_col( X, Xcol );

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

  // Redistribute AX from Col -> Row format
  bdist.redistribute_col_to_row( AXcol, AX );


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
    detail::row_dist_inner_replicate( heightLocal, width, width, 
                                      X.Data(),  heightLocal, 
                                      AX.Data(), heightLocal, XTX.Data(), 
                                      mpi_comm );

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

    bdist.redistribute_row_to_col( Xtemp, Xcol );

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

    bdist.redistribute_col_to_row( Wcol, W );
    bdist.redistribute_col_to_row( AWcol, AW );


    // W = W - X(X'W), AW = AW - AX(X'W)
    detail::row_dist_inner_replicate( heightLocal, width, width,
                                      X.Data(), heightLocal,
                                      W.Data(), heightLocal, XTX.Data(), 
                                      mpi_comm );


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

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for norm1 in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;
#endif


    // P = P - X(X'P), AP = AP - AX(X'P)
    if( numSet == 3 ){
      detail::row_dist_inner_replicate( heightLocal, width, width,
                                        X.Data(), heightLocal,
                                        P.Data(), heightLocal, XTX.Data(), 
                                        mpi_comm );

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

#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for norm2 in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;
#endif

    }

    // Perform the sweep
    GetTime( timeSta );
    Int sbSize = esdfParam.PPCGsbSize, nsb = width/sbSize; // this should be generalized to subblocks 
    DblNumMat  AMat( 3*sbSize, 3*sbSize ), BMat( 3*sbSize, 3*sbSize );
    DblNumMat  AMatAll( 3*sbSize, 3*sbSize*nsb ), BMatAll( 3*sbSize, 3*sbSize*nsb ); // contains all nsb 3-by-3 matrices
    DblNumMat  AMatAllLocal( 3*sbSize, 3*sbSize*nsb ), BMatAllLocal( 3*sbSize, 3*sbSize*nsb ); // contains local parts of all nsb 3-by-3 matrices

    SetValue( AMat, 0.0 ); SetValue( BMat, 0.0 );
    SetValue( AMatAll, 0.0 ); SetValue( BMatAll, 0.0 );
    SetValue( AMatAllLocal, 0.0 ); SetValue( BMatAllLocal, 0.0 );

    // LOCKING NOT SUPPORTED, loop over all columns 
    for( Int k = 0; k < nsb; k++ ){

      // fetch indiviual columns
      DblNumMat  x( heightLocal, sbSize, false, X.VecData(sbSize*k) );
      DblNumMat  w( heightLocal, sbSize, false, W.VecData(sbSize*k) );
      DblNumMat ax( heightLocal, sbSize, false, AX.VecData(sbSize*k) );
      DblNumMat aw( heightLocal, sbSize, false, AW.VecData(sbSize*k) );

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
      DblNumMat  x( heightLocal, sbSize, false, X.VecData(sbSize*k) );
      DblNumMat  w( heightLocal, sbSize, false, W.VecData(sbSize*k) );
      DblNumMat  p( heightLocal, sbSize, false, P.VecData(sbSize*k) );
      DblNumMat ax( heightLocal, sbSize, false, AX.VecData(sbSize*k) );
      DblNumMat aw( heightLocal, sbSize, false, AW.VecData(sbSize*k) );
      DblNumMat ap( heightLocal, sbSize, false, AP.VecData(sbSize*k) );

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
    detail::replicated_cholesky_qr_row_dist( heightLocal, width, X.Data(), heightLocal,
                                             XTX.Data(), mpi_comm );


    //            // Copy the eigenvalues
    //            SetValue( eigValS, 0.0 );
    //            for( Int i = 0; i < numKeep; i++ ){
    //                eigValS[i] = eigs[i];
    //            }


    //#if ( _DEBUGlevel_ >= 1 )
    //        statusOFS << "numLocked = " << numLocked << std::endl;
    //        statusOFS << "eigValS   = " << eigValS << std::endl;
    //#endif

  } while( (iter < (10 * eigMaxIter)) && ( (iter < eigMaxIter) || (resMin > eigMinTolerance) ) );



  // *********************************************************************
  // Post processing
  // *********************************************************************

  // Obtain the eigenvalues and eigenvectors
  // if isConverged==true then XTX should contain the matrix X' * (AX); and X is an
  // orthonormal set

  if (!isConverged){
    detail::row_dist_inner_replicate( heightLocal, width, width,
                                      X.Data(), heightLocal,
                                      AX.Data(), heightLocal, XTX.Data(), 
                                      mpi_comm );
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
  }
  else //esdfParam.PWSolver == "PPCG"
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

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "Time for Xtemp in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;
#endif

  SetValue( resNormLocal, 0.0 );
  GetTime( timeSta );
  for( Int k = 0; k < width; k++ ){
    resNormLocal(k) = Energy(DblNumVec(heightLocal, false, Xtemp.VecData(k)));
  }
  GetTime( timeEnd );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd - timeSta );

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "Time for resNorm in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;
#endif

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

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "Time for resMax and resMin in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;
#endif

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "resNorm = " << resNorm << std::endl;
  statusOFS << "eigValS = " << eigValS << std::endl;
  statusOFS << "maxRes  = " << resMax  << std::endl;
  statusOFS << "minRes  = " << resMin  << std::endl;
#endif



#if ( _DEBUGlevel_ >= 2 )

  detail::row_dist_inner_replicate( heightLocal, width, width,
                                    X.Data(), heightLocal,
                                    X.Data(), heightLocal, XTX.Data(), 
                                    mpi_comm );

  statusOFS << "After the PPCG, XTX = " << XTX << std::endl;

#endif

  // Save the eigenvalues and eigenvectors back to the eigensolver data
  // structure

  eigVal_ = DblNumVec( width, true, eigValS.Data() );
  resVal_ = resNorm;

  bdist.redistribute_row_to_col( X, Xcol );

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


} // namespace dgdft
