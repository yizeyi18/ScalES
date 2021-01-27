//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Jack Poulson and Lin Lin

/// @file lapack.cpp
/// @brief Thin interface to LAPACK
/// @date 2012-09-12
#include "lapack.hpp"

namespace scales {
namespace lapack {

extern "C" {

  // Safely compute a Givens rotation
  void LAPACK(slartg)
    ( const float* phi, const float* gamma,
      float* c, float* s, float* rho );
  void LAPACK(dlartg)
    ( const double* phi, const double* gamma,
      double* c, double* s, double* rho );
  void LAPACK(clartg)
    ( const scomplex* phi, const scomplex* gamma,
      float* c, scomplex* s, scomplex* rho );
  void LAPACK(zlartg)
    ( const dcomplex* phi, const dcomplex* gamma,
      double* c, dcomplex* s, dcomplex* rho );

  // Cholesky factorization
  void LAPACK(spotrf)
    ( const char* uplo, const Int* n, const float* A, const Int* lda,
      Int* info );
  void LAPACK(dpotrf)
    ( const char* uplo, const Int* n, const double* A, const Int* lda,
      Int* info );
  void LAPACK(cpotrf)
    ( const char* uplo, const Int* n, const scomplex* A,
      const Int* lda, Int* info );
  void LAPACK(zpotrf)
    ( const char* uplo, const Int* n, const dcomplex* A,
      const Int* lda, Int* info );

  // LU factorization (with partial pivoting)
  void LAPACK(sgetrf)
    ( const Int* m, const Int* n,
      float* A, const Int* lda, Int* p, Int* info );
  void LAPACK(dgetrf)
    ( const Int* m, const Int* n,
      double* A, const Int* lda, Int* p, Int* info );
  void LAPACK(cgetrf)
    ( const Int* m, const Int* n,
      scomplex* A, const Int* lda, Int* p, Int* info );
  void LAPACK(zgetrf)
    ( const Int* m, const Int* n,
      dcomplex* A, const Int* lda, Int* p, Int* info );

  // For reducing well-conditioned Hermitian generalized EVP to Hermitian 
  // standard form
  void LAPACK(ssygst)
    ( const Int* itype, const char* uplo, const Int* n,
      float* A, Int* lda, const float* B, Int* ldb, Int* info );
  void LAPACK(dsygst)
    ( const Int* itype, const char* uplo, const Int* n,
      double* A, Int* lda, const double* B, Int* ldb, Int* info );
  void LAPACK(chegst)
    ( const Int* itype, const char* uplo, const Int* n,
      scomplex* A, const Int* lda,
      const scomplex* B, const Int* ldb, Int* info );
  void LAPACK(zhegst)
    ( const Int* itype, const char* uplo, const Int* n,
      dcomplex* A, const Int* lda,
      const dcomplex* B, const Int* ldb, Int* info );

  // For solving the standard eigenvalue problem using the divide and
  // conquer algorithm
  // TODO all versions
  void LAPACK(dsyevd)
    ( const char *jobz, const char *uplo, const Int *n, 
      double *A, const Int *lda, double *W, double *work, 
      const int *lwork, Int *iwork, const int *liwork, int *info );

  void LAPACK(zheevd)
    ( const char *jobz, const char *uplo, const Int *n, 
      dcomplex *A, const Int *lda, double *W, dcomplex* work, 
      const int *lwork, double* rwork, int * lrwork, 
      Int *iwork, const int *liwork, int *info );

  // For solving the generalized eigenvalue problem using the divide and
  // conquer algorithm  --- added by Eugene
  void LAPACK(dsygvd)
    ( const Int *itype, const char *jobz, const char *uplo, const Int *n, 
      double *A, const Int *lda, double* B, const Int *ldb, double *W, double *work, 
      const int *lwork, Int *iwork, const int *liwork, int *info );


  // For solving the generalized eigenvalue problem using the divide and
  // conquer algorithm  --- added by Eugene
  void LAPACK(zhegvd)
    ( const Int *itype, const char *jobz, const char *uplo, const Int *n, 
      dcomplex*A, const Int *lda, dcomplex* B, const Int *ldb, double *W, dcomplex *work, 
      const int *lwork, double *rwork, int *lrwork, Int *iwork, const int *liwork, int *info );

  // Triangular inversion
  void LAPACK(strtri)
    ( const char* uplo, const char* diag,
      const Int* n, const float* A, const Int* lda, Int* info );
  void LAPACK(dtrtri)
    ( const char* uplo, const char* diag,
      const Int* n, const double* A, const Int* lda, Int* info );
  void LAPACK(ctrtri)
    ( const char* uplo, const char* diag,
      const Int* n, const scomplex* A, const Int* lda, Int* info );
  void LAPACK(ztrtri)
    ( const char* uplo, const char* diag,
      const Int* n, const dcomplex* A, const Int* lda, Int* info );

  // Bidiagonal QR
  void LAPACK(sbdsqr)
    ( const char* uplo, const Int* n, const Int* numColsVTrans, const Int* numRowsU,
      const Int* numColsC, float* d, float* e, float* VTrans, const Int* ldVTrans,
      float* U, const Int* ldU, float* C, const Int* ldC, float* work, Int* info );
  void LAPACK(dbdsqr)
    ( const char* uplo, const Int* n, const Int* numColsVTrans, const Int* numRowsU,
      const Int* numColsC, double* d, double* e,
      double* VTrans, const Int* ldVTrans, double* U, const Int* ldU,
      double* C, const Int* ldC, double* work, Int* info );
  void LAPACK(cbdsqr)
    ( const char* uplo, const Int* n, const Int* numColsVAdj, const Int* numRowsU,
      const Int* numColsC, float* d, float* e,
      scomplex* VAdj, const Int* ldVAdj, scomplex* U, const Int* ldU,
      scomplex* C, const Int* ldC, float* work, Int* info );
  void LAPACK(zbdsqr)
    ( const char* uplo, const Int* n, const Int* numColsVAdj, const Int* numRowsU,
      const Int* numColsC, double* d, double* e,
      dcomplex* VAdj, const Int* ldVAdj, dcomplex* U, const Int* ldU,
      dcomplex* C, const Int* ldC, double* work, Int* info );

  // Divide and Conquer SVD
  void LAPACK(sgesdd)
    ( const char* jobz, const Int* m, const Int* n, float* A, const Int* lda,
      float* s, float* U, const Int* ldu, float* VTrans, const Int* ldvt,
      float* work, const Int* lwork, Int* iwork, Int* info );
  void LAPACK(dgesdd)
    ( const char* jobz, const Int* m, const Int* n, double* A, const Int* lda,
      double* s, double* U, const Int* ldu, double* VTrans, const Int* ldvt,
      double* work, const Int* lwork, Int* iwork, Int* info );
  void LAPACK(cgesdd)
    ( const char* jobz, const Int* m, const Int* n,
      scomplex* A, const Int* lda, float* s,
      scomplex* U, const Int* ldu, scomplex* VTrans, const Int* ldvt,
      scomplex* work, const Int* lwork, float* rwork,
      Int* iwork, Int* info );
  void LAPACK(zgesdd)
    ( const char* jobz, const Int* m, const Int* n,
      dcomplex* A, const Int* lda, double* s,
      dcomplex* U, const Int* ldu, dcomplex* VAdj, const Int* ldva,
      dcomplex* work, const Int* lwork, double* rwork,
      Int* iwork, Int* info );

  // QR-algorithm SVD
  void LAPACK(sgesvd)
    ( const char* jobu, const char* jobvt, const Int* m, const Int* n,
      float* A, const Int* lda,
      float* s, float* U, const Int* ldu, float* VTrans, const Int* ldvt,
      float* work, const Int* lwork, Int* info );
  void LAPACK(dgesvd)
    ( const char* jobu, const char* jobvt, const Int* m, const Int* n,
      double* A, const Int* lda,
      double* s, double* U, const Int* ldu, double* VTrans, const Int* ldvt,
      double* work, const Int* lwork, Int* info );
  void LAPACK(cgesvd)
    ( const char* jobu, const char* jobva, const Int* m, const Int* n,
      scomplex* A, const Int* lda, float* s,
      scomplex* U, const Int* ldu, scomplex* VTrans, const Int* ldvt,
      scomplex* work, const Int* lwork, float* rwork, Int* info );
  void LAPACK(zgesvd)
    ( const char* jobu, const char* jobva, const Int* m, const Int* n,
      dcomplex* A, const Int* lda, double* s,
      dcomplex* U, const Int* ldu, dcomplex* VAdj, const Int* ldva,
      dcomplex* work, const Int* lwork, double* rwork, Int* info );


  // SVD based least square
  void LAPACK(sgelss)
    ( const Int *m, const Int *n, const Int *nrhs, float *A, const Int *lda,
      float *B, const Int *ldb, float *S, const float *rcond, Int *rank,
      float *work, const Int *lwork, Int *info );    
  void LAPACK(dgelss)
    ( const Int *m, const Int *n, const Int *nrhs, double *A, const Int *lda,
      double *B, const Int *ldb, double *S, const double *rcond, Int *rank,
      double *work, const Int *lwork, Int *info );    
  void LAPACK(cgelss)
    ( const Int *m, const Int *n, const Int *nrhs, scomplex *A, const Int *lda,
      scomplex *B, const Int *ldb, float *S, const float *rcond, Int *rank,
      scomplex *work, const Int *lwork, float *rwork, Int *info );    
  void LAPACK(zgelss)
    ( const Int *m, const Int *n, const Int *nrhs, dcomplex *A, const Int *lda,
      dcomplex *B, const Int *ldb, double *S, const double *rcond, Int *rank,
      dcomplex *work, const Int *lwork, double *rwork, Int *info );    

  // Copy

  void LAPACK(dlacpy)
    ( const char* uplo, const Int* m, const Int* n, 
      const double* A, const Int *lda, 
      double* B, const Int *ldb );
  void LAPACK(zlacpy)
    ( const char* uplo, const Int* m, const Int* n, 
      const dcomplex* A, const Int *lda, 
      dcomplex* B, const Int *ldb );

  // Triangular solve : Trsm
  //void LAPACK(ztrsm)
  //    ( const char* side, const char* uplo, const char* transa, const char * diag,
  //        const Int* m, const Int* n, const dcomplex* alpha, const dcomplex* A, const Int* lda,
  //        dcomplex* B, const Int* ldb );

  // Inverting a factorized matrix: Getri
  void LAPACK(dgetri)
    ( const Int* n, double* A, const Int* lda, const Int* ipiv, double* work,
      const Int* lwork, Int* info );
  void LAPACK(zgetri)
    ( const Int* n, dcomplex* A, const Int* lda, const Int* ipiv, dcomplex* work,
      const Int* lwork, Int* info );

  // Compute the factorization of a real symmetric matrix: Sytrf
  void LAPACK(dsytrf)
    ( const char* uplo, const Int* N, double* A, const Int* lda,
      Int* ipiv, double* work, const Int* lwork, Int* info );



  // Estimating the reciprocal of the condition number
  // TODO Add Pocon and Lansy

  // QR decomposition with column pivoting
  void LAPACK(dgeqp3) (const Int* m, const Int* n, double* A, 
      const Int* lda, Int* jpvt, double* tau, double* work, 
      const Int* lwork, Int* info);

  // QR factorization
  void LAPACK(dgeqrf) (const Int* m, const Int* n, double* A, 
      const Int* lda, double* tau, double* work, const Int* lwork,
      Int* info);

  // Construct the Q factor from QR factorization
  void LAPACK(dorgqr) (const Int* m, const Int* n, const Int* k, 
      double* A, const Int* lda, double* tau, double* work, 
      const Int* lwork, Int* info);

} // extern "C"


// *********************************************************************
// Cholesky factorization
// *********************************************************************

void Potrf( char uplo, Int n, const float* A, Int lda )
{
  Int info;
  LAPACK(spotrf)( &uplo, &n, A, &lda, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "spotrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
    ErrorHandling("Matrix is not HPD.");
}

void Potrf( char uplo, Int n, const double* A, Int lda )
{
  Int info;
  LAPACK(dpotrf)( &uplo, &n, A, &lda, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "dpotrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 ){
    std::ostringstream msg;
    msg << "dpotrf returned with info = " << info << std::endl;
    msg << "A(info,info) = " << A[info-1+(info-1)*lda] << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
}

void Potrf( char uplo, Int n, const scomplex* A, Int lda )
{
  Int info;
  LAPACK(cpotrf)( &uplo, &n, A, &lda, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "cpotrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
    ErrorHandling("Matrix is not HPD.");
}

void Potrf( char uplo, Int n, const dcomplex* A, Int lda )
{
  Int info;
  LAPACK(zpotrf)( &uplo, &n, A, &lda, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "zpotrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
    ErrorHandling("Matrix is not HPD.");
}

// *********************************************************************
// LU factorization (with partial pivoting)
// *********************************************************************

void Getrf( Int m, Int n, float* A, Int lda, Int* p )
{
  Int info;
  LAPACK(sgetrf)( &m, &n, A, &lda, p, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "sgetrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
    ErrorHandling("Matrix is singular.");
}

void Getrf( Int m, Int n, double* A, Int lda, Int* p )
{
  Int info;
  LAPACK(dgetrf)( &m, &n, A, &lda, p, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "dgetrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
    ErrorHandling("Matrix is singular.");
}

void Getrf( Int m, Int n, scomplex* A, Int lda, Int* p )
{
  Int info;
  LAPACK(cgetrf)( &m, &n, A, &lda, p, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "cgetrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
    ErrorHandling("Matrix is singular.");
}

void Getrf( Int m, Int n, dcomplex* A, Int lda, Int* p )
{
  Int info;
  LAPACK(zgetrf)( &m, &n, A, &lda, p, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "zgetrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
    ErrorHandling("Matrix is singular.");
}

//
// Reduced a well-conditioned Hermitian generalized definite EVP to 
// standard form
//

void Hegst
( Int itype, char uplo, Int n,
  float* A, Int lda, const float* B, Int ldb )
{
  Int info;
  LAPACK(ssygst)( &itype, &uplo, &n, A, &lda, B, &ldb, &info );
  if( info != 0 )
  {
    std::ostringstream msg;
    msg << "ssygst returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
}

void Hegst
( Int itype, char uplo, Int n,
  double* A, Int lda, const double* B, Int ldb )
{
  Int info;
  LAPACK(dsygst)( &itype, &uplo, &n, A, &lda, B, &ldb, &info );
  if( info != 0 )
  {
    std::ostringstream msg;
    msg << "dsygst returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
}

void Hegst
( Int itype, char uplo, Int n,
  scomplex* A, Int lda, const scomplex* B, Int ldb )
{
  Int info;
  LAPACK(chegst)( &itype, &uplo, &n, A, &lda, B, &ldb, &info );
  if( info != 0 )
  {
    std::ostringstream msg;
    msg << "chegst returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
}

void Hegst
( Int itype, char uplo, Int n,
  dcomplex* A, Int lda, const dcomplex* B, Int ldb )
{
  Int info;
  LAPACK(zhegst)( &itype, &uplo, &n, A, &lda, B, &ldb, &info );
  if( info != 0 )
  {
    std::ostringstream msg;
    msg << "zhegst returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
}

// *********************************************************************
// For solving the standard eigenvalue problem using the divide and
// conquer algorithm
// *********************************************************************

void Syevd
( char jobz, char uplo, Int n, double* A, Int lda, double* eigs ){
  Int lwork = -1, info;
  Int liwork = -1;
  std::vector<double> work(1);
  std::vector<int>    iwork(1);

  LAPACK(dsyevd)( &jobz, &uplo, &n, A, &lda, eigs, &work[0],
      &lwork, &iwork[0], &liwork, &info );
  lwork = (Int)work[0];
  work.resize(lwork);
  liwork = iwork[0];
  iwork.resize(liwork);

  LAPACK(dsyevd)( &jobz, &uplo, &n, A, &lda, eigs, &work[0],
      &lwork, &iwork[0], &liwork, &info );

  if( info != 0 )
  {
    std::ostringstream msg;
    msg << "syevd returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
}


// *********************************************************************
// For solving the generalized eigenvalue problem using the divide and
// conquer algorithm  --- added by Eugene
// *********************************************************************

void Sygvd
( Int itype, char jobz, char uplo, Int n, double* A, Int lda, double* B, Int ldb, double* eigs ){
  Int lwork = -1, info;
  Int liwork = -1;
  std::vector<double> work(1);
  std::vector<int>    iwork(1);

  LAPACK(dsygvd)( &itype, &jobz, &uplo, &n, A, &lda, B, &ldb, eigs, &work[0],
      &lwork, &iwork[0], &liwork, &info );
  lwork = (Int)work[0];
  work.resize(lwork);
  liwork = iwork[0];
  iwork.resize(liwork);

  LAPACK(dsygvd)( &itype, &jobz, &uplo, &n, A, &lda, B, &ldb, eigs, &work[0],
      &lwork, &iwork[0], &liwork, &info );

  if( info != 0 )
  {
    std::ostringstream msg;
    msg << "sygvd returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
}

void Syevd
( char jobz, char uplo, Int n, dcomplex* A, Int lda, double* eigs ){
  Int lwork = -1, info;
  Int liwork = -1;
  Int lrwork = -1;
  std::vector<dcomplex> work(1);
  std::vector<int>    iwork(1);
  std::vector<double> rwork(1);

  LAPACK(zheevd)( &jobz, &uplo, &n, A, &lda, eigs, &work[0],
      &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info );
  lwork = (Int)work[0].real();
  work.resize(lwork);
  liwork = iwork[0];
  iwork.resize(liwork);
  lrwork = (Int) rwork[0];
  rwork.resize(lrwork);

  LAPACK(zheevd)( &jobz, &uplo, &n, A, &lda, eigs, &work[0],
      &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info );

  if( info != 0 )
  {
    std::ostringstream msg;
    msg << "syevd returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
}

void Sygvd
( Int itype, char jobz, char uplo, Int n, dcomplex* A, Int lda, dcomplex* B, Int ldb, double* eigs ){
  Int lwork = -1, info;
  Int liwork = -1;
  Int lrwork = -1;
  std::vector<dcomplex> work(1);
  std::vector<int>    iwork(1);
  std::vector<double> rwork(1);

  LAPACK(zhegvd)( &itype, &jobz, &uplo, &n, A, &lda, B, &ldb, eigs, &work[0],
      &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info );
  lwork = (Int)work[0].real();
  work.resize(lwork);
  liwork = iwork[0];
  iwork.resize(liwork);
  lrwork = (Int) rwork[0];
  rwork.resize(lrwork);

  LAPACK(zhegvd)( &itype, &jobz, &uplo, &n, A, &lda, B, &ldb, eigs, &work[0],
      &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info );

  if( info != 0 )
  {
    std::ostringstream msg;
    msg << "zhegvd returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
}



// *********************************************************************
// For computing the inverse of a triangular matrix
// *********************************************************************

void Trtri( char uplo, char diag, Int n, const float* A, Int lda )
{
  Int info;
  LAPACK(strtri)( &uplo, &diag, &n, A, &lda, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "strtri returned with info = " << info;
    throw std::logic_error( msg.str().c_str() );
  }
  else if( info > 0 )
    throw std::runtime_error("Matrix is singular.");
}

void Trtri( char uplo, char diag, Int n, const double* A, Int lda )
{
  Int info;
  LAPACK(dtrtri)( &uplo, &diag, &n, A, &lda, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "dtrtri returned with info = " << info;
    throw std::logic_error( msg.str().c_str() );
  }
  else if( info > 0 )
    throw std::runtime_error("Matrix is singular.");
}

void Trtri
( char uplo, char diag, Int n, const scomplex* A, Int lda )
{
  Int info;
  LAPACK(ctrtri)( &uplo, &diag, &n, A, &lda, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "ctrtri returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
    ErrorHandling("Matrix is singular.");
}

void Trtri
( char uplo, char diag, Int n, const dcomplex* A, Int lda )
{
  Int info;
  LAPACK(ztrtri)( &uplo, &diag, &n, A, &lda, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "ztrtri returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
    ErrorHandling("Matrix is singular.");
}

//
// Bidiagonal QR algorithm for SVD
//

void BidiagQRAlg
( char uplo, Int n, Int numColsVTrans, Int numRowsU,
  float* d, float* e, float* VTrans, Int ldVTrans, float* U, Int ldU )
{
  if( n==0 )
  {
    return;
  }

  Int info;
  float* C=0;
  const Int numColsC=0, ldC=1;
  std::vector<float> work( 4*n );
  LAPACK(sbdsqr)
    ( &uplo, &n, &numColsVTrans, &numRowsU, &numColsC, d, e, VTrans, &ldVTrans,
      U, &ldU, C, &ldC, &work[0], &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "sbdsqr had " << info << " elements of e not converge";
    ErrorHandling( msg.str().c_str() );
  }
}

void BidiagQRAlg
( char uplo, Int n, Int numColsVTrans, Int numRowsU, 
  double* d, double* e, double* VTrans, Int ldVTrans, double* U, Int ldU )
{
  if( n==0 )
  {
    return;
  }

  Int info;
  double* C=0;
  const Int numColsC=0, ldC=1;
  std::vector<double> work( 4*n );
  LAPACK(dbdsqr)
    ( &uplo, &n, &numColsVTrans, &numRowsU, &numColsC, d, e, VTrans, &ldVTrans,
      U, &ldU, C, &ldC, &work[0], &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "dbdsqr had " << info << " elements of e not converge";
    ErrorHandling( msg.str().c_str() );
  }
}

void BidiagQRAlg
( char uplo, Int n, Int numColsVAdj, Int numRowsU, 
  float* d, float* e, scomplex* VAdj, Int ldVAdj, scomplex* U, Int ldU )
{
  if( n==0 )
  {
    return;
  }

  Int info;
  scomplex* C=0;
  const Int numColsC=0, ldC=1;
  std::vector<float> work( 4*n );
  LAPACK(cbdsqr)
    ( &uplo, &n, &numColsVAdj, &numRowsU, &numColsC, d, e, VAdj, &ldVAdj,
      U, &ldU, C, &ldC, &work[0], &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "cbdsqr had " << info << " elements of e not converge";
    ErrorHandling( msg.str().c_str() );
  }
}

void BidiagQRAlg
( char uplo, Int n, Int numColsVAdj, Int numRowsU, 
  double* d, double* e, dcomplex* VAdj, Int ldVAdj, dcomplex* U, Int ldU )
{
  if( n==0 )
  {
    return;
  }

  Int info;
  dcomplex* C=0;
  const Int numColsC=0, ldC=1;
  std::vector<double> work( 4*n );
  LAPACK(zbdsqr)
    ( &uplo, &n, &numColsVAdj, &numRowsU, &numColsC, d, e, VAdj, &ldVAdj,
      U, &ldU, C, &ldC, &work[0], &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "zbdsqr had " << info << " elements of e not converge";
    ErrorHandling( msg.str().c_str() );
  }
}

// *********************************************************************
// Compute the SVD of a general matrix using a divide and conquer algorithm
// *********************************************************************

void DivideAndConquerSVD
( Int m, Int n, float* A, Int lda, 
  float* s, float* U, Int ldu, float* VTrans, Int ldvt )
{
  if( m==0 || n==0 )
  {
    return;
  }

  const char jobz='S';
  Int lwork=-1, info;
  float dummyWork;
  const Int k = std::min(m,n);
  std::vector<Int> iwork(8*k);

  LAPACK(sgesdd)
    ( &jobz, &m, &n, A, &lda, s, U, &ldu, VTrans, &ldvt, &dummyWork, &lwork,
      &iwork[0], &info );

  lwork = dummyWork;
  std::vector<float> work(lwork);
  LAPACK(sgesdd)
    ( &jobz, &m, &n, A, &lda, s, U, &ldu, VTrans, &ldvt, &work[0], &lwork,
      &iwork[0], &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("sgesdd's updating process failed");
  }
}

void DivideAndConquerSVD
( Int m, Int n, double* A, Int lda, 
  double* s, double* U, Int ldu, double* VTrans, Int ldvt )
{
  if( m==0 || n==0 )
  {
    return;
  }

  const char jobz='S';
  Int lwork=-1, info;
  double dummyWork;
  const Int k = std::min(m,n);
  std::vector<Int> iwork(8*k);

  LAPACK(dgesdd)
    ( &jobz, &m, &n, A, &lda, s, U, &ldu, VTrans, &ldvt, &dummyWork, &lwork,
      &iwork[0], &info );

  lwork = dummyWork;
  std::vector<double> work(lwork);
  LAPACK(dgesdd)
    ( &jobz, &m, &n, A, &lda, s, U, &ldu, VTrans, &ldvt, &work[0], &lwork,
      &iwork[0], &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("dgesdd's updating process failed");
  }
}

void DivideAndConquerSVD
( Int m, Int n, scomplex* A, Int lda, 
  float* s, scomplex* U, Int ldu, scomplex* VAdj, Int ldva )
{
  if( m==0 || n==0 )
  {
    return;
  }

  const char jobz='S';
  Int lwork=-1, info;
  const Int k = std::min(m,n);
  const Int K = std::max(m,n);
  const Int lrwork = k*std::max(5*k+7,2*K+2*k+1);
  std::vector<float> rwork(lrwork);
  std::vector<Int> iwork(8*k);

  scomplex dummyWork;
  LAPACK(cgesdd)
    ( &jobz, &m, &n, A, &lda, s, U, &ldu, VAdj, &ldva, &dummyWork, &lwork,
      &rwork[0], &iwork[0], &info );

  lwork = dummyWork.real();
  std::vector<scomplex> work(lwork);
  LAPACK(cgesdd)
    ( &jobz, &m, &n, A, &lda, s, U, &ldu, VAdj, &ldva, &work[0], &lwork,
      &rwork[0], &iwork[0], &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("cgesdd's updating process failed");
  }
}

void DivideAndConquerSVD
( Int m, Int n, dcomplex* A, Int lda, 
  double* s, dcomplex* U, Int ldu, dcomplex* VAdj, Int ldva )
{
  if( m==0 || n==0 )
  {
    return;
  }

  const char jobz='S';
  Int lwork=-1, info;
  dcomplex dummyWork;
  const Int k = std::min(m,n);
  const Int K = std::max(m,n);
  const Int lrwork = k*std::max(5*k+7,2*K+2*k+1);
  std::vector<double> rwork(lrwork);
  std::vector<Int> iwork(8*k);

  LAPACK(zgesdd)
    ( &jobz, &m, &n, A, &lda, s, U, &ldu, VAdj, &ldva, &dummyWork, &lwork,
      &rwork[0], &iwork[0], &info );

  lwork = dummyWork.real();
  std::vector<dcomplex> work(lwork);
  LAPACK(zgesdd)
    ( &jobz, &m, &n, A, &lda, s, U, &ldu, VAdj, &ldva, &work[0], &lwork,
      &rwork[0], &iwork[0], &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("zgesdd's updating process failed");
  }
}

//
// QR-algorithm SVD
//

void QRSVD
( Int m, Int n, float* A, Int lda, 
  float* s, float* U, Int ldu, float* VTrans, Int ldvt )
{
  if( m==0 || n==0 )
  {
    return;
  }

  const char jobu='S', jobvt='S';
  Int lwork=-1, info;
  float dummyWork;

  LAPACK(sgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, s, U, &ldu, VTrans, &ldvt, 
      &dummyWork, &lwork, &info );

  lwork = dummyWork;
  std::vector<float> work(lwork);
  LAPACK(sgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, s, U, &ldu, VTrans, &ldvt, 
      &work[0], &lwork, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("sgesvd's updating process failed");
  }
}

void QRSVD
( Int m, Int n, double* A, Int lda, 
  double* s, double* U, Int ldu, double* VTrans, Int ldvt )
{
  if( m==0 || n==0 )
  {
    return;
  }

  const char jobu='S', jobvt='S';
  Int lwork=-1, info;
  double dummyWork;

  LAPACK(dgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, s, U, &ldu, VTrans, &ldvt, 
      &dummyWork, &lwork, &info );

  lwork = dummyWork;
  std::vector<double> work(lwork);
  LAPACK(dgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, s, U, &ldu, VTrans, &ldvt, 
      &work[0], &lwork, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("dgesvd's updating process failed");
  }
}

void QRSVD
( Int m, Int n, scomplex* A, Int lda, 
  float* s, scomplex* U, Int ldu, scomplex* VAdj, Int ldva )
{
  if( m==0 || n==0 )
  {
    return;
  }

  const char jobu='S', jobva='S';
  Int lwork=-1, info;
  const Int k = std::min(m,n);
  std::vector<float> rwork(5*k);

  scomplex dummyWork;
  LAPACK(cgesvd)
    ( &jobu, &jobva, &m, &n, A, &lda, s, U, &ldu, VAdj, &ldva, 
      &dummyWork, &lwork, &rwork[0], &info );

  lwork = dummyWork.real();
  std::vector<scomplex> work(lwork);
  LAPACK(cgesvd)
    ( &jobu, &jobva, &m, &n, A, &lda, s, U, &ldu, VAdj, &ldva, 
      &work[0], &lwork, &rwork[0], &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("cgesvd's updating process failed");
  }
}

void QRSVD
( Int m, Int n, dcomplex* A, Int lda, 
  double* s, dcomplex* U, Int ldu, dcomplex* VAdj, Int ldva )
{
  if( m==0 || n==0 )
  {
    return;
  }

  const char jobu='S', jobva='S';
  Int lwork=-1, info;
  dcomplex dummyWork;
  const Int k = std::min(m,n);
  std::vector<double> rwork(5*k);

  LAPACK(zgesvd)
    ( &jobu, &jobva, &m, &n, A, &lda, s, U, &ldu, VAdj, &ldva, 
      &dummyWork, &lwork, &rwork[0], &info );

  lwork = dummyWork.real();
  std::vector<dcomplex> work(lwork);
  LAPACK(zgesvd)
    ( &jobu, &jobva, &m, &n, A, &lda, s, U, &ldu, VAdj, &ldva, 
      &work[0], &lwork, &rwork[0], &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("zgesvd's updating process failed");
  }
}

//
// Compute singular values with QR algorithm
//

void SingularValues( Int m, Int n, float* A, Int lda, float* s )
{
  if( m==0 || n==0 )
  {
    return;
  }

  const char jobu='N', jobvt='N';
  Int fakeLDim=1, lwork=-1, info;
  float dummyWork;

  LAPACK(sgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, s, 0, &fakeLDim, 0, &fakeLDim, 
      &dummyWork, &lwork, &info );

  lwork = dummyWork;
  std::vector<float> work(lwork);
  LAPACK(sgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, s, 0, &fakeLDim, 0, &fakeLDim, 
      &work[0], &lwork, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("sgesvd's updating process failed");
  }
}

void SingularValues( Int m, Int n, double* A, Int lda, double* s )
{
  if( m==0 || n==0 )
  {
    return;
  }

  const char jobu='N', jobvt='N';
  Int fakeLDim=1, lwork=-1, info;
  double dummyWork;

  LAPACK(dgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, s, 0, &fakeLDim, 0, &fakeLDim, 
      &dummyWork, &lwork, &info );

  lwork = dummyWork;
  std::vector<double> work(lwork);
  LAPACK(dgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, s, 0, &fakeLDim, 0, &fakeLDim, 
      &work[0], &lwork, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("dgesvd's updating process failed");
  }
}

void SingularValues( Int m, Int n, scomplex* A, Int lda, float* s )
{
  if( m==0 || n==0 )
  {
    return;
  }

  const char jobu='N', jobva='N';
  Int fakeLDim=1, lwork=-1, info;
  scomplex dummyWork;
  const Int k = std::min(m,n);
  std::vector<float> rwork(5*k);

  LAPACK(cgesvd)
    ( &jobu, &jobva, &m, &n, A, &lda, s, 0, &fakeLDim, 0, &fakeLDim, 
      &dummyWork, &lwork, &rwork[0], &info );

  lwork = dummyWork.real();
  std::vector<scomplex> work(lwork);
  LAPACK(cgesvd)
    ( &jobu, &jobva, &m, &n, A, &lda, s, 0, &fakeLDim, 0, &fakeLDim, 
      &work[0], &lwork, &rwork[0], &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("cgesvd's updating process failed");
  }
}

void SingularValues( Int m, Int n, dcomplex* A, Int lda, double* s )
{
  if( m==0 || n==0 )
  {
    return;
  }

  const char jobu='N', jobva='N';
  Int fakeLDim=1, lwork=-1, info;
  dcomplex dummyWork;
  const Int k = std::min(m,n);
  std::vector<double> rwork(5*k);

  LAPACK(zgesvd)
    ( &jobu, &jobva, &m, &n, A, &lda, s, 0, &fakeLDim, 0, &fakeLDim, 
      &dummyWork, &lwork, &rwork[0], &info );

  lwork = dummyWork.real();
  std::vector<dcomplex> work(lwork);
  LAPACK(zgesvd)
    ( &jobu, &jobva, &m, &n, A, &lda, s, 0, &fakeLDim, 0, &fakeLDim, 
      &work[0], &lwork, &rwork[0], &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("zgesvd's updating process failed");
  }
}


// *********************************************************************
// Compute the linear least square problem using SVD
// *********************************************************************
void SVDLeastSquare( Int m, Int n, Int nrhs, float * A, Int lda,
    float * B, Int ldb, float * S, float rcond,
    Int* rank )
{
  if( m==0 || n==0 )
  {
    return;
  }

  Int  lwork=-1, info;
  float dummyWork;

  LAPACK(sgelss)
    ( &m, &n, &nrhs, A, &lda, B, &ldb, S,
      &rcond, rank, &dummyWork, &lwork, &info );

  lwork = dummyWork;

  std::vector<float> work(lwork);
  LAPACK(sgelss)
    ( &m, &n, &nrhs, A, &lda, B, &ldb, S,
      &rcond, rank, &work[0], &lwork, &info );

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("sgelss's svd failed to converge.");
  }
}

void SVDLeastSquare( Int m, Int n, Int nrhs, double * A, Int lda,
    double * B, Int ldb, double * S, double rcond,
    Int* rank )
{
  if( m==0 || n==0 )
  {
    return;
  }

  Int  lwork=-1, info;
  double dummyWork;

  LAPACK(dgelss)
    ( &m, &n, &nrhs, A, &lda, B, &ldb, S,
      &rcond, rank, &dummyWork, &lwork, &info );

  lwork = (Int) (dummyWork * 2);

  std::vector<double> work(lwork);
  LAPACK(dgelss)
    ( &m, &n, &nrhs, A, &lda, B, &ldb, S,
      &rcond, rank, &work[0], &lwork, &info );

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("dgelss's svd failed to converge.");
  }
}

void SVDLeastSquare( Int m, Int n, Int nrhs, scomplex * A, Int lda,
    scomplex * B, Int ldb, float * S, float rcond,
    Int* rank )
{
  if( m==0 || n==0 )
  {
    return;
  }

  Int  lwork=-1, info;
  Int  lrwork = 5*m;
  std::vector<float> rwork(lrwork);
  scomplex dummyWork;

  LAPACK(cgelss)
    ( &m, &n, &nrhs, A, &lda, B, &ldb, S,
      &rcond, rank, &dummyWork, &lwork, &rwork[0], &info );

  lwork = dummyWork.real();

  std::vector<scomplex> work(lwork);
  LAPACK(cgelss)
    ( &m, &n, &nrhs, A, &lda, B, &ldb, S,
      &rcond, rank, &work[0], &lwork, &rwork[0], &info );

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("cgelss's svd failed to converge.");
  }
}

void SVDLeastSquare( Int m, Int n, Int nrhs, dcomplex * A, Int lda,
    dcomplex * B, Int ldb, double * S, double rcond,
    Int* rank )
{
  if( m==0 || n==0 )
  {
    return;
  }

  Int  lwork=-1, info;
  Int  lrwork = 5*m;
  std::vector<double> rwork(lrwork);
  dcomplex dummyWork;

  LAPACK(zgelss)
    ( &m, &n, &nrhs, A, &lda, B, &ldb, S,
      &rcond, rank, &dummyWork, &lwork, &rwork[0], &info );

  lwork = dummyWork.real();

  std::vector<dcomplex> work(lwork);
  LAPACK(zgelss)
    ( &m, &n, &nrhs, A, &lda, B, &ldb, S,
      &rcond, rank, &work[0], &lwork, &rwork[0], &info );

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("zgelss's svd failed to converge.");
  }
}

// *********************************************************************
// Copy
// *********************************************************************

void Lacpy( char uplo, Int m, Int n, const double* A, Int lda,
    double* B, Int ldb    ){
  LAPACK(dlacpy)( &uplo, &m, &n, A, &lda, B, &ldb );
}

void Lacpy( char uplo, Int m, Int n, const dcomplex* A, Int lda,
    dcomplex* B, Int ldb    ){
  LAPACK(zlacpy)( &uplo, &m, &n, A, &lda, B, &ldb );
}

// *********************************************************************
// Inverting a factorized matrix: Getri
// *********************************************************************
void
Getri ( Int n, double* A, Int lda, const Int* ipiv )
{
  Int lwork = -1, info;
  double dummyWork;

  LAPACK(dgetri)( &n, A, &lda, ipiv, &dummyWork, &lwork, &info );

  lwork = dummyWork;
  std::vector<double> work(lwork);

  LAPACK(dgetri)( &n, A, &lda, ipiv, &work[0], &lwork, &info );

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "U(" << info << ", " << info << ") = 0. The matrix is singular and cannot be inverted.";
    ErrorHandling( msg.str().c_str() );
  }


  return ;
}        // -----  end of function Getri  ----- 


void
Getri ( Int n, dcomplex* A, Int lda, const Int* ipiv )
{
  Int lwork = -1, info;
  dcomplex dummyWork;

  LAPACK(zgetri)( &n, A, &lda, ipiv, &dummyWork, &lwork, &info );

  lwork = dummyWork.real();
  std::vector<dcomplex> work(lwork);

  LAPACK(zgetri)( &n, A, &lda, ipiv, &work[0], &lwork, &info );

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "U(" << info << ", " << info << ") = 0. The matrix is singular and cannot be inverted.";
    ErrorHandling( msg.str().c_str() );
  }


  return ;
}        // -----  end of function Getri  ----- 



// Compute the factorization of a real symmetric matrix: Sytrf

void
Sytrf ( char uplo, Int n, double* A, Int lda, Int* ipiv ){
  Int lwork = -1, info;
  double dummyWork;

  LAPACK(dsytrf) ( &uplo, &n, A, &lda,
      ipiv, &dummyWork, &lwork, &info );

  lwork = dummyWork;
  std::vector<double> work(lwork);

  LAPACK(dsytrf) ( &uplo, &n, A, &lda,
      ipiv, &work[0], &lwork, &info );

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "U(" << info << ", " << info << ") = 0. "
      << "The matrix is singular and factorization cannot proceed.";
    ErrorHandling( msg.str().c_str() );
  }


  return ;
}        // -----  end of function Sytrf  ----- 

// *********************************************************************
// Orthogonalization (MATLAB's orth, using SVD)
// *********************************************************************

void Orth( Int m, Int n, double* A, Int lda ){
  if( m==0 || n==0 )
    return;

  // Overwrite mode
  const char jobu='O', jobvt='N';
  std::vector<double> s(n);
  Int lwork=-1, info;
  double dummyWork;
  double dummyReal;
  Int dummyInt = 1;

  LAPACK(dgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, &s[0], &dummyReal, &dummyInt, 
      &dummyReal, &dummyInt, &dummyWork, &lwork, &info );

  lwork = dummyWork;
  std::vector<double> work(lwork);
  LAPACK(dgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, &s[0], &dummyReal, &dummyInt, 
      &dummyReal, &dummyInt, &work[0], &lwork, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("dgesvd's updating process failed");
  }
  return ;
}


// *********************************************************************
// QR with column pivoting. Use dgeqp3 routine
// *********************************************************************

void QRCP( Int m, Int n, double* A, Int lda, Int * piv, 
    double* tau )
{
  if( m==0 || n==0 )
  {
    return;
  }

  Int lwork=-1, info;
  double dummyWork;

  LAPACK(dgeqp3)
    ( &m, &n, A, &lda, piv, tau, &dummyWork, &lwork, &info );

  lwork = dummyWork;
  std::vector<double> work(lwork);
  LAPACK(dgeqp3)
    ( &m, &n, A, &lda, piv, tau, &work[0], &lwork, &info );

  // Important: fortran index is 1-based. Change to 0-based
  for( Int i = 0; i < n; i++ ){
    piv[i]--;
  }

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }

  return;
}



// *********************************************************************
// QR without column pivoting. Use dgeqrf routine
// *********************************************************************

void QR( Int m, Int n, double* A, Int lda, double* tau )
{
  if( m==0 || n==0 )
  {
    return;
  }

  Int lwork=-1, info;
  double dummyWork;

  LAPACK(dgeqrf) ( &m, &n, A, &lda, tau, &dummyWork, &lwork, &info );

  lwork = dummyWork;
  std::vector<double> work(lwork);
  LAPACK(dgeqrf) ( &m, &n, A, &lda, tau, &work[0], &lwork, &info );


  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }

  return;
}


void QRCP( Int m, Int n, double* A, double* Q, double* R, 
    Int lda, Int * piv )
{
  if( m==0 || n==0 )
  {
    return;
  }

  Int lwork=-1, info;
  double dummyWork;
  std::vector<double> tau(n);


  LAPACK(dgeqp3)
    ( &m, &n, A, &lda, piv, &tau[0], &dummyWork, &lwork, &info );

  lwork = dummyWork;
  std::vector<double> work(lwork);
  LAPACK(dgeqp3)
    ( &m, &n, A, &lda, piv, &tau[0], &work[0], &lwork, &info );

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "xGEQP3 Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }



  // Important: fortran index is 1-based. Change to 0-based
  for( Int i = 0; i < n; i++ ){
    piv[i]--;
  }

  // Copy A to R. Assumes that R has been allocated with the same
  // distribution as A
  Lacpy( 'A', m, n, A, lda, R, lda ); 

  // Delete the lower triangular entries of R
  for( Int j = 0; j < n; j++ ){
    for( Int i = 0; i < m; i++ ){
      if( i > j )
        R[i+j*lda] = 0.0;
    }
  }

  // Reconstruct the Q factor
  Int k = std::min( m, n );
  lwork = -1;
  LAPACK(dorgqr)
    ( &m, &k, &k, A, &lda, &tau[0], &dummyWork, &lwork, &info );
  lwork = dummyWork;
  work.resize(lwork);
  LAPACK(dorgqr)
    ( &m, &k, &k, A, &lda, &tau[0], &work[0], &lwork, &info );

  Lacpy( 'A', m, k, A, lda, Q, lda ); 

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "xORGQR Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }

  return;
}


} // namespace lapack
} // namespace scales
