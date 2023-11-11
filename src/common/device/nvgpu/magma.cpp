//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

/// @file magma.cpp
/// @brief Thin interface to magma
/// @date 2020-08-21
#include "magma.h"

namespace scales {
namespace MAGMA {
#ifdef GPU
void Init(void)
{
  magma_int_t st = magma_init();
  if( st != MAGMA_SUCCESS)
  {
    std::ostringstream msg;
    msg << " MAGMA init wrong " << std::endl;
  }
}

void Destroy(void)
{
  magma_int_t st = magma_finalize();
  if( st != MAGMA_SUCCESS)
  {
    std::ostringstream msg;
    msg << " MAGMA destroy wrong " << std::endl;
  }
}

// *********************************************************************
// Cholesky factorization
// *********************************************************************
void Potrf( char uplo, Int n, const cuDoubleComplex* A, Int lda )
{
  magma_int_t info;
  magma_uplo_t job; 

  if (uplo == 'u' || uplo == 'U')
       job = MagmaUpper;
  else 
       job = MagmaLower;

  magmaDoubleComplex_ptr Aptr = (magmaDoubleComplex_ptr) A;
  magma_zpotrf_gpu( job, n, Aptr, lda, &info );

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "zpotrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 ){
    std::ostringstream msg;
    msg << "zpotrf returned with info = " << info << std::endl;
    //msg << "A(info,info) = " << A[info-1+(info-1)*lda] << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
}

void Potrf( char uplo, Int n, const double* A, Int lda )
{
  magma_int_t info;
  magma_uplo_t job; 

  if (uplo == 'u' || uplo == 'U')
       job = MagmaUpper;
  else 
       job = MagmaLower;

  magmaDouble_ptr Aptr = (magmaDouble_ptr) A;
  magma_dpotrf_gpu( job, n, Aptr, lda, &info );

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
/*
*/
// *********************************************************************
// LU factorization (with partial pivoting)
// *********************************************************************

/*
void Getrf( Int m, Int n, double* A, Int lda, Int* p )
{
  Int info;
  MAGMA(dgetrf)( &m, &n, A, &lda, p, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "dgetrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
    ErrorHandling("Matrix is singular.");
}
*/

// *********************************************************************
// For solving the standard eigenvalue problem using the divide and
// conquer algorithm
// *********************************************************************
void Syevd
( char jobz, char uplo, Int n, cuDoubleComplex* A, Int lda, double* eigs ){

////
    magma_int_t *iwork, *isuppz, *ifail, aux_iwork[1];
    magma_int_t N, n2, info, lwork, liwork, ldda;
    N = n;
    ldda = lda;
    cuDoubleComplex aux_work[1];
    double *rwork, aux_rwork[1];
    magma_int_t lrwork;
////

  magma_vec_t zjob;
  if (jobz == 'v' || jobz == 'V') zjob = MagmaVec; else  zjob = MagmaNoVec;
  
  magma_uplo_t m_uplo;
  if(uplo == 'U') m_uplo = MagmaUpper; else m_uplo = MagmaLower;
  magmaDoubleComplex_ptr dA = (magmaDoubleComplex_ptr) A;

  /*
  Int lwork = -1, info;
  Int liwork = -1;
  std::vector<double> work(1);
  std::vector<int>    iwork(1);
  */
  magma_zheevd_gpu( zjob, m_uplo,
                    N, NULL, lda, NULL,  // A, w
                    NULL, lda,            // host A
                    aux_work,  -1,
                    aux_rwork, -1,
                    aux_iwork, -1,
                    &info );

  lwork  = (magma_int_t) ( aux_work[0].x );
  liwork = aux_iwork[0];
  cuDoubleComplex * h_R, *h_work;

  magma_zmalloc_pinned( &h_R,    N*lda  );
  magma_zmalloc_pinned( &h_work, lwork  );
  magma_imalloc_cpu( &iwork,  liwork );

  lrwork = (magma_int_t) aux_rwork[0];
  //double * rwork;
  magma_dmalloc_pinned( &rwork,    lrwork );



  magma_zheevd_gpu( zjob, m_uplo,
                    N, dA, ldda, eigs,
                    h_R, lda,          // h_R
                    h_work, lwork,
                    rwork, lrwork,
                    iwork, liwork,
                    &info );
  if( info != 0 )
  {
    std::ostringstream msg;
    msg << "magma_dsyevd_gpu returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }

  magma_free_pinned( h_R    );
  magma_free_pinned( h_work );
  magma_free_cpu( iwork );
  magma_free_pinned( rwork );
}

void Syevd
( char jobz, char uplo, Int n, double* A, Int lda, double* eigs ){

////
    magma_int_t *iwork, *isuppz, *ifail, aux_iwork[1];
    magma_int_t N, n2, info, lwork, liwork, ldda;
    N = n;
    ldda = lda;
    double aux_work[1];
#ifdef COMPLEX
    double *rwork, aux_rwork[1];
    magma_int_t lrwork;
#endif
////

  magma_vec_t zjob;
  if (jobz == 'v' || jobz == 'V') zjob = MagmaVec; else  zjob = MagmaNoVec;
  
  magma_uplo_t m_uplo;
  if(uplo == 'U') m_uplo = MagmaUpper; else m_uplo = MagmaLower;
  magmaDouble_ptr dA = (magmaDouble_ptr) A;  

  /*
  Int lwork = -1, info;
  Int liwork = -1;
  std::vector<double> work(1);
  std::vector<int>    iwork(1);
  */
  magma_dsyevd_gpu( zjob, m_uplo,
                    N, NULL, lda, NULL,  // A, w
                    NULL, lda,            // host A
                    aux_work,  -1,
                    #ifdef COMPLEX
                    aux_rwork, -1,
                    #endif
                    aux_iwork, -1,
                    &info );

  lwork  = (magma_int_t) MAGMA_D_REAL( aux_work[0] );
  liwork = aux_iwork[0];
  double * h_R, *h_work;

  magma_dmalloc_pinned( &h_R,    N*lda  );
  magma_dmalloc_pinned( &h_work, lwork  );
  magma_imalloc_cpu( &iwork,  liwork );

#ifdef COMPLEX
  lrwork = (magma_int_t) aux_rwork[0];
  double * rwork;
  magma_dmalloc_pinned( &rwork,    lrwork );
#endif



  magma_dsyevd_gpu( zjob, m_uplo,
                    N, dA, ldda, eigs,
                    h_R, lda,          // h_R
                    h_work, lwork,
                    #ifdef COMPLEX
                    rwork, lrwork,
                    #endif
                    iwork, liwork,
                    &info );
  if( info != 0 )
  {
    std::ostringstream msg;
    msg << "magma_dsyevd_gpu returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }

  magma_free_pinned( h_R    );
  magma_free_pinned( h_work );
  magma_free_cpu( iwork );
#ifdef COMPLEX
  magma_free_pinned( rwork );
#endif
}


/*
 * magma_zgels_gpu 
*/

void Zgels( Int m, Int n, Int nrhs, cuDoubleComplex * A, Int lda, 
       cuDoubleComplex * B, Int ldb)
{
 
  Int info;
  magma_int_t lworkgpu, lhwork;
  magmaDoubleComplex tmp[1], *h_work;

  Int nb   = magma_get_zgeqrf_nb( m, n );
  lworkgpu = (m - n + nb)*(nrhs + nb) + nrhs*nb;

  magma_zgels_gpu( MagmaNoTrans, m, n, nrhs, NULL, lda,
                   NULL, ldb, tmp, -1, &info );

  if ( info != 0) 
  {
    std::ostringstream msg;
    msg << "magma_zgels_gpu get work array size info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  lhwork = (magma_int_t) MAGMA_Z_REAL( tmp[0] );
  lhwork = lhwork > lworkgpu ? lhwork:lworkgpu;

  magma_zmalloc_cpu( &h_work, lhwork );
/*
  magma_zgels_gpu( MagmaNoTrans, m, n, nrhs, A, lda,
                   B, ldb, h_work, lworkgpu, &info );
*/
  magma_zgels3_gpu( MagmaNoTrans, m, n, nrhs, A, lda,
                   B, ldb, h_work, lworkgpu, &info );
  if ( info != 0) 
  {
    std::ostringstream msg;
    msg << "magma_zgels_gpu error info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  magma_free_cpu( h_work );
}


// *********************************************************************
// For computing the inverse of a triangular matrix
// *********************************************************************


/*
void Trtri( char uplo, char diag, Int n, const double* A, Int lda )
{
  Int info;
  MAGMA(dtrtri)( &uplo, &diag, &n, A, &lda, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "dtrtri returned with info = " << info;
    throw std::logic_error( msg.str().c_str() );
  }
  else if( info > 0 )
    throw std::runtime_error("Matrix is singular.");
}
*/
// *********************************************************************
// Copy
// *********************************************************************

void Lacpy( char uplo, Int m, Int n, const double* A, Int lda, double* B, Int ldb )
{
  //MAGMA(dlacpy)( &uplo, &m, &n, A, &lda, B, &ldb );
 /*
  magma_uplo_t magmaUplo;
  magmaDouble_const_ptr dA = (magmaDouble_const_ptr ) A;
  magmaDouble_const_ptr dB = (magmaDouble_const_ptr ) B;
  magma_queue_t queue ;
  
  magmablas_dlacpy(magmaUplo, m, n, dA, lda, dB, ldb, queue);
  */
}
#endif

} // namespace MAGMA
} // namespace scales
