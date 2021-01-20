#pragma once

#include "environment.hpp"
#include <lapack.hh>

namespace dgdft {


template <typename T>
void QRCP( Int m, Int n, T* A, Int lda, Int * piv, T* tau ) {


  if( !( m and n ) ) return;

  statusOFS << "IN QRCP - NO Q" << std::endl;
  std::vector<int64_t> _piv( n );
  auto info = lapack::geqp3( m, n, A, lda, _piv.data(), tau );


  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "xGEQP3 " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }

  for( auto i = 0; i < n; ++i ) piv[i] = _piv[i] - 1;

}

template <typename T>
void QRCP( Int m, Int n, T* A, T* Q, T* R, Int lda, Int * piv ) {

  if( !( m and n ) ) return;

  statusOFS << "IN QRCP - with Q" << std::endl;
  std::vector<T> tau(std::min(m,n));
  QRCP( m, n, A, lda, piv, tau.data() ); // Perform QR factorization

  // Copy A to R. Assumes that R has been allocated with the same
  // distribution as A
  lapack::lacpy( lapack::MatrixType::General, m, n, A, lda, R, lda ); 

  // Delete the lower triangular entries of R
  // XXX: This can be done directly via lacpy
  for( Int j = 0; j < n; j++ ){
    for( Int i = 0; i < m; i++ ){
      if( i > j )
        R[i+j*lda] = 0.0;
    }
  }

  // Reconstruct the Q factor
  // TODO: Make driver to handle ungqr for complex
  lapack::lacpy( lapack::MatrixType::General, m, n, A, lda, Q, lda );
  auto info = lapack::orgqr( m, n, std::min(m,n), Q, lda, tau.data() );

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "xORGQR Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
}


// *********************************************************************
// Orthogonalization (MATLAB's orth, using SVD)
// *********************************************************************
//
template <typename T>
void Orth( Int m, Int n, T* A, Int lda ){
  if( not (m and n) ) return;
  std::vector<T> S(n); // TODO: Complex -> Real Type
  lapack::gesvd( lapack::Job::OverwriteVec, lapack::Job::NoVec,
                 m, n, A, lda, S.data(), A, 1, A, 1 );
}




}
