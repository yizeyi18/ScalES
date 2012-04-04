/* TESTGE solves the generalized eigenvalue problem
 *   A x = \lambda B x
 * by factorizing B.
 *
 * Date of revision: 10/11/2011
 */
#include "comobject.hpp"
#include "vec3t.hpp"
#include "numtns.hpp"
#include "parvec.hpp"
#include "cblacs.h"

int main(int argc, char** argv){
  int N = 3;
  DblNumMat A(N,N), B(N,N);
  DblNumVec W(N);
  for(int i = 0; i < N; i++){
    A(i,i) = 2.0;
    if(i < N-1)
      A(i,i+1) = -1.0;
    if(i > 0 )
      A(i,i-1) = -1.0;
  }
  for(int i = 0; i < N; i++){
    B(i,i) = 4.0;
    if(i < N-1)
      B(i,i+1) = -1.0;
    if(i > 0 )
      B(i,i-1) = -1.0;
  }
  /* Factorization of matrix B */
  {
    char   uplo    = 'L';
    int    ldb     =  N;
    int    info;
    dpotrf_(&uplo, &N, B.data(), &ldb, &info);
  }
  /* Form the standard eigenvalue problem */
  {
    int    itype   = 1;
    char   uplo    = 'L';
    int    lda     = N;
    int    ldb     = N;
    int    info;
    dsygst_(&itype, &uplo, &N, A.data(), &lda, B.data(), &ldb, &info); 
  }
  /* Solve the standard eigenvalue problem */
  {
    char    jobz    = 'V';
    char    uplo    = 'L';
    int     lda     = N;
    int     lwork, liwork;
    int     info;
    DblNumVec      work;
    IntNumVec      iwork;

    lwork = 1 + 6*N + 2*N*N;
    liwork = 3 + 5 * N;
    
    work.resize(lwork);
    iwork.resize(liwork);

    dsyevd_(&jobz, &uplo, &N, A.data(), &lda, W.data(), work.data(), 
	    &lwork, iwork.data(), &liwork, &info);
  }
  /* Obtain the correct eigenfunctions and save them in A */
  {
    char    side   = 'L';
    char    uplo   = 'L';
    char    transa = 'T';
    char    diag   = 'N';
    double  one    = 1.0;
    dtrsm_(&side, &uplo, &transa, &diag, &N, &N, &one,
	   B.data(), &N, A.data(), &N);
  }
  cout << "eigenvalues " << endl;
  cout << W << endl;
  cout << "eigenvectors " << endl;
  cout << A << endl;
  cout << "B matrix after factorization " << endl;
  cout << B << endl;
   
  return 1; 
}

