/* TESTPGE solves the generalized eigenvalue problem
 *   A x = \lambda B x
 * by factorizing B and use ScaLAPACK.  Run with 4 processors.
 *
 * A = [2  -1   0  0
 *      -1  2  -1  0
 *      0  -1   2 -1
 *      0   0  -1  2]
 *
 * B = [4  -1   0  0
 *      -1  4  -1  0
 *      0  -1   4 -1
 *      0   0  -1  4]
 * 
 * [V,D] = eig(A,B) gives
 *

    V =
    
       -0.2409    0.3271   -0.2799    0.1568
       -0.3897    0.2021    0.1730   -0.2538
       -0.3897   -0.2021    0.1730    0.2538
       -0.2409   -0.3271   -0.2799   -0.1568
    
    
    D =
    
        0.1604         0         0         0
             0    0.4086         0         0
             0         0    0.5669         0
             0         0         0    0.6440

 * Date of revision: 10/11/2011
 */
#include "comobject.hpp"
#include "vec3t.hpp"
#include "numtns.hpp"
#include "parvec.hpp"
#include "cblacs.h"



int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  int mpirank; MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize; MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  int myid = mpirank, nprocs = mpisize;
  int myprow, mypcol;
  int CONTXT;
  
  /* BLACS initialization */
  char colorder = 'C';
  int nprow = 0, npcol = 0;
  for(int i = round(sqrt(double(nprocs))); i <= nprocs; i++){
    nprow = i; npcol = nprocs / nprow;
    if( nprow * npcol == nprocs) break;
  } 
  if(myid==MASTER) {
    cerr<<"nprow "<<nprow<<" npcol "<<npcol<<endl;
  }
  Cblacs_get(0,0, &CONTXT);
  Cblacs_gridinit(&CONTXT, &colorder, nprow, npcol);
  Cblacs_gridinfo(CONTXT,  &nprow, &npcol, &myprow, &mypcol);
 

  int MB = 2;
  int loc_m = MB;
  int nbblck = 2;
  int N = loc_m * nbblck;

  int desca[DLEN_], descb[DLEN_], descz[DLEN_];
  int izero = 0, ione = 1;
  double zero = 0.0;
  int info;
  descinit_(desca, &N, &N, &MB, &MB, &izero, &izero, &CONTXT, &loc_m, &info);
  descinit_(descb, &N, &N, &MB, &MB, &izero, &izero, &CONTXT, &loc_m, &info);
  descinit_(descz, &N, &N, &MB, &MB, &izero, &izero, &CONTXT, &loc_m, &info);

  DblNumMat locA(loc_m, loc_m);
  DblNumMat locB(loc_m, loc_m);
  DblNumMat locZ(loc_m, loc_m);
  setvalue(locA, 0.0);
  setvalue(locB, 0.0);
  setvalue(locZ, 0.0);


  DblNumMat A(N,N),  B(N,N);
  DblNumVec W(N);
  setvalue(A, 0.0);  setvalue(B, 0.0);
  // Simple case
  if(0){ 
    for(int i = 0; i < N; i++){
      A(i,i) = 1.0;  B(i,i) = 2.0;
    } 
  }
  // Slightly more complicated case
  if(1){
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
  }

  if(1)
  {
    for(int ja = 0; ja < nbblck; ja++){
      for(int ia = 0; ia < nbblck; ia++){
	if( (ia == myprow) && (ja == mypcol) ){
	  for(int j = 0; j < MB; j++){
	    for(int i = 0; i < MB; i++){
	      locA(i,j) = A(i+ia*MB, j+ja*MB);
	    }
	  }
	}
      }
    }  
    for(int ja = 0; ja < nbblck; ja++){
      for(int ia = 0; ia < nbblck; ia++){
	if( (ia == myprow) && (ja == mypcol) ){
	  for(int j = 0; j < MB; j++){
	    for(int i = 0; i < MB; i++){
	      locB(i,j) = B(i+ia*MB, j+ja*MB);
	    }
	  }
	}
      }
    }  

  } 
  cout << "myrow = " << myprow << "mycol = " << mypcol << endl;

  /* Factorization of matrix B */
  if(1)
  {
    char   uplo    = 'L';
    int    ldb     =  N;
    int    ione    = 1;
    int    info;
    pdpotrf_(&uplo, &N, locB.data(), &ione,
             &ione, descb, &info);
  }

  /* Form the standard eigenvalue problem */
  if(1)
  {
    int    itype   = 1;
    char   uplo    = 'L';
    int    lda     = N;
    int    ldb     = N;
    int    info;
    double scale;
    int lwork = 1000;  // might be changed
    DblNumVec work(lwork);

    pdsygst_(&itype, &uplo, &N, locA.data(), &ione, &ione,
	     desca,  locB.data(), &ione, &ione, descb, &scale, &info); 


    if( mpirank == MASTER )
      cout << "info = " << info;
  }

  /* Solve the standard eigenvalue problem */
  if(1)
  {
    char    jobz    = 'V';
    char    uplo    = 'L';
    int     lda     = N;
    int     lwork, liwork;
    int     info;
    DblNumVec      work;
    IntNumVec      iwork;

    lwork  = 1000;
    liwork = 1000;
    
    work.resize(lwork);
    iwork.resize(liwork);

    pdsyevd_(&jobz, &uplo, &N, locA.data(), &ione, &ione, desca,
	     W.data(), locZ.data(), &ione, &ione, descz, work.data(), 
	     &lwork, iwork.data(), &liwork, &info);
  }

  /* Obtain the correct eigenfunctions and save them in A */
  if(1)
  {
    char    side   = 'L';
    char    uplo   = 'L';
    char    transa = 'T';
    char    diag   = 'N';
    double  one    = 1.0;
    pdtrsm_(&side, &uplo, &transa, &diag, &N, &N, &one,
	    locB.data(), &ione, &ione, descb, locZ.data(), 
	    &ione, &ione, descz);
  }

  if(1)
  {
    int   nout   = 6; 
    DblNumVec work(MB);
    pdlaprnt_(&N, &N, locA.data(), &ione, &ione, descb, 
	     & izero, &izero, "A", &nout, work.data(), 1);
    pdlaprnt_(&N, &N, locB.data(), &ione, &ione, descb, 
	     & izero, &izero, "B", &nout, work.data(), 1);
    pdlaprnt_(&N, &N, locZ.data(), &ione, &ione, descz, 
	     & izero, &izero, "Z", &nout, work.data(), 1);
    if( mpirank == MASTER )
      cout << "eigenvalue: " << W << endl;
  } 
  
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
   
  return 1; 
}

