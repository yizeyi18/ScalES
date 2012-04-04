/* TESTPQR performs QR factorization of a matrix A, threshold the R
 * value as Rt and obtain the updated matrix
 *   At = Q * Rt
 * Run with 4 processors.
 *
 * A = [2  -1   0  -1
 *     -1   2  -1   0
 *      0  -1   2  -1
 *     -1   0  -1   2]
 *
 * The corresponding MATLAB routine goes as follows
 * >> [Q,R]=qr(A,0)

    Q =
    
        0.8165    0.1826   -0.2236   -0.5000
       -0.4082    0.7303    0.2236   -0.5000
             0   -0.5477    0.6708   -0.5000
       -0.4082   -0.3651   -0.6708   -0.5000
    
    
    R =
    
        2.4495   -1.6330    0.8165   -1.6330
             0    1.8257   -1.4606   -0.3651
             0         0    1.7889   -1.7889
             0         0         0    0.0000
   >> Q*R(:,1:3)
    
    ans =
    
        2.0000   -1.0000    0.0000
       -1.0000    2.0000   -1.0000
             0   -1.0000    2.0000
       -1.0000    0.0000   -1.0000   

 * Date of revision: 10/23/2011
 */
#include "comobject.hpp"
#include "vec3t.hpp"
#include "numtns.hpp"
#include "parvec.hpp"
#include "cblacs.h"
#include "vecmatop.hpp"



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
 

  int MB = 2, NB = MB;
  int loc_m = MB;
  int nbblck = 2;
  int N = loc_m * nbblck;

  int desca[DLEN_];
  int izero = 0, ione = 1;
  double zero = 0.0;
  int info;
  descinit_(desca, &N, &N, &MB, &MB, &izero, &izero, &CONTXT, &loc_m, &info);

  DblNumMat locA(loc_m, loc_m);
  setvalue(locA, 0.0);

  DblNumMat A(N,N);
  setvalue(A, 0.0);  
  // Slightly more complicated case
  if(1){
    for(int i = 0; i < N; i++){
      A(i,i) = 2.0;
      if(i < N-1)
	A(i,i+1) = -1.0;
      if(i > 0 )
	A(i,i-1) = -1.0;
    }
    A(0,N-1) = -1.0;
    A(N-1,0) = -1.0;
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
  } 
 
  // Test purpose
  int gj, j, IZERO = 0;
  j = 0;
  gj = npcol*NB*(j/NB) + mypcol*NB + j%NB;
  int id = 2;
  if(mpirank == id){
    cout << "g[" << j << "] = " << gj << endl;
  }
  j = 1;
  gj = npcol*NB*(j/NB) + mypcol*NB + j%NB;
  if(mpirank == id){
    cout << "g[" << j << "] = " << gj << endl;
  }
  j = loc_m;
  gj = npcol*NB*(j/NB) + mypcol*NB + j%NB;
//  gj = indxl2g_(&j, &NB, &mypcol, &IZERO, &npcol);
  if(mpirank == id){
    cout << "g[" << j << "] = " << gj << endl;
  }


  DblNumMat locQ(loc_m, loc_m);
  DblNumVec tau(loc_m);
  IntNumVec ipiv(loc_m);
  locQ = locA;
  DblNumMat locR(loc_m, loc_m);
  setvalue(locR, 0.0);

  /* QR factorization by pdgeqpf */
  if(1){
    int AM = N, AN = N;
    int IONE = 1, IZERO = 0;
    int lwork, info;
    DblNumVec work(1);

    lwork = -1;
    pdgeqpf_(&AM, &AN, locQ.data(), &IONE, &IONE, desca,
	     ipiv.data(), tau.data(), work.data(),
	     &lwork, &info);

    lwork = (int)work(0);
    work.resize(lwork);
    pdgeqpf_(&AM, &AN, locQ.data(), &IONE, &IONE, desca,
	     ipiv.data(), tau.data(), work.data(),
	     &lwork, &info);
  }

  /* Get the R matrix */
  int ncolsav, tncol;
  if(1){
    locR = locQ;
    int IZERO = 0;
    int gi, gj;
    double EPS = 1e-8;
    tncol = N;
    for(int j = 0; j < locR.n(); j++){
      gj = npcol*NB*(j/NB) + mypcol*NB + j%NB;
//      gj = indxl2g_(&j, &NB, &mypcol, &IZERO, &npcol);
      for(int i = 0; i < locR.m(); i++){
	gi = nprow*MB*(i/MB) + myprow*MB + i%MB;
//	gi = indxl2g_(&i, &MB, &myprow, &IZERO, &nprow);
	/* Only get the upper triangular part */
        if(gi > gj){
	  locR(i,j) = 0.0;
	} 
	/* Threshold the diagonal elements of R */
	if(gi == gj){
	  if( abs(locR(i,i)) < EPS ){
	    tncol = min(tncol, gi);
	  }
	}
      }
    } 
    
    MPI_Allreduce(&tncol, &ncolsav, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if( mpirank == 0 ){
      cout << "ncolsav = " << ncolsav << endl;
    } 
  }
  
  /* Multiply Q*R back and test whether one obtains A */
  int descRrdc[DLEN_]; 
  descinit_(descRrdc, &N, &ncolsav, &MB, &MB, &izero, &izero, &CONTXT, &loc_m, &info);
  /* B = Q * R(:,1:gud) */
  DblNumMat locB(loc_m, loc_m);
  locB = locR;
  if(1){
    char side     = 'L';
    char trans    = 'N';  
    int  IZERO = 0, IONE = 1;
    int  lwork, info;
    DblNumVec work(1);

    lwork = -1;
    pdormqr_(&side, &trans, &N, &ncolsav, &N, locQ.data(), 
	     &ione, &ione, desca, tau.data(), locB.data(),
	     &ione, &ione, descRrdc, work.data(), &lwork, &info);
    lwork = (int)work(0);
    work.resize(lwork);
    pdormqr_(&side, &trans, &N, &ncolsav, &N, locQ.data(), 
	     &ione, &ione, desca, tau.data(), locB.data(),
	     &ione, &ione, descRrdc, work.data(), &lwork, &info);
  }

  int descAR[DLEN_];
  DblNumMat AR;
  int ARM, ARN;
  char ntrans = 'N';
  pdgemm(&ntrans, &ntrans, locA, N, N, desca, locR, N, N, desca, AR,
	 ARM, ARN, descAR, MB, MB, CONTXT);


  if(1)
  {
    int   nout   = 6; 
    DblNumVec work(MB);
    pdlaprnt_(&N, &N, locQ.data(), &ione, &ione, desca, 
	     & izero, &izero, "Q", &nout, work.data(), 1);
    cout << endl;
    pdlaprnt_(&N, &N, locR.data(), &ione, &ione, desca, 
	     & izero, &izero, "R", &nout, work.data(), 1);
    cout << endl;
    pdlaprnt_(&N, &ncolsav, locR.data(), &ione, &ione, descRrdc, 
	     & izero, &izero, "r", &nout, work.data(), 1);
    cout << endl;
    pdlaprnt_(&N, &ncolsav, locB.data(), &ione, &ione, descRrdc, 
	     & izero, &izero, "B", &nout, work.data(), 1);
    
    cout << endl;
    pdlaprnt_(&N, &ncolsav, AR.data(), &ione, &ione, descAR, 
	     & izero, &izero, "AR", &nout, work.data(), 2);
  } 
  
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
   
  return 1; 
}

