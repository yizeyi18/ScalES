#include "blas.h"
#include "lapack.h"
#include "numvec.hpp"
#include "nummat.hpp"
#include "numtns.hpp"
#include "vecmatop.hpp"
#include "cblacs.h"

using std::cerr;
using std::abs;
//typedef long int int;

//Y <- a M X + b Y
// ---------------------------------------------------------------------- 
int dgemm(double alpha, const DblNumMat& A, const DblNumMat& B, double beta, DblNumMat& C)
{
  assert( A.m() == C.m() );  assert( A.n() == B.m() );  assert( B.n() == C.n() );
  iC( dgemm(C.m(), C.n(), A.n(), alpha, A.data(), B.data(), beta, C.data()) );
  return 0;
}
// ---------------------------------------------------------------------- 
int dgemm(int m, int n, int k, double alpha, double* A, double* B, double beta, double* C)
{
  char transa = 'N';
  char transb = 'N';
  assert(m!=0 && n!=0 && k!=0);
  dgemm_(&transa, &transb, &m, &n, &k,
	 &alpha, A, &m, B, &k, &beta, C, &m);
  return 0;
}
//Y <- a M X + b Y
// ---------------------------------------------------------------------- 
int dgemv(double alpha, const DblNumMat& A, const DblNumVec& X, double beta, DblNumVec& Y)
{
  assert(Y.m() == A.m());
  assert(A.n() == X.m());
  iC( dgemv(A.m(), A.n(), alpha, A.data(), X.data(), beta, Y.data()) );
  return 0;
}
// ---------------------------------------------------------------------- 
int dgemv(int m, int n, double alpha, double* A, double* X, double beta, double* Y)
{
  char trans = 'N';
  assert(m!=0 && n!=0);
  int incx = 1;
  int incy = 1;
  dgemv_(&trans, &m, &n, &alpha, A, &m, X, &incx, &beta, Y, &incy);
  return 0;
}

// ---------------------------------------------------------------------- 
int zgemm(cpx alpha, const CpxNumMat& A, const CpxNumMat& B, cpx beta, CpxNumMat& C)
{
  assert( A.m() == C.m() );  assert( A.n() == B.m() );  assert( B.n() == C.n() );
  iC( zgemm(C.m(), C.n(), A.n(), alpha, A.data(), B.data(), beta, C.data()) );
  return 0;
}
// ---------------------------------------------------------------------- 
int zgemm(int m, int n, int k, cpx alpha, cpx* A, cpx* B, cpx beta, cpx* C)
{
  char transa = 'N';
  char transb = 'N';
  assert(m!=0 && n!=0 && k!=0);
  zgemm_(&transa, &transb, &m, &n, &k,
	 &alpha, A, &m, B, &k, &beta, C, &m);
  return 0;
}
//Y <- a M X + b Y
// ---------------------------------------------------------------------- 
int zgemv(cpx alpha, const CpxNumMat& A, const CpxNumVec& X, cpx beta, CpxNumVec& Y)
{
  assert(Y.m() == A.m());
  assert(A.n() == X.m());
  iC( zgemv(A.m(), A.n(), alpha, A.data(), X.data(), beta, Y.data()) );
  return 0;
}
// ---------------------------------------------------------------------- 
int zgemv(int m, int n, cpx alpha, cpx* A, cpx* X, cpx beta, cpx* Y)
{
  char trans = 'N';
  int incx = 1;
  int incy = 1;
  assert(m!=0 && n!=0);
  //cerr<<sizeof(int)<<" "<<sizeof(long int)<<endl;
  zgemv_(&trans, &m, &n, &alpha, A, &m, X, &incx, &beta, Y, &incy);
  return 0;
}


// ---------------------------------------------------------------------- 
int dgmres(int (*A)(const DblNumVec&, DblNumVec&), const DblNumVec& b, const DblNumVec& x0,
	   int restart, double tol, int maxit, int print,
	   DblNumVec& x, int& flag, double& relres, int& niter, vector<double>& resvec)
{
  int n = b.m();
  int m = restart;
  
  DblNumMat V(n,m+1);  setvalue(V,0.0);
  DblNumMat H(m+1,m);  setvalue(H,0.0);
  
  double bnrm2 = 0;  for(int a=0; a<n; a++)    bnrm2 = bnrm2 + b(a)*b(a);
  bnrm2 = sqrt(bnrm2);
  resvec.clear();
  
  DblNumVec tmp(n);  for(int a=0; a<n; a++)    tmp(a) = 0;
  x = x0;
  double xnrm2 = 0;  for(int a=0; a<n; a++)    xnrm2 = xnrm2 + x(a)*x(a);
  xnrm2 = sqrt(xnrm2);
  if(xnrm2 > 1e-16) {
    iC( (*A)(x,tmp) );
  }
  DblNumVec r(n);  for(int a=0; a<n; a++)    r(a) = b(a) - tmp(a);
  double beta=0;  for(int a=0; a<n; a++)    beta = beta + r(a)*r(a);
  beta = sqrt(beta);
  double res = beta; 
  //if(print==1) cerr<<"Iter "<<resvec.size()<<": "<<res<<endl;
  if(print==1) printf("Iter %d %.10e\n", (int)resvec.size(), double(res));
  resvec.push_back(res);
  double err = res/bnrm2;
  
  int iter = 0;
  while(1) {
    for(int a=0; a<n; a++)      V(a,0) = r(a)/beta;
    int j = 0;
    DblNumVec y;
    DblNumMat Hj;
    while(1) {
      DblNumVec Vj(n);      for(int a=0; a<n; a++)	Vj(a) = V(a,j);
      DblNumVec w(n);      setvalue(w,0.0);
      iC( (*A)(Vj,w) );
      for(int k=0; k<=j; k++) {
	double sum = 0;	for(int a=0; a<n; a++)	  sum = sum + V(a,k)*w(a);
	H(k,j) = sum;
	for(int a=0; a<n; a++)	  w(a) = w(a) - sum*V(a,k);
      }
      double nw=0;      for(int a=0; a<n; a++)	nw = nw + w(a)*w(a);
      nw = sqrt(nw);
      H(j+1,j) = nw;
      for(int a=0; a<n; a++)	V(a,j+1) = w(a) / nw;
      DblNumVec be(j+2);      for(int a=0; a<j+2; a++)	be(a) = 0;
      be(0) = beta;
      y.resize(j+1);
      Hj.resize(j+2,j+1);
      for(int a=0; a<j+2; a++)	for(int c=0; c<j+1; c++)	  Hj(a,c) = H(a,c);
      //SOLVE
      {
	int m = j+2;
	int n = j+1;
	int nrhs = 1;
	DblNumMat Hjtmp(Hj);	//cpx* aptr = Hjtmp.data();
	int lda = j+2;
	DblNumVec betmp(be);	//cpx* bptr = betmp.data();
	int ldb = j+2;
	DblNumVec s(j+1);	setvalue(s,0.0);
	double rcond = 0;
	int rank;
	DblNumVec work(10*(j+2));
	int lwork = 10*(j+2);
	//DblNumVec rwork(10*(j+2));
	int info;
	dgelss_(&m,&n,&nrhs,Hjtmp.data(),&lda,betmp.data(),&ldb,s.data(),&rcond,&rank,work.data(),&lwork,&info);	//cerr<<"info "<<info<<endl;
	for(int a=0; a<j+1; a++)	  y(a) = betmp(a);
	//cerr<<"Hj "<<Hj<<endl;	//cerr<<"y "<<y<<endl;
      }
      iC( dgemv(-1.0, Hj, y, 1, be) );
      double res=0;      for(int a=0; a<j+2; a++)	res = res + be(a)*be(a);
      res = sqrt(res);
      //if(print==1) cerr<<"Iter "<<resvec.size()<<": "<<res<<endl;
      if(print==1) printf("Iter %d %.10e\n", (int)resvec.size(), double(res));
      resvec.push_back(res);
      err = res/bnrm2;
      if((err<tol) | (j==m-1))
	break;
      j=j+1;
    }
    DblNumMat Vj(n,j+1,false,V.data());
    iC( dgemv(1.0, Vj, y, 1.0, x) );
    DblNumVec tmp(j+2);
    iC( dgemv(1.0, Hj, y, 0.0, tmp) );
    DblNumMat Vj1(n,j+2,false,V.data());
    iC( dgemv(-1.0, Vj1, tmp, 1.0, r) );
    beta = 0;    for(int a=0; a<n; a++)      beta = beta + r(a)*r(a);
    beta = sqrt(beta);
    if((err<tol) | (iter==maxit-1))
      break;
    iter++;
  }
  flag = (err>tol);
  relres = beta;
  niter = iter+1;
  
  return 0;
}

// ---------------------------------------------------------------------- 
int zgmres(int (*A)(const CpxNumVec&, CpxNumVec&), const CpxNumVec& b, const CpxNumVec& x0,
	   int restart, double tol, int maxit, int print,
	   CpxNumVec& x, int& flag, double& relres, int& niter, vector<double>& resvec)
{
  int n = b.m();
  int m = restart;
  
  CpxNumMat V(n,m+1);  setvalue(V, cpx(0,0));
  CpxNumMat H(m+1,m);  setvalue(H, cpx(0,0));
  
  double bnrm2 = 0;  for(int a=0; a<n; a++)    bnrm2 = bnrm2 + abs(b(a)*b(a));
  bnrm2 = sqrt(bnrm2);
  resvec.clear();
  
  CpxNumVec tmp(n);  for(int a=0; a<n; a++)    tmp(a) = 0;
  x = x0;
  double xnrm2 = 0;  for(int a=0; a<n; a++)    xnrm2 = xnrm2 + abs(x(a)*x(a));
  xnrm2 = sqrt(xnrm2);
  if(xnrm2 > 1e-16) {
    iC( (*A)(x,tmp) );
  }
  CpxNumVec r(n);  for(int a=0; a<n; a++)    r(a) = b(a) - tmp(a);
  double beta=0;  for(int a=0; a<n; a++)    beta = beta + abs(r(a)*r(a));
  beta = sqrt(beta);
  double res = beta; if(print==1) cerr<<"Iter "<<resvec.size()<<": "<<res<<endl;
  resvec.push_back(res);
  double err = res/bnrm2;
  
  int iter = 0;
  while(1) {
    for(int a=0; a<n; a++)      V(a,0) = r(a)/beta;
    int j = 0;
    CpxNumVec y;
    CpxNumMat Hj;
    while(1) {
      CpxNumVec Vj(n);      for(int a=0; a<n; a++)	Vj(a) = V(a,j);
      CpxNumVec w(n);      setvalue(w,cpx(0,0));
      iC( (*A)(Vj,w) );
      for(int k=0; k<=j; k++) {
	cpx sum = 0;	for(int a=0; a<n; a++)	  sum = sum + conj(V(a,k))*w(a);
	H(k,j) = sum;
	for(int a=0; a<n; a++)	  w(a) = w(a) - sum*V(a,k);
      }
      double nw=0;      for(int a=0; a<n; a++)	nw = nw + abs(w(a)*w(a));
      nw = sqrt(nw);
      H(j+1,j) = nw;
      for(int a=0; a<n; a++)	V(a,j+1) = w(a) / nw;
      CpxNumVec be(j+2);      for(int a=0; a<j+2; a++)	be(a) = 0;
      be(0) = beta;
      y.resize(j+1);
      Hj.resize(j+2,j+1);
      for(int a=0; a<j+2; a++)	for(int c=0; c<j+1; c++)	  Hj(a,c) = H(a,c);
      //SOLVE
      {
	int m = j+2;
	int n = j+1;
	int nrhs = 1;
	CpxNumMat Hjtmp(Hj);	//cpx* aptr = Hjtmp.data();
	int lda = j+2;
	CpxNumVec betmp(be);	//cpx* bptr = betmp.data();
	int ldb = j+2;
	DblNumVec s(j+2);
	double rcond = 0;
	int rank;
	CpxNumVec work(10*(j+2));
	int lwork = 10*(j+2);
	DblNumVec rwork(10*(j+2));
	int info;
	zgelss_(&m,&n,&nrhs,Hjtmp.data(),&lda,betmp.data(),&ldb,s.data(),&rcond,&rank,work.data(),
		&lwork,rwork.data(),&info);
	for(int a=0; a<j+1; a++)	  y(a) = betmp(a);
      }
      iC( zgemv(-1.0, Hj, y, 1, be) );
      double res=0;      for(int a=0; a<j+2; a++)	res = res + abs(be(a)*be(a));
      res = sqrt(res);      if(print==1) cerr<<"Iter "<<resvec.size()<<": "<<res<<endl;
      resvec.push_back(res);
      err = res/bnrm2;
      if((err<tol) | (j==m-1))
	break;
      j=j+1;
    }
    CpxNumMat Vj(n,j+1,false,V.data());
    iC( zgemv(1.0, Vj, y, 1.0, x) );
    CpxNumVec tmp(j+2);
    iC( zgemv(1.0, Hj, y, 0.0, tmp) );
    CpxNumMat Vj1(n,j+2,false,V.data());
    iC( zgemv(-1.0, Vj1, tmp, 1.0, r) );
    beta = 0;    for(int a=0; a<n; a++)      beta = beta + abs(r(a)*r(a));
    beta = sqrt(beta);
    if((err<tol) | (iter==maxit-1))
      break;
    iter++;
  }
  flag = (err>tol);
  relres = beta;
  niter = iter+1;
  
  return 0;
}

// ---------------------------------------------------------------------- 
// LLIN: C = A * B.  
// Fills in the dimension and the descriptor of C 
//
// Uses cblacs.h
int pdgemm(char* transa,       char* transb, 
	   DblNumMat& Aloc,    int  AlocM,    int  AlocN,    int* descA,
	   DblNumMat& Bloc,    int  BlocM,    int  BlocN,    int* descB,
	   DblNumMat& Cloc,    int& ClocM,    int& ClocN,    int* descC, 
	   int MB,             int NB,        int CONTXT){
  int ione = 1, izero = 0;
  double one = 1.0, zero = 0.0;
  int nrowa, ncola, nrowb, ncolb;
  int info;
  int nprow, npcol, myprow, mypcol;
  Cblacs_gridinfo(CONTXT,  &nprow, &npcol, &myprow, &mypcol);
  
  if( *transa == 'T' || *transa == 't' ){
    nrowa = AlocN;  ncola = AlocM;
  }
  else if( *transa == 'N' || *transa == 'n' ){
    nrowa = AlocM; ncola = AlocN;
  }
  else{
    return 1;
  }

  if( *transb == 'T' || *transb == 't' ){
    nrowb = BlocN;  ncolb = BlocM;
  }
  else if( *transb == 'N' || *transb == 'n' ){
    nrowb = BlocM;  ncolb = BlocN;
  }
  else{
    return 1;
  }
  
  if(ncola != nrowb)
    return 1;

  ClocM = nrowa;  ClocN = ncolb;
  
  int mbblck = ((ClocM+(MB-1))/MB);  //LLIN: NOTE the index
  int nbblck = ((ClocN+(NB-1))/NB);  //LLIN: NOTE the index
  int loc_blkm = (mbblck + nprow - 1 ) / nprow;
  int loc_blkn = (nbblck + npcol - 1 ) / npcol;
  int loc_m = loc_blkm * MB;
  int loc_n = loc_blkn * NB;
 
  Cloc.resize(loc_m, loc_n);   setvalue(Cloc, 0.0); //LLIN: IMPORTANT
  descinit_(descC, &ClocM, &ClocN, &MB, &NB, &izero, &izero, &CONTXT,
	    &loc_m, &info); 

  pdgemm_(transa, transb, &nrowa, &ncolb, &ncola, &one, 
	  Aloc.data(), &ione, &ione, descA,
	  Bloc.data(), &ione, &ione, descB, &zero,
	  Cloc.data(), &ione, &ione, descC);
   
  return 0;
}
