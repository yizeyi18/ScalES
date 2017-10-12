/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin

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
/// @file utility.cpp
/// @brief Utility subroutines
/// @date 2012-08-12
#include "utility.hpp"

namespace dgdft{

// *********************************************************************
// Spline functions
// *********************************************************************

void spline(int n, double* x, double* y, double* b, double* c, double* d){
  /* 
     the coefficients b(i), c(i), and d(i), i=1,2,...,n are computed
     for a cubic interpolating spline

     s(x) = y(i) + b(i)*(x-x(i)) + c(i)*(x-x(i))**2 + d(i)*(x-x(i))**3

     for  x(i) .le. x .le. x(i+1)

     input..

     n = the number of data points or knots (n.ge.2)
     x = the abscissas of the knots in strictly increasing order
     y = the ordinates of the knots

     output..

     b, c, d  = arrays of spline coefficients as defined above.

     using  p  to denote differentiation,

     y(i) = s(x(i))
     b(i) = sp(x(i))
     c(i) = spp(x(i))/2
     d(i) = sppp(x(i))/6  (derivative from the right)

     the accompanying function subprogram  seval  can be used
     to evaluate the spline.
   */
  int nm1, i;
  double t;

  for(i = 0; i < n; i++){
    b[i] = 0.0;
    c[i] = 0.0;
    d[i] = 0.0;
  }
  nm1 = n-1;
  if ( n < 2 ) {
    ErrorHandling(" SPLINE REQUIRES N >= 2!" );
  }
  if ( n < 3 ){
    b[0] = (y[1]-y[0])/(x[1]-x[0]);
    c[0] = 0;
    d[0] = 0;
    b[1] = b[0];
    c[1] = 0;
    d[1] = 0;
    return;
  }

  /*
     set up tridiagonal system

     b = diagonal, d = offdiagonal, c = right hand side.
   */ 

  d[0] = x[1] - x[0];
  c[1] = (y[1] - y[0])/d[0];
  for(i = 1; i <  nm1; i++){
    d[i] = x[i+1] - x[i];
    b[i] = 2.*(d[i-1] + d[i]);
    c[i+1] = (y[i+1] - y[i])/d[i];
    c[i] = c[i+1] - c[i];
  }

  /* 
     end  onditions.  third derivatives at  x(1)  and  x(n)
     obtained from divided differences.
   */ 
  b[0] = -d[0];
  b[n-1] = -d[n-2];
  c[0] = 0.;
  c[n-1] = 0.;
  if ( n > 3 ){
    c[0] = c[2]/(x[3]-x[1]) - c[1]/(x[2]-x[0]);
    c[n-1] = c[n-2]/(x[n-1]-x[n-3]) - c[n-3]/(x[n-2]-x[n-4]);
    c[0] = c[0]*d[0]*d[0]/(x[3]-x[0]);
    c[n-1] = -c[n-1]*d[n-2]*d[n-2]/(x[n-1]-x[n-4]);
  }

  /* forward elimination */

  for( i = 1; i < n; i++ ){
    t = d[i-1] / b[i-1];
    b[i] = b[i] - t * d[i-1];
    c[i] = c[i] - t * c[i-1];
  }

  /* backward substitution */
  c[n-1] = c[n-1] / b[n-1];
  for( i = n-2; i >= 0; i-- ){
    c[i] = (c[i] - d[i]*c[i+1]) / b[i];
  }

  /* compute polynomial coefficients */
  b[n-1] = (y[n-1] - y[nm1-1])/d[nm1-1] + d[nm1-1]*(c[nm1-1] + 2.*c[n-1]);
  for(i = 0; i < nm1; i++){
    b[i] = (y[i+1] - y[i])/d[i] - d[i]*(c[i+1] + 2.*c[i]);
    d[i] = (c[i+1] - c[i])/d[i];
    c[i] = 3.*c[i];
  }
  c[n-1] = 3.*c[n-1];
  d[n-1] = d[n-2];

}


void seval(double* v, int m, double* u, int n, double* x, 
    double* y, double* b, double* c, double* d){

  /* ***************************************************
   * This SPLINE function is designed specifically for the interpolation
   * part for pseudopotential generation in the electronic structure
   * calculation.  Therefore if u is outside the range [min(x), max(x)],
   * the corresponding v value will be an extrapolation.
   * ***************************************************

   this subroutine evaluates the  spline function

   seval = y(i) + b(i)*(u-x(i)) +  (i)*(u-x(i))**2 + d(i)*(u-x(i))**3

   where  x(i) .lt. u .lt. x(i+1), using horner's rule

   if  u .lt. x(1) then  i = 1  is used.
   if  u .ge. x(n) then  i = n  is used.

   input..

   m = the number of output data points
   n = the number of input data points
   u = the abs issa at which the spline is to be evaluated
   v = the value of the spline function at u
   x,y = the arrays of data abs issas and ordinates
   b,c,d = arrays of spline coefficients  omputed by spline

   if  u  is not in the same interval as the previous  all, then a
   binary sear h is performed to determine the proper interval.
   */

  int i, j, k, l;
  double dx;
  if( n < 2 ){
    ErrorHandling(" SPLINE REQUIRES N >= 2!" );
  }

  for(l = 0; l < m; l++){
    v[l] = 0.0;
  }

  for(l = 0; l < m; l++){
    i = 0;
    if( u[l] < x[0] ){
      i = 0;
    }
    else if( u[l] > x[n-1] ){
      i = n-1;
    }
    else{
      /* calculate the index of u[l] */
      i = 0;
      j = n;
      while( j > i+1 ) {
        k = (i+j)/2;
        if( u[l] < x[k] ) j = k;
        if( u[l] >= x[k] ) i = k;
      }
    }
    /* evaluate spline */
    dx = u[l] - x[i];
    v[l] = y[i] + dx*(b[i] + dx*(c[i] + dx*d[i]));
  }
  return;
}

// *********************************************************************
// Generating grids in a domain
// *********************************************************************

void GenerateLGLMeshWeightOnly(double* x, double* w, int Nm1)
{
  int i, j;
  double pi = 4.0 * atan(1.0);
  double err, tol = 1e-15;
  std::vector<double> xold;
  int N = Nm1;
  int N1 = N + 1;
  // Only for the three-term recursion
  DblNumMat PMat(N1, 3);
  SetValue( PMat, 0.0 );

  xold.resize(N1);

  double *P0 = PMat.VecData(0);
  double *P1 = PMat.VecData(1);
  double *P2 = PMat.VecData(2);

  for (i=0; i<N1; i++){
    x[i] = cos(pi*(N1-i-1)/(double)N);
  }

  do{
    for (i=0; i<N1; i++){
      xold[i] = x[i];
      P0[i] = 1.0;
      P1[i] = x[i];
    }
    for (j=2; j<N1; j++){
      for (i=0; i<N1; i++){
        P2[i] = ((2.0*j-1.0)*x[i]*P1[i] - (j-1)*P0[i])/j;
        P0[i] = P1[i];
        P1[i] = P2[i];
      }
    }

    for (i=0; i<N1; i++){
      x[i] = xold[i] - (x[i]*P1[i] - P0[i])/(N1*P1[i]);
    }

    err = 0.0;
    for (i=0; i<N1; i++){
      if (err < fabs(xold[i] - x[i])){
        err = fabs(xold[i] - x[i]);
      }
    }
  } while(err>tol);

  for (i=0; i<N1; i++){
    w[i] = 2.0/(N*N1*P1[i]*P1[i]);
  }

  return;
}

void GenerateLGLMeshWeightOnly(
    DblNumVec&         x, 
    DblNumVec&         w, 
    Int                N)
{
  x.Resize( N );
  w.Resize( N );
  GenerateLGLMeshWeightOnly( x.Data(), w.Data(), N-1 );

  return;
}


void GenerateLGL(double* x, double* w, double* P, double* D, int Nm1)
{
  int i, j;
  double pi = 4.0 * atan(1.0);
  double err, tol = 1e-15;
  std::vector<double> xold;
  int N = Nm1;
  int N1 = N + 1;

  xold.resize(N1);

  for (i=0; i<N1; i++){
    x[i] = cos(pi*(N1-i-1)/(double)N);
  }

  for (j=0; j<N1; j++){
    for (i=0; i<N1; i++){
      P[j*N1+i] = 0;
    }
  }

  do{
    for (i=0; i<N1; i++){
      xold[i] = x[i]; 
      P[i] = 1.0; 
      P[N1+i] = x[i];
    }
    for (j=2; j<N1; j++){
      for (i=0; i<N1; i++){
        P[j*N1+i] = ((2*j-1)*x[i]*P[(j-1)*N1+i] - (j-1)*P[(j-2)*N1+i])/j;
      }
    }

    for (i=0; i<N1; i++){
      x[i] = xold[i] - (x[i]*P[N*N1+i] - P[(N-1)*N1+i])/(N1*P[N*N1+i]);
    }

    err = 0.0;
    for (i=0; i<N1; i++){
      if (err < fabs(xold[i] - x[i])){
        err = fabs(xold[i] - x[i]);
      }
    }
  } while(err>tol);

  for (i=0; i<N1; i++){
    w[i] = 2.0/(N*N1*P[N*N1+i]*P[N*N1+i]);
  }

  for (j=0; j<N1; j++){
    for (i=0; i<N1; i++){
      if (i!=j) {
        D[j*N1+i] = P[N*N1+i]/P[N*N1+j]/(x[i] - x[j]);
      }
      else if (i==0){
        D[j*N1+i] = - N*N1/4.0;
      }
      else if (i==N1-1){
        D[j*N1+i] = N*N1/4.0;
      }
      else{
        D[j*N1+i] = 0.0;      
      }
    }
  }

  return;
}

void GenerateLGL(
    DblNumVec&         x, 
    DblNumVec&         w, 
    DblNumMat&         P,
    DblNumMat&         D,
    Int                N)
{
  x.Resize( N );
  w.Resize( N );
  P.Resize( N, N );
  D.Resize( N, N );
  GenerateLGL( x.Data(), w.Data(), P.Data(), D.Data(), N-1 );

  return;
}



void
UniformMesh ( const Domain &dm, std::vector<DblNumVec> &gridpos )
{
  gridpos.resize(DIM);
  for (Int d=0; d<DIM; d++) {
    gridpos[d].Resize(dm.numGrid[d]);
    Real h = dm.length[d] / dm.numGrid[d];
    for (Int i=0; i < dm.numGrid[d]; i++) {
      gridpos[d](i) = dm.posStart[d] + Real(i)*h;
    }
  }

  return ;
}        // -----  end of function UniformMesh  ----- 


void
UniformMeshFine ( const Domain &dm, std::vector<DblNumVec> &gridpos )
{
  gridpos.resize(DIM);
  for (Int d=0; d<DIM; d++) {
    gridpos[d].Resize(dm.numGridFine[d]);
    Real h = dm.length[d] / dm.numGridFine[d];
    for (Int i=0; i < dm.numGridFine[d]; i++) {
      gridpos[d](i) = dm.posStart[d] + Real(i)*h;
    }
  }

  return ;
}        // -----  end of function UniformMesh  ----- 


void
LGLMesh ( const Domain &dm, const Index3& numGrid, std::vector<DblNumVec> &gridpos )
{
  gridpos.resize(DIM);
  for (Int d=0; d<DIM; d++) {
    gridpos[d].Resize( numGrid[d] );

    DblNumVec  mesh;
    DblNumVec  dummyW;
    DblNumMat  dummyP, dummyD;
    GenerateLGL( mesh, dummyW, dummyP, dummyD, numGrid[d] );
    for( Int i = 0; i < numGrid[d]; i++ ){
      gridpos[d][i] = dm.posStart[d] + 
        ( mesh[i] + 1.0 ) * dm.length[d] * 0.5;
    }
  }

  return ;
}        // -----  end of function LGLMesh  ----- 

// *********************************************************************
// IO functions
// *********************************************************************
//---------------------------------------------------------
Int SeparateRead(std::string name, std::istringstream& is)
{
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  char filename[100];
  sprintf(filename, "%s_%d_%d", name.c_str(), mpirank, mpisize);  
  std::ifstream fin(filename);
  if( !fin.good() ){
    ErrorHandling( "File cannot be open!" );
  }

  is.str( std::string(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>()) );
  fin.close();
  return 0;
}

//---------------------------------------------------------
Int SeparateRead(std::string name, std::istringstream& is, Int outputIndex)
{
  char filename[100];
  sprintf(filename, "%s_%d", name.c_str(), outputIndex);
  std::ifstream fin(filename);
  if( !fin.good() ){
    ErrorHandling( "File cannot be open!" );
  }

  is.str( std::string(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>()) );
  fin.close();
  return 0;
}


//---------------------------------------------------------
Int SeparateWrite(std::string name, std::ostringstream& os)
{
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  char filename[100];
  sprintf(filename, "%s_%d_%d", name.c_str(), mpirank, mpisize);
  std::ofstream fout(filename);
  if( !fout.good() ){
    ErrorHandling( "File cannot be open!" );
  }
  fout<<os.str();
  fout.close();
  return 0;
}


//---------------------------------------------------------
Int SeparateWrite(std::string name, std::ostringstream& os, Int outputIndex)
{
  char filename[100];
  sprintf(filename, "%s_%d", name.c_str(), outputIndex);
  std::ofstream fout(filename);
  if( !fout.good() ){
    ErrorHandling( "File cannot be open!" );
  }
  fout<<os.str();
  fout.close();
  return 0;
}

//---------------------------------------------------------
Int SharedRead(std::string name, std::istringstream& is)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  std::vector<char> tmpstr;
  if(mpirank==0) {
    std::ifstream fin(name.c_str());
    if( !fin.good() ){
      ErrorHandling( "File cannot be open!" );
    }
    //std::string str(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    //tmpstr.insert(tmpstr.end(), str.begin(), str.end());
    tmpstr.insert(tmpstr.end(), std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    fin.close();
    int size = tmpstr.size();    
    MPI_Bcast((void*)&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)&(tmpstr[0]), size, MPI_BYTE, 0, MPI_COMM_WORLD);
  } else {
    int size;
    MPI_Bcast((void*)&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    tmpstr.resize(size);
    MPI_Bcast((void*)&(tmpstr[0]), size, MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  is.str( std::string(tmpstr.begin(), tmpstr.end()) );
  //
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

//---------------------------------------------------------
Int SharedWrite(std::string name, std::ostringstream& os)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  if(mpirank==0) {
    std::ofstream fout(name.c_str());
    if( !fout.good() ){
      ErrorHandling( "File cannot be open!" );
    }
    fout<<os.str();
    fout.close();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}


//---------------------------------------------------------
Int SeparateWriteAscii(std::string name, std::ostringstream& os)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  char filename[100];
  sprintf(filename, "%s_%d_%d", name.c_str(), mpirank, mpisize);
  std::ofstream fout(filename, std::ios::trunc);
  if( !fout.good() ){
    ErrorHandling( "File cannot be open!" );
  }
  fout<<os;
  fout.close();
  //
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}
#ifdef _COMPLEX_
void AlltoallForward( CpxNumMat& A, CpxNumMat& B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = A.m();
  Int widthTemp = A.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

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
  
  CpxNumVec sendbuf(height*widthLocal); 
  CpxNumVec recvbuf(heightLocal*width);
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

  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      sendbuf[sendk(i, j)] = A(i, j); 
    }
  }
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE_COMPLEX, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE_COMPLEX, comm );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      B(i, j) = recvbuf[recvk(i, j)];
    }
  }


  return ;
}        // -----  end of function AlltoallForward ----- 


#endif

void AlltoallForward( DblNumMat& A, DblNumMat& B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = A.m();
  Int widthTemp = A.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

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

  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      sendbuf[sendk(i, j)] = A(i, j); 
    }
  }
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, comm );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      B(i, j) = recvbuf[recvk(i, j)];
    }
  }


  return ;
}        // -----  end of function AlltoallForward ----- 
#ifdef _COMPLEX_
// AlltoallBackward change from G-para to Band-para
void AlltoallBackward( CpxNumMat& A, CpxNumMat& B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = B.m();
  Int widthTemp = B.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

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

  CpxNumVec sendbuf(height*widthLocal); 
  CpxNumVec recvbuf(heightLocal*width);
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

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvbuf[recvk(i, j)] = A(i, j);
    }
  }
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE_COMPLEX, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE_COMPLEX, comm );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      B(i, j) = sendbuf[sendk(i, j)]; 
    }
  }


  return ;
}        // -----  end of function AlltoallBackward ----- 


#endif
void AlltoallBackward( DblNumMat& A, DblNumMat& B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = B.m();
  Int widthTemp = B.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

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

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvbuf[recvk(i, j)] = A(i, j);
    }
  }
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, comm );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      B(i, j) = sendbuf[sendk(i, j)]; 
    }
  }


  return ;
}        // -----  end of function AlltoallBackward ----- 

// serialize/deserialize the pseudopot

Int serialize(const PseudoPot& val, std::ostream& os, const std::vector<Int>& mask)
{
  serialize( val.pseudoCharge,        os, mask );
  serialize( val.vnlList,             os, mask );
  serialize( val.vnlListFine,         os, mask );
  // No need to serialize the communicator
  return 0;
}

Int deserialize(PseudoPot& val, std::istream& is, const std::vector<Int>& mask)
{
  deserialize( val.pseudoCharge,      is, mask );
  deserialize( val.vnlList,           is, mask );
  deserialize( val.vnlListFine,       is, mask );
  return 0;
}

void findMin(NumMat<Real>& A, const int Dim, NumVec<Int>& Imin){
  int n = A.n_;
  int m = A.m_;
  if (Dim == 0){ 
    Imin.Resize(n);
    Int* Iptr = Imin.Data();
    for (int i = 0; i < n; i++){
      double* temp = A.VecData(i);
      Iptr[i] = std::distance(temp,std::min_element(temp,temp+m));
    }
  } else {
    Real* Aptr = A.Data();
    DblNumVec amin(m,1,Aptr);
    Imin.Resize(m);
    SetValue(Imin,0);
    Int* Iptr = Imin.Data();
    Real* aptr = amin.Data();
    for (int j = 1; j < n; j++){
      for (int i = 0; i < m; i++){
        if (Aptr[i+j*m] < aptr[i]){
          aptr[i] = Aptr[i+j*m];
          Iptr[i] = j;
        }
      }
    }
  }
}

void findMin(NumMat<Real>& A, const int Dim, NumVec<Int>& Imin, NumVec<Real>& amin){
  int n = A.n_;
  int m = A.m_;
  if (Dim == 0){ 
    Imin.Resize(n);
    amin.Resize(n);
    Int* Iptr = Imin.Data();
    Real* aptr = amin.Data();
    Int d;
    for (int i = 0; i < n; i++){
      double* temp = A.VecData(i);
      d = std::distance(temp,std::min_element(temp,temp+m));
      Iptr[i] = d;
      aptr[i] = temp[d];
    }
  } else {
    Real* Aptr = A.Data();
    amin = DblNumVec(m,1,Aptr);
    Imin.Resize(m);
    SetValue(Imin,0);
    Int* Iptr = Imin.Data();
    Real* aptr = amin.Data();
    for (int j = 1; j < n; j++){
      for (int i = 0; i < m; i++){
        if (Aptr[i+j*m] < aptr[i]){
          aptr[i] = Aptr[i+j*m];
          Iptr[i] = j;
        }
      }
    } 
  }
}

void pdist2(NumMat<Real>& A, NumMat<Real>& B, NumMat<Real>& D){
  D.Resize(A.m_, B.m_);
  Int Am = A.m_;
  Int Bm = B.m_;
  Real* Dptr = D.Data();
  Real* Aptr = A.Data();
  Real* Bptr = B.Data();
  
  Real d1,d2,d3;
  for (int j = 0; j < Bm;  j++) {
    for (int i = 0; i < Am; i++) {
      d1 = Aptr[i] - Bptr[j];
      d2 = Aptr[i+Am] - Bptr[j+Bm];
      d3 = Aptr[i+2*Am] - Bptr[j+2*Bm];
      Dptr[j*Am+i] = d1*d1 + d2*d2 + d3*d3;
    }
  }
}

void unique(NumVec<Int>& Index){
  Sort(Index);
  Int* Ipt = Index.Data();
  Int* it = std::unique(Ipt, Ipt + Index.m_);
  std::vector<Int> temp(Ipt, it); 
  delete[] Index.Data();
  Index.m_ = temp.size();
  Index.data_ = new Int[Index.m_];
  Ipt = Index.Data();
  for (int i = 0; i < Index.m_; i++){
    Ipt[i] = temp[i];
  }
}

void KMEAN(Int n, NumVec<Real>& weight, Int& rk, Real KmeansTolerance, 
    Int KmeansMaxIter, Real DFTolerance,  const Domain &dm, Int* piv)
{
  MPI_Barrier(dm.comm);
  int mpirank; MPI_Comm_rank(dm.comm, &mpirank);
  int mpisize; MPI_Comm_size(dm.comm, &mpisize);
  
  Real timeSta, timeEnd;
  Real timeSta2, timeEnd2;
  Real timeDist=0.0;
  Real timeMin=0.0;
  Real timeComm=0.0;
  Real time0 = 0.0;

  GetTime(timeSta);
  Real* wptr = weight.Data();
  int npt;
  std::vector<int> index(n);
  double maxW = 0.0;
  if(DFTolerance > 1e-16){
    maxW = findMax(weight);
    npt = 0;
    for (int i = 0; i < n;i++){
      if (wptr[i] > DFTolerance*maxW){
        index[npt] = i;
        npt++;
      }
    }
    index.resize(npt);
  } else {
    npt = n;
    for (int i = 0; i < n; i++){
      index[i] = i;
    }
  }

  if(npt < rk){
    int k0 = 0;
    int k1 = 0;
    for (int i = 0; i < npt; i++){
      if ( i == index[k0] ){
        piv[k0] = i;
        k0 = std::min(k0+1, rk-1);
      } else {
        piv[npt+k1] = i;
        k1++;
      }
    }
    std::random_shuffle(piv+npt,piv+n);
    return;
  } 

  int nptLocal = n/mpisize;
  int res = n%mpisize;
  if (mpirank < res){
    nptLocal++;
  }
  int indexSta = mpirank*nptLocal;
  if (mpirank >= res){
    indexSta += res;
  }
  std::vector<int> indexLocal(nptLocal);
  DblNumMat GridLocal(nptLocal,3);
  Real* glptr = GridLocal.Data();
  DblNumVec weightLocal(nptLocal);
  Real* wlptr = weightLocal.Data();

  int tmp;
  double len[3];
  double dx[3];
  int nG[3];
  for (int i = 0; i < 3; i++){
    len[i] = dm.length[i];
    nG[i] = dm.numGrid[i];
    dx[i] = len[i]/nG[i];
  }

  for (int i = 0; i < nptLocal; i++){
    tmp = index[indexSta+i];
    indexLocal[i] = tmp;
    wlptr[i] = wptr[tmp];
    glptr[i] = (tmp%nG[1])*dx[0];
    glptr[i+nptLocal] = (tmp%(nG[1]*nG[2])-glptr[i])/nG[1]*dx[2];
    glptr[i+2*nptLocal] = (tmp-glptr[i]-glptr[i+nptLocal]*nG[1])/(nG[1]*nG[2])*dx[2];
  }
  DblNumMat C(rk,3);
  Real* Cptr = C.Data();
  std::vector<int> Cind = index;
  std::vector<int> Cinit;
  Cinit.reserve(rk);
  std::random_shuffle(Cind.begin(), Cind.end());
  GetTime(timeEnd);
  statusOFS << "After Setup: " << timeEnd-timeSta << "[s]" << std::endl;

  if (piv[0]!= piv[1]){
    statusOFS << "Used previous initialization." << std::endl;
    for (int i = 0; i < rk; i++){
      if(wptr[piv[i]] > DFTolerance*maxW){
        Cinit.push_back(piv[i]);
      }
    }
    statusOFS << "Reusable pivots: " << Cinit.size() << std::endl;
    GetTime(timeEnd);
    statusOFS << "After load: " << timeEnd-timeSta << "[s]" << std::endl;
    int k = 0;
    while(Cinit.size() < rk && k < npt){
      bool flag = 1;
      int it = 0; 
      while (flag && it < Cinit.size()){
        if (Cinit[it] == Cind[k]){
          flag = 0;
        }
        it++;
      }
      if(flag){
        Cinit.push_back(Cind[k]);
      }
      k++;
    }
  } else {
    Cinit = Cind;
    Cinit.resize(rk);
  }
  GetTime(timeEnd);
  statusOFS << "After Initialization: " << timeEnd-timeSta << "[s]" << std::endl;

  for (int i = 0; i < rk; i++){
    tmp = Cinit[i];
    Cptr[i] = (tmp%nG[1])*dx[0];
    Cptr[i+rk] = (tmp%(nG[1]*nG[2])-Cptr[i])/nG[1]*dx[2];
    Cptr[i+2*rk] = (tmp-Cptr[i]-Cptr[i+rk]*nG[1])/(nG[1]*nG[2])*dx[2];
  }

  int s = 0;
  int flag = n;
  int flagrecv = 0;
  IntNumVec label(nptLocal);
  Int* lbptr = label.Data();
  IntNumVec last(nptLocal);
  Int* laptr = last.Data();
  DblNumVec count(rk);
  Real* cptr = count.Data();
  DblNumMat DLocal(nptLocal, rk);
  DblNumMat Crecv(rk,3);
  Real* Crptr = Crecv.Data();
  DblNumVec countrecv(rk);
  Real* crptr = countrecv.Data();

  GetTime(timeSta2);
  pdist2(GridLocal, C, DLocal);
  GetTime(timeEnd2);
  timeDist += (timeEnd2-timeSta2);
  
  GetTime(timeSta2);
  findMin(DLocal, 1, label);
  GetTime(timeEnd2);
  timeMin+=(timeEnd2-timeSta2);
  lbptr = label.Data();

  double maxF = KmeansTolerance*n;
  while (flag > maxF && s < KmeansMaxIter){
    SetValue(count, 0.0);
    SetValue(C, 0.0);
    for (int i = 0; i < nptLocal; i++){
      tmp = lbptr[i];
      cptr[tmp] += wlptr[i];
      Cptr[tmp] += wlptr[i]*glptr[i];
      Cptr[tmp+rk] += wlptr[i]*glptr[i+nptLocal];
      Cptr[tmp+2*rk] += wlptr[i]*glptr[i+2*nptLocal];
    }
    MPI_Barrier(dm.comm);
    GetTime(timeSta2);
    MPI_Reduce(cptr, crptr, rk, MPI_DOUBLE, MPI_SUM, 0, dm.comm);
    MPI_Reduce(Cptr, Crptr, rk*3, MPI_DOUBLE, MPI_SUM, 0, dm.comm);
    GetTime(timeEnd2);
    timeComm += (timeEnd2-timeSta2);

    GetTime(timeSta2);
    if (mpirank == 0){
      tmp = rk;
      for (int i = 0; i < rk; i++){
        if(crptr[i] != 0.0){
          Crptr[i] = Crptr[i]/crptr[i];
          Crptr[i+tmp] = Crptr[i+tmp]/crptr[i];
          Crptr[i+2*tmp] = Crptr[i+2*tmp]/crptr[i];
        } else {
          rk--;
          Crptr[i] = Crptr[rk];
          Crptr[i+tmp] = Crptr[rk+tmp];
          Crptr[i+2*tmp] = Crptr[rk+2*tmp];
          crptr[i] = crptr[rk];
          i--;
        }
      }
      C.Resize(rk,3);
      Cptr = C.Data();
      for (int i = 0; i < rk; i++){
        Cptr[i] = Crptr[i];
        Cptr[i+rk] = Crptr[i+tmp];
        Cptr[i+2*rk] = Crptr[i+2*tmp];
      }
    }
    GetTime(timeEnd2);
    time0 += (timeEnd2-timeSta2);

    MPI_Bcast(&rk, 1, MPI_INT, 0, dm.comm);
    if (mpirank != 0){
      C.Resize(rk,3);
      Cptr= C.Data();
    }
    GetTime(timeSta2);
    MPI_Bcast(Cptr, rk*3, MPI_DOUBLE, 0, dm.comm);
    GetTime(timeEnd2);
    timeComm += (timeEnd2-timeSta2);

    count.Resize(rk);
    GetTime(timeSta2);
    pdist2(GridLocal, C, DLocal);
    GetTime(timeEnd2);
    timeDist += (timeEnd2-timeSta2);

    last = label;
    laptr = last.Data();
    GetTime(timeSta2);
    findMin(DLocal, 1, label);
    GetTime(timeEnd2);
    timeMin +=(timeEnd2-timeSta2);
    lbptr = label.Data();
    flag = 0;
    for (int i = 0; i < label.m_; i++){
      if(laptr[i]!=lbptr[i]){
        flag++;
      }
    }
    MPI_Barrier(dm.comm);
    MPI_Reduce(&flag, &flagrecv, 1, MPI_INT, MPI_SUM, 0, dm.comm);
    MPI_Bcast(&flagrecv, 1, MPI_INT, 0, dm.comm);
    flag = flagrecv;
    statusOFS<< flag << " ";
    s++;
  }
  statusOFS << std::endl << "Converged in " << s << " iterations." << std::endl;
  GetTime(timeEnd);
  statusOFS << "After iteration: " << timeEnd-timeSta << "[s]" << std::endl;
  IntNumVec Imin(rk);
  Int* imptr = Imin.Data();
  DblNumVec amin(rk);
  findMin(DLocal, 0, Imin, amin);
  for (int i = 0; i < rk; i++){
    imptr[i] = indexLocal[imptr[i]];
  }
  IntNumMat Iminrecv(rk, mpisize);
  Int* imrptr = Iminrecv.Data();
  DblNumMat aminrecv(rk, mpisize);
  MPI_Barrier(dm.comm);
  
  GetTime(timeSta2);
  MPI_Gather(imptr, rk, MPI_INT, imrptr, rk, MPI_INT, 0, dm.comm);
  MPI_Gather(amin.Data(), rk, MPI_DOUBLE, aminrecv.Data(), rk, MPI_DOUBLE, 0, dm.comm);
  GetTime(timeEnd2);
  timeComm += (timeEnd2-timeSta2);
  IntNumVec pivTemp(rk);
  Int* pvptr = pivTemp.Data();
  
  GetTime(timeSta2);
  if (mpirank == 0) {
    findMin(aminrecv,1,pivTemp);
    for (int i = 0; i <rk; i++){
      pvptr[i] = imrptr[i+rk*pvptr[i]];
    }
  }
  GetTime(timeEnd2);
  time0 += (timeEnd2-timeSta2);

  GetTime(timeSta2);
  MPI_Bcast(pvptr, rk, MPI_INT, 0, dm.comm);
  GetTime(timeEnd2);
  timeComm += (timeEnd2-timeSta2);

  unique(pivTemp);
  pvptr = pivTemp.Data();
  rk = pivTemp.m_;
  int k0 = 0;
  int k1 = 0;
  for (int i = 0; i < n; i++){
    if(i == pvptr[k0]){
      piv[k0] = i;
      k0 = std::min(k0+1, rk-1);
    } else {
      piv[rk+k1] = i;
      k1++;
    }
  }
  statusOFS << "Dist time: " << timeDist << "[s]" << std::endl;
  statusOFS << "Min time: " << timeMin << "[s]" << std::endl;
  statusOFS << "Comm time: " << timeComm << "[s]" << std::endl;
  statusOFS << "core0 time: " << time0 << "[s]" << std::endl;
}

}  // namespace dgdft
