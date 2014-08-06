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
/// @file ex38.cpp
/// @brief Simple test of the matrix matrix multiplication routine.
/// @date 2014-06-12
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;

// SCALAPACK routines
extern "C"{
void SCALAPACK(descinit)(Int* desc, const Int* m, const Int * n, const Int* mb,
    const Int* nb, const Int* irsrc, const Int* icsrc,
    const Int* contxt, const Int* lld, Int* info);

void SCALAPACK(pdgemm)(const char* transA, const char* transB,
    const Int* m, const Int* n, const Int* k,
    const double* alpha,
    const double* A, const Int* ia, const Int* ja, const Int* desca, 
    const double* B, const Int* ib, const Int* jb, const Int* descb,
    const double* beta,
    double* C, const Int* ic, const Int* jc, const Int* descc,
    const Int* contxt);

}

void Usage(){
  std::cout 
		<< "ex38 " << std::endl;
}


int main(int argc, char **argv) 
{
	MPI_Init(&argc, &argv);
	int mpirank, mpisize;
	MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
	MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
	Real timeSta, timeEnd;

	if( mpirank == 0 )
		Usage();


	try
	{
		SetRandomSeed(mpirank);

    Int height = 100000, width = 100;

    Int widthBlocksize = width / mpisize;
    Int heightBlocksize = height / mpisize;
    Int widthLocal = widthBlocksize;
    Int heightLocal = heightBlocksize;

    if(mpirank < (width % mpisize)){
      widthLocal = widthBlocksize + 1;
    }
    
    if(mpirank == (mpisize - 1)){
      heightLocal = heightBlocksize + height % mpisize;
    }

    DblNumMat X1(height, widthLocal), Y1(height, widthLocal);
    DblNumMat X1TY1(width, width);
    DblNumMat X1TX1(width, width);

    UniformRandom( X1 );
    UniformRandom( Y1 );

    DblNumMat X2(heightLocal, width), Y2(heightLocal, width);
    DblNumMat X2TY2(width, width);
    DblNumMat X2TX2(width, width);
    
    UniformRandom( X2 );
    UniformRandom( Y2 );
   
    DblNumMat X3(heightLocal, width), Y3(heightLocal, width);
    DblNumMat X3TY3(width, width);
    DblNumMat X3TX3(width, width);
    
    UniformRandom( X3 );
    UniformRandom( Y3 );
    
    Int descX1[9];
    Int desc_width[9];

    Int nprow = 1;
    Int npcol = mpisize;
    Int myrow, mycol;

    Int contxt;
    Cblacs_get(0, 0, &contxt);
    Cblacs_gridinit(&contxt, "C", nprow, npcol); 
    Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);

    Int info;
    Int irsrc = 0;
    Int icsrc = 0;
    Int MBX1 = height;
    Int NBX1 = 1;
    SCALAPACK(descinit)(&descX1[0], &height, &width, &MBX1, &NBX1, &irsrc, &icsrc, &contxt, &height, &info);
    SCALAPACK(descinit)(&desc_width[0], &width, &width, &width, &width, &irsrc, &icsrc, &contxt, &width, &info);

    
    Real timeSta, timeEnd;
    char TT = 'T';
    char NN = 'N';
    double D_ONE = 1.0;
    double D_ZERO = 0.0;
    int I_ONE = 1;

    
    MPI_Barrier( MPI_COMM_WORLD );

    GetTime( timeSta );
    SCALAPACK(pdgemm)(&TT, &NN, &width, &width, &height, 
        &D_ONE,
        X1.Data(), &I_ONE, &I_ONE, &descX1[0],
        X1.Data(), &I_ONE, &I_ONE, &descX1[0], 
        &D_ZERO,
        X1TX1.Data(), &I_ONE, &I_ONE, &desc_width[0], 
        &contxt );
    GetTime( timeEnd );

    MPI_Barrier( MPI_COMM_WORLD );

    if ( mpirank == 0) {
      std::cout << "The time for pdgemm X1'*X1 is " << timeEnd - timeSta << " sec." << std::endl;
    }

    SetValue( X2, 0.0 ); 
    int sendk = height*widthLocal;
    int recvk = heightLocal*width;
   
    double sendbuf[sendk]; 
    double recvbuf[recvk];
    int sendcounts[mpisize];
    int recvcounts[mpisize];
    int senddispls[mpisize];
    int recvdispls[mpisize];

    for( Int k = 0; k < mpisize; k++ ){ 
      if( k < (mpisize - 1)){
        sendcounts[k] = heightBlocksize * widthLocal;
      }
      else {
        sendcounts[mpisize - 1] = (heightBlocksize + (height % mpisize)) * widthLocal;  
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

   
    GetTime( timeSta );
    int kk;

    if((height % heightBlocksize) == 0){
      for( Int j = 0; j < widthLocal; j++ ){ 
        for( Int i = 0; i < height; i++ ){
          kk = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
          sendbuf[kk] = X1(i, j);
        } 
      }
    }
    else{
      for( Int j = 0; j < widthLocal; j++ ){ 
        for( Int i = 0; i < height; i++ ){
          if((i / heightBlocksize) < (mpisize - 1)){
            kk = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
          }
          else {
            kk = senddispls[mpisize -1] + j * (heightBlocksize + height % heightBlocksize) 
              + (i - (mpisize - 1) * heightBlocksize) % (heightBlocksize + height % heightBlocksize);
          }
          sendbuf[kk] = X1(i, j);
        }
      }
    }
    
    GetTime( timeEnd );
    double t1 = timeEnd - timeSta;
    
    GetTime( timeSta );
    MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, MPI_COMM_WORLD );
    GetTime( timeEnd );

    if ( mpirank == 0) {
      std::cout << "The time for Alltoallv X1 to X2 is " << timeEnd - timeSta << " sec." << std::endl;
    }    

    GetTime( timeSta );
    for( Int j = 0; j < width; j++ ){ 
      for( Int i = 0; i < heightLocal; i++ ){
        kk = recvdispls[j % mpisize] + (j / mpisize) * heightLocal + i;
        X2(i, j) = recvbuf[kk];
      }
    }
    GetTime( timeEnd );

    if ( mpirank == 0) {
    std::cout << "The time for X1 to X2 is " << t1 + timeEnd - timeSta << " sec." << std::endl;
    }    
   
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X2.Data(), heightLocal, X2.Data(), heightLocal, 0.0, X2TX2.Data(), width );
    GetTime( timeEnd );
    
    if ( mpirank == 0) {
    std::cout << "The time for Gemm X2'*X2 is " << timeEnd - timeSta << " sec." << std::endl;
    }

    DblNumMat X2TX2Temp(width, width);
    SetValue( X2TX2Temp, 0.0 ); 
   
    GetTime( timeSta );
    MPI_Allreduce( X2TX2.Data(), X2TX2Temp.Data(), width*width, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    GetTime( timeEnd );

    if ( mpirank == 0) {
    std::cout << "The time for Allreduce X2'X2 is " << timeEnd - timeSta << " sec." << std::endl;
    }

    if ( mpirank == 0) {
    
    double sum;
    for( Int j = 0; j < width; j++ ){ 
      for( Int i = 0; i < width; i++ ){ 
        sum = sum + std::abs( X2TX2Temp(i, j) - X1TX1(i, j)) ;
      }
    }
    
    std::cout << "Sum of X2'X2 - X1'X1 is " << sum << std::endl;
    
    }



    GetTime( timeSta );
    AlltoallForward (X1, X3, MPI_COMM_WORLD);
    GetTime( timeEnd );

    if ( mpirank == 0) {
    std::cout << "The time for X1 to X3 is " << timeEnd - timeSta << " sec." << std::endl;
    }    
   
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X3.Data(), heightLocal, X3.Data(), heightLocal, 0.0, X3TX3.Data(), width );
    GetTime( timeEnd );
    
    if ( mpirank == 0) {
    std::cout << "The time for Gemm X3'*X3 is " << timeEnd - timeSta << " sec." << std::endl;
    }

    DblNumMat X3TX3Temp(width, width);
    SetValue( X3TX3Temp, 0.0 ); 
   
    GetTime( timeSta );
    MPI_Allreduce( X3TX3.Data(), X3TX3Temp.Data(), width*width, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    GetTime( timeEnd );

    if ( mpirank == 0) {
    std::cout << "The time for Allreduce X3'X3 is " << timeEnd - timeSta << " sec." << std::endl;
    }

    if ( mpirank == 0) {
    
    double sum;
    for( Int j = 0; j < width; j++ ){ 
      for( Int i = 0; i < width; i++ ){ 
        sum = sum + std::abs( X3TX3Temp(i, j) - X1TX1(i, j)) ;
      }
    }
    
    std::cout << "Sum of X3'X3 - X1'X1 is " << sum << std::endl;
    
    }

    GetTime( timeSta );
    AlltoallBackward (X3, X1, MPI_COMM_WORLD);
    GetTime( timeEnd );

    if ( mpirank == 0) {
    std::cout << "The time for X3 to X1 is " << timeEnd - timeSta << " sec." << std::endl;
    }    
    
    GetTime( timeSta );
    SCALAPACK(pdgemm)(&TT, &NN, &width, &width, &height, 
        &D_ONE,
        X1.Data(), &I_ONE, &I_ONE, &descX1[0],
        X1.Data(), &I_ONE, &I_ONE, &descX1[0], 
        &D_ZERO,
        X1TX1.Data(), &I_ONE, &I_ONE, &desc_width[0], 
        &contxt );
    GetTime( timeEnd );

    MPI_Barrier( MPI_COMM_WORLD );

    if ( mpirank == 0) {
      std::cout << "The time for pdgemm X1'*X1 is " << timeEnd - timeSta << " sec." << std::endl;
    }

    
    if ( mpirank == 0) {
    
    double sum;
    for( Int j = 0; j < width; j++ ){ 
      for( Int i = 0; i < width; i++ ){ 
        sum = sum + std::abs( X3TX3Temp(i, j) - X1TX1(i, j)) ;
      }
    }
    
    std::cout << "Sum of X3'X3 - X1'X1 is " << sum << std::endl;
    
    }
   
  
 
    Cblacs_gridexit	(	contxt );	

	}
	catch( std::exception& e )
	{
		std::cerr << " caught exception with message: "
			<< e.what() << std::endl;
#ifndef _RELEASE_
		DumpCallStack();
#endif
	}

	MPI_Finalize();

	return 0;
}
