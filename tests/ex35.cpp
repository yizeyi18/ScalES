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
/// @file ex35.cpp
/// @brief Simple test of the matrix matrix multiplication routine but
/// performed with GEMR2D.
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


void SCALAPACK(pdgemr2d)(const Int* m, const Int* n, const double* A, const Int* ia, 
    const Int* ja, const Int* desca, double* B,
    const Int* ib, const Int* jb, const Int* descb,
    const Int* contxt);
}

void Usage(){
  std::cout 
		<< "ex34 " << std::endl;
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

    Int M = 1000000, N = 32;
    Int MB = M;
    Int NB = 1;
    Int MB2D = 100;
    Int NB2D = 8;

    Int NLocal = N / mpisize;
    Int M2DLocal = M / MB2D;
    Int N2DLocal = N / NB2D;

    DblNumMat XLocal(M, NLocal), YLocal(M, NLocal);
    DblNumMat X2DLocal( M2DLocal, N2DLocal );
    DblNumMat Y2DLocal( M2DLocal, N2DLocal );
    DblNumMat XTY(N, N);

    UniformRandom( XLocal );

    UniformRandom( YLocal );

    Int descX[9];
    Int descX2D[9];
    Int descM[9];

    Int nprow = 1;
    Int npcol = mpisize;
    Int myrow, mycol;

    Int contxt;
    Cblacs_get(0, 0, &contxt);

    Cblacs_gridinit(&contxt, "C", nprow, npcol); 

    Int info;
    Int irsrc = 0;
    Int icsrc = 0;
    SCALAPACK(descinit)(&descX[0], &M, &N, &MB, &NB, &irsrc, &icsrc, &contxt, &M, &info);
    SCALAPACK(descinit)(&descX2D[0], &M, &N, &MB2D, &NB2D, &irsrc, &icsrc, &contxt, &M, &info);
    SCALAPACK(descinit)(&descM[0], &N, &N, &NB, &NB, &irsrc, &icsrc, &contxt, &N, &info);

    Real timeSta, timeEnd;
    char TT = 'T';
    char NN = 'N';
    double D_ONE = 1.0;
    double D_ZERO = 0.0;
    int I_ONE = 1;

    MPI_Barrier( MPI_COMM_WORLD );

    GetTime( timeSta );


    SCALAPACK(pdgemr2d)(&M, &N, 
        XLocal.Data(), &I_ONE, &I_ONE, &descX[0],
        X2DLocal.Data(), &I_ONE, &I_ONE, &descX2D[0],
        &contxt );

    SCALAPACK(pdgemr2d)(&M, &N, 
        YLocal.Data(), &I_ONE, &I_ONE, &descX[0],
        Y2DLocal.Data(), &I_ONE, &I_ONE, &descX2D[0],
        &contxt );
        

    SCALAPACK(pdgemm)(&TT, &NN, &N, &N, &M, 
        &D_ONE,
        X2DLocal.Data(), &I_ONE, &I_ONE, &descX2D[0],
        Y2DLocal.Data(), &I_ONE, &I_ONE, &descX2D[0], 
        &D_ZERO,
        XTY.Data(), &I_ONE, &I_ONE, &descM[0], 
        &contxt );

    GetTime( timeEnd );

    MPI_Barrier( MPI_COMM_WORLD );

    if( mpirank == 0 ){
      std::cout << "The time for the X'*Y is " << timeEnd - timeSta << " sec." << std::endl;
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
