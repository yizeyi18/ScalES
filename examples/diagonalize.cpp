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
/// @file diagonalize.cpp
/// @brief Read a matrix generated from DGDFT in parallel, convert to
/// block cyclic format and diagonalize it using ScaLAPACK.
///
/// @author Lin Lin
/// @date 2013-10-15
#include "sparse_matrix_decl.hpp"
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;


void Usage(){
	cout << "Read a matrix generated from DGDFT in parallel, " 
		<< "convert to block cyclic format and diagonalize it using ScaLAPACK."
		<< endl << endl;
}

int main(int argc, char **argv) 
{
	MPI_Init(&argc, &argv);
	int mpirank, mpisize;
	MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
	MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

	if( mpirank == 0 )
		Usage();

	try
	{

		Real timeSta, timeEnd;

    // *********************************************************************
    // Input parameter
    // *********************************************************************
    std::map<std::string,std::string> options;

    OptionsCreate(argc, argv, options);

		stringstream  ss;
		ss << "logTest" << mpirank;
		statusOFS.open( ss.str().c_str() );

    // Default processor number 
		Int nprow = 1;
    Int npcol = mpisize;

    if( options.find("-r") != options.end() ){
      if( options.find("-c") != options.end() ){
        nprow= atoi(options["-r"].c_str());
        npcol= atoi(options["-c"].c_str());
        if(nprow*npcol > mpisize){
          throw std::runtime_error("The number of used processors cannot be higher than the total number of available processors." );
        } 
      }
      else{
        throw std::runtime_error( "When using -r option, -c also needs to be provided." );
      }
    }
    else if( options.find("-c") != options.end() ){
      if( options.find("-r") != options.end() ){
        nprow= atoi(options["-r"].c_str());
        npcol= atoi(options["-c"].c_str());
        if(nprow*npcol > mpisize){
          throw std::runtime_error("The number of used processors cannot be higher than the total number of available processors." );
        } 
      }
      else{
        throw std::runtime_error( "When using -c option, -r also needs to be provided." );
      }
    }


		string Hfile;
		if( options.find("-H") != options.end() ){ 
			Hfile = options["-H"];
		}
		else{
			throw std::logic_error("Hfile must be provided.");
		}

		Int blockSize;
		if( options.find("-MB") != options.end() ){ 
			blockSize = atoi(options["-MB"].c_str());
		}
		else{
			// Default block size for ScaLAPACK
			blockSize = 32;  
		}

		Int numEig;
		if( options.find("-NE") != options.end() ){ 
			numEig = atoi(options["-NE"].c_str());
			if( numEig < 0 ){
				throw std::runtime_error("Number of eigenvalues must be 0 (comput all eigenvalues) or > 0 (partial diagonalization)" );
			}
		}
		else{
			// Default NE for computing all eigenvalues
			numEig = 0;
		}

		Int routine;
		if( options.find("-D") != options.end() ){ 
			routine = atoi(options["-D"].c_str());
			if( routine != 0 && routine != 1 ){
				throw std::runtime_error("Diagonalization must be 0 (Syevr) or 1 (Syevd).");
			}
		}
		else{
			// Default routine: SYEVR
			routine = 0;
		}

		// Output input parameters
		PrintBlock( statusOFS, "Input parameters" );
		Print( statusOFS, "nprow                   = ", nprow );
		Print( statusOFS, "npcol                   = ", npcol ); 
		Print( statusOFS, "H file                  = ", Hfile );
		Print( statusOFS, "Block size              = ", blockSize );
		if( numEig == 0 )
			Print( statusOFS, "Number of eigs          = ALL" );
		else	
			Print( statusOFS, "Number of eigs          = ", numEig );

		if( routine == 0 )
			Print( statusOFS, "Diagonalization routine = PDSYEVR" );
		if( routine == 1 )
			Print( statusOFS, "Diagonalization routine = PDSYEVD" );

		statusOFS << endl;

		// *********************************************************************
		// Read input matrix
		// *********************************************************************


		DistSparseMatrix<Real> HMat;
		
		GetTime( timeSta );
		ParaReadDistSparseMatrix( Hfile.c_str(), HMat, MPI_COMM_WORLD); 

		GetTime( timeEnd );
		
		statusOFS << "Time for reading H and S is " << timeEnd - timeSta << endl;
		statusOFS << "H.size = " << HMat.size << endl;
		statusOFS << "H.nnz  = " << HMat.nnz  << endl;

		// *********************************************************************
		// Convert the H matrix into the ScaLAPACK format
		// *********************************************************************
		GetTime( timeSta );

		// Initialize BLACS
		Int contxt;
		Cblacs_get(0, 0, &contxt);
		Cblacs_gridinit(&contxt, "C", nprow, npcol);

		scalapack::Descriptor descH( HMat.size, HMat.size, blockSize, blockSize, 
				0, 0, contxt );

		scalapack::ScaLAPACKMatrix<Real>  scaH, scaZ;
		std::vector<Real> eigs;

		DistSparseMatToScaMat( HMat, descH, scaH, MPI_COMM_WORLD );

		GetTime( timeEnd );

		statusOFS << "Time for converting the matrix is " << timeEnd - timeSta << endl;

		// *********************************************************************
		// Solve using ScaLAPACK
		// *********************************************************************

		GetTime( timeSta );

		if( routine == 0 && numEig > 0 )
			scalapack::Syevr('U', scaH, eigs, scaZ, 1, numEig );
		if( routine == 0 && numEig == 0 )
			scalapack::Syevr('U', scaH, eigs, scaZ );
		if( routine == 1 )
			scalapack::Syevd('U', scaH, eigs, scaZ );

		GetTime( timeEnd );

		statusOFS << "Time for diagonalizing the matrix is " << timeEnd -
			timeSta << endl;

		// *********************************************************************
		// Post-processing
		// *********************************************************************

		statusOFS << "Eigenvalues = " << eigs << endl;

		

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
