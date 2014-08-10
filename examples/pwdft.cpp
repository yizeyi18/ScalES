/*
	 Copyright (c) 2012 The Regents of the University of California,
	 through Lawrence Berkeley National Laboratory.  

   Author: Lin Lin and Wei Hu
	 
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
/// @file pwdft.cpp
/// @brief Main driver for self-consistent field iteration using plane
/// wave basis set.  
///
/// The current version of pwdft is a sequential code and is used for
/// testing purpose, both for energy and for force.
/// @date 2013-10-16 Original implementation
/// @date 2014-02-01 Dual grid implementation
/// @date 2014-07-15 Parallelization of PWDFT.
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;


void Usage(){
  std::cout 
		<< "pwdft -in [inFile]" << std::endl
		<< "in:             Input file (default: pwdft.in)" << std::endl;
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
		// *********************************************************************
		// Input parameter
		// *********************************************************************
		
		// Initialize log file
		stringstream  ss;
		ss << "statfile." << mpirank;
		statusOFS.open( ss.str().c_str() );

    Print( statusOFS, "mpirank = ", mpirank );
    Print( statusOFS, "mpisize = ", mpisize );
		
    // Initialize input parameters
		std::map<std::string,std::string> options;
		OptionsCreate(argc, argv, options);

		std::string inFile;                   
		if( options.find("-in") != options.end() ){ 
			inFile = options["-in"];
		}
		else{
			inFile = "pwdft.in";
		}


		// Read ESDF input file
		ESDFInputParam  esdfParam;

		ESDFReadInput( esdfParam, inFile.c_str() );

		// Print the initial state
		{
			PrintBlock(statusOFS, "Basic information");

			Print(statusOFS, "Super cell        = ",  esdfParam.domain.length );
			Print(statusOFS, "Grid Wavefunction = ",  esdfParam.domain.numGrid );
      Print(statusOFS, "Grid Density      = ",  esdfParam.domain.numGridFine ); 
			Print(statusOFS, "Mixing dimension  = ",  esdfParam.mixMaxDim );
			Print(statusOFS, "Mixing variable   = ",  esdfParam.mixVariable );
			Print(statusOFS, "Mixing type       = ",  esdfParam.mixType );
			Print(statusOFS, "Mixing Steplength = ",  esdfParam.mixStepLength);
			Print(statusOFS, "SCF Outer Tol     = ",  esdfParam.scfOuterTolerance);
			Print(statusOFS, "SCF Outer MaxIter = ",  esdfParam.scfOuterMaxIter);
			Print(statusOFS, "Eig Tolerence     = ",  esdfParam.eigTolerance);
			Print(statusOFS, "Eig MaxIter       = ",  esdfParam.eigMaxIter);
			Print(statusOFS, "Eig Tolerance Dyn = ",  esdfParam.isEigToleranceDynamic);
			Print(statusOFS, "Num unused state  = ",  esdfParam.numUnusedState);

			Print(statusOFS, "RestartDensity    = ",  esdfParam.isRestartDensity);
			Print(statusOFS, "RestartWfn        = ",  esdfParam.isRestartWfn);
			Print(statusOFS, "OutputDensity     = ",  esdfParam.isOutputDensity);

			Print(statusOFS, "EcutWavefunction  = ",  esdfParam.ecutWavefunction);
			Print(statusOFS, "Density GridFactor= ",  esdfParam.densityGridFactor);

			Print(statusOFS, "Temperature       = ",  au2K / esdfParam.Tbeta, "[K]");
			Print(statusOFS, "Extra states      = ",  esdfParam.numExtraState );
			Print(statusOFS, "PeriodTable File  = ",  esdfParam.periodTableFile );
			Print(statusOFS, "Pseudo Type       = ",  esdfParam.pseudoType );
			Print(statusOFS, "PW Solver         = ",  esdfParam.PWSolver );
			Print(statusOFS, "XC Type           = ",  esdfParam.XCType );

			PrintBlock(statusOFS, "Atom Type and Coordinates");

			const std::vector<Atom>&  atomList = esdfParam.atomList;
			for(Int i=0; i < atomList.size(); i++) {
				Print(statusOFS, "Type = ", atomList[i].type, "Position  = ", atomList[i].pos);
			}

			statusOFS << std::endl;
		}


		// *********************************************************************
		// Preparation
		// *********************************************************************
		SetRandomSeed(mpirank);

		Domain&  dm = esdfParam.domain;
		PeriodTable ptable;
		Fourier fft;
    Fourier fftFine;
		Spinor  spn;
		KohnSham hamKS;
		EigenSolver eigSol;
		SCF  scf;

		ptable.Setup( esdfParam.periodTableFile );

    fft.Initialize( dm );

    fft.InitializeFine( dm );

		// Hamiltonian

		hamKS.Setup( dm, esdfParam.atomList, esdfParam.pseudoType, 
				esdfParam.XCType, esdfParam.numExtraState );

		DblNumVec& vext = hamKS.Vext();
		SetValue( vext, 0.0 );

		hamKS.CalculatePseudoPotential( ptable );

		statusOFS << "Hamiltonian constructed." << std::endl;

		// Wavefunctions
    int numStateTotal = hamKS.NumStateTotal();
    int numStateLocal, blocksize;

    if ( numStateTotal <=  mpisize ) {
      blocksize = 1;

      if ( mpirank < numStateTotal ){
        numStateLocal = 1; // blocksize == 1;
      }
      else { 
        // FIXME Throw an error here.
        numStateLocal = 0;
      }
  
    }
  
    else {  // numStateTotal >  mpisize
  
      if ( numStateTotal % mpisize == 0 ){
        blocksize = numStateTotal / mpisize;
        numStateLocal = blocksize ;
      }
      else {
        // blocksize = ((numStateTotal - 1) / mpisize) + 1;
        blocksize = numStateTotal / mpisize;
        numStateLocal = blocksize ;
        if ( mpirank < ( numStateTotal % mpisize ) ) {
          numStateLocal = numStateLocal + 1 ;
        }
      }    

    }
     
    spn.Setup( dm, 1, hamKS.NumStateTotal(), numStateLocal, 0.0 );
    UniformRandom( spn.Wavefun() );

//    MPI_Comm mpi_comm = dm.comm;
    
//		Spinor  spnTemp;
//    spnTemp.Setup( dm, 1, hamKS.NumStateTotal(), hamKS.NumStateTotal(), 0.0 );
   
//    if (mpirank == 0){
//      UniformRandom( spnTemp.Wavefun() );
//    }
//    MPI_Bcast(spnTemp.Wavefun().Data(), spnTemp.Wavefun().m()*spnTemp.Wavefun().n()*spnTemp.Wavefun().p(), MPI_DOUBLE, 0, mpi_comm);

//	Int size = spn.Wavefun().m() * spn.Wavefun().n();
//	Int nocc = spn.Wavefun().p();
 
//  IntNumVec& wavefunIdx = spn.WavefunIdx();
//  NumTns<Scalar>& wavefun = spn.Wavefun();

//	for (Int k=0; k<nocc; k++) {
//		Scalar *ptr = spn.Wavefun().MatData(k);
//		Scalar *ptr1 = spnTemp.Wavefun().MatData(wavefunIdx(k));
//		for (Int i=0; i<size; i++) {
//      *ptr = *ptr1;
//		  ptr = ptr + 1;
//		  ptr1 = ptr1 + 1;
//    }
//	}

		// Eigensolver class
		eigSol.Setup( esdfParam, hamKS, spn, fft );

		scf.Setup( esdfParam, eigSol, ptable );

		GetTime( timeSta );

		// *********************************************************************
		// Solve
		// *********************************************************************

		scf.Iterate();

		// Print out the force
		PrintBlock( statusOFS, "Atomic Force" );
		{
      hamKS.CalculateForce( spn, fft );
			Point3 forceCM(0.0, 0.0, 0.0);
			std::vector<Atom>& atomList = hamKS.AtomList();
			Int numAtom = atomList.size();
			for( Int a = 0; a < numAtom; a++ ){
				Print( statusOFS, "atom", a, "force", atomList[a].force );
				forceCM += atomList[a].force;
			}
			statusOFS << std::endl;
			Print( statusOFS, "force for centroid: ", forceCM );
			statusOFS << std::endl;
		}

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
