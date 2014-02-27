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
/// @file pwdft.cpp
/// @brief Main driver for self-consistent field iteration using plane
/// wave basis set.  
///
/// The current version of pwdft is a sequential code and is used for
/// testing purpose, both for energy and for force.
/// @date 2013-10-16
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

	if( mpisize > 1 ){
		std::cout 
			<< "The current version of pwdft is a sequential code." << std::endl;
		MPI_Finalize();
		return -1;
	}

	try
	{
		// *********************************************************************
		// Input parameter
		// *********************************************************************
		
		// Initialize log file
		stringstream  ss;
		ss << "statfile." << mpirank;
		statusOFS.open( ss.str().c_str() );

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

		ESDFReadInput( esdfParam, inFile.c_str() ); //ZGGTODO: add in nsw and dt

		// Print the initial state
		{
			PrintBlock(statusOFS, "Basic information");

			Print(statusOFS, "MD Steps          = ",  esdfParam.nsw );//ZGG
			Print(statusOFS, "MD time Step      = ",  esdfParam.dt );//ZGG

			Print(statusOFS, "Super cell        = ",  esdfParam.domain.length );
			Print(statusOFS, "Grid size         = ",  esdfParam.domain.numGrid ); 
			Print(statusOFS, "Mixing dimension  = ",  esdfParam.mixMaxDim );
			Print(statusOFS, "Mixing variable   = ",  esdfParam.mixVariable );
			Print(statusOFS, "Mixing type       = ",  esdfParam.mixType );
			Print(statusOFS, "Mixing Steplength = ",  esdfParam.mixStepLength);
			Print(statusOFS, "SCF Outer Tol     = ",  esdfParam.scfOuterTolerance);
			Print(statusOFS, "SCF Outer MaxIter = ",  esdfParam.scfOuterMaxIter);
			Print(statusOFS, "Eig Tolerence     = ",  esdfParam.eigTolerance);
			Print(statusOFS, "Eig MaxIter       = ",  esdfParam.eigMaxIter);

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
		SetRandomSeed(1);

		Domain&  dm = esdfParam.domain;
		PeriodTable ptable;
		Fourier fft;
		Spinor  spn;
		KohnSham hamKS;
		EigenSolver eigSol;
		SCF  scf;

		Int NSW=esdfParam.nsw;
		Int dt=esdfParam.dt;

		ptable.Setup( esdfParam.periodTableFile );
		fft.Initialize( dm );

		const std::vector<Atom>&  atomList = esdfParam.atomList;//ZG: need this?
		Int numAtom = atomList.size();
		Real *atomMass;
		atomMass = new Real[numAtom];
		for(Int a=0; a < numAtom; a++) {
			Int atype = atomList[a].type;
			if (ptable.ptemap().find(atype)==ptable.ptemap().end() ){
				throw std::logic_error( "Cannot find the atom type." );
			}
			atomMass[a]=amu2au*ptable.ptemap()[atype].params(PTParam::MASS); //amu2au = 1822.8885
			Print(statusOFS, "atom Mass  = ", atomMass[a]);
		}

		// Hamiltonian
		statusOFS << "Hamiltonian begin." << std::endl;

		hamKS.Setup( dm, esdfParam.atomList, esdfParam.pseudoType, 
				esdfParam.XCId, esdfParam.numExtraState );

		DblNumVec& vext = hamKS.Vext();
		SetValue( vext, 0.0 );

		hamKS.CalculatePseudoPotential( ptable );

		statusOFS << "Hamiltonian constructed." << std::endl;

		// Wavefunctions
		spn.Setup( dm, 1, hamKS.NumStateTotal(), 0.0 );
		UniformRandom( spn.Wavefun() );

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

	
//ZG:*********MD starts***********

		//MD Velocity Verlet geometry update if NSW!=0
    if (NSW != 0) {

						std::vector<Atom>& atomList1 = esdfParam.atomList;
            std::vector<Atom>& atomList2 = hamKS.AtomList();
            Int numAtom = atomList2.size();

            std::vector<Point3>  atompos;
						std::vector<Point3>  atomv;
						std::vector<Point3>  atomforce;
						std::vector<Point3>		atomforcem;
		            
            atompos.resize( numAtom );
            atomv.resize( numAtom );
            atomforce.resize( numAtom );
            atomforcem.resize( numAtom );

            for(Int i=0; i<numAtom; i++) {
            	for(Int j = 0; j<3; j++)
              	atomv[i][j]=0.;
						}

//            Point3 atompos[500];     //x2 ZG: fix: 500 to numAtom
//            Real atomv[500][3]={0.};         //v2
//            Point3 atomforce[500];    //f2
//            Point3 atomforcem[500];   //f1

            for( Int i = 0; i < numAtom; i++ ){
                    atompos[i]   = atomList1[i].pos;
                    atomforcem[i]=atomList2[i].force;
            }//x1, f1, v1=0

/*debug*/
						for (Int i=0;i<numAtom;i++)
								Print(statusOFS, "debug: position",atompos[i]);

						for (Int i=0;i<numAtom;i++)
								Print(statusOFS, "debug: force",atomforcem[i]); //debug: OK

            for (Int n=0; n<NSW; n++){

                    Print(statusOFS, "Num of MD step = ", n);

                    for(Int i=0; i<numAtom; i++) {
                            for(Int j = 0; j<3; j++)
                                    atompos[i][j]=atompos[i][j]+atomv[i][j]*dt+atomforcem[i][j]*dt*dt/atomMass[i]/2;

                    }//x2=x1+v1*dt+f1*dt^2/2M

                    for(Int i = 0; i < numAtom; i++){
                            atomList1[i].pos = atompos[i];
                            Print(statusOFS, "Current Position    = ",  atomList[i].pos);
                    }//print out OK

                    /* Add the block to calculate force again, output as atomforce[i][j], copied from original code */

										hamKS.Update( esdfParam.atomList ); //ZG:updated atomList.pos

//										DblNumVec& vext = hamKS.Vext();//ZG:what is hamKS.Vext()?
//										SetValue( vext, 0.0 );//ZG: initial hamKS.Vext as 0.0?
										hamKS.UpdatePseudoPotential( ptable );//hamiltonian.cpp

										statusOFS << "Hamiltonian updated." << std::endl;

										// Wavefunctions
//											spn.Setup( dm, 1, hamKS.NumStateTotal(), 0.0 );//skip
//											UniformRandom( spn.Wavefun() );//skip

										// Eigensolver class
//										eigSol.Setup( esdfParam, hamKS, spn, fft );

										scf.Update( esdfParam, eigSol, ptable ); //ZG: Update scf


										GetTime( timeSta ); //debug

										// ************************************************
										// Solve
										// ************************************************

										scf.Iterate();

										hamKS.CalculateForce( spn, fft ); //new

								    /* Force block ends here*/
								    std::vector<Atom>& atomList = hamKS.AtomList();
								    Int numAtom = atomList.size();
								    for( Int i = 0; i < numAtom; i++ ){
						    			atomforce[i]=atomList[i].force;
						      	  Print( statusOFS, "Atom", i, "Force", atomList[i].force );
								    }//f2

								    for(Int i = 0; i < numAtom; i++ ){
							        for (Int j=0; j<3; j++){
							        atomv[i][j] = atomv[i][j]+(atomforcem[i][j]+atomforce[i][j])*dt/atomMass[i]/2.;
							        atomforcem[i][j]=atomforce[i][j];//f2->f1
							        Print(statusOFS, "Current Velocity = ", atomv[i][j]); //debug
						        }
						     }//v2, -> ex

					   }//for(n<NSW) loop ends here
					 }//if(NSW!=0) ends
	}
//****MD end*******

//******************************
//Clean up
//******************************
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
