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

		ESDFReadInput( esdfParam, inFile.c_str() );

		// Print the initial state
		{
			PrintBlock(statusOFS, "Basic information");

			Print(statusOFS, "MD Steps          = ",  esdfParam.nsw );//ZGG
			Print(statusOFS, "MD time Step      = ",  esdfParam.dt );//ZGG
			Print(statusOFS, "Thermostat mass   = ",  esdfParam.qmass);//ZGG

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
    Fourier fftFine;
		Spinor  spn;
		KohnSham hamKS;
		EigenSolver eigSol;
		SCF  scf;

		Int NSW=esdfParam.nsw;
		Int dt=esdfParam.dt;
		Real Q1=esdfParam.qmass;
		Real Q2=esdfParam.qmass;

		ptable.Setup( esdfParam.periodTableFile );
		fft.Initialize( dm );

    fft.InitializeFine( dm );

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

//Nose Hoover Parameters//
  	Int L=3*numAtom; 
		Real K=0.;
    Real Efree = 0.;
    Real Ktot1= 0.;
		Real Ktot2= 0.; //Kinetic energy including ionic part and thermostat
    Real Edrift= 0.;
  	Real xi1=0., xi2=0.;
  	Real vxi1 = 0.0, vxi2=0.0;
  	Real G1, G2;
  	Real s; //scale factor

		Print(statusOFS, "debug: Q1",Q1);
		Print(statusOFS, "debug: Q2",Q2);

		Real T = 1. / esdfParam.Tbeta;
		Print(statusOFS, "debug: Temperature",T);

//ZG:*********MD starts***********

		//MD Velocity Verlet geometry update if NSW!=0
    if (NSW != 0) {

						std::vector<Atom>& atomList1 = esdfParam.atomList;
            std::vector<Atom>& atomList2 = hamKS.AtomList();
            Int numAtom = atomList2.size();

            std::vector<Point3>  atompos;
						std::vector<Point3>  atomv;
						std::vector<Point3>  atomforce;
		            
            atompos.resize( numAtom );
            atomv.resize( numAtom );
            atomforce.resize( numAtom );

            for(Int i=0; i<numAtom; i++) {
            	for(Int j = 0; j<3; j++)
              	atomv[i][j]=0.;
						}


            for( Int i = 0; i < numAtom; i++ ){
                    atompos[i]   = atomList1[i].pos;
                    atomforce[i]=atomList2[i].force;
            }//x1, f1, v1=0

/*debug*/
						for (Int i=0;i<numAtom;i++)
								Print(statusOFS, "debug: position",atompos[i]);

						for (Int i=0;i<numAtom;i++)
								Print(statusOFS, "debug: force",atomforce[i]);

            for (Int n=0; n<NSW; n++){
                    Print(statusOFS, "Num of MD step = ", n);

//  chain(K[i], T, dt, vxi1[i], vxi2[i], xi1[i], xi2[i], v[i]);//
    								G2 = (Q1*vxi1*vxi1-T)/Q2; //add /Q2
										Print(statusOFS, "debug: G2",G2);//debug
								    vxi2 = vxi2+G2*dt/4.;
										Print(statusOFS, "debug: vxi2",vxi2);//debug
								    vxi1 = vxi1*exp(-vxi2*dt/8.);
										Print(statusOFS, "debug: vxi1",vxi1);//debug
								    G1 = (2*K-L*T)/Q1;
										Print(statusOFS, "debug: G1",G1);//debug
								    vxi1 = vxi1+G1*dt/4.;
										Print(statusOFS, "debug: vxi1",vxi1);//debug
								    vxi1 = vxi1*exp(-vxi2*dt/8.);
										Print(statusOFS, "debug: vxi1",vxi1);//debug
								    xi1 = xi1+vxi1*dt/2.;
										Print(statusOFS, "debug: xi1",xi1);//debug
								    xi2 = xi2+vxi2*dt/2.;
										Print(statusOFS, "debug: xi2",xi2);//debug
								    s = exp(-vxi1*dt/2.);
										Print(statusOFS, "debug: s",s);//debug
										for(Int i=0; i<numAtom; i++){
												for(Int j=0; j<3; j++)
														atomv[i][j]=s*atomv[i][j];
										} // v = s*v;
								    K=K*s*s;
								    vxi1 = vxi1*exp(-vxi2*dt/8.);
								    G1 = (2*K-L*T)/Q1;
								    vxi1 = vxi1+G1*dt/4.;
								    vxi1 = vxi1*exp(-vxi2*dt/8.);
								    G2 = (Q1*vxi1*vxi1-T)/Q2;
								    vxi2 = vxi2+G2*dt/4.;


/*
//numchain=1 start//
										vxi1 = vxi1+(K*2-L*T)/Q1*dt/4.;
										Print(statusOFS, "debug: vxi1 ",vxi1);//debug
										xi1 = xi1+vxi1*dt/2.;
										Print(statusOFS, "debug: xi1 ",xi1);//debug
										s=exp(-vxi1*dt/2.);
										Print(statusOFS, "debug: s ",s);//debug
										for(Int i=0;i<numAtom;i++){
												for(Int j=0;j<3;j++)
														atomv[i][j]=s*atomv[i][j];
										}

										K=K*s*s;
										vxi1=vxi1+(2*K-L*T)/Q1*dt/4.;
										Print(statusOFS, "debug: vxi1 ",vxi1);//debug
//numchain=1 end//
*/

//posvel//

										K=0.;
                    for(Int i=0; i<numAtom; i++) {
                            for(Int j = 0; j<3; j++)
                                    atompos[i][j]=atompos[i][j]+atomv[i][j]*dt/2.;
                    }

                    for(Int i = 0; i < numAtom; i++){
                            atomList1[i].pos = atompos[i];
                            Print(statusOFS, "Current Position before SCF    = ",  atomList1[i].pos);
                    }//x=x+v*dt/2

										hamKS.Update( esdfParam.atomList ); //ZG:updated atomList.pos

										hamKS.UpdatePseudoPotential( ptable );//hamiltonian.cpp

										statusOFS << "Hamiltonian updated." << std::endl;

										scf.Update( esdfParam, eigSol, ptable ); //ZG: Update scf

										GetTime( timeSta ); //debug

										scf.Iterate();

										hamKS.CalculateForce( spn, fft ); //new

								    std::vector<Atom>& atomList = hamKS.AtomList();
								    Int numAtom = atomList.size();
								    for( Int i = 0; i < numAtom; i++ ){
						    			atomforce[i]=atomList[i].force;
						      	  Print( statusOFS, "Atom", i, "Force", atomList[i].force );
								    }//update f

/*debug block//

						for (Int i=0;i<numAtom;i++)
								Print(statusOFS, "debug: check position after SCF ",atompos[i]); //checked! correct

						std::vector<Atom>& atomList3 = esdfParam.atomList;
                    for(Int i = 0; i < numAtom; i++){
                            Print(statusOFS, "Current Position after SCF   = ",  atomList3[i].pos); //checked! correct
                    }//x=x+v*dt/2
//debug block ends*/

										for(Int i=0; i<numAtom; i++){
												for(Int j=0; j<3; j++){
														atomv[i][j]=atomv[i][j]+atomforce[i][j]*dt/atomMass[i]; //v=v+f*dt
														atompos[i][j]=atompos[i][j]+atomv[i][j]*dt/2.; //x=x+v*dt/2
														K += atomMass[i]*atomv[i][j]*atomv[i][j]/2.;
												}
										}
//debug block//
									
										for (Int i=0;i<numAtom;i++)
												Print(statusOFS, "debug: velocity",atomv[i]);

										for (Int i=0;i<numAtom;i++)
												Print(statusOFS, "debug: position",atompos[i]);
//end of debug block//	

//										Print(statusOFS, "debug: Kinetic energy of ions ",K);

//  chain(K[i], T, dt, vxi1[i], vxi2[i], xi1[i], xi2[i], v[i]);//
    								G2 = (Q1*vxi1*vxi1-T)/Q2;
								    vxi2 = vxi2+G2*dt/4.;
								    vxi1 = vxi1*exp(-vxi2*dt/8.);
								    G1 = (2*K-L*T)/Q1;
								    vxi1 = vxi1+G1*dt/4.;
								    vxi1 = vxi1*exp(-vxi2*dt/8.);
								    xi1 = xi1+vxi1*dt/2.;
								    xi2 = xi2+vxi2*dt/2.;
								    s = exp(-vxi1*dt/2.);
										for(Int i=0; i<numAtom; i++){
												for(Int j=0; j<3; j++)
														atomv[i][j]=s*atomv[i][j];
										} // v = s*v;
								    K=K*s*s;
								    vxi1 = vxi1*exp(-vxi2*dt/8.);
								    G1 = (2*K-L*T)/Q1;
								    vxi1 = vxi1+G1*dt/4.;
								    vxi1 = vxi1*exp(-vxi2*dt/8.);
								    G2 = (Q1*vxi1*vxi1-T)/Q2;
								    vxi2 = vxi2+G2*dt/4.;

										Print(statusOFS, "debug: K*s*s: Kinetic energy II of ions ",K);
//    E=Et(x,v);
/*
//nuchain=1//
										vxi1 = vxi1+(K*2-L*T)/Q1*dt/4.;
										xi1 = xi1+vxi1*dt/2.;
										s=exp(-vxi1*dt/2.);
										for(Int i=0;i<numAtom;i++){
												for(Int j=0;j<3;j++)
														atomv[i][j]=s*atomv[i][j];
										}
										K=K*s*s;
										Print(statusOFS, "debug: Kinetic energy II of ions ",K);

										vxi1=vxi1+(2*K-L*T)/Q1*dt/4.;
//numchain=1 end//
*/
//										Ktot =K+Q1*vxi1*vxi1/2.+Q2*vxi2*vxi2/2.+T*L*xi1+T*L*xi2;
//										Print(statusOFS, "debug: Ktot =  ",Ktot);

                    Efree = scf.getEfree();
                    Print(statusOFS, "MD_Efree =  ",Efree);
                    Print(statusOFS, "MD_K =  ",K);

//                    Ktot2 =K+Efree+Q1*vxi1*vxi1/2.+Q2*vxi2*vxi2/2.+T*numAtom*xi1+T*xi2;
                    Ktot2 =K+Efree+Q1*vxi1*vxi1/2.+Q2*vxi2*vxi2/2.+L*T*xi1+T*xi2;
                    if(NSW == 0)
                      Ktot1 = Ktot2;

                    Edrift= (Ktot2-Ktot1)/Ktot1;
                    Print(statusOFS, "Conserved Energy: Ktot =  ",Ktot2);
                    Print(statusOFS, "Drift of Conserved Energy: Edrift =  ",Edrift);
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
