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
/// @file dgdft.cpp
/// @brief Main driver for DGDFT for self-consistent field iteration.
/// @date 2013-02-11
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;


void Usage(){
  std::cout 
    << "dgdft -in [inFile] -fftsize [fftsize]" << std::endl
    << "in:             Input file (default: dgdft.in)" << std::endl
    << "fftsize:        Number of procs used for distributed memory fft (default:mpisize)" << std::endl;
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

    // Initialize FFTW
    fftw_mpi_init();

    // Initialize BLACS
    Int nprow, npcol;
    Int contxt;
    Cblacs_get(0, 0, &contxt);
    for( Int i = IRound(sqrt(double(mpisize))); i <= mpisize; i++){
      nprow = i; npcol = mpisize / nprow;
      if( nprow * npcol == mpisize ) break;
    } 
    Print( statusOFS, "nprow = ", nprow );
    Print( statusOFS, "npcol = ", npcol ); 
    Cblacs_gridinit(&contxt, "C", nprow, npcol);

    // Initialize input parameters
    std::map<std::string,std::string> options;
    OptionsCreate(argc, argv, options);

    Int distfftSize;
    if( options.find("-fftsize") != options.end() ){
      distfftSize = std::atoi(options["-fftsize"].c_str());
    }
    else{
      distfftSize = mpisize;
    }

    std::string inFile;                   
    if( options.find("-in") != options.end() ){ 
      inFile = options["-in"];
    }
    else{
      inFile = "dgdft.in";
    }


    // Read ESDF input file
    GetTime( timeSta );
    ESDFInputParam  esdfParam;

    ESDFReadInput( esdfParam, inFile.c_str() );

    GetTime( timeEnd );
    statusOFS << "Time for reading the input file is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;

    // The number of grid points have already been adjusted in ESDFReadInput.
    //		{
    //			bool isGridAdjusted = false;
    //			Index3& numGrid = esdfParam.domain.numGrid;
    //			Index3& numElem = esdfParam.numElem;
    //			for( Int d = 0; d < DIM; d++ ){
    //				if( numGrid[d] % numElem[d] != 0 ){
    //					numGrid[d] = IRound( (Real)numGrid[d] / numElem[d] ) * numElem[d];
    //					isGridAdjusted = true;
    //				}
    //			}
    //			if( isGridAdjusted ){
    //				statusOFS << std::endl 
    //					<< "Grid size is adjusted to be a multiple of the number of elements." 
    //					<< std::endl;
    //			}
    //		}

    // Print the initial state
    GetTime( timeSta );
    {
      PrintBlock(statusOFS, "Basic information");

      Print(statusOFS, "MD Steps          = ",  esdfParam.nsw );//ZGG
      Print(statusOFS, "MD time Step      = ",  esdfParam.dt );//ZGG
      Print(statusOFS, "Thermostat mass   = ",  esdfParam.qmass);//ZGG

      Print(statusOFS, "Super cell        = ",  esdfParam.domain.length );
      Print(statusOFS, "Grid wfc size     = ",  esdfParam.domain.numGrid ); 
      Print(statusOFS, "Grid rho size     = ",  esdfParam.domain.numGridFine );
      Print(statusOFS, "Mixing dimension  = ",  esdfParam.mixMaxDim );
      Print(statusOFS, "Mixing variable   = ",  esdfParam.mixVariable );
      Print(statusOFS, "Mixing type       = ",  esdfParam.mixType );
      Print(statusOFS, "Mixing Steplength = ",  esdfParam.mixStepLength);
      Print(statusOFS, "SCF Outer Tol     = ",  esdfParam.scfOuterTolerance);
      Print(statusOFS, "SCF Outer MaxIter = ",  esdfParam.scfOuterMaxIter);
      Print(statusOFS, "SCF Inner Tol     = ",  esdfParam.scfInnerTolerance);
      Print(statusOFS, "SCF Inner MaxIter = ",  esdfParam.scfInnerMaxIter);
      Print(statusOFS, "Eig Tolerence     = ",  esdfParam.eigTolerance);
      Print(statusOFS, "Eig MaxIter       = ",  esdfParam.eigMaxIter);
      Print(statusOFS, "SVD Basis Tol     = ",  esdfParam.SVDBasisTolerance);

      Print(statusOFS, "RestartDensity    = ",  esdfParam.isRestartDensity);
      Print(statusOFS, "RestartWfn        = ",  esdfParam.isRestartWfn);
      Print(statusOFS, "RestartPosition   = ",  esdfParam.isRestartPosition);
      Print(statusOFS, "RestartThermostate= ",  esdfParam.isRestartThermostate);
      Print(statusOFS, "OutputDensity     = ",  esdfParam.isOutputDensity);
      Print(statusOFS, "OutputWfnElem     = ",  esdfParam.isOutputWfnElem);
      Print(statusOFS, "OutputWfnExtElem  = ",  esdfParam.isOutputWfnExtElem);
      Print(statusOFS, "OutputPotExtElem  = ",  esdfParam.isOutputPotExtElem);
      Print(statusOFS, "OutputHMatrix     = ",  esdfParam.isOutputHMatrix );
      Print(statusOFS, "OutputPosition    = ",  esdfParam.isOutputPosition );
      Print(statusOFS, "OutputThermostate = ",  esdfParam.isOutputThermostate );

      Print(statusOFS, "PeriodizePotential= ",  esdfParam.isPeriodizePotential);
      Print(statusOFS, "DistancePeriodize = ",  esdfParam.distancePeriodize);

      //			Print(statusOFS, "Barrier W         = ",  esdfParam.potentialBarrierW);
      //			Print(statusOFS, "Barrier S         = ",  esdfParam.potentialBarrierS);
      //			Print(statusOFS, "Barrier R         = ",  esdfParam.potentialBarrierR);
      Print(statusOFS, "EcutWavefunction  = ",  esdfParam.ecutWavefunction);
      Print(statusOFS, "Density GridFactor= ",  esdfParam.densityGridFactor);
      Print(statusOFS, "LGL GridFactor    = ",  esdfParam.LGLGridFactor);

      Print(statusOFS, "Temperature       = ",  au2K / esdfParam.Tbeta, "[K]");
      Print(statusOFS, "Extra states      = ",  esdfParam.numExtraState );
      Print(statusOFS, "PeriodTable File  = ",  esdfParam.periodTableFile );
      Print(statusOFS, "Pseudo Type       = ",  esdfParam.pseudoType );
      Print(statusOFS, "PW Solver         = ",  esdfParam.PWSolver );
      Print(statusOFS, "XC Type           = ",  esdfParam.XCType );

      Print(statusOFS, "Penalty Alpha     = ",  esdfParam.penaltyAlpha );
      Print(statusOFS, "Element size      = ",  esdfParam.numElem ); 
      Print(statusOFS, "Wfn Elem GridSize = ",  esdfParam.numGridWavefunctionElem ); 
      Print(statusOFS, "Rho Elem GridSize = ",  esdfParam.numGridDensityElem ); 
      Print(statusOFS, "LGL Grid size     = ",  esdfParam.numGridLGL ); 
      Print(statusOFS, "ScaLAPACK block   = ",  esdfParam.scaBlockSize); 
      statusOFS << "Number of ALB for each element: " << std::endl 
        << esdfParam.numALBElem << std::endl;

      Print(statusOFS, "Solution Method   = ",  esdfParam.solutionMethod );
      if( esdfParam.solutionMethod == "pexsi" ){
        Print(statusOFS, "Number of poles   = ",  esdfParam.numPole); 
        Print(statusOFS, "Nproc row PEXSI   = ",  esdfParam.numProcRowPEXSI); 
        Print(statusOFS, "Nproc col PEXSI   = ",  esdfParam.numProcColPEXSI); 
        Print(statusOFS, "Nproc for symbfact= ",  esdfParam.npSymbFact); 
        Print(statusOFS, "Energy gap        = ",  esdfParam.energyGap); 
        Print(statusOFS, "Spectral radius   = ",  esdfParam.spectralRadius); 
        Print(statusOFS, "Matrix ordering   = ",  esdfParam.matrixOrdering); 
        Print(statusOFS, "Inertia before SCF= ",  esdfParam.inertiaCountSteps);
        Print(statusOFS, "Max PEXSI iter    = ",  esdfParam.maxPEXSIIter); 
        Print(statusOFS, "MuMin0            = ",  esdfParam.muMin); 
        Print(statusOFS, "MuMax0            = ",  esdfParam.muMax); 
        Print(statusOFS, "NumElectron tol   = ",  esdfParam.numElectronPEXSITolerance); 
        Print(statusOFS, "mu Inertia tol    = ",  esdfParam.muInertiaTolerance); 
        Print(statusOFS, "mu Inertia expand = ",  esdfParam.muInertiaExpansion); 
        Print(statusOFS, "mu PEXSI safeguard= ",  esdfParam.muPEXSISafeGuard); 
      }



      Print(statusOFS, "Calculate force at each step                        = ",  
          esdfParam.isCalculateForceEachSCF );
      Print(statusOFS, "Calculate A Posteriori error estimator at each step = ",  
          esdfParam.isCalculateAPosterioriEachSCF);


      //read position from lastPos.out into esdfParam.atomList[i].pos if isRestartPosition=1
      if(esdfParam.isRestartPosition){
         Point3 pos;
         std::vector<Atom>&  atomList0 = esdfParam.atomList;
         fstream fin;
         fin.open("lastPos.out",ios::in);
         for(Int i=0; i<atomList0.size(); i++){
           fin>> pos[0];
           fin>> pos[1];
           fin>> pos[2];
           atomList0[i].pos = pos;
//           Print(statusOFS, "Type = ", atomList0[i].type, "Position  = ", atomList0[i].pos);
         }

        PrintBlock( statusOFS, "Read in Atomic Position" );
        {
          for( Int a = 0; a < atomList0.size(); a++ ){
            Print( statusOFS, "atom", a, "Position   ", atomList0[a].pos );
          }
        }
       }//restart position read in

      // Only master processor output information containing all atoms
      else{
        if( mpirank == 0 ){
          PrintBlock(statusOFS, "Atom Type and Coordinates");
  
          const std::vector<Atom>&  atomList = esdfParam.atomList;
          for(Int i=0; i < atomList.size(); i++) {
            Print(statusOFS, "Type = ", atomList[i].type, "Position  = ", atomList[i].pos);
          }
        }
      }

      statusOFS << std::endl;
    }
    GetTime( timeEnd );
    statusOFS << "Time for outputing the initial state variable is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;


    // *********************************************************************
    // Preparation
    // *********************************************************************
    SetRandomSeed(1);

    GetTime( timeSta );

    Domain&  dm = esdfParam.domain;
    PeriodTable ptable;
    ptable.Setup( esdfParam.periodTableFile );

		//MD parameters
		Int NSW=esdfParam.nsw;
		Int dt=esdfParam.dt;
		Real Q1=esdfParam.qmass;
		Real Q2=esdfParam.qmass;

    GetTime( timeEnd );
    statusOFS << "Time for setting up the periodic table is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;

    GetTime( timeSta );

    // Setup the element and extended element information
    DistVec<Index3, EigenSolver, ElemPrtn>  distEigSol; 
    DistVec<Index3, KohnSham, ElemPrtn>     distHamKS;
    DistVec<Index3, Spinor, ElemPrtn>       distPsi;
    // All extended elements share the same Fourier structure.
    Fourier fftExtElem;

    // Setup the eigenvalue solvers in each extended element
    {
      // Element partition information
      Index3  numElem = esdfParam.numElem;

      IntNumTns& elemPrtnInfo = distEigSol.Prtn().ownerInfo;
      elemPrtnInfo.Resize( numElem[0], numElem[1], numElem[2] );

      if( mpisize != numElem.prod() ){
        std::ostringstream msg;
        msg << "The number of processors is not equal to the total number of elements." << std::endl;
        throw std::runtime_error( msg.str().c_str() );
      }

      Int cnt = 0;
      for( Int k=0; k< numElem[2]; k++ )
        for( Int j=0; j< numElem[1]; j++ )
          for( Int i=0; i< numElem[0]; i++ ) {
            elemPrtnInfo(i,j,k) = cnt++;
          }

      distHamKS.Prtn() = distEigSol.Prtn();
      distPsi.Prtn()   = distEigSol.Prtn(); 


      for( Int k=0; k< numElem[2]; k++ )
        for( Int j=0; j< numElem[1]; j++ )
          for( Int i=0; i< numElem[0]; i++ ) {
            Index3 key (i,j,k);
            if( distEigSol.Prtn().Owner(key) == mpirank ){
              // Setup the domain in the extended element
              Domain dmExtElem;
              for( Int d = 0; d < DIM; d++ ){
                // Assume the global domain starts from 0.0
                if( numElem[d] == 1 ){
                  dmExtElem.length[d]     = dm.length[d];
                  dmExtElem.numGrid[d]    = esdfParam.numGridWavefunctionElem[d];
                  dmExtElem.numGridFine[d] = esdfParam.numGridDensityElem[d];
                  dmExtElem.posStart[d]   = 0.0;
                }
                else if ( numElem[d] >= 3 ){
                  dmExtElem.length[d]     = dm.length[d]  / numElem[d] * 3;
                  dmExtElem.numGrid[d]    = esdfParam.numGridWavefunctionElem[d] * 3;
                  dmExtElem.numGridFine[d] = esdfParam.numGridDensityElem[d] * 3;
                  dmExtElem.posStart[d]   = dm.length[d]  / numElem[d] * ( key[d] - 1 );
                }
                else{
                  throw std::runtime_error( "numElem[d] is either 1 or >=3." );
                }

                // Do not specify the communicator for the domain yet
                // since it is not used for parallelization
              }


              // Atoms	
              std::vector<Atom>&  atomList = esdfParam.atomList;
              std::vector<Atom>   atomListExtElem;

              Int numAtom = atomList.size();
              for( Int a = 0; a < numAtom; a++ ){
                Point3 pos = atomList[a].pos;
//                Print(statusOFS, "pos before adjust=",pos);//debug 0304
                if( IsInSubdomain( pos, dmExtElem, dm.length ) ){
                  // Update the coordinate relative to the extended
                  // element
                  for( Int d = 0; d < DIM; d++ ){
                    pos[d] -= floor( ( pos[d] - dmExtElem.posStart[d] ) / 
                        dm.length[d] )* dm.length[d]; 
//                    Print(statusOFS, "number of atom=",a);//debug 0304
//                    Print(statusOFS, "pos in element=",pos[d]);//debug 0304
                  }
                  atomListExtElem.push_back( Atom( atomList[a].type,
                        pos, atomList[a].vel, atomList[a].force ) );

//       	          Print(statusOFS, "atomListExtElem_pos    = ",  atomListExtElem[a].pos);
//                 	Print(statusOFS, "atomListExtElem_vel    = ",  atomListExtElem[a].vel);
//                 	Print(statusOFS, "atomListExtElem_force    = ",  atomListExtElem[a].force);
                } // Atom is in the extended element
              }

              // Fourier
              fftExtElem.Initialize( dmExtElem );
              fftExtElem.InitializeFine( dmExtElem );

              // Wavefunction
              Spinor& spn = distPsi.LocalMap()[key];
              spn.Setup( dmExtElem, 1, esdfParam.numALBElem(i,j,k), 0.0 );

//              UniformRandom( spn.Wavefun() );
              // Hamiltonian
              // The exchange-correlation type and numExtraState is not
              // used in the extended element calculation
              statusOFS << "Hamiltonian begin." << std::endl;
              KohnSham& hamKS = distHamKS.LocalMap()[key];

              hamKS.Setup( dmExtElem, atomListExtElem, 
                  esdfParam.pseudoType, esdfParam.XCId );

              // Setup the external barrier potential in the extended element
              Real barrierR = esdfParam.potentialBarrierR;
              Real barrierW = esdfParam.potentialBarrierW;
              Real barrierS = esdfParam.potentialBarrierS;
              std::vector<DblNumVec> gridpos(DIM);
              UniformMesh ( dmExtElem, gridpos );
              // Barrier potential along each dimension
              std::vector<DblNumVec> vBarrier(DIM);

              for( Int d = 0; d < DIM; d++ ){
                Real length   = dmExtElem.length[d];
                Int numGrid   = dmExtElem.numGrid[d];
                Real posStart = dmExtElem.posStart[d]; 
                Real center   = posStart + length / 2.0;
                Real EPS      = 1.0;           // For stability reason
                Real dist;
                statusOFS << "center = " << center << std::endl;
                vBarrier[d].Resize( numGrid );
                SetValue( vBarrier[d], 0.0 );
                for( Int p = 0; p < numGrid; p++ ){
                  dist = std::abs( gridpos[d][p] - center );
                  // Only apply the barrier for region outside barrierR
                  if( dist > barrierR ){
                    vBarrier[d][p] = barrierS * std::exp( - barrierW / 
                        ( dist - barrierR ) ) / std::pow( dist - length / 2.0 - EPS, 2.0 );
                  }
                }
              } // for (d)

#if 0
              statusOFS << "gridpos[0] = " << std::endl << gridpos[0] << std::endl;
              statusOFS << "gridpos[1] = " << std::endl << gridpos[1] << std::endl;
              statusOFS << "gridpos[2] = " << std::endl << gridpos[2] << std::endl;
              statusOFS << "vBarrier[0] = " << std::endl << vBarrier[0] << std::endl;
              statusOFS << "vBarrier[1] = " << std::endl << vBarrier[1] << std::endl;
              statusOFS << "vBarrier[2] = " << std::endl << vBarrier[2] << std::endl;
#endif

              DblNumVec& vext = hamKS.Vext();
              SetValue( vext, 0.0 );
#if 0
              for( Int gk = 0; gk < dmExtElem.numGrid[2]; gk++)
                for( Int gj = 0; gj < dmExtElem.numGrid[1]; gj++ )
                  for( Int gi = 0; gi < dmExtElem.numGrid[0]; gi++ ){
                    Int idx = gi + gj * dmExtElem.numGrid[0] + 
                      gk * dmExtElem.numGrid[0] * dmExtElem.numGrid[1];
                    vext[idx] = vBarrier[0][gi] + vBarrier[1][gj] + vBarrier[2][gk];
                  } // for (gi)
#endif

              hamKS.CalculatePseudoPotential( ptable );

              statusOFS << "Hamiltonian constructed." << std::endl;

              // Eigensolver class
              EigenSolver& eigSol = distEigSol.LocalMap()[key];
              eigSol.Setup( esdfParam, 
                  hamKS, 
                  spn, 
                  fftExtElem );

            } // own this element
          } // for(i)
    }
    GetTime( timeEnd );
    statusOFS << "Time for setting up extended element is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;

    // Setup Fourier
    GetTime( timeSta );

    DistFourier distfft;
    distfft.Initialize( dm, distfftSize );

    GetTime( timeEnd );
    statusOFS << "Time for setting up Distributed Fourier is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;

    // Setup HamDG
    GetTime( timeSta );

    HamiltonianDG hamDG( esdfParam );
    hamDG.CalculatePseudoPotential( ptable );

    GetTime( timeEnd );
    statusOFS << "Time for setting up the DG Hamiltonian is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;

    // Setup SCFDG
    GetTime( timeSta );

    SCFDG  scfDG;
    scfDG.Setup( esdfParam, hamDG, distEigSol, distfft, ptable, contxt );

    GetTime( timeEnd );
    statusOFS << "Time for setting up SCFDG is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;

    // *********************************************************************
    // Solve
    // *********************************************************************

    // Main SCF iteration
    GetTime( timeSta );
    scfDG.Iterate();
    GetTime( timeEnd );
    statusOFS << "Time for SCF iteration is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;

    // Compute force
    if( esdfParam.solutionMethod == "diag" ){
      GetTime( timeSta );

      hamDG.CalculateForce( distfft );
      // Print out the force. 
      // Only master processor output information containing all atoms
//    if( mpirank == 0 ){//
      PrintBlock( statusOFS, "Atomic Force" );
      {
        Point3 forceCM(0.0, 0.0, 0.0);
        std::vector<Atom>& atomList = hamDG.AtomList();
        Int numAtom = atomList.size();
        for( Int a = 0; a < numAtom; a++ ){
          Print( statusOFS, "atom", a, "force", atomList[a].force );
          forceCM += atomList[a].force;
        }
        statusOFS << std::endl;
        Print( statusOFS, "force for centroid: ", forceCM );
        statusOFS << std::endl;
      }
//    }//

      GetTime( timeEnd );
      statusOFS << "Time for computing the force is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

      // Compute the a posteriori error estimator
      GetTime( timeSta );
      DblNumTns  eta2Total, eta2Residual, eta2GradJump, eta2Jump;
      hamDG.CalculateAPosterioriError( 
        eta2Total, eta2Residual, eta2GradJump, eta2Jump );
      GetTime( timeEnd );
      statusOFS << "Time for computing the a posteriori error is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

    // Only master processor output information containing all atoms
    if( mpirank == 0 ){
      PrintBlock( statusOFS, "A Posteriori error" );
      {
        statusOFS << std::endl << "Total a posteriori error:" << std::endl;
        statusOFS << eta2Total << std::endl;
        statusOFS << std::endl << "Residual term:" << std::endl;
        statusOFS << eta2Residual << std::endl;
        statusOFS << std::endl << "Face term:" << std::endl;
        statusOFS << eta2GradJump << std::endl;
        statusOFS << std::endl << "Jump term:" << std::endl;
        statusOFS << eta2Jump << std::endl;
      }
    }
  }
#ifdef _USE_PEXSI_
    if( esdfParam.solutionMethod == "pexsi" ){
      // FIXME Introduce distDMMat to hamDG
      //      hamDG.CalculateForceDM( *distfftPtr_, distDMMat );
    }
#endif

		//Nose-Hoover Parameters
    Int L;
    Real K=0.;
    Real Efree=0.;
    Real Ktot1=0.;
    Real Ktot2=0.; //Kinetic energy including ionic part and thermostat of previous and current step
    Real Edrift=0.0;
    Real xi1=0., xi2=0.;
    Real vxi1 = 0.0, vxi2=0.0;
    Real G1, G2;
    Real s; //scale factor

//    Print(statusOFS, "debug: Q1=",Q1);
//    Print(statusOFS, "debug: Q2=",Q2);

    Real T = 1. / esdfParam.Tbeta;
    Print(statusOFS, "debug: Temperature ",T);

		//*********MD starts***********

    //NHC-MD propagate if NSW!=0
    if (NSW != 0) {
      std::vector<Atom>& atomList1 = esdfParam.atomList;
      std::vector<Atom>& atomList2 = hamDG.AtomList(); //TODO hamKS->hamDG?
      Int numAtom = atomList2.size();

			Real *atomMass;
			atomMass = new Real[numAtom];
			for(Int a=0; a < numAtom; a++) {
			  Int atype = atomList1[a].type;
			  if (ptable.ptemap().find(atype)==ptable.ptemap().end() ){
			   throw std::logic_error( "Cannot find the atom type." );
			  }
			  atomMass[a]=amu2au*ptable.ptemap()[atype].params(PTParam::MASS); //amu2au = 1822.8885
			  Print(statusOFS, "atom Mass  = ", atomMass[a]);
    	}

      std::vector<Point3>  atompos;
      std::vector<Point3>  atomv;
     	std::vector<Point3>  atomforce;

     	atompos.resize( numAtom );
     	atomv.resize( numAtom );
     	atomforce.resize( numAtom );

		 	L=3*numAtom;
//		 	Print(statusOFS, "debug: L", L);

      if(esdfParam.isRestartThermostate){
         fstream fin_v;
         fin_v.open("lastthermo.out",ios::in);
         for(Int i=0; i<numAtom; i++){
           fin_v>> atomv[i][0];
           fin_v>> atomv[i][1];
           fin_v>> atomv[i][2];
//           Print(statusOFS, "Type = ", atomList0[i].type, "Position  = ", atomList0[i].pos);
         }
         fin_v>> vxi1;
         fin_v>> vxi2;
         fin_v>> K;
         fin_v>>xi1;
         fin_v>>xi2;

      PrintBlock( statusOFS, "Read in Atomic Velocity" );
        {
          for( Int a = 0; a < numAtom; a++ ){
            Print( statusOFS, "atom", a, "Velocity   ", atomv[a] );
          }
        }

      Print( statusOFS, "vxi1= ", vxi1);
      Print( statusOFS, "vxi2= ", vxi2 );
      Print( statusOFS, "K= ", K );
      Print( statusOFS, "xi1= ", xi1 );
      Print( statusOFS, "xi2= ", xi2 );

       }//restart read in last velocity of atoms

      else{
        for(Int i=0; i<numAtom; i++) {
      	  for(Int j = 0; j<3; j++)
        	  atomv[i][j]=0.;
     	  }
      }

     	for( Int i = 0; i < numAtom; i++ ){
      	atompos[i]   = atomList1[i].pos;
        atomforce[i] = atomList2[i].force;
      }//x1, f1, v1=0

      for (Int i=0;i<numAtom;i++)
      	Print(statusOFS, "debug: position",atompos[i]);

//      for (Int i=0;i<numAtom;i++)
//        Print(statusOFS, "debug: force",atomforce[i]);

      for (Int n=0; n<NSW; n++){
        Print(statusOFS, "Num of MD step = ", n);

//  chain(K[i], T, dt, vxi1[i], vxi2[i], xi1[i], xi2[i], v[i]);//
        G2 = (Q1*vxi1*vxi1-T)/Q2; 
        Print( statusOFS, "Q1= ", Q1);
        Print( statusOFS, "vxi1= ", vxi1);
        Print( statusOFS, "T= ", T);
        Print( statusOFS, "Q2= ", Q2);
        Print( statusOFS, "G2 ", G2);
        vxi2 = vxi2+G2*dt/4.;
        Print( statusOFS, "vxi2= ", vxi2);
        vxi1 = vxi1*exp(-vxi2*dt/8.);
        Print( statusOFS, "vxi1= ", vxi1);
        G1 = (2*K-L*T)/Q1;
        Print( statusOFS, "K= ", K);
        Print( statusOFS, "L= ", L);
        Print( statusOFS, "G1= ", G1);

        vxi1 = vxi1+G1*dt/4.;
        Print( statusOFS, "vxi1= ", vxi1);
        vxi1 = vxi1*exp(-vxi2*dt/8.);
        Print( statusOFS, "vxi1= ", vxi1);
        xi1 = xi1+vxi1*dt/2.;
        Print( statusOFS, "xi1= ", xi1);
        xi2 = xi2+vxi2*dt/2.;
        Print( statusOFS, "xi2= ", xi2);
        s = exp(-vxi1*dt/2.);
        Print( statusOFS, "s= ", s);

        for(Int i=0; i<numAtom; i++){
        	for(Int j=0; j<3; j++)
          	atomv[i][j]=s*atomv[i][j];
        } // v = s*v;
        K=K*s*s;
        Print( statusOFS, "K= ", K);
        vxi1 = vxi1*exp(-vxi2*dt/8.);
        Print( statusOFS, "vxi1= ", vxi1);
        G1 = (2*K-L*T)/Q1;
        Print( statusOFS, "G1= ", G1);
        vxi1 = vxi1+G1*dt/4.;
        Print( statusOFS, "vxi1= ", vxi1);
        vxi1 = vxi1*exp(-vxi2*dt/8.);
        Print( statusOFS, "vxi1= ", vxi1);
        G2 = (Q1*vxi1*vxi1-T)/Q2;
        Print( statusOFS, "G2= ", G2);
        vxi2 = vxi2+G2*dt/4.;
        Print( statusOFS, "vxi2= ", vxi2);

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
//          	Print(statusOFS, "Current Position before SCF    = ",  atomList1[i].pos);
        }//x=x+v*dt/2

        PrintBlock( statusOFS, "Atomic Position before SCF" );
        {
          for( Int a = 0; a < numAtom; a++ ){
            Print( statusOFS, "atom", a, "Position   ", atomList1[a].pos );
          }
        }


//debug
//				std::vector<Atom>& atomList4 = esdfParam.atomList;
//        for(Int i = 0; i < numAtom; i++){
//        	Print(statusOFS, "Write in esdfParam? position   = ",  atomList4[i].pos); //checked! correct
//        }//x=x+v*dt/2
//new add-ons//

//update atomListExtElem with new atomList.pos

//      std::vector<Atom>&  atomList = esdfParam.atomList; //atomList1
			  std::vector<Atom>   atomListExtElem;
	      Index3  numElem = esdfParam.numElem;
   		  	for( Int k=0; k< numElem[2]; k++ )
        		for( Int j=0; j< numElem[1]; j++ )
          		for( Int i=0; i< numElem[0]; i++ ) {
            		Index3 key (i,j,k);
		            if( distEigSol.Prtn().Owner(key) == mpirank ){
    		          // Setup the domain in the extended element
        		      Domain dmExtElem;
            		  for( Int d = 0; d < DIM; d++ ){
                		// Assume the global domain starts from 0.0
		                if( numElem[d] == 1 ){
    		              dmExtElem.length[d]     = dm.length[d];
        		          dmExtElem.numGrid[d]    = esdfParam.numGridWavefunctionElem[d];
											dmExtElem.numGridFine[d] = esdfParam.numGridDensityElem[d];
            		      dmExtElem.posStart[d]   = 0.0;
                		}
		                else if ( numElem[d] >= 3 ){
    		              dmExtElem.length[d]     = dm.length[d]  / numElem[d] * 3;
        		          dmExtElem.numGrid[d]    = esdfParam.numGridWavefunctionElem[d] * 3;
											dmExtElem.numGridFine[d] = esdfParam.numGridDensityElem[d] * 3;
            		      dmExtElem.posStart[d]   = dm.length[d]  / numElem[d] * ( key[d] - 1 );
                		}
		                else{
    		              throw std::runtime_error( "numElem[d] is either 1 or >=3." );
        		        }

            		    // Do not specify the communicator for the domain yet
                		// since it is not used for parallelization
		              } //(d)

		              for( Int a = 0; a < numAtom; a++ ){
		              	Point3 pos = atomList1[a].pos;
//                    Print(statusOFS, "pos before adjust=",pos);//debug 0304
			              if( IsInSubdomain( pos, dmExtElem, dm.length ) ){
  	  		             // Update the coordinate relative to the extended
    	    		         // element
      	      		   	for( Int d = 0; d < DIM; d++ ){
        	        	  	pos[d] -= floor( ( pos[d] - dmExtElem.posStart[d] ) / 
          	         		    dm.length[d] )* dm.length[d];
//                        Print(statusOFS, "number of atom=",a);//debug 0304
//                        Print(statusOFS, "pos in element=",pos[d]);//debug 0304
            	      	}
		          	       atomListExtElem.push_back( Atom( atomList1[a].type, //atomList[a].type
    		        	          pos, atomList1[a].vel, atomList1[a].force ) );

//          	          Print(statusOFS, "atomListExtElem_pos    = ",  atomListExtElem[a].pos);
//                    	Print(statusOFS, "atomListExtElem_vel    = ",  atomListExtElem[a].vel);
//                    	Print(statusOFS, "atomListExtElem_force    = ",  atomListExtElem[a].force);
        		      	} // Atom is in the extended element
            		  }

//debug
//				          std::vector<Atom>& atomList5 = esdfParam.atomList;
//                  for(Int i = 0; i < numAtom; i++){
//                          Print(statusOFS, "After partition position   = ",  atomList5[i].pos); //checked! correct
//                  }

/*		              Spinor& spn = distPsi.LocalMap()[key];//debug_9
    		          spn.Setup( dmExtElem, 1, esdfParam.numALBElem(i,j,k), 0.0 );//debug_9

		              UniformRandom( spn.Wavefun() );//debug_9
*/
			            KohnSham& hamKS = distHamKS.LocalMap()[key];
		
//    			        hamKS.Setup( dmExtElem, atomListExtElem, 
//          			      esdfParam.pseudoType, esdfParam.XCId );

  			          hamKS.Update( atomListExtElem );

//	             		hamKS.CalculatePseudoPotential( ptable );

  	             	hamKS.UpdatePseudoPotential( ptable );

					        statusOFS << "Hamiltonian updated." << std::endl;

/*		              EigenSolver& eigSol = distEigSol.LocalMap()[key];//debug_9
     		          eigSol.Setup( esdfParam,
         		          hamKS,
             		      spn,
                 		  fftExtElem );//debug_9*/

					      }//own this element
              }//(i)


			  statusOFS << "Finish hamKS UpdatePseudoPotential" << std::endl;

				hamDG.UpdateHamiltonianDG( esdfParam.atomList );

				statusOFS << "Finish HamiltonianDG Update" << std::endl;

				hamDG.UpdatePseudoPotential( ptable );

				statusOFS << "Finish UpdatePseudoPotential DG." << std::endl;

    		scfDG.Update( esdfParam, hamDG, distEigSol, distfft, ptable, contxt );

				statusOFS << "Finish Update scfDG" << std::endl;

    		scfDG.Iterate();

				statusOFS << "Finish scfDG Iterate" << std::endl;

        if( esdfParam.solutionMethod == "diag" ){
          hamDG.CalculateForce( distfft );

//				if( mpirank == 0 ){//
				  PrintBlock( statusOFS, "Atomic Force" );
				  {
				    Point3 forceCM(0.0, 0.0, 0.0);
				    std::vector<Atom>& atomList = hamDG.AtomList();
				    Int numAtom = atomList.size();
				    for( Int a = 0; a < numAtom; a++ ){
          		atomforce[a]=atomList[a].force;
				      Print( statusOFS, "atom", a, "force", atomList[a].force );
				      forceCM += atomList[a].force;
				    }
				    statusOFS << std::endl;
				    Print( statusOFS, "force for centroid: ", forceCM );
				    statusOFS << std::endl;
				  }
        }

        if( esdfParam.solutionMethod == "pexsi" ){
          // FIXME Introduce distDMMat to hamDG
          //      hamDG.CalculateForceDM( *distfftPtr_, distDMMat );
        }
//				}//Update Force//

//debug block//
//        for (Int i=0;i<numAtom;i++)
//        	Print(statusOFS, "debug: check position after SCF ",atompos[i]); //checked! correct

//	      std::vector<Atom>& atomList3 = esdfParam.atomList;
//        for(Int i = 0; i < numAtom; i++){
//          Print(statusOFS, "Current Position after SCF   = ",  atomList3[i].pos); //checked! correct
//        }//x=x+v*dt/2
//debug block ends//

        for(Int i=0; i<numAtom; i++){
        	for(Int j=0; j<3; j++){
          	atomv[i][j]=atomv[i][j]+atomforce[i][j]*dt/atomMass[i]; //v=v+f*dt
            atompos[i][j]=atompos[i][j]+atomv[i][j]*dt/2.; //x=x+v*dt/2
            K += atomMass[i]*atomv[i][j]*atomv[i][j]/2.;
          }
        }

//debug block//
//        for (Int i=0;i<numAtom;i++)
//            Print(statusOFS, "debug: velocity",atomv[i]);

/*        PrintBlock( statusOFS, "Updated Atomic Velocity" );
        {
          for( Int a = 0; a < numAtom; a++ ){
            Print( statusOFS, "atom", a, "Velocity   ", atomv[a] );
          }
        }*/

        PrintBlock( statusOFS, "Updated Atomic Position" );
        {
          for( Int a = 0; a < numAtom; a++ ){
            Print( statusOFS, "atom", a, "Position   ", atompos[a] );
          }
        }

        if(esdfParam.isOutputPosition){
          fstream fout;
          fout.open("lastPos.out",ios::out);
          if( !fout.good() ){
             throw std::logic_error( "File cannot be open!" );
           }
          for(Int i=0; i<numAtom; i++){
            fout<< setw(16)<< atompos[i][0];
            fout<< setw(16)<< atompos[i][1];
            fout<< setw(16)<< atompos[i][2];
            fout<< "\n";
          }
          fout.close();
        }

//        for (Int i=0;i<numAtom;i++)
//            Print(statusOFS, "debug: position",atompos[i]);
//end of debug block//  

//        Print(statusOFS, "debug: Kinetic energy of ions ",K);

				//  chain(K[i], T, dt, vxi1[i], vxi2[i], xi1[i], xi2[i], v[i]);//
        G2 = (Q1*vxi1*vxi1-T)/Q2;
        Print( statusOFS, "Q1= ", Q1);
        Print( statusOFS, "vxi1= ", vxi1);
        Print( statusOFS, "T= ", T);
        Print( statusOFS, "G2= ", G2);
        vxi2 = vxi2+G2*dt/4.;
        Print( statusOFS, "vxi2= ", vxi2);
        vxi1 = vxi1*exp(-vxi2*dt/8.);
        Print( statusOFS, "vxi1= ", vxi1);
        G1 = (2*K-L*T)/Q1;
        Print( statusOFS, "G1= ", G1);
        vxi1 = vxi1+G1*dt/4.;
        Print( statusOFS, "vxi1= ", vxi1);
        vxi1 = vxi1*exp(-vxi2*dt/8.);
        Print( statusOFS, "vxi1= ", vxi1);
        xi1 = xi1+vxi1*dt/2.;
        Print( statusOFS, "xi= ", xi1);
        xi2 = xi2+vxi2*dt/2.;
        Print( statusOFS, "xi2= ", xi2);
        s = exp(-vxi1*dt/2.);
        Print( statusOFS, "s= ", s);
        for(Int i=0; i<numAtom; i++){
            for(Int j=0; j<3; j++)
                atomv[i][j]=s*atomv[i][j];
        } // v = s*v;
        K=K*s*s;
        Print( statusOFS, "K= ", K);
        vxi1 = vxi1*exp(-vxi2*dt/8.);
        Print( statusOFS, "vxi1= ", vxi1);
        G1 = (2*K-L*T)/Q1;
        Print( statusOFS, "G1= ", G1);
        vxi1 = vxi1+G1*dt/4.;
        Print( statusOFS, "vxi1= ", vxi1);
        vxi1 = vxi1*exp(-vxi2*dt/8.);
        Print( statusOFS, "vxi1= ", vxi1);
        G2 = (Q1*vxi1*vxi1-T)/Q2;
        Print( statusOFS, "G2= ", G2);
        vxi2 = vxi2+G2*dt/4.;
        Print( statusOFS, "vxi2= ", vxi2);

        PrintBlock( statusOFS, "Scaled Atomic Velocity" );
        {
          for( Int a = 0; a < numAtom; a++ ){
            Print( statusOFS, "atom", a, "Velocity   ", atomv[a] );
          }
        }

        Print(statusOFS, "debug: K*s*s: Kinetic energy II of ions ",K);

        if(esdfParam.isOutputThermostate){
          fstream fout_v;
          fout_v.open("lastthermo.out",ios::out);
          if( !fout_v.good() ){
             throw std::logic_error( "File cannot be open!" );
           }
          for(Int i=0; i<numAtom; i++){
            fout_v<< setw(16)<< atomv[i][0];
            fout_v<< setw(16)<< atomv[i][1];
            fout_v<< setw(16)<< atomv[i][2];
            fout_v<< "\n";
          }
          fout_v<<setw(16)<< vxi1<<"\n";
          fout_v<<setw(16)<< vxi2<<"\n";
          fout_v<<setw(16)<< K<<"\n";
          fout_v<<setw(16)<<xi1<<"\n";
          fout_v<<setw(16)<<xi2<<"\n";
          fout_v.close();
        }
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
        Efree = scfDG.getEfree();
        Print(statusOFS, "MD_Efree =  ",Efree);
        Print(statusOFS, "MD_K =  ",K);

//        Ktot2 =K+Efree+Q1*vxi1*vxi1/2.+Q2*vxi2*vxi2/2.+T*L*xi1+T*L*xi2;
        Ktot2 =K+Efree+Q1*vxi1*vxi1/2.+Q2*vxi2*vxi2/2.+L*T*xi1+T*xi2;
        if(n == 0)
          Ktot1 = Ktot2;

        Edrift= (Ktot2-Ktot1)/Ktot1;
        Print(statusOFS, "Conserved Energy: Ktot =  ",Ktot2);
        Print(statusOFS, "Drift of Conserved Energy: Edrift =  ",Edrift);

//        Ktot2 = 0.;


      }//for(n<NSW) loop ends here
    }//if(NSW!=0) ends
//****MD end*******


    // *********************************************************************
    // Clean up
    // *********************************************************************

    // Finish Cblacs
    Cblacs_gridexit( contxt );

    // Finish fftw
    fftw_mpi_cleanup();

  }//(try)
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
