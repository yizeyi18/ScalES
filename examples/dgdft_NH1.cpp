/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  
   
   Author: Lin Lin and Gaigong Zhang
	 
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
/// @file dgdft_NH1.cpp
/// @brief Main driver for DGDFT using Nose-Hoover chain level 1
/// @date Original 2013-02-11
/// @date Revision 2014-10-24
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;


void Usage(){
  std::cout 
    << "dgdft -in [inFile]" << std::endl
    << "in:             Input file (default: dgdft.in)" << std::endl;
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
#ifdef _RELEASE_
    // In the release mode, only the master processor outputs information
    if( mpirank == 0 ){
      stringstream  ss;
      ss << "statfile." << mpirank;
      statusOFS.open( ss.str().c_str() );
    }
#else
    // Every processor outputs information
    {
      stringstream  ss;
      ss << "statfile." << mpirank;
      statusOFS.open( ss.str().c_str() );
    }
#endif
    


    // Initialize FFTW
    fftw_mpi_init();

    // Initialize BLACS

    Int nprow, npcol;
    Int contxt;
    //Cblacs_get(0, 0, &contxt);
    
    Print( statusOFS, "mpisize = ", mpisize );
    Print( statusOFS, "mpirank = ", mpirank );

    // Initialize input parameters
    std::map<std::string,std::string> options;
    OptionsCreate(argc, argv, options);

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


      Print(statusOFS, "MD Steps          = ",  esdfParam.MDMaxStep );
      Print(statusOFS, "MD time Step      = ",  esdfParam.MDTimeStep );
      Print(statusOFS, "Thermostat mass   = ",  esdfParam.qMass);
      Print(statusOFS, "RestartPosition   = ",  esdfParam.isRestartPosition);
      Print(statusOFS, "RestartThermostat = ",  esdfParam.isRestartThermostat);
      Print(statusOFS, "OutputPosition    = ",  esdfParam.isOutputPosition );
      Print(statusOFS, "OutputThermostat  = ",  esdfParam.isOutputThermostat );
      Print(statusOFS, "Output XYZ format = ",  esdfParam.isOutputXYZ );


      Print(statusOFS, "Super cell        = ",  esdfParam.domain.length );
      Print(statusOFS, "Grid wfc size     = ",  esdfParam.domain.numGrid ); 
      Print(statusOFS, "Grid rho size     = ",  esdfParam.domain.numGridFine );
      Print(statusOFS, "Mixing dimension  = ",  esdfParam.mixMaxDim );
      Print(statusOFS, "Mixing variable   = ",  esdfParam.mixVariable );
      Print(statusOFS, "Mixing type       = ",  esdfParam.mixType );
      Print(statusOFS, "Mixing Steplength = ",  esdfParam.mixStepLength);
      Print(statusOFS, "SCF Outer Tol     = ",  esdfParam.scfOuterTolerance);
      Print(statusOFS, "SCF Outer MaxIter = ",  esdfParam.scfOuterMaxIter);
      Print(statusOFS, "SCF Free Energy Per Atom Tol = ",  esdfParam.scfOuterEnergyTolerance);
      Print(statusOFS, "SCF Inner Tol     = ",  esdfParam.scfInnerTolerance);
      Print(statusOFS, "SCF Inner MaxIter = ",  esdfParam.scfInnerMaxIter);
      Print(statusOFS, "Eig Tolerence     = ",  esdfParam.eigTolerance);
      Print(statusOFS, "Eig MaxIter       = ",  esdfParam.eigMaxIter);
			Print(statusOFS, "Eig Tolerance Dyn = ",  esdfParam.isEigToleranceDynamic);
			Print(statusOFS, "Num unused state  = ",  esdfParam.numUnusedState);
      Print(statusOFS, "SVD Basis Tol     = ",  esdfParam.SVDBasisTolerance);

      Print(statusOFS, "RestartDensity    = ",  esdfParam.isRestartDensity);
      Print(statusOFS, "RestartWfn        = ",  esdfParam.isRestartWfn);
      Print(statusOFS, "OutputDensity     = ",  esdfParam.isOutputDensity);
      Print(statusOFS, "OutputALBElemLGL  = ",  esdfParam.isOutputALBElemLGL);
      Print(statusOFS, "OutputALBElemUniform  = ",  esdfParam.isOutputALBElemUniform);
      Print(statusOFS, "OutputWfnExtElem  = ",  esdfParam.isOutputWfnExtElem);
      Print(statusOFS, "OutputPotExtElem  = ",  esdfParam.isOutputPotExtElem);
      Print(statusOFS, "OutputHMatrix     = ",  esdfParam.isOutputHMatrix );

      Print(statusOFS, "PeriodizePotential= ",  esdfParam.isPeriodizePotential);
      Print(statusOFS, "DistancePeriodize = ",  esdfParam.distancePeriodize);

      //			Print(statusOFS, "Barrier W         = ",  esdfParam.potentialBarrierW);
      //			Print(statusOFS, "Barrier S         = ",  esdfParam.potentialBarrierS);
      //			Print(statusOFS, "Barrier R         = ",  esdfParam.potentialBarrierR);
      Print(statusOFS, "EcutWavefunction  = ",  esdfParam.ecutWavefunction);
      Print(statusOFS, "Density GridFactor= ",  esdfParam.densityGridFactor);
      Print(statusOFS, "LGL GridFactor    = ",  esdfParam.LGLGridFactor);

      Print(statusOFS, "Temperature       = ",  au2K / esdfParam.Tbeta, "[K]");
      Print(statusOFS, "Ion Temperature   = ",  esdfParam.ionTemperature, "[K]");
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
      Print(statusOFS, "Number of procs for DistFFT  = ",  esdfParam.numProcDistFFT ); 

      Print(statusOFS, "Solution Method   = ",  esdfParam.solutionMethod );
      if( esdfParam.solutionMethod == "diag" ){
        Print(statusOFS, "Number of procs for ScaLAPACK  = ",  esdfParam.numProcScaLAPACK); 
      }
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


      // Read position from lastPos.out into esdfParam.atomList[i].pos if isRestartPosition=1
      if(esdfParam.isRestartPosition){
        std::vector<Atom>&  atomList = esdfParam.atomList;
        Int numAtom = atomList.size();
        DblNumVec atomposRead(3*numAtom);
        // Only master processor read and then distribute
        if( mpirank == 0 ){
          fstream fin;
          fin.open("lastPos.out",ios::in);
          for(Int a=0; a<numAtom; a++){
            fin>> atomposRead[3*a];
            fin>> atomposRead[3*a+1];
            fin>> atomposRead[3*a+2];
          }
          fin.close();
        }
        // Broadcast the atomic position
        MPI_Bcast( atomposRead.Data(), 3*numAtom, MPI_DOUBLE, 0, MPI_COMM_WORLD );
        Point3 pos;
        for(Int a=0; a<numAtom; a++){
          pos = Point3( atomposRead[3*a], atomposRead[3*a+1], atomposRead[3*a+2] );
          atomList[a].pos = pos;
        }

      } //position read in for restart
      if(mpirank == 0){
        const std::vector<Atom>&  atomList = esdfParam.atomList;
        Int numAtom = atomList.size();
        PrintBlock( statusOFS, "Initial Atomic Position" );
        {
          for( Int a = 0; a < numAtom; a++ ){
            Print(statusOFS, "Type = ", atomList[a].type, "Position  = ", atomList[a].pos);
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

    // FIXME IMPORTANT: RandomSeed cannot be the same.
    // SetRandomSeed(1);
    SetRandomSeed(mpirank);

    GetTime( timeSta );

    Domain&  dm = esdfParam.domain;
    PeriodTable ptable;
    ptable.Setup( esdfParam.periodTableFile );

    GetTime( timeEnd );
    statusOFS << "Time for setting up the periodic table is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;

    GetTime( timeSta );

    // Setup the element and extended element information
    DistVec<Index3, EigenSolver, ElemPrtn>  distEigSol; 
    DistVec<Index3, KohnSham, ElemPrtn>     distHamKS;
    DistVec<Index3, Spinor, ElemPrtn>       distPsi;
    // FIXME no use?
    distEigSol.SetComm( dm.comm );
    distHamKS.SetComm( dm.comm );
    distPsi.SetComm( dm.comm );

    // All extended elements share the same Fourier structure.
    Fourier fftExtElem;

    // Setup the eigenvalue solvers in each extended element
    {
      // Element partition information
      Index3  numElem = esdfParam.numElem;

      IntNumTns& elemPrtnInfo = distEigSol.Prtn().ownerInfo;
      elemPrtnInfo.Resize( numElem[0], numElem[1], numElem[2] );

      // Note the usage of notation can be a bit misleading here:
      // dmRow is the number of processors per row, which normally is
      // denoted by number of column processors
      // dmCol is the number of processors per column, which normally is
      // denoted by number of row processors
      int dmCol = numElem[0] * numElem[1] * numElem[2];
      int dmRow = mpisize / dmCol;

      // Cblacs_gridinit(&contxt, "C", dmRow, dmCol);
      Int numProcScaLAPACK = esdfParam.numProcScaLAPACK;
      
      // Here nprow, npcol is for the usage of ScaLAPACK in
      // diagonalization
      for( Int i = IRound(sqrt(double(numProcScaLAPACK))); 
          i <= numProcScaLAPACK; i++){
        nprow = i; npcol = numProcScaLAPACK / nprow;
        if( nprow * npcol == numProcScaLAPACK ) break;
      }

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "nprowSca = " << nprow << std::endl;
      statusOFS << "npcolSca = " << npcol << std::endl;
#endif
      
      Int ldpmap = npcol;
      IntNumVec pmap(numProcScaLAPACK);
      // Take the first numProcScaLAPACK processors for diagonalization
      for ( Int i = 0; i < numProcScaLAPACK; i++ ){
        pmap[i] = i;
      }

      contxt = -1; 
      Cblacs_get( 0, 0, &contxt );
      Cblacs_gridmap(&contxt, &pmap[0], nprow, nprow, npcol);

      if( (mpisize % dmCol) != 0 ){
        std::ostringstream msg;
        msg << "Total number of processors do not fit to the number processors per element." << std::endl;
        throw std::runtime_error( msg.str().c_str() );
      }

      GetTime( timeSta );
      
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
            if( distEigSol.Prtn().Owner(key) == (mpirank / dmRow) ){
              // Setup the domain in the extended element
              Domain dmExtElem;
              dmExtElem.comm    = dm.rowComm;
              dmExtElem.rowComm = dm.rowComm;
              dmExtElem.colComm = dm.rowComm;
              for( Int d = 0; d < DIM; d++ ){
               
              
                // Assume the global domain starts from 0.0
                if( numElem[d] == 1 ){
                  dmExtElem.length[d]      = dm.length[d];
                  dmExtElem.numGrid[d]     = esdfParam.numGridWavefunctionElem[d];
                  dmExtElem.numGridFine[d] = esdfParam.numGridDensityElem[d];
                  dmExtElem.posStart[d]    = 0.0;
                }
                else if ( numElem[d] >= 3 ){
                  dmExtElem.length[d]      = dm.length[d]  / numElem[d] * 3;
                  dmExtElem.numGrid[d]     = esdfParam.numGridWavefunctionElem[d] * 3;
                  dmExtElem.numGridFine[d] = esdfParam.numGridDensityElem[d] * 3;
                  dmExtElem.posStart[d]    = dm.length[d]  / numElem[d] * ( key[d] - 1 );
                }
                else{
                  throw std::runtime_error( "numElem[d] is either 1 or >=3." );
                }

                // Do not specify the communicator for the domain yet
                // since it is not used for parallelization
              } // for d

              // Atoms	
              std::vector<Atom>&  atomList = esdfParam.atomList;
              std::vector<Atom>   atomListExtElem;

              Int numAtom = atomList.size();
              for( Int a = 0; a < numAtom; a++ ){
                Point3 pos = atomList[a].pos;
                if( IsInSubdomain( pos, dmExtElem, dm.length ) ){
                  // Update the coordinate relative to the extended
                  // element
                  for( Int d = 0; d < DIM; d++ ){
                    pos[d] -= floor( ( pos[d] - dmExtElem.posStart[d] ) / 
                        dm.length[d] )* dm.length[d]; 
                  }
                  atomListExtElem.push_back( Atom( atomList[a].type,
                        pos, atomList[a].vel, atomList[a].force ) );
                } // Atom is in the extended element
              }

              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
              statusOFS << "Time for Initialize ExtElem = " << timeEnd - timeSta << std::endl;
#endif

              // Fourier
              GetTime( timeSta );
              fftExtElem.Initialize( dmExtElem );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "Time for fftExtElem.Initialize = " << timeEnd - timeSta << std::endl;
#endif
              
              GetTime( timeSta );
              fftExtElem.InitializeFine( dmExtElem );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "Time for fftExtElem.InitializeFine = " << timeEnd - timeSta << std::endl;
#endif
              // Wavefunction
              //Spinor& spn = distPsi.LocalMap()[key];
              //spn.Setup( dmExtElem, 1, esdfParam.numALBElem(i,j,k), 0.0 );

              GetTime( timeSta );
              
              int dmExtElemMpirank, dmExtElemMpisize;
              MPI_Comm_rank( dmExtElem.comm, &dmExtElemMpirank );
              MPI_Comm_size( dmExtElem.comm, &dmExtElemMpisize );
              int numStateTotal = esdfParam.numALBElem(i,j,k);
              int numStateLocal, blocksize;

              if ( numStateTotal <=  dmExtElemMpisize ) {
                blocksize = 1;

                if ( dmExtElemMpirank < numStateTotal ){
                  numStateLocal = 1; // blocksize == 1;
                }
                else { 
                  // FIXME Throw an error here
                  numStateLocal = 0;
                }
  
              } 
    
              else {  // numStateTotal >  mpisize
      
                if ( numStateTotal % dmExtElemMpisize == 0 ){
                  blocksize = numStateTotal / dmExtElemMpisize;
                  numStateLocal = blocksize ;
                }
                else {
                  // blocksize = ((numStateTotal - 1) / mpisize) + 1;
                  blocksize = numStateTotal / dmExtElemMpisize;
                  numStateLocal = blocksize ;
                  if ( dmExtElemMpirank < ( numStateTotal % dmExtElemMpisize ) ) {
                    numStateLocal = numStateLocal + 1 ;
                  }
                }    

              }

              Spinor& spn = distPsi.LocalMap()[key];
              
              spn.Setup( dmExtElem, 1, numStateTotal, numStateLocal, 0.0 );

              UniformRandom( spn.Wavefun() );
              
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
              statusOFS << "Time for Initialize Spinor = " << timeEnd - timeSta << std::endl;
#endif
              // Hamiltonian
              // The exchange-correlation type and numExtraState is not
              // used in the extended element calculation
              statusOFS << "Hamiltonian begin." << std::endl;
              KohnSham& hamKS = distHamKS.LocalMap()[key];

              GetTime( timeSta );
              hamKS.Setup( dmExtElem, atomListExtElem, 
                  esdfParam.pseudoType, esdfParam.XCType );
              GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 1 )
              statusOFS << "Time for hamKS.Setup = " << timeEnd - timeSta << std::endl;
#endif
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
    distfft.Initialize( dm, esdfParam.numProcDistFFT );

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

    // FIXME
    // This SCF iteration is NOT NECESSARY, but can be more stable
    // especially for the initial step
    // In the future this should be substituted either by a more complete restart scheme,
    // or by an option to do more SCF steps initially, or both
    scfDG.Iterate();
    scfDG.set_Cheby_MD_schedule_flag();

    // LLIN: 11/3/2014. NO SCF calculation here, postpone to MD stpes
    
		// *********************************************************************
		// Nose-Hoover Thermostat: Level 1
    // Algorithm: Frenkel-Smit, pp 540-542
		// *********************************************************************
    if(1){
      Int L;
      Real K=0.; //Kinetic Energy
      Real Efree=0.; 
      Real KtotInitial=0.; //Conserve Quantity of the first step
      Real Ktot=0.; //Conserve Quantity at current step
      Real Edrift=0.0;
      Real xi1=0.;
      Real vxi1 = 0.0;
      Real G1;
      Real s; //scale factor

      Int MDMaxStep = esdfParam.MDMaxStep;
      Real dt = esdfParam.MDTimeStep;
      Real Q1 = esdfParam.qMass;
    


      Real T = esdfParam.ionTemperature;

      //*********MD starts***********
      //NHC-MD propagate if NSW!=0

      if (MDMaxStep > 0) {
        std::vector<Atom>& atomList = hamDG.AtomList(); 
        Int numAtom = atomList.size();

        DblNumVec atomMass( numAtom );
        for(Int a=0; a < numAtom; a++) {
          Int atype = atomList[a].type;
          if (ptable.ptemap().find(atype)==ptable.ptemap().end() ){
            throw std::logic_error( "Cannot find the atom type." );
          }
          atomMass[a]=amu2au*ptable.ptemap()[atype].params(PTParam::MASS); 
        }
#if ( _DEBUGlevel_ >= 0 )
        for(Int a=0; a < numAtom; a++) {
          Print( statusOFS, "atom", a, "Type  ", atomList[a].type, "Mass  ", atomMass[a] );
        }
#endif

        std::vector<Point3>  atompos(numAtom);
        std::vector<Point3>  atomvel(numAtom);
        std::vector<Point3>  atomforce(numAtom);

        // History of density for extrapolation
        Int maxHist = 1; // Linear extrapolation
        std::vector<DistDblNumVec>    densityHist(maxHist);
        // Initialize the history
        for( Int l = 0; l < maxHist; l++ ){
          DistDblNumVec& den = densityHist[l];
          DistDblNumVec& denCur = hamDG.Density();
          Index3& numElem = esdfParam.numElem;
          // FIXME the communicator and partitioning of densityHist is
          // not set since only the localmap is used.
          int dmCol = numElem[0] * numElem[1] * numElem[2];
          int dmRow = mpisize / dmCol;
          for( Int k=0; k< numElem[2]; k++ )
            for( Int j=0; j< numElem[1]; j++ )
              for( Int i=0; i< numElem[0]; i++ ) {
                Index3 key = Index3(i,j,k);
                if( distEigSol.Prtn().Owner(key) == (mpirank / dmRow) ){
                  den.LocalMap()[key]     = denCur.LocalMap()[key];
                } // own this element
              }  // for (i)
        }

        // Degree of freedom
        L=3*numAtom;

        // One processor read and then distribute
        if(esdfParam.isRestartThermostat){
          DblNumVec atomvelRead(3*numAtom);
          if( mpirank == 0 ){
            fstream fin;
            fin.open("lastthermo.out",ios::in);
            for(Int a=0; a<numAtom; a++){
              fin>> atomvelRead[3*a+0];
              fin>> atomvelRead[3*a+1];
              fin>> atomvelRead[3*a+2];
            }
            fin >> vxi1;
            fin >> K;
            fin >> xi1;

            fin.close();
          }
          // Broadcast thermostat information
          MPI_Bcast( atomvelRead.Data(), 3*numAtom, MPI_DOUBLE, 0, MPI_COMM_WORLD );
          MPI_Bcast( &vxi1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
          MPI_Bcast( &K, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
          MPI_Bcast( &xi1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 

          Point3 vel;
          for(Int a=0; a<numAtom; a++){
            vel = Point3( atomvelRead[3*a], atomvelRead[3*a+1], atomvelRead[3*a+2] );
            atomvel[a] = vel;
          }

          if( mpirank == 0 ){
            PrintBlock( statusOFS, "Read in Atomic Velocity" );
            {
              for( Int a = 0; a < numAtom; a++ ){
                Print( statusOFS, "atom", a, "Velocity   ", atomvel[a] );
              }
            }
          }

          Print( statusOFS, "vxi1= ", vxi1);
          Print( statusOFS, "K= ", K );
          Print( statusOFS, "xi1= ", xi1 );

        }//restart read in last velocities of atoms
        else{
          for(Int i=0; i<numAtom; i++) {
            for(Int j = 0; j< DIM; j++)
              atomvel[i][j]=0.;
          }
        }

        for( Int i = 0; i < numAtom; i++ ){
          atompos[i]   = atomList[i].pos;
        }

        for (Int iStep=1; iStep <= MDMaxStep; iStep++){
          {
            std::ostringstream msg;
            msg << "MD step # " << iStep;
            PrintBlock( statusOFS, msg.str() );
          }

          // Propagate the chain
          //numchain=1 start//
          vxi1 = vxi1+(K*2-L*T)/Q1*dt/4.;
          xi1 = xi1+vxi1*dt/2.;
          s=std::exp(-vxi1*dt/2.);
          for(Int i=0;i<numAtom;i++){
            atomvel[i] = s * atomvel[i];
          }

          K=K*s*s;
          vxi1=vxi1+(2*K-L*T)/Q1*dt/4.;
          //numchain=1 end//

          //posvel//
          for(Int i=0; i<numAtom; i++) {
            atompos[i] = atompos[i] + atomvel[i] * dt/2.;
          }

          // Update atomic position in the extended element
          // FIXME This part should be modulated
          if(1){
            std::vector<Atom>   atomListExtElem;
            Index3  numElem = esdfParam.numElem;

            Int dmCol = numElem[0] * numElem[1] * numElem[2];
            Int dmRow = mpisize / dmCol;
            for( Int k=0; k< numElem[2]; k++ )
              for( Int j=0; j< numElem[1]; j++ )
                for( Int i=0; i< numElem[0]; i++ ) {
                  Index3 key (i,j,k);
                  if( distEigSol.Prtn().Owner(key) == (mpirank / dmRow) ){
                    // Setup the domain in the extended element
                    Domain dmExtElem;
                    dmExtElem.comm    = dm.rowComm;
                    dmExtElem.rowComm = dm.rowComm;
                    dmExtElem.colComm = dm.rowComm;
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
                      Point3 pos = atompos[a];
                      if( IsInSubdomain( pos, dmExtElem, dm.length ) ){
                        // Update the coordinate relative to the extended
                        // element
                        for( Int d = 0; d < DIM; d++ ){
                          pos[d] -= floor( ( pos[d] - dmExtElem.posStart[d] ) / 
                              dm.length[d] )* dm.length[d];
                        }
                        atomListExtElem.push_back( Atom( atomList[a].type, 
                              pos, atomList[a].vel, atomList[a].force ) );
                      } // Atom is in the extended element
                    }

                    KohnSham& hamKS = distHamKS.LocalMap()[key];

                    hamKS.Update( atomListExtElem );

                    hamKS.CalculatePseudoPotential( ptable );

                    statusOFS << "Hamiltonian updated." << std::endl;

                  }//own this element
                }//(i)
          }

          statusOFS << "Finish hamKS UpdatePseudoPotential" << std::endl;

          std::vector<Atom> atomListTmp;
          for( Int a = 0; a < numAtom; a++ ){
            atomListTmp.push_back( Atom( atomList[a].type, 
                  atompos[a], atomList[a].vel, atomforce[a] ) );
          }

          hamDG.UpdateHamiltonianDG( atomListTmp );

          statusOFS << "Finish HamiltonianDG Update" << std::endl;

          hamDG.CalculatePseudoPotential( ptable );

          statusOFS << "Finish UpdatePseudoPotential DG." << std::endl;

          scfDG.Update( );

          // Update the density through linear extrapolation
          if(1)
          {
            Index3  numElem = esdfParam.numElem;
            Int dmCol = numElem[0] * numElem[1] * numElem[2];
            Int dmRow = mpisize / dmCol;
            DistDblNumVec& denCur = hamDG.Density();
            for( Int k=0; k< numElem[2]; k++ )
              for( Int j=0; j< numElem[1]; j++ )
                for( Int i=0; i< numElem[0]; i++ ) {
                  Index3 key = Index3(i,j,k);
                  if( distEigSol.Prtn().Owner(key) == (mpirank / dmRow) ){
                    DblNumVec& denCurVec = denCur.LocalMap()[key];
                    DblNumVec& denHist0Vec = densityHist[0].LocalMap()[key];
                    DblNumVec denSaveVec = denCurVec;
                    for( Int a = 0; a < denCurVec.m(); a++ ){
                      denCurVec(a) = 2.0 * denCurVec(a) - denHist0Vec(a);
                      denHist0Vec(a) = denSaveVec(a);
                    }
                  } // own this element
                }  // for (i)
          }



          statusOFS << "Finish Update scfDG" << std::endl;

          scfDG.Iterate();
	  
	  

          statusOFS << "Finish scfDG Iterate" << std::endl;

          std::vector<Atom>& atomList = hamDG.AtomList(); 
          Real VDWEnergy = 0.0;
          DblNumMat VDWForce;
          VDWForce.Resize( atomList.size(), DIM );
          SetValue( VDWForce, 0.0 );

          if( esdfParam.VDWType == "DFT-D2"){
            scfDG.CalculateVDW ( VDWEnergy, VDWForce );
          } 

          // Force calculation
          {
            GetTime( timeSta );
            if( esdfParam.solutionMethod == "diag" ){
              hamDG.CalculateForce( distfft );
            }
            else if( esdfParam.solutionMethod == "pexsi" ){
              hamDG.CalculateForceDM( distfft, scfDG.DMMat() );
            }

            if( esdfParam.VDWType == "DFT-D2"){
              for( Int a = 0; a < atomList.size(); a++ ){
                atomList[a].force += Point3( VDWForce(a,0), VDWForce(a,1), VDWForce(a,2) );
              }
            } 

            for( Int a = 0; a < numAtom; a++ ){
              atomforce[a]=atomList[a].force;
            }
            GetTime( timeEnd );
            statusOFS << "Time for computing the force is " <<
              timeEnd - timeSta << " [s]" << std::endl << std::endl;
          }

          // Output intermediate information
#if ( _DEBUGlevel_ >= 0 )
          if( mpirank == 0 ){
            PrintBlock( statusOFS, "Atomic Position, Velocity, Force at the same time step" ); 
            for( Int a = 0; a < numAtom; a++ ){
              Print( statusOFS, "atom", a, "Position   ", atompos[a] );
            }
            statusOFS << std::endl;
            for( Int a = 0; a < numAtom; a++ ){
              Print( statusOFS, "atom", a, "Velocity   ", atomvel[a] );
            }
            statusOFS << std::endl;

            Point3 forceCM(0.0, 0.0, 0.0);
            for( Int a = 0; a < numAtom; a++ ){
              forceCM += atomList[a].force;
              Print( statusOFS, "atom", a, "Force      ", atomforce[a] );
            }
            statusOFS << std::endl;
            Print( statusOFS, "force for centroid: ", forceCM );
            statusOFS << std::endl;
          }
#endif

          // Update kinetic energy, position and velocity
          K=0.;
          for(Int i=0; i<numAtom; i++){
            atomvel[i] = atomvel[i] + atomforce[i]*dt/atomMass[i]; 
            atompos[i] = atompos[i] + atomvel[i]*dt/2.; 
            for(Int j=0; j<3; j++){
              K += atomMass[i]*atomvel[i][j]*atomvel[i][j]/2.;
            }
          }


          //nuchain=1//
          vxi1 = vxi1+(K*2.-L*T)/Q1*dt/4.;
          xi1  = xi1+vxi1*dt/2.;
          s    = std::exp(-vxi1*dt/2.);
          for(Int i=0;i<numAtom;i++){
            atomvel[i]=s*atomvel[i];
          }
          K=K*s*s;

          vxi1=vxi1+(2*K-L*T)/Q1*dt/4.;
          //numchain=1 end//

          // Output the XYZ format for movie
          if( esdfParam.isOutputXYZ & mpirank == 0 ){
            fstream fout;
            fout.open("MD.xyz",ios::out | ios::app) ;
            if( !fout.good() ){
              throw std::logic_error( "Cannot open MD.xyz!" );
            }
            fout << numAtom << std::endl;
            fout << "MD step # "<< iStep << std::endl;
            for(Int a=0; a<numAtom; a++){
              fout<< setw(6)<< atomList[a].type
                << setw(16)<< atompos[a][0]*au2ang
                << setw(16)<< atompos[a][1]*au2ang
                << setw(16)<< atompos[a][2]*au2ang
                << std::endl;
            }
            fout.close();
          }

          // Output position and thermostat
          if( mpirank == 0 ){
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
                fout<< std::endl;
              }
              fout.close();
            }
          }


          if( mpirank == 0 ){
            if(esdfParam.isOutputThermostat){
              fstream fout_v;
              fout_v.open("lastthermo.out",ios::out);
              if( !fout_v.good() ){
                throw std::logic_error( "File cannot be open!" );
              }
              for(Int i=0; i<numAtom; i++){
                fout_v<< setw(16)<< atomvel[i][0];
                fout_v<< setw(16)<< atomvel[i][1];
                fout_v<< setw(16)<< atomvel[i][2];
                fout_v<< std::endl;
              }
              fout_v<<setw(16)<< vxi1<<std::endl;
              fout_v<<setw(16)<< K<<std::endl;
              fout_v<<setw(16)<<xi1<<std::endl;
              fout_v.close();
            }
          }

          // Output conserved quantities
          Efree = scfDG.Efree();

          Print(statusOFS, "MD_Efree =  ",Efree);
          Print(statusOFS, "MD_K     =  ",K);
          Print(statusOFS, "MD_Fermi =  ",scfDG.Fermi());

          Ktot =K+Efree+Q1*vxi1*vxi1/2.+L*T*xi1;
          if(iStep == 1)
            KtotInitial = Ktot;

          Edrift= (Ktot-KtotInitial)/KtotInitial;
          Print(statusOFS, "Conserved Energy: Ktot =  ",Ktot);
          Print(statusOFS, "Drift of Conserved Energy: Edrift =  ",Edrift);

        }//for(iStep<MDMaxStep) loop ends here
      }//if(MDMaxStep>0)
    }


    // *********************************************************************
    // Clean up
    // *********************************************************************

    // Finish Cblacs
    if(contxt >= 0) {
      Cblacs_gridexit( contxt );
    }
    
    // Finish fftw
    fftw_mpi_cleanup();

    MPI_Comm_free( &dm.rowComm );
    MPI_Comm_free( &dm.colComm );

  }
  catch( std::exception& e )
  {
    std::cerr << "Processor " << mpirank << " caught exception with message: "
      << e.what() << std::endl;
#ifndef _RELEASE_
    DumpCallStack();
#endif
  }

  MPI_Finalize();

  return 0;
}
