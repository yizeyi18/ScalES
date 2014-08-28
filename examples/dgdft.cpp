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
/// @file dgdft.cpp
/// @brief Main driver for DGDFT for self-consistent field iteration.
/// @date 2012-09-16 Original version
/// @date 2014-02-11 Dual grid implementation
/// @date 2014-08-06 Intra-element parallelization
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
    Int contxt, contxt_C;
    Cblacs_get(0, 0, &contxt);
    
    Print( statusOFS, "mpisize = ", mpisize );
    Print( statusOFS, "mpirank = ", mpirank );

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
			Print(statusOFS, "Eig Tolerance Dyn = ",  esdfParam.isEigToleranceDynamic);
			Print(statusOFS, "Num unused state  = ",  esdfParam.numUnusedState);
      Print(statusOFS, "SVD Basis Tol     = ",  esdfParam.SVDBasisTolerance);

      Print(statusOFS, "RestartDensity    = ",  esdfParam.isRestartDensity);
      Print(statusOFS, "RestartWfn        = ",  esdfParam.isRestartWfn);
      Print(statusOFS, "OutputDensity     = ",  esdfParam.isOutputDensity);
      Print(statusOFS, "OutputWfnElem     = ",  esdfParam.isOutputWfnElem);
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


      // Only master processor output information containing all atoms
      if( mpirank == 0 ){
        PrintBlock(statusOFS, "Atom Type and Coordinates");

        const std::vector<Atom>&  atomList = esdfParam.atomList;
        for(Int i=0; i < atomList.size(); i++) {
          Print(statusOFS, "Type = ", atomList[i].type, "Position  = ", atomList[i].pos);
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
    SetRandomSeed(mpirank+10);

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

      Cblacs_gridinit(&contxt, "C", dmRow, dmCol);
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
      Int pmap[numProcScaLAPACK];
      // Take the first numProcScaLAPACK processors for diagonalization
      for ( Int i = 0; i < numProcScaLAPACK; i++ ){
        pmap[i] = i;
      }

      Cblacs_get( contxt, 0, &contxt_C );
      Cblacs_gridmap(&contxt_C, &pmap[0], nprow, nprow, npcol);

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
              for( Int d = 0; d < DIM; d++ ){
               
                dmExtElem.comm    = dm.rowComm;
                dmExtElem.rowComm = dm.rowComm;
                dmExtElem.colComm = dm.rowComm;
              
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
    scfDG.Setup( esdfParam, hamDG, distEigSol, distfft, ptable, contxt_C );

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
      if( mpirank == 0 ){
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
      }

      GetTime( timeEnd );
      statusOFS << "Time for computing the force is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

      // Compute the a posteriori error estimator
//      GetTime( timeSta );
//      DblNumTns  eta2Total, eta2Residual, eta2GradJump, eta2Jump;
//      hamDG.CalculateAPosterioriError( 
//          eta2Total, eta2Residual, eta2GradJump, eta2Jump );
//      GetTime( timeEnd );
//      statusOFS << "Time for computing the a posteriori error is " <<
//        timeEnd - timeSta << " [s]" << std::endl << std::endl;
//
//      // Only master processor output information containing all atoms
//      if( mpirank == 0 ){
//        PrintBlock( statusOFS, "A Posteriori error" );
//        {
//          statusOFS << std::endl << "Total a posteriori error:" << std::endl;
//          statusOFS << eta2Total << std::endl;
//          statusOFS << std::endl << "Residual term:" << std::endl;
//          statusOFS << eta2Residual << std::endl;
//          statusOFS << std::endl << "Face term:" << std::endl;
//          statusOFS << eta2GradJump << std::endl;
//          statusOFS << std::endl << "Jump term:" << std::endl;
//          statusOFS << eta2Jump << std::endl;
//        }
//      }
    }
#ifdef _USE_PEXSI_
    if( esdfParam.solutionMethod == "pexsi" ){
      // FIXME Introduce distDMMat to hamDG
      //      hamDG.CalculateForceDM( *distfftPtr_, distDMMat );
    }
#endif



    // *********************************************************************
    // Clean up
    // *********************************************************************

    // Finish Cblacs
    if(contxt_C >= 0) {
      Cblacs_gridexit( contxt_C );
    }
    
    Cblacs_gridexit( contxt );

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
