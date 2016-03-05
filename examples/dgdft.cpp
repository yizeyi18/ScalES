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

    // Print the initial state
    ESDFPrintInput( esdfParam );


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
      Int dmCol = numElem[0] * numElem[1] * numElem[2];
      Int dmRow = mpisize / dmCol;
      Int numALBElement = esdfParam.numALBElem(0,0,0);

      if( mpisize > (dmCol * numALBElement) ){
        std::ostringstream msg;
        msg << "Total number of processors is too large! " << std::endl;
        msg << "The maximum number of processors is " << dmCol * numALBElement << std::endl;
        throw std::runtime_error( msg.str().c_str() );
      }
    
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
              hamKS.Setup( esdfParam, dmExtElem, atomListExtElem );
              GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 1 )
              statusOFS << "Time for hamKS.Setup = " << timeEnd - timeSta << std::endl;
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

    // *********************************************************************
    // Solve
    // *********************************************************************

    // Main SCF iteration
    GetTime( timeSta );
    scfDG.Iterate();
    GetTime( timeEnd );
    statusOFS << "Time for SCF iteration is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;

    Real efreeHarris, etot, efree, ekin, ehart, eVxc, exc, evdw,
         eself, ecor, fermi, scfOuterNorm, efreeDifPerAtom;

    scfDG.LastSCF(efreeHarris, etot, efree, ekin, ehart, eVxc, exc, evdw,
        eself, ecor, fermi, scfOuterNorm, efreeDifPerAtom );

    std::vector<Atom>& atomList = hamDG.AtomList(); 
    Real VDWEnergy = 0.0;
    DblNumMat VDWForce;
    VDWForce.Resize( atomList.size(), DIM );
    SetValue( VDWForce, 0.0 );

    if( esdfParam.VDWType == "DFT-D2"){
      scfDG.CalculateVDW ( VDWEnergy, VDWForce );
    } 
   
    efreeHarris += VDWEnergy;
    etot        += VDWEnergy;
    efree       += VDWEnergy;
    ecor        += VDWEnergy;

    // Print out the energy
    PrintBlock( statusOFS, "Energy" );
    statusOFS 
      << "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself + Evdw" << std::endl
      << "       Etot  = Ekin + Ecor" << std::endl
      << "       Efree = Etot	+ Entropy" << std::endl << std::endl;
    Print(statusOFS, "EfreeHarris           = ",  efreeHarris, "[au]");
    Print(statusOFS, "Etot                  = ",  etot, "[au]");
    Print(statusOFS, "Efree                 = ",  efree, "[au]");
    Print(statusOFS, "Ekin                  = ",  ekin, "[au]");
    Print(statusOFS, "Ehart                 = ",  ehart, "[au]");
    Print(statusOFS, "EVxc                  = ",  eVxc, "[au]");
    Print(statusOFS, "Exc                   = ",  exc, "[au]"); 
    Print(statusOFS, "Evdw                  = ",  VDWEnergy, "[au]"); 
    Print(statusOFS, "Eself                 = ",  eself, "[au]");
    Print(statusOFS, "Ecor                  = ",  ecor, "[au]");
    Print(statusOFS, "Fermi                 = ",  fermi, "[au]");
	  Print(statusOFS, "norm(out-in)/norm(in) = ",  scfOuterNorm ); 
		Print(statusOFS, "Efree diff per atom   = ",  efreeDifPerAtom, "[au]"); 

    // Print out the force

    PrintBlock( statusOFS, "Atomic Force" );
    // Compute force
    {
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
   
    }

    if( mpirank == 0 ){
      Point3 forceCM(0.0, 0.0, 0.0);
      std::vector<Atom>& atomList = hamDG.AtomList(); 
      Int numAtom = atomList.size(); 
      for( Int a = 0; a < numAtom; a++ ){
        forceCM += atomList[a].force;
        Print( statusOFS, "atom", a, "Force      ", atomList[a].force );
      }
      statusOFS << std::endl;
      Print( statusOFS, "force for centroid: ", forceCM );
      statusOFS << std::endl;
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
