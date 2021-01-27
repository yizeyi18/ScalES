/// @file ex23.cpp
/// @brief Test for SCFDG class.
///
/// @author Lin Lin
/// @date 2013-02-06
#include "scales.hpp"

using namespace scales;
using namespace std;
using namespace scales::esdf;
using namespace scales::scalapack;

#define _DEBUGlevel_ 1

void Usage(){
  cout << "Test for SCFDG" << endl << endl;
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
    // Initialize log file
    stringstream  ss;
    ss << "statfile." << mpirank;
    cout << "The filename for the statfile is " << ss.str() << endl;
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

    ESDFInputParam  esdfParam;

    ESDFReadInput( esdfParam, "scales.in" );

    // Adjust the input parameters
    {
      bool isGridAdjusted = false;
      Index3& numGrid = esdfParam.domain.numGrid;
      Index3& numElem = esdfParam.numElem;
      for( Int d = 0; d < DIM; d++ ){
        if( numGrid[d] % numElem[d] != 0 ){
          numGrid[d] = IRound( (Real)numGrid[d] / numElem[d] ) * numElem[d];
          isGridAdjusted = true;
        }
      }
      if( isGridAdjusted ){
        statusOFS << std::endl 
          << "Grid size is adjusted to be a multiple of the number of elements." 
          << std::endl;
      }
    }

    // Print the initial state
    {
      PrintBlock(statusOFS, "Basic information");

      Print(statusOFS, "Super cell        = ",  esdfParam.domain.length );
      Print(statusOFS, "Grid size         = ",  esdfParam.domain.numGrid ); 
      Print(statusOFS, "Mixing dimension  = ",  esdfParam.mixMaxDim );
      Print(statusOFS, "Mixing type       = ",  esdfParam.mixType );
      Print(statusOFS, "Mixing Steplength = ",  esdfParam.mixStepLength);
      Print(statusOFS, "SCF Tolerence     = ",  esdfParam.scfTolerance);
      Print(statusOFS, "SCF MaxIter       = ",  esdfParam.scfMaxIter);
      Print(statusOFS, "Eig Tolerence     = ",  esdfParam.eigTolerance);
      Print(statusOFS, "Eig MaxIter       = ",  esdfParam.eigMaxIter);

      Print(statusOFS, "RestartDensity    = ",  esdfParam.isRestartDensity);
      Print(statusOFS, "RestartWfn        = ",  esdfParam.isRestartWfn);
      Print(statusOFS, "OutputDensity     = ",  esdfParam.isOutputDensity);
      Print(statusOFS, "OutputWfn         = ",  esdfParam.isOutputWfn);

      Print(statusOFS, "Temperature       = ",  au2K / esdfParam.Tbeta, "[K]");
      Print(statusOFS, "Extra states      = ",  esdfParam.numExtraState );
      Print(statusOFS, "PeriodTable File  = ",  esdfParam.periodTableFile );
      Print(statusOFS, "Pseudo Type       = ",  esdfParam.pseudoType );
      Print(statusOFS, "PW Solver         = ",  esdfParam.PWSolver );
      Print(statusOFS, "XC Type           = ",  esdfParam.XCType );

      Print(statusOFS, "Penalty Alpha     = ",  esdfParam.penaltyAlpha );
      Print(statusOFS, "Element size      = ",  esdfParam.numElem ); 
      Print(statusOFS, "LGL Grid size     = ",  esdfParam.numGridLGL ); 
      Print(statusOFS, "ScaLAPACK block   = ",  esdfParam.scaBlockSize); 
      Print(statusOFS, "ALB per element   = ",  esdfParam.numALB ); 


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
    ptable.Setup( esdfParam.periodTableFile );


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
                  dmExtElem.numGrid[d]    = dm.numGrid[d];
                  dmExtElem.posStart[d]   = 0.0;
                }
                else if ( numElem[d] >= 4 ){
                  dmExtElem.length[d]     = dm.length[d]  / numElem[d] * 3;
                  dmExtElem.numGrid[d]    = dm.numGrid[d] / numElem[d] * 3;
                  dmExtElem.posStart[d]   = dm.length[d]  / numElem[d] * ( key[d] - 1 );
                }
                else{
                  throw std::runtime_error( "numElem[d] is either 1 or >=4." );
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

#if ( _DEBUGlevel_ >= 1 )
              statusOFS << "Atoms in the extended element" << std::endl;
              for(std::vector<Atom>::iterator vi = atomListExtElem.begin();
                  vi != atomListExtElem.end(); vi++ ){
                statusOFS << (*vi).pos << std::endl;
              }
#endif

              // Fourier
              fftExtElem.Initialize( dmExtElem );

              // Wavefunction
              Spinor& spn = distPsi.LocalMap()[key];
              spn.Setup( dmExtElem, 1, esdfParam.numALB, 0.0 );

              UniformRandom( spn.Wavefun() );

#if ( _DEBUGlevel_ >= 1 )
              statusOFS << "Energy of wavefunction = " << 
                Energy( spn.Wavefun() ) << std::endl;
#endif

              // Hamiltonian
              // The exchange-correlation type and numExtraState is not
              // used in the extended element calculation
              statusOFS << "Hamiltonian begin." << std::endl;
              KohnSham& hamKS = distHamKS.LocalMap()[key];

              hamKS.Setup( dmExtElem, atomListExtElem, 
                  esdfParam.pseudoType, esdfParam.XCId );

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


    DistFourier distfft;

    // FIXME
    distfft.Initialize( dm, mpisize );

    // Setup HamDG
    // FIXME for the time being, calculate PP twice
    HamiltonianDG hamDG( esdfParam );

    hamDG.CalculatePseudoPotential( ptable );

    SCFDG  scfDG;

    scfDG.Setup( esdfParam, hamDG, distEigSol, distfft, ptable, contxt );

    scfDG.Iterate();


    // Finish Cblacs
    Cblacs_gridexit( contxt );

    // Finish fftw
    fftw_mpi_cleanup();

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
