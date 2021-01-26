// *********************************************************************
// Test for the SCF procedure for KohnSham class
// *********************************************************************
#include "scales.hpp"
#include "utility.hpp"
#include "eigensolver.hpp"
#include "periodtable.hpp"
#include "esdf.hpp"
#include "scf.hpp"

using namespace scales;
using namespace scales::esdf;
using namespace std;

void Usage(){
  cout << "Test for the SCF procedure for KohnSham class" << endl;
}

int main(int argc, char **argv) 
{
  MPI_Init(&argc, &argv);
  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  Usage();

  try
  {

    // *********************************************************************
    // Reading files
    // *********************************************************************


    stringstream  ss;
    ss << "statfile." << mpirank;
    cout << "The filename for the statfile is " << ss.str() << endl;
    statusOFS.open( ss.str().c_str() );

    ESDFInputParam  esdfParam;

    ESDFReadInput( esdfParam, "scales.in" );

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

      PrintBlock(statusOFS, "Atom Type and Coordinates");

      const std::vector<Atom>&  atomList = esdfParam.atomList;
      for(Int i=0; i < atomList.size(); i++) {
        Print(statusOFS, "Type = ", atomList[i].type, "Position  = ", atomList[i].pos);
      }
    }


    // *********************************************************************
    // Preparation
    // *********************************************************************
#ifndef  _RELEASE_
    PushCallStack( "Preparation" );
#endif

    Domain&  dm = esdfParam.domain;

    PeriodTable ptable;
    ptable.Setup( esdfParam.periodTableFile );

    PrintBlock( statusOFS, "Preparing the Hamiltonian" );
    Print( statusOFS, "Periodic table setup finished." );

    KohnSham hamKS( esdfParam, 1 );

    hamKS.CalculatePseudoPotential( ptable );

    Print( statusOFS, "Pseudopotential setup finished." );

    Fourier fft;
    fft.Initialize( dm );

#ifdef _USE_COMPLEX_
    Spinor  spn( dm, 1, hamKS.NumStateTotal(), Complex(1.0, 1.0) );
#else
    Spinor  spn( dm, 1, hamKS.NumStateTotal(), 1.0 );
#endif

    SetRandomSeed(1);

    UniformRandom( spn.Wavefun() );

    EigenSolver eigSol;

    eigSol.Setup( esdfParam, hamKS, spn, fft );

    Print( statusOFS, "Eigensolver setup finished." );

    SCF   scf;
    scf.Setup( esdfParam, eigSol, ptable );


    Print( statusOFS, "SCF setup finished." );

#ifndef  _RELEASE_
    PopCallStack();
#endif

    // *********************************************************************
    // Solve
    // *********************************************************************
    scf.Iterate();

    scf.OutputState();

    statusOFS.close();

  }
  catch( std::exception& e )
  {
    std::cerr << " caught exception with message: "
      << e.what() << std::endl;
#ifndef  _RELEASE_
    DumpCallStack();
#endif
  }

  MPI_Finalize();

  return 0;
}
