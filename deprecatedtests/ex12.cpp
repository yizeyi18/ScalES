// *********************************************************************
// Test for utilities and input interface 
// 
// This test subroutine is to be revised in the next step when
// periodictable is updated.
// *********************************************************************
#include "scales.hpp"
#include "spinor.hpp"
#include "fourier.hpp"
#include "esdf.hpp"
#include "utility.hpp"


using namespace scales;
using namespace std;
using namespace esdf;
using scales::esdf::ESDFInputParam;
using scales::esdf::ESDFReadInput;

static char help[] = "Test the utilities and input interface";

int main(int argc, char **argv) 
{
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help); CHKERRQ(ierr);

  int mpirank, mpisize;
  MPI_Comm_rank( PETSC_COMM_WORLD, &mpirank );
  MPI_Comm_size( PETSC_COMM_WORLD, &mpisize );

  try
  {
#ifdef  _RELEASE_
    throw std::runtime_error("Test should be run under debug mode");
#endif
    stringstream  ss;
    ss << "statfile." << mpirank;
    cout << "The filename for the statfile is " << ss.str() << endl;
    statusOFS.open( ss.str().c_str() );



    ESDFInputParam  esdfParam;

    ESDFReadInput( esdfParam, "scales.in" );


    Print(statusOFS, "Etot              = ",  1.23234, "[Ry]");

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
    Print(statusOFS, "Extra states      = ",  esdfParam.numExtraStates );
    Print(statusOFS, "PeriodTable File  = ",  esdfParam.periodTableFile );
    Print(statusOFS, "Pseudo Type       = ",  esdfParam.pseudoType );
    Print(statusOFS, "PW Solver         = ",  esdfParam.PWSolver );
    Print(statusOFS, "XC Type           = ",  esdfParam.XCType );


    statusOFS << "************************************************************" <<endl;
    statusOFS << "          Atom Type and Coordinates                         " <<endl;

    std::vector<Atom>&  atomList = esdfParam.atomList;
    for(int i=0; i < atomList.size(); i++) {
      Print(statusOFS, "Type = ", atomList[i].type, "Position  = ", atomList[i].pos);
    }



    statusOFS.close();

    std::vector<Real> a(5);
    for(int i = 0; i < 5; i++){
      a[i] = 2.3*i;
    }

    cout << a << endl;

  }
  catch( std::exception& e )
  {
    std::cerr << " caught exception with message: "
      << e.what() << std::endl;
    DumpCallStack();
  }
  catch ( PetscErrorCode ierr ){
    std::cerr << "Caught PETSc error message " << std::endl;
    DumpCallStack();
    CHKERRQ(ierr);
  }

  ierr = SlepcFinalize();CHKERRQ(ierr);

  return 0;
}

