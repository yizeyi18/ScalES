// *********************************************************************
// Test for using the periodic table structure
// 
// This test subroutine is to be revised in the next step when
// periodictable is updated.
// *********************************************************************
#include "dgdft.hpp"
#include "spinor.hpp"
#include "fourier.hpp"
#include "utility.hpp"

#define EPSMODIFIEDBLOPEX      "modified_blopex"

using namespace dgdft;
using namespace std;

static char help[] = "Test the periodtable class";


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


    Print(statusOFS, "Etot           = ",  1.23234, "[Ry]");
    Print(statusOFS, "Super cell     = ",  Point3(2.0,2.0,3.0) );
    Print(statusOFS, "Grid size      = ",  Index3(20,20,30) );


    statusOFS.close();


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

