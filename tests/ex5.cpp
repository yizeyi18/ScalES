// *********************************************************************
// Test spinors
// *********************************************************************
#include "dgdft.hpp"
#include "spinor.hpp"

using namespace dgdft;
using namespace std;

static char help[] = "Test spinors";

int main(int argc, char **argv) 
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);

  try
  {
#ifndef _USE_COMPLEX_
    throw std::runtime_error("This test program require the usage of complex");
#endif
#ifdef  _RELEASE_
    throw std::runtime_error("Test should be run under debug mode");
#endif
    PushCallStack("Main");

    int mpirank, mpisize;
    MPI_Comm_rank( PETSC_COMM_WORLD, &mpirank );
    MPI_Comm_size( PETSC_COMM_WORLD, &mpisize );

    Domain  dm1;
    dm1.length = Point3( 1.0, 2.0, 3.0 );
    dm1.numGrid = Index3( 1, 2, 4 );
    dm1.typeGrid = UNIFORM;
    dm1.posStart = Point3( 1.0, 0.1, 0.0 );

    int numGridLocal = dm1.NumGridTotal() / mpisize;
    Spinor  spn2( dm1, numGridLocal, 1, 2, Complex(1.0, 1.0) );
    Spinor  spn3( dm1, spn2.NumGridLocal(), spn2.NumComponent(),
        spn2.NumState(), false, spn2.LocalWavefun().Data() );

    cout << "Spn2 wavefun(0,0)" << endl;
    cout << " localWavefun[0] = " << spn2.LocalWavefun()[0] << endl;
    VecView( spn2.LockedWavefun(0,0),PETSC_VIEWER_STDOUT_WORLD );
    cout << endl;

    cout << "Modify the local data of spn2 by direct modification" << endl;
    spn2.LocalWavefun()[0] = Complex(1.0, 2.0);
    cout << " localWavefun[0] = " << spn2.LocalWavefun()[0] << endl;
    cout << "Spn2 wavefun(0,0)" << endl;
    VecView( spn2.LockedWavefun(0,0),PETSC_VIEWER_STDOUT_WORLD );
    cout << endl;

    cout << "Modify the local data of spn2 again by VecSetValue" << endl;
    VecSetValue( spn2.Wavefun(0,0), 0, Complex( 1.0, 3.0 ), INSERT_VALUES );
    VecAssemblyBegin( spn2.Wavefun(0,0) );
    VecAssemblyEnd( spn2.Wavefun(0,0) );
    cout << " localWavefun[0] = " << spn2.LocalWavefun()[0] << endl;
    cout << "Spn2 wavefun(0,0)" << endl;
    VecView( spn2.LockedWavefun(0,0),PETSC_VIEWER_STDOUT_WORLD );
    cout << endl;

    cout << "Modify spn3 to change spn2" << endl;
    ierr = VecAXPY( spn3.Wavefun(0,0), 2.0, spn3.Wavefun(0, 1) );
    if( ierr ) throw ierr;
    cout << "Spn2 wavefun(0,0)" << endl;
    ierr = VecView( spn2.LockedWavefun(0,0),PETSC_VIEWER_STDOUT_WORLD );
    if( ierr ) throw ierr;
    cout << "Spn3 wavefun(0,0)" << endl;
    ierr = VecView( spn3.LockedWavefun(0,0),PETSC_VIEWER_STDOUT_WORLD );
    if( ierr ) throw ierr;
    cout << endl;

    PopCallStack();
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


  ierr = PetscFinalize();

  return 0;
}
