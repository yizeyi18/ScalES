// *********************************************************************
// Test for solving the eigenvalue problem for periodic Laplacian
// operator using EigenSolver class
//
//   A = -\Delta / 2 + I
//
//
// *********************************************************************
#include "scales.hpp"
#include "eigensolver.hpp"

using namespace scales;
using namespace std;

static char help[] = "Test for solving the eigenvalue problem for periodic Laplacian operator using SLEPc";

// *********************************************************************
// Matrix operations
// *********************************************************************

int main(int argc, char **argv) 
{
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help); CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nLaplace operator 3D\n\n");CHKERRQ(ierr);

  int mpirank, mpisize;
  MPI_Comm_rank( PETSC_COMM_WORLD, &mpirank );
  MPI_Comm_size( PETSC_COMM_WORLD, &mpisize );

  try
  {
#ifndef _USE_COMPLEX_
    throw std::runtime_error("This test program require the usage of complex");
#endif
#ifdef  _RELEASE_
    throw std::runtime_error("Test should be run under debug mode");
#endif


    // *********************************************************************
    // Preparation
    // *********************************************************************

    PushCallStack( "Preparation" );

    Domain  dm;
    dm.length  = Point3( 1.0, 1.0, 1.0 );
    dm.numGrid = Index3( 24, 72, 72);

    Hamiltonian ham;
    Fourier fft;
    PrepareFourier( fft, dm );

    // FIXME the magic number  10 here.
    Int ncv = 10;
    Spinor  spn( dm, fft.numGridLocal, 1, ncv, Complex(1.0, 0.0) ); 	

    PopCallStack();

    EigenSolver eigSol( ham, spn, fft );
    eigSol.Setup();


    // *********************************************************************
    // Solve
    // *********************************************************************
    PushCallStack( "Solving" );
    eigSol.Solve(false);
    eigSol.PostProcessing();
    eigSol.Solve(false);
    eigSol.PostProcessing();
    PopCallStack();


  }
  catch( std::exception& e )
  {
    std::cerr << " caught exception with message: "
      << e.what() << std::endl;
    DumpCallStack();
  }
  catch ( PetscErrorCode ierr ){
    std::cerr 
      << "\n\n************************************************************************\n"  
      << "Caught PETSc error message " << std::endl;
    DumpCallStack();
    CHKERRQ(ierr);
  }

  ierr = SlepcFinalize();CHKERRQ(ierr);

  return 0;
}


