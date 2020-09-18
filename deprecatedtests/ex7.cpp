// *********************************************************************
// Test for solving the eigenvalue problem for periodic Laplacian
// operator using SLEPc without using preconditioner
//
//   A = -\Delta / 2
// 
//
// *********************************************************************
#include "dgdft.hpp"
#include "spinor.hpp"
#include "fourier.hpp"

using namespace dgdft;
using namespace std;

static char help[] = "Test for solving the eigenvalue problem for periodic Laplacian operator using SLEPc";


// *********************************************************************
// Matrix operations
// *********************************************************************
PetscErrorCode MatLap_Mult(Mat,Vec,Vec);

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

    Mat            A;               /* eigenvalue problem matrix */
    EPS            eps;             /* eigenproblem solver context */
    const EPSType  type;
    Int            nev;

    // *********************************************************************
    // Preparation
    // *********************************************************************

    PushCallStack( "Preparation" );

    Domain  dm;
    dm.length  = Point3( 1.0, 1.0, 1.0 );
    dm.numGrid = Index3( 8, 8, 8 );

    Fourier fft;
    PrepareFourier( fft, dm );

    Spinor  spn( dm, fft.numGridLocal, 1, 1, Complex(1.0, 0.0) ); 	

    PopCallStack();


    // *********************************************************************
    // Shell matrix context
    // *********************************************************************

    // TODO numGridLocal probably should be changed to
    // numGridLocalTotal to avoid confusion with future changes in
    // r2c/c2r.

    ierr = MatCreateShell(
        PETSC_COMM_WORLD,
        fft.numGridLocal,
        fft.numGridLocal,
        dm.NumGridTotal(),
        dm.NumGridTotal(),
        (void*) (&fft), &A );
    CHKERRQ(ierr);

    ierr = MatShellSetOperation(A,MATOP_MULT,(void(*)())MatLap_Mult);CHKERRQ(ierr);

    // *********************************************************************
    // Create the eigensolver and set various options
    // *********************************************************************
    ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
    ierr = EPSSetOperators(eps,A,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);
    ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);
    ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

    // *********************************************************************
    // Solve the eigenvalue problem
    // *********************************************************************
    ierr = EPSSolve(eps);CHKERRQ(ierr);


    // *********************************************************************
    // Display solutions
    // *********************************************************************
    ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
    ierr = EPSGetDimensions(eps,&nev,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);


    // *********************************************************************
    // Clean up
    // *********************************************************************
    PushCallStack("Clean up");
    ierr = EPSPrintSolution(eps,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSDestroy(&eps);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
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

  ierr = SlepcFinalize();CHKERRQ(ierr);

  return 0;
}


#undef  __FUNCT__
#define __FUNCT__ "MatLap_Mult"
PetscErrorCode MatLap_Mult(Mat A, Vec x, Vec y)
{
  Int numGridTotal, numGridLocal;
  Scalar*    xArray;
  Scalar*  	 yArray; 
  Fourier*   fft;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)(&fft));CHKERRQ(ierr);
  numGridTotal = fft->domain.NumGridTotal();
  numGridLocal = fft->numGridLocal;
  VecGetArray( x, reinterpret_cast<PetscScalar**>(&xArray) );
  VecGetArray( y, reinterpret_cast<PetscScalar**>(&yArray) );

  fftw_mpi_execute_dft( fft->forwardPlan, 
      reinterpret_cast<fftw_complex*>( xArray ),  
      reinterpret_cast<fftw_complex*>( fft->outputComplexVec.Data() ) );

  Index3 numGrid = fft->domain.numGrid;
  //TODO change this to NumTns later.
  for( Int k = 0; k < numGrid[2]; k++ ){
    for( Int j = 0; j < numGrid[1]; j++ ){
      for( Int i = fft->localN0Start; 
          i < fft->localN0Start + fft->localN0; i++ ){
        Int idx1 = (i - fft->localN0Start) + j * fft->localN0 + k * (fft->localN0 * numGrid[1]);
        Int idx2 = i + j * numGrid[0] + k * numGrid[0] * numGrid[1];
        fft->outputComplexVec[ idx1 ] *= fft->gkk( idx2 );
      }
    }
  }

  fftw_mpi_execute_dft( fft->backwardPlan, 
      reinterpret_cast<fftw_complex*>( fft->outputComplexVec.Data() ),  
      reinterpret_cast<fftw_complex*>( yArray ) );

  //	for( Int i = 0; i < numGridLocal; i++ ){
  //		yArray[i] = xArray[i];
  //	}

  VecRestoreArray( x, reinterpret_cast<PetscScalar**>(&xArray) );
  VecRestoreArray( y, reinterpret_cast<PetscScalar**>(&yArray) );
  ierr = VecScale(y, 1.0 / numGridTotal);	 CHKERRQ(ierr);
  //	ierr = VecAXPY(y, 1.0, x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__