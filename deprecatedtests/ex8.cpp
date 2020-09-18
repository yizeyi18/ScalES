// *********************************************************************
// Test for solving the eigenvalue problem for periodic Laplacian
// operator using SLEPc using preconditioner
//
//   A = -\Delta / 2 + I
//
// *********************************************************************
#include <slepc-private/epsimpl.h>              // Trick to use eps->
#include "dgdft.hpp"
#include "spinor.hpp"
#include "fourier.hpp"

#define EPSMODIFIEDBLOPEX      "modified_blopex"

using namespace dgdft;
using namespace std;

static char help[] = "Test for solving the eigenvalue problem for periodic Laplacian operator using SLEPc";

extern "C"{
extern PetscErrorCode EPSCreate_MODIFIED_BLOPEX(EPS eps);
}

// *********************************************************************
// Matrix operations
// *********************************************************************
PetscErrorCode MatLap_Mult(Mat,Vec,Vec);
PetscErrorCode MatPrecond_Mult(Mat,Vec,Vec);

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

    Mat            A;                           // eigenvalue problem matrix
    Mat            P;                           // Preconditioning matrix
    EPS            eps;                         // eigenproblem solver context
    ST             st;                          // Spectral transformation
    const EPSType  type;
    Int            nev, ncv;
    Int            numIter;
    Int            numConv;
    EPSConvergedReason reason;

    // *********************************************************************
    // Preparation
    // *********************************************************************

    PushCallStack( "Preparation" );

    Domain  dm;
    dm.length  = Point3( 1.0, 1.0, 1.0 );
    dm.numGrid = Index3( 24, 72, 72);

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

    ierr = MatCreateShell(
        PETSC_COMM_WORLD,
        fft.numGridLocal,
        fft.numGridLocal,
        dm.NumGridTotal(),
        dm.NumGridTotal(),
        (void*) (&fft), &P );
    CHKERRQ(ierr);

    ierr = MatShellSetOperation(P,MATOP_MULT,(void(*)())MatPrecond_Mult);CHKERRQ(ierr);


    // *********************************************************************
    // Create the eigensolver and set various options
    // *********************************************************************

    // Register for a new solver: the modified blopex

    EPSRegister( "modified_blopex", 0, "EPSCreate_MODIFIED_BLOPEX", 
        EPSCreate_MODIFIED_BLOPEX );

    ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);


    ierr = EPSSetOperators(eps,A,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);
    ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);
    ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

    ierr = EPSGetType(eps,&type);CHKERRQ(ierr);

    // *********************************************************************
    // Set for the preconditioner
    // *********************************************************************
    if( strcmp( type, EPSMODIFIEDBLOPEX ) ){
      if( mpirank == 0 ){
        cout << "EPSType != modified_blopex, "
          << "Preconditioner is not used" << endl;
      }
    }
    else{
      ierr = EPSGetST( eps, &st ); CHKERRQ( ierr ); 
      ierr = STPrecondSetMatForPC( st, P ); CHKERRQ( ierr );
    }


    // *********************************************************************
    // Solve the eigenvalue problem
    // *********************************************************************
    Real timeSolveStart = MPI_Wtime();
    ierr = EPSSolve(eps);CHKERRQ(ierr);
    Real timeSolveEnd = MPI_Wtime();
    if( mpirank == 0 ){
      std::cout << "Time for solving the eigenvalue problem is " <<
        timeSolveEnd - timeSolveStart << std::endl;
    }


    // *********************************************************************
    // Display solutions
    // *********************************************************************
    ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
    ierr = EPSGetDimensions(eps,&nev,&ncv,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSGetIterationNumber(eps, &numIter); CHKERRQ(ierr);
    ierr = EPSGetConverged(eps, &numConv); CHKERRQ(ierr);	
    ierr = EPSGetConvergedReason(eps, &reason); CHKERRQ(ierr);


    ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Number of subspace dimension: %D\n",ncv);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenvalues: %D\n",numConv);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations taken: %D\n", numIter);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Converged reason: %D\n", reason);CHKERRQ(ierr);

    // Print out eigenvalues
    {
      eps->nconv = eps->nev;
      NumVec<Scalar>  eigr(nev), eigi(nev);
      for( Int i = 0; i < nev; i++ ){
        ierr = EPSGetEigenvalue(eps, i, &eigr(i), &eigi(i)); CHKERRQ(ierr);
      }
      if( mpirank == 0 ){
        cout << "EPSGetEigenvalue does not work if not converged" << endl;
        for( Int i = 0; i < nev; i++ ){
          cout << "eig[" << i << "]" << eigr(i) << endl;
        }
        cout << endl;
        cout << "eps->eigr[i] works" << endl; 
        for( Int i = 0; i < nev; i++ ){
          cout << "eig[" << i << "]" << eps->eigr[i] << endl;
        }
      }
    }

    // *********************************************************************
    // Clean up
    // *********************************************************************
    PushCallStack("Clean up");
    //		ierr = EPSPrintSolution(eps,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSDestroy(&eps);CHKERRQ(ierr);
    ierr = MatDestroy(&P); CHKERRQ(ierr);
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

  Int ownLB, ownUB;
  ierr = VecGetOwnershipRange( x, &ownLB, &ownUB ); if( ierr ) throw ierr;
  for( Int i = ownLB; i < ownUB; i++ ){
    fft->outputComplexVec[ i - ownLB ] 
      *= fft->gkk( i );
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

  // Add the identity operator
  ierr = VecAXPY(y, 1.0, x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__


#undef  __FUNCT__
#define __FUNCT__ "MatPrecond_Mult"
PetscErrorCode MatPrecond_Mult(Mat P, Vec x, Vec y)
{
  Int numGridTotal, numGridLocal;
  Scalar*    xArray;
  Scalar*  	 yArray; 
  Fourier*   fft;
  PetscErrorCode    ierr;

  PetscFunctionBegin;

  ierr = MatShellGetContext(P,(void**)(&fft));CHKERRQ(ierr);
  numGridTotal = fft->domain.NumGridTotal();
  numGridLocal = fft->numGridLocal;
  VecGetArray( x, reinterpret_cast<PetscScalar**>(&xArray) );
  VecGetArray( y, reinterpret_cast<PetscScalar**>(&yArray) );

  fftw_mpi_execute_dft( fft->forwardPlan, 
      reinterpret_cast<fftw_complex*>( xArray ),  
      reinterpret_cast<fftw_complex*>( fft->outputComplexVec.Data() ) );

  Int ownLB, ownUB;
  ierr = VecGetOwnershipRange( x, &ownLB, &ownUB ); if( ierr ) throw ierr;
  for( Int i = ownLB; i < ownUB; i++ ){
    fft->outputComplexVec[ i - ownLB ] 
      *= fft->TeterPrecond( i );
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

  PetscFunctionReturn(0);
}
#undef __FUNCT__