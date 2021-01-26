// *********************************************************************
// Test FFT
// *********************************************************************
#include "scales.hpp"
#include "spinor.hpp"
#include "fourier.hpp"

using namespace scales;
using namespace std;

static char help[] = "Test FFT";

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
    dm1.numGrid = Index3( 8, 8, 8 );
    dm1.typeGrid = UNIFORM;
    dm1.posStart = Point3( 0.0, 0.0, 0.0 );

    Fourier fft;
    PrepareFourier( fft, dm1 );

    Spinor  spn1( dm1, fft.numGridLocal, 1, 1, Z_ONE ); 	
    NumVec<Scalar>  out( fft.numGridLocal );

    fftw_mpi_execute_dft( fft.forwardPlan, 
        reinterpret_cast<fftw_complex*>( spn1.LocalWavefunData(0,0) ),  
        reinterpret_cast<fftw_complex*>( out.Data() ) );

    fftw_mpi_execute_dft( fft.backwardPlan, 
        reinterpret_cast<fftw_complex*>( out.Data() ),  
        reinterpret_cast<fftw_complex*>( spn1.LocalWavefunData(0,0) ) );

    VecScale( spn1.Wavefun(0,0), 1.0 / dm1.NumGridTotal() );
    VecView( spn1.LockedWavefun(0,0), PETSC_VIEWER_STDOUT_WORLD );

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
