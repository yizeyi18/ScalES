// *********************************************************************
// Test spinors without Petsc for real and complex case
// *********************************************************************
#include  "scales.hpp"
#include  "utility.hpp"
#include  "domain.hpp"
#include  "spinor.hpp"

using namespace scales;
using namespace std;

static char help[] = "Test spinors";

int main(int argc, char **argv) 
{
  MPI_Init(&argc, &argv);
  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  try
  {
#ifdef  _RELEASE_
    throw std::runtime_error("Test should be run under debug mode");
#endif
    PushCallStack("Main");


    Domain  dm1;
    dm1.length = Point3( 1.0, 2.0, 3.0 );
    dm1.numGrid = Index3( 1, 2, 4 );
    dm1.posStart = Point3( 1.0, 0.1, 0.0 );


    // *********************************************************************
    // Test lifecycle
    // *********************************************************************

    cout << "Test Lifecycle" << endl << endl;

    int numGridLocal = dm1.NumGridTotal() / mpisize;
#ifdef _USE_COMPLEX_
    Spinor  spn1( dm1, 1, 2, Complex(1.0, 1.0) );
#else
    Spinor  spn1( dm1, 1, 2, 1.0 );
#endif

    Spinor  spn2( dm1, spn1.NumComponent(),
        spn1.NumState(), false, spn1.Wavefun().Data() );

    cout << "Spn1 wavefun(0,0)" << endl;
    cout << NumVec<Scalar>( spn1.NumGridTotal(), false, 
        spn1.Wavefun().VecData(0,0) ) << endl;

#ifdef _USE_COMPLEX_
    cout << "spn1(0,0)[1] = (1.0, 2.0)" << endl;
    (spn1.Wavefun().VecData(0,0))[1] = Complex(1.0, 2.0);
#else
    cout << "spn1(0,0)[1] = 2.0" << endl;
    (spn1.Wavefun().VecData(0,0))[1] = 2.0;
#endif
    cout << "Spn2 is viewing spn1" << endl;
    cout << "Spn2 wavefun(0,0)" << endl;
    cout << NumVec<Scalar>( spn2.NumGridTotal(), false, 
        spn2.Wavefun().VecData(0,0) ) << endl;


    // *********************************************************************
    // Test AddScalarDiag
    // *********************************************************************

    cout << "Test AddScalarDiag" << endl << endl;


    spn2.Normalize();
    cout << "Spn2 wavefun(0,0) after normalization" << endl;
    cout << NumVec<Scalar>( spn2.NumGridTotal(), false, 
        spn2.Wavefun().VecData(0,0) ) << endl;

    DblNumVec val(dm1.NumGridTotal() );
    SetValue(val, 2.0);

    NumTns<Scalar>  outTns(spn1.Wavefun().m(), spn1.Wavefun().n(),
        spn1.Wavefun().p());
    SetValue( outTns, SCALAR_ZERO );

    spn1.AddScalarDiag(1, val, outTns);
    cout << "val = " << val << endl;
    cout << "outTns(:,1) += val(:) .* spn1(:,1)" << endl;
    cout << "outTns(0,0)" << endl;
    cout << NumVec<Scalar>( spn1.NumGridTotal(), false, 
        outTns.VecData(0,0) ) << endl;
    cout << "outTns(0,1)" << endl;
    cout << NumVec<Scalar>( spn1.NumGridTotal(), false, 
        outTns.VecData(0,1) ) << endl;

    SetValue( outTns, SCALAR_ZERO );
    spn1.AddScalarDiag( val, outTns );
    cout << "outTns = 0; outTns(:,i) += val(:) .* spn1(:,i), i=1,2" << endl;
    cout << "outTns(0,0)" << endl;
    cout << NumVec<Scalar>( spn1.NumGridTotal(), false, 
        outTns.VecData(0,0) ) << endl;
    cout << "outTns(0,1)" << endl;
    cout << NumVec<Scalar>( spn1.NumGridTotal(), false, 
        outTns.VecData(0,1) ) << endl;


    // *********************************************************************
    // Test Laplacian
    // *********************************************************************

    cout << "Test Laplacian" << endl << endl;
    Fourier  fft;
    PrepareFourier( fft, dm1 );

    SetValue( outTns, SCALAR_ZERO );
    spn1.AddLaplacian( outTns, &fft );
    cout << "gkk = " << fft.gkk << endl;
    cout << "outTns = 0; outTns = -1/2 Delta spn1" << endl;
    cout << "outTns(0,0)" << endl;
    cout << NumVec<Scalar>( spn1.NumGridTotal(), false, 
        outTns.VecData(0,0) ) << endl;
    cout << "outTns(0,1)" << endl;
    cout << NumVec<Scalar>( spn1.NumGridTotal(), false, 
        outTns.VecData(0,1) ) << endl;


    PopCallStack();
  }
  catch( std::exception& e )
  {
    std::cerr << " caught exception with message: "
      << e.what() << std::endl;
    DumpCallStack();
  }

  MPI_Finalize();

  return 0;
}
