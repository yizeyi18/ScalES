/// @file ex22.cpp
/// @brief Test for ScaLAPACK using a 3 by 3 matrix as example.
///
/// The matrix to be diagonalized is
///
/// [ 5.0  0.0  0.0
///   0.0  2.0  0.0
///   0.0  0.0  1.0 ]
///
/// Distributed among 1 * 2 processors.
///
/// @author Lin Lin
/// @date 2013-01-29
#include "scales.hpp"

using namespace scales;
using namespace std;
using namespace scales::esdf;
using namespace scales::scalapack; 

void Usage(){
  cout << "Test for the ScaLAPACK" << endl;
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
    // Initialize BLACS

    if( mpisize != 2 ){
      throw std::runtime_error("mpisize must be 2.");
    }

    int contxt;
    Cblacs_get(0, 0, &contxt);
    const int nprow = 1;
    const int npcol = 2;
    Cblacs_gridinit(&contxt, "C", nprow, npcol);

    ScaLAPACKMatrix<Real>  A, Z;

    Int m = 3, n = m;
    int MB = 1;
    A.SetDescriptor( Descriptor( m, n, MB, MB, 0, 0, contxt ) );

    DblNumMat  localMat( A.LocalHeight(), A.LocalWidth(), false, A.Data() );

    SetValue( localMat, 0.0 );

    if( mpirank == 0 ){
      localMat(0, 0) = 5.0;
      localMat(2, 1) = 1.0;
    }
    if( mpirank == 1 ){
      localMat(1, 0) = 2.0;
    }

    std::vector<double> eigs;

    scalapack::Syevd('U', A, eigs, Z);

    if( mpirank == 0 ){
      cout << "Eigs = " << eigs << endl;
      cout << "Z.localMatrix = " << Z.LocalMatrix() << endl;
    }


    Cblacs_gridexit( contxt );

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
