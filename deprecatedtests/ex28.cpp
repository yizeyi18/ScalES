/// @file ex28.cpp
/// @brief Test the OpenMP performance of some LAPACK routines
///
/// @author Lin Lin
/// @date 2013-09-28
#include "scales.hpp"

using namespace scales;
using namespace std;


int main(int argc, char **argv) 
{

  MPI_Init(&argc, &argv);
  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );


  if( argc != 3 ){
    cout << "Run the code with " << endl << "ex28 {height} {width}" << endl <<
      "height:      the size of the matrix" << endl <<
      "width:       the width of the matrix" << endl; 
    MPI_Finalize();
    return -1;
  }

  Int height = atoi(argv[1]);
  Int width  = atoi(argv[2]);

  cout << "Matrix of size " << height << " x " << width << endl;

  Real timeSta, timeEnd;

  cout << "SVD version 1: Compute svd(A)" << endl;
  {
    DblNumMat x0(height, width);
    DblNumMat U(height, width);
    DblNumVec S(width);
    DblNumMat VT(width, width);

    SetRandomSeed( 1 );
    UniformRandom( x0 );

    GetTime( timeSta );

    lapack::QRSVD( height, width, x0.Data(), height,
        S.Data(), U.Data(), height, VT.Data(), width );

    GetTime( timeEnd );

    cout << "Time elapsed = " << timeEnd - timeSta << endl; 
  }


  cout << "SVD version 2: Compute svd(A^T A)" << endl;
  {
    DblNumMat x0(height, width);
    DblNumMat xTx( width, width );


    DblNumMat U(height, width);
    DblNumMat VT(width, height);
    DblNumVec S(width);
    DblNumMat tU(width, width);

    SetRandomSeed( 1 );
    UniformRandom( x0 );



    GetTime( timeSta );

    blas::Gemm( 'T', 'N', width, width, height, 1.0,
        x0.Data(), height, x0.Data(), height, 0.0,
        xTx.Data(), width );

    lapack::QRSVD( width, width, xTx.Data(), width,
        S.Data(), tU.Data(), width, VT.Data(), width );

    GetTime( timeEnd );

    cout << "Time elapsed = " << timeEnd - timeSta << endl; 
  }



  MPI_Finalize();

  return 0;
}
