/// @file ex29.cpp
/// @brief Test the correctness of OpenMP over multithreaded BLAS.
///
/// @author Lin Lin
/// @date 2013-09-30
#include "scales.hpp"

using namespace scales;
using namespace std;


int main(int argc, char **argv) 
{

  MPI_Init(&argc, &argv);
  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  Int height;

  if( argc != 2 ){
    cout << "Run the code with " << endl << "ex29 {height}" << endl <<
      "height:      the size of the matrix" << endl;
    MPI_Finalize();
    return -1;
  }

  height = atoi(argv[1]);

  Real timeSta, timeEnd;

  Int omprank, ompsize;
#pragma omp parallel shared(ompsize) private(omprank)
  {
    omprank = omp_get_thread_num();
    ompsize = omp_get_num_threads();
  }

  cout << "Number of threads = " << ompsize << endl;

  Int numMat = 24;

  cout << "Generate " << numMat << " matrices of size = " 
    << height << " x " << height << endl;


  DblNumTns x0(height, height, numMat);
  DblNumTns x1(height, height, numMat);
  DblNumTns x2(height, height, numMat);

  SetValue( x0, 0.0 );
  SetValue( x1, 0.0 );
  SetValue( x2, 0.0 );
  SetRandomSeed( 1 );
  UniformRandom( x0 );


  cout << "Get the accurate result using " << ompsize 
    << " number of consequetive multithreaded BLAS..." << endl;

  {
    GetTime( timeSta );
    for( Int i = 0; i < numMat; i++ ){
      blas::Gemm( 'N', 'N', height, height, height, 1.0, 
          x0.MatData(i), height, x0.MatData(i), height, 0.0, 
          x1.MatData(i), height );
    }
    GetTime( timeEnd );
    cout << "Method 1 Time elapsed = " << timeEnd - timeSta << endl; 
  }

  cout << "Get the result using " << ompsize 
    << " processors using OPENMP, each thread calling sequential BLAS..." << endl;

  {
    GetTime( timeSta );
    Int i;
    Int CHUNK_SIZE = 1;
#pragma omp parallel shared(ompsize, numMat, x0, x2, height, CHUNK_SIZE) private(i)
    {
#pragma omp for schedule(static,CHUNK_SIZE) 
      for( i = 0; i < numMat; i++ ){
        blas::Gemm( 'N', 'N', height, height, height, 1.0, 
            x0.MatData(i), height, x0.MatData(i), height, 0.0, 
            x2.MatData(i), height );
      }
    }

    GetTime( timeEnd );
    cout << "Method 2 Time elapsed = " << timeEnd - timeSta << endl; 

#pragma omp parallel shared(ompsize) private(omprank)
    {
      omprank = omp_get_thread_num();
      ompsize = omp_get_num_threads();
    }

    cout << "After calling the second method, number of threads = " << ompsize << endl;

    for( Int i = 0; i < numMat; i++ ){
      DblNumMat err(height, height);
      blas::Copy( height * height, x1.MatData(i), 1, err.Data(), 1 );
      blas::Axpy( height * height, -1.0, x2.MatData(i), 1, 
          err.Data(), 1 );
      cout << "Matrix #" << i << endl;
      //			cout << "1st result = " << DblNumMat( height, height, false, x1.MatData(i) ) << endl;
      //			cout << "2nd result = " << DblNumMat( height, height, false, x2.MatData(i) ) << endl;
      cout << "Error      = " << Energy( err ) << endl;
    }

  }

  MPI_Finalize();

  return 0;
}
