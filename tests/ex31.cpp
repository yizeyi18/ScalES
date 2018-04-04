/// @file ex31.cpp
/// @brief Test the OpenMP parallelization of the ThreeDotProduct
/// process.
///
/// @author Lin Lin
/// @date 2013-10-24
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;


int main(int argc, char **argv) 
{

  MPI_Init(&argc, &argv);
  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  Int height;

  if( argc != 2 ){
    cout << "Run the code with " << endl << "ex31 {height}" << endl <<
      "height:      the size of the matrix" << endl;
    MPI_Finalize();
    return -1;
  }

  height = atoi(argv[1]);


  Int omprank, ompsize;
#pragma omp parallel shared(ompsize) private(omprank)
  {
    omprank = omp_get_thread_num();
    ompsize = omp_get_num_threads();
  }

  cout << "Number of threads = " << ompsize << endl;

  cout << "Matrices of size = " << height << " x " << height << endl;


  DblNumMat x0(height, height);
  DblNumMat x1(height, height);
  DblNumMat x2(height, height);
  DblNumVec v0(height);
  DblNumMat v1(height, ompsize);

  SetValue( x0, 0.0 );
  SetValue( x1, 0.0 );
  SetValue( x2, 0.0 );
  SetValue( v0, 0.0 );
  SetRandomSeed( 1 );
  UniformRandom( x0 );
  UniformRandom( x1 );
  UniformRandom( v0 );
  // Replicate the vector v0
  for( Int i = 0; i < ompsize; i++ ){
    blas::Copy( height, v0.Data(), 1, v1.VecData(i), 1 );
  }



  // Straightforward parallelization with pointers
  {
    Int i, j, k;
    Real res;
    Real *ptrx0, *ptrx1, *ptrx2, *ptrv0;
    Real timeSta, timeEnd;
    Real timeTotalSta, timeTotalEnd;

    GetTime( timeTotalSta );
#pragma omp parallel private(i, j, k, ptrx0, ptrx1, ptrx2, ptrv0)
    {
      GetTime( timeSta );
#pragma omp for
      for( j = 0; j < height; j++ ){
        ptrx2 = x2.VecData(j);
        for( i = 0; i < height; i++ ){
          ptrx0 = x0.VecData(i);
          ptrx1 = x1.VecData(j);
          ptrv0 = v0.Data();
          res = 0.0;
          for( k = 0; k < height; k++ )
            res += ptrx0[k] * ptrx1[k] * ptrv0[k];
          *(ptrx2++) = res;
        }
      }

      GetTime( timeEnd );
    }
    GetTime( timeTotalEnd );
    cout << "Implementation 1" <<endl;
    cout << "Time elapsed for computation = " << 
      timeEnd - timeSta << endl; 
    cout << "Time elapsed in total (including OPENMP overhead) = " << 
      timeTotalEnd - timeTotalSta << endl; 

  }


  // Manual copy of the commonly shared arrays
  {
    Int i, j, k;
    Real res;
    Real *ptrx0, *ptrx1, *ptrx2, *ptrv1;
    Real timeSta, timeEnd;
    Real timeTotalSta, timeTotalEnd;

    GetTime( timeTotalSta );
#pragma omp parallel private(i, j, k, ptrx0, ptrx1, ptrx2, ptrv1, omprank)
    {
      omprank = omp_get_thread_num();
      GetTime( timeSta );
#pragma omp for
      for( j = 0; j < height; j++ ){
        ptrx2 = x2.VecData(j);
        for( i = 0; i < height; i++ ){
          ptrx0 = x0.VecData(i);
          ptrx1 = x1.VecData(j);
          ptrv1 = v1.VecData(omprank);
          res = 0.0;
          for( k = 0; k < height; k++ )
            res += ptrx0[k] * ptrx1[k] * ptrv1[k];
          *(ptrx2++) = res;
        }
      }

      GetTime( timeEnd );
    }
    GetTime( timeTotalEnd );
    cout << "Implementation 2 (replicate the vector v)" <<endl;
    cout << "Time elapsed for computation = " << 
      timeEnd - timeSta << endl; 
    cout << "Time elapsed in total (including OPENMP overhead) = " << 
      timeTotalEnd - timeTotalSta << endl; 

  }

  // Do not use pointers for collecting the results.  There might be
  // possible parallelization issue when operating on x2.
  {
    Int i, j, k;
    Real res;
    Real timeSta, timeEnd;
    Real timeTotalSta, timeTotalEnd;
    Real *ptrx0, *ptrx1, *ptrv0;

    GetTime( timeTotalSta );
#pragma omp parallel private(i, j, k, omprank, ptrx0, ptrx1, ptrv0)
    {
      omprank = omp_get_thread_num();
      GetTime( timeSta );
#pragma omp for
      for( j = 0; j < height; j++ ){
        for( i = 0; i < height; i++ ){
          ptrx0 = x0.VecData(i);
          ptrx1 = x1.VecData(j);
          ptrv0 = v0.Data();

          res = 0.0;
          for( k = 0; k < height; k++ )
            res += ptrx0[k] * ptrx1[k] * ptrv0[k];
          x2(i,j) = res;
        }
      }

      GetTime( timeEnd );
    }
    GetTime( timeTotalEnd );
    cout << "Implementation 3 (Use x2(i,j) directly)" <<endl;
    cout << "Time elapsed for computation = " << 
      timeEnd - timeSta << endl; 
    cout << "Time elapsed in total (including OPENMP overhead) = " << 
      timeTotalEnd - timeTotalSta << endl; 

  }

  // Do not use pointers for collecting the results.  There might be
  // possible parallelization issue when operating on x2. Use the inline
  // function ThreeDotProduct
  {
    Int i, j, k;
    Real res;
    Real timeSta, timeEnd;
    Real timeTotalSta, timeTotalEnd;
    Real *ptrx0, *ptrx1, *ptrv0;

    GetTime( timeTotalSta );
#pragma omp parallel private(i, j, k, omprank)
    {
      omprank = omp_get_thread_num();
      GetTime( timeSta );
#pragma omp for
      for( j = 0; j < height; j++ ){
        for( i = 0; i < height; i++ ){
          x2(i,j) = ThreeDotProduct( 
              x0.VecData(i), x1.VecData(j), v0.Data(), height );
        }
      }

      GetTime( timeEnd );
    }
    GetTime( timeTotalEnd );
    cout << "Implementation 4 (Use x2(i,j) directly with ThreeDotProduct)" <<endl;
    cout << "Time elapsed for computation = " << 
      timeEnd - timeSta << endl; 
    cout << "Time elapsed in total (including OPENMP overhead) = " << 
      timeTotalEnd - timeTotalSta << endl; 

  }


  MPI_Finalize();

  return 0;
}
