/// @file ex30.cpp
/// @brief Test the performance of manually implemented OPENMP
/// procedure for matrix multiplication.
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
		cout << "Run the code with " << endl << "ex30 {height}" << endl <<
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

	cout << "Matrices of size = " << height << " x " << height << endl;


	DblNumMat x0(height, height);
	DblNumMat x1(height, height);
	DblNumMat x2(height, height);
	DblNumMat x3(height, height);
	
	SetValue( x0, 0.0 );
	SetValue( x1, 0.0 );
	SetValue( x2, 0.0 );
	SetValue( x3, 0.0 );
	SetRandomSeed( 1 );
	UniformRandom( x0 );
	UniformRandom( x1 );

	cout << "Get the accurate result using multithreaded BLAS..." << endl;

	{
		GetTime( timeSta );
		blas::Gemm( 'T', 'N', height, height, height, 1.0, 
				x0.Data(), height, x1.Data(), height, 0.0, 
				x2.Data(), height );
		GetTime( timeEnd );
		cout << "Multithreaded BLAS, Time elapsed = " << timeEnd - timeSta << endl; 
	}

	cout << "Get the result using manually implemented GEMM process" << endl;
 
	{
		Int i, j, k;
		Int CHUNK_SIZE = 1;
		Real res;
		Real *ptr0, *ptr1, *ptr3;
#pragma omp parallel private(i, j, k, ptr0, ptr1, ptr3)
		{
			GetTime( timeSta );
#pragma omp for
			for( j = 0; j < height; j++ ){
				ptr3 = x3.VecData(j);
				for( i = 0; i < height; i++ ){
					ptr0 = x0.VecData(i);
					ptr1 = x1.VecData(j);
					res = 0.0;
					for( k = 0; k < height; k++ )
						res += (*(ptr0++)) * (*(ptr1++));
					*(ptr3++) = res;
				}
			}


			GetTime( timeEnd );
			cout << "Manual GEMM, Implemention 1, Time elapsed = " << 
				timeEnd - timeSta << endl; 

			GetTime( timeSta );
			ptr0 = x0.Data();
			ptr1 = x1.Data();
			ptr3 = x3.Data();
#pragma omp for
			for( j = 0; j < height; j++ ){
				for( i = 0; i < height; i++ ){
					res = 0.0;
					for( k = 0; k < height; k++ )
						res += ptr0[k+i*height] * ptr1[k+j*height];
					ptr3[i+height*j] = res;
				}
			}

			GetTime( timeEnd );
			cout << "Manual GEMM, Implemention 2, Time elapsed = " << 
				timeEnd - timeSta << endl; 

		}


	}


	cout << "Check the accuracy" << endl;

	DblNumMat err(height, height);
	blas::Copy( height * height, x2.Data(), 1, err.Data(), 1 );
	blas::Axpy( height * height, -1.0, x3.Data(), 1, 
			err.Data(), 1 );
	cout << "Error      = " << Energy( err ) << endl;

	MPI_Finalize();

	return 0;
}
