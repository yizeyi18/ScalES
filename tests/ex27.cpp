/// @file ex27.cpp
/// @brief Test the OpenMP performance of just the BLAS solver.
///
/// @author Lin Lin
/// @date 2013-09-28
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
		cout << "Run the code with " << endl << "ex27 {height}" << endl <<
			"height:      the size of the matrix" << endl;
		MPI_Finalize();
		return -1;
	}

	height = atoi(argv[1]);

	cout << "Height = " << height << endl;

	Real timeSta, timeEnd;

	DblNumMat x0(height, height);
	DblNumMat x1(height, height); 
	DblNumMat x2(height, height); 
	
	SetRandomSeed( 1 );
	UniformRandom( x0 );
	UniformRandom( x1 ); 

	GetTime( timeSta );

	blas::Gemm( 'N', 'N', height, height, height, 1.0, 
			x0.Data(), height, x1.Data(), height, 0.0, 
			x2.Data(), height );

	GetTime( timeEnd );

	cout << "Time elapsed = " << timeEnd - timeSta << endl; 

	MPI_Finalize();

	return 0;
}
