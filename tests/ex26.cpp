/// @file ex26.cpp
/// @brief Test the OpenMP performance of the blopex routine using a
/// dense solver
///
/// @author Lin Lin
/// @date 2013-09-27
#include "dgdft.hpp"
#include <omp.h>

using namespace dgdft;
using namespace std;

const	Int CHUNK_SIZE   = 1;


void Mult
(void *A, void *X, void *Y) {
	serial_Multi_Vector*  x = (serial_Multi_Vector*) X;
	serial_Multi_Vector*  y = (serial_Multi_Vector*) Y;

  Int height      = x->size;
	Int width       = x->num_vectors;


	DblNumMat xMat( height, width, false, x->data );
  DblNumMat yMat( height, width, false, y->data );	

#pragma omp parallel shared(CHUNK_SIZE, xMat, yMat, width, height)
	{
#pragma omp for schedule(static, CHUNK_SIZE)
		for( Int j = 0; j < width; j++ ){
			for( Int i = 0; i < height; i++ ){
				yMat(i,j) = xMat(i,j) * Real(i+1);
			}
		}
	}

}


int main(int argc, char **argv) 
{

	MPI_Init(&argc, &argv);
	int mpirank, mpisize;
	MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
	MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
	
	Int MAXIT  = 100;

	if( argc != 3 ){
		cout << "Run the code with " << endl << "ex26 {height} {width}" << endl <<
			"height:      the height of the matrix" << endl << 
			"width:       the width of the matrix" << endl; 
		MPI_Finalize();
		return -1;
	}

	Int height = atoi(argv[1]);
	Int width  = atoi(argv[2]);

	cout << "Matrix of size " << height << " x " << width << endl;

	Real timeSta, timeEnd;

	DblNumMat x0(height, width);

	SetRandomSeed( 1 );
	UniformRandom( x0 );

  serial_Multi_Vector *x;
  x = (serial_Multi_Vector*)malloc(sizeof(serial_Multi_Vector));

  x->data = x0.Data();
  x->owns_data = 0;
  x->size        = height;
  x->num_vectors = width;
  x->num_active_vectors = x->num_vectors;
  x->active_indices = (BlopexInt*)malloc(sizeof(BlopexInt)*x->num_active_vectors);
  for (Int i=0; i<x->num_active_vectors; i++) x->active_indices[i] = i;

  mv_MultiVectorPtr xx;
  mv_InterfaceInterpreter ii;
  lobpcg_Tolerance lobpcg_tol;
  lobpcg_BLASLAPACKFunctions blap_fn;

  lobpcg_tol.absolute = 1e-20;
  lobpcg_tol.relative = 1e-20;

  SerialSetupInterpreter ( &ii );
  xx = mv_MultiVectorWrap( &ii, x, 0);

  Int iterations;

	DblNumVec eigs( width ), resid( width );

  blap_fn.dpotrf = LAPACK(dpotrf);
  blap_fn.dsygv  = LAPACK(dsygv);

	GetTime(timeSta);
	
	lobpcg_solve_double ( 
			xx,
			NULL,
			Mult,
			NULL,
			NULL,
			NULL,
			NULL,
			NULL,
			blap_fn,
			lobpcg_tol,
			MAXIT,
			0,
			&iterations,
			eigs.Data(),
			NULL,
			0,
			resid.Data(),
			NULL,
			0);

	GetTime(timeEnd);

	cout << "Time elapsed = " << timeEnd - timeSta << endl; 
	cout << "eigs = " << eigs << std::endl;


  serial_Multi_VectorDestroy(x);
  mv_MultiVectorDestroy(xx);

	MPI_Finalize();

	return 0;
}
