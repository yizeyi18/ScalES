/// @file ex26.cpp
/// @brief Test the OpenMP performance of the blopex routine using a
/// dense solver
///
/// @author Lin Lin
/// @date 2013-09-27
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;

Int CHUNK_SIZE   = 1;


void Mult
(void *A, void *X, void *Y) {
  serial_Multi_Vector*  x = (serial_Multi_Vector*) X;
  serial_Multi_Vector*  y = (serial_Multi_Vector*) Y;

  Int height      = x->size;
  Int width       = x->num_vectors;


  DblNumMat xMat( height, width, false, x->data ); //double numerical matrix view(false) data
  DblNumMat yMat( height, width, false, y->data );	

#ifdef _USE_OPENMP_
#pragma omp parallel shared(CHUNK_SIZE, xMat, yMat, width, height)
#endif
  {
#ifdef _USE_OPENMP_
#pragma omp for schedule(static, CHUNK_SIZE)
#endif
    for( Int j = 0; j < width; j++ ){
      Real* xdata = xMat.VecData(j);
      Real* ydata = yMat.VecData(j);
      for( Int i = 0; i < height; i++ ){
        *(ydata++) = *(xdata++) * Real(i+1);
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

  mv_MultiVectorPtr xx; //Hypre_Lobpcgsolver(hypresolver, mv_MultiVectorPtr, mv_multiVectorPtr, double*)?
  mv_InterfaceInterpreter ii; //Hypre_LobpcgCreate?
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
