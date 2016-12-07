/// @file ex25.cpp
/// @brief Test the read/write of sparse / distsparse matrix format.
///
/// @author Lin Lin
/// @date 2013-07-14
#include "sparse_matrix_decl.hpp"
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;

#define _DEBUGlevel_ 1

void Usage(){
  cout << "Test the read/write of sparse / distsparse matrix format."
    << endl << endl;
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
    string  fname = "g20.matrix";

    if( mpisize > 1 ){
      if( mpirank == 0 )
        cout << "Reading the matrix as a DistSparseMatrix..." << endl;

      DistSparseMatrix<Real> AMat;

      ReadDistSparseMatrixFormatted<Real>( fname.c_str(), AMat, MPI_COMM_WORLD );

      if( mpirank == 1 & 0 ){
        cout << "Finish reading the matrix the data structure." << endl;
        cout << "Size = " << AMat.size << endl;
        cout << "nnzLocal = " << AMat.nnzLocal << endl;
        cout << "nnz = " << AMat.nnz << endl;
        cout << "firstCol = " << AMat.firstCol << endl;
        cout << "colptrLocal = " << AMat.colptrLocal;
        cout << "rowindLocal = " << AMat.rowindLocal;
        cout << "nzvalLocal = " << AMat.nzvalLocal;
      }

      WriteDistSparseMatrixFormatted( "g20out.matrix", AMat );
    }
    if( mpisize == 1 ){
      cout << "Reading the matrix as a SparseMatrix..." << endl;

      SparseMatrix<Real> AMat;

      ReadSparseMatrixFormatted<Real>( fname.c_str(), AMat );

      cout << "Finish reading the matrix the data structure." << endl;
      cout << "Size = " << AMat.size << endl;
      cout << "nnz = " << AMat.nnz << endl;
      cout << "colptr = " << AMat.colptr;
      cout << "rowind = " << AMat.rowind;
      cout << "nzval = " << AMat.nzval;

    }

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
