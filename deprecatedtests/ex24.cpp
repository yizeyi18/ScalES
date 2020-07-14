/// @file ex24.cpp
/// @brief Test for converting a DistElemMatrix to a DistSparseMatrix.
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
  cout << "Test for converting a DistElemMatrix to a DistSparseMatrix."
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
    if( mpisize != 2 ){
      throw std::logic_error("mpisize has to be 2 for this example.");
    }


    Int sizeMat = 5;
    DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>       distElemMat;
    DistSparseMatrix<Real>                               distSparseMat;
    NumTns<std::vector<Int> >                            basisIdx;

    // The matrix takes the form
    // [   0   1   2   0   0  ]
    // [  10  11  12   0   0  ]
    // [  20  21  22  23   0  ]
    // [   0   0  32  33  34  ]
    // [   0   0   0  43  44  ]
    //
    // The matrix is partitioned into 4 elements, separated by stars
    //
    // [   0   1 *  2 *  0 *  0  ]
    // [  10  11 * 12 *  0 *  0  ]
    //  *************************
    // [  20  21 * 22 * 23 *  0  ]
    //  ************************
    // [   0   0 * 32 * 33 * 34  ]
    //  *************************
    // [   0   0 *  0 * 43 * 44  ]

    basisIdx.Resize( 1, 1, 4 );
    basisIdx(0,0,0).push_back(0);
    basisIdx(0,0,0).push_back(1);
    basisIdx(0,0,1).push_back(2);
    basisIdx(0,0,2).push_back(3);
    basisIdx(0,0,3).push_back(4);

    ElemMatPrtn prtn;
    prtn.ownerInfo.Resize( 1, 1, 4 );
    IntNumTns& owner = prtn.ownerInfo;
    for( Int i = 0; i < 4; i++ ){
      if( i < 3 )
        owner(0,0,i) = 0;
      else
        owner(0,0,i) = 1;
    }

    distElemMat.Prtn() = prtn;
    if( mpirank == 0 ){
      {
        DblNumMat& mat = distElemMat.LocalMap()[std::pair<Index3,Index3>(Index3(0,0,0), Index3(0,0,0))];
        mat.Resize(2,2);
        mat(0,0) = 0;
        mat(0,1) = 1;
        mat(1,0) = 10;
        mat(1,1) = 11;
      }
      {
        DblNumMat& mat = distElemMat.LocalMap()[std::pair<Index3,Index3>(Index3(0,0,0), Index3(0,0,1))];
        mat.Resize(2,1);
        mat(0,0) = 2;
        mat(1,0) = 12;
      }
      {
        DblNumMat& mat = distElemMat.LocalMap()[std::pair<Index3,Index3>(Index3(0,0,1), Index3(0,0,0))];
        mat.Resize(1,2);
        mat(0,0) = 20;
        mat(0,1) = 21;
      }
      {
        DblNumMat& mat = distElemMat.LocalMap()[std::pair<Index3,Index3>(Index3(0,0,1), Index3(0,0,1))];
        mat.Resize(1,1);
        mat(0,0) = 22;
      }
      {
        DblNumMat& mat = distElemMat.LocalMap()[std::pair<Index3,Index3>(Index3(0,0,1), Index3(0,0,2))];
        mat.Resize(1,1);
        mat(0,0) = 23;
      }
      {
        DblNumMat& mat = distElemMat.LocalMap()[std::pair<Index3,Index3>(Index3(0,0,2), Index3(0,0,1))];
        mat.Resize(1,1);
        mat(0,0) = 32;
      }
      {
        DblNumMat& mat = distElemMat.LocalMap()[std::pair<Index3,Index3>(Index3(0,0,2), Index3(0,0,2))];
        mat.Resize(1,1);
        mat(0,0) = 33;
      }
      {
        DblNumMat& mat = distElemMat.LocalMap()[std::pair<Index3,Index3>(Index3(0,0,2), Index3(0,0,3))];
        mat.Resize(1,1);
        mat(0,0) = 34;
      }
    }

    if( mpirank == 1 ){
      {
        DblNumMat& mat = distElemMat.LocalMap()[std::pair<Index3,Index3>(Index3(0,0,3), Index3(0,0,2))];
        mat.Resize(1,1);
        mat(0,0) = 43;
      }
      {
        DblNumMat& mat = distElemMat.LocalMap()[std::pair<Index3,Index3>(Index3(0,0,3), Index3(0,0,3))];
        mat.Resize(1,1);
        mat(0,0) = 44;
      }
    }


    DistElemMatToDistSparseMat( distElemMat,
        sizeMat,
        distSparseMat,
        basisIdx, 
        MPI_COMM_WORLD );

    if( mpirank == 1 ){
      cout << "Finish converting the data structure." << endl;
      cout << "Remember that the row and column indices are swapped." << endl << endl;
      cout << "Size = " << distSparseMat.size << endl;
      cout << "nnzLocal = " << distSparseMat.nnzLocal << endl;
      cout << "nnz = " << distSparseMat.nnz << endl;
      cout << "firstCol = " << distSparseMat.firstCol << endl;
      cout << "colptrLocal = " << distSparseMat.colptrLocal;
      cout << "rowindLocal = " << distSparseMat.rowindLocal;
      cout << "nzvalLocal = " << distSparseMat.nzvalLocal;
    }


    WriteDistSparseMatrixFormatted( "5x5.matrix", distSparseMat );

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
