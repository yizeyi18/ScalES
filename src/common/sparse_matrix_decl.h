//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin

/// @file sparse_matrix_decl.hpp
/// @brief Sparse matrix and Distributed sparse matrix in compressed
/// column format.
/// @date 2012-11-10
#ifndef _SPARSE_MATRIX_DECL_HPP_
#define _SPARSE_MATRIX_DECL_HPP_

#include "environment.h"
#include "numvec_impl.hpp"

namespace  scales{


/// @struct SparseMatrix
/// 
/// @brief SparseMatrix describes a sequential sparse matrix saved in
/// compressed sparse column format.
///
/// Note
/// ----
///
/// Since in PEXSI and PPEXSI only symmetric matrix is considered, the
/// compressed sparse row format will also be represented by the
/// compressed sparse column format.
template <class F> struct SparseMatrix{
  Int          size;                            // Matrix dimension
  Int          nnz;                             // Number of nonzeros
  IntNumVec    colptr;                          // Column index pointer
  IntNumVec    rowind;                          // Starting row index pointer
  NumVec<F>    nzval;                           // Nonzero values for the sparse matrix
};

// Commonly used
typedef SparseMatrix<Real>       DblSparseMatrix;
typedef SparseMatrix<Complex>    CpxSparseMatrix;

/// @struct DistSparseMatrix
///
/// @brief DistSparseMatrix describes a Sparse matrix in the compressed
/// sparse column format (CSC) and distributed with column major partition. 
///
/// Note
/// ----
/// 
/// Since only symmetric matrix is considered here, the compressed
/// sparse row format will also be represented by the compressed sparse
/// column format.
///
template <class F> struct DistSparseMatrix{
  /// @brief Matrix dimension.
  Int          size;         

  /// @brief Total number of nonzeros elements.
  Int          nnz;                             

  /// @brief Local number of local nonzeros elements on this processor.
  Int          nnzLocal;                        

  /// @brief The starting column on this processor.  The indices are
  /// 1-based (FORTRAN-convention).  Usually firstCol can be computed
  /// directly through the formula
  ///
  /// firstCol = mpirank * (size/mpisize) + 1
  /// 
  /// FIXME: The definition of firstCol is different in ScalES and in
  /// PEXSI.  In PEXSI, firstCol is 0-based (C-convention).
  Int          firstCol;

  /// @brief Dimension numColLocal + 1, storing the pointers to the
  /// nonzero row indices and nonzero values in rowptrLocal and
  /// nzvalLocal, respectively.  numColLocal is the number
  /// of local columns saved on this processor. The indices are 1-based
  /// (FORTRAN-convention), i.e.  colptrLocal[0] = 1. 
  IntNumVec    colptrLocal;                     

  /// @brief Dimension nnzLocal, storing the nonzero indices.
  /// The indices are 1-based (FORTRAN-convention), i.e. the first row
  /// index is 1. 
  IntNumVec    rowindLocal;                    

  /// @brief Dimension nnzLocal, storing the nonzero values.
  NumVec<F>    nzvalLocal;                      

  /// @brief MPI communicator
  MPI_Comm     comm;        
};

// Commonly used
typedef DistSparseMatrix<Real>       DblDistSparseMatrix;
typedef DistSparseMatrix<Complex>    CpxDistSparseMatrix;

// Utility subroutines

/// @brief Read a sparse matrix from a binary file using one processor.
template<typename F>
void ReadSparseMatrix ( const char* filename, SparseMatrix<F>& spmat );

/// @brief Read a sparse matrix from a formatted file using one processor.
template <class F> void
ReadSparseMatrixFormatted    ( const char* filename, SparseMatrix<F>& spmat );

/// @brief Read a distributed sparse matrix from a binary file.
template<typename F>
void ReadDistSparseMatrix ( const char* filename, 
    DistSparseMatrix<F>& pspmat, MPI_Comm comm );

/// @brief Read a distributed sparse matrix from a formatted file.
template<typename F>
void ReadDistSparseMatrixFormatted ( const char* filename, 
    DistSparseMatrix<F>& pspmat, MPI_Comm comm );

/// @brief Write distributed sparse matrix to a formatted file.
template<typename F>
void WriteDistSparseMatrixFormatted ( 
    const char* filename, 
    DistSparseMatrix<F>& pspmat );


/// @brief Read distributed sparse matrix from an unformatted file using
/// MPI-IO.
template<typename F>
void ParaReadDistSparseMatrix( 
    const char* filename, 
    DistSparseMatrix<F>& pspmat,
    MPI_Comm comm    );


/// @brief Write distributed sparse matrix to an unformatted file using
/// MPI-IO.
template<typename F>
void ParaWriteDistSparseMatrix( 
    const char* filename, 
    DistSparseMatrix<F>& pspmat );



} // namespace scales




#endif // _SPARSE_MATRIX_DECL_HPP_
