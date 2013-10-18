/*
	 Copyright (c) 2012 The Regents of the University of California,
	 through Lawrence Berkeley National Laboratory.  

   Author: Lin Lin
	 
   This file is part of DGDFT. All rights reserved.

	 Redistribution and use in source and binary forms, with or without
	 modification, are permitted provided that the following conditions are met:

	 (1) Redistributions of source code must retain the above copyright notice, this
	 list of conditions and the following disclaimer.
	 (2) Redistributions in binary form must reproduce the above copyright notice,
	 this list of conditions and the following disclaimer in the documentation
	 and/or other materials provided with the distribution.
	 (3) Neither the name of the University of California, Lawrence Berkeley
	 National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
	 be used to endorse or promote products derived from this software without
	 specific prior written permission.

	 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
	 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
	 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	 You are under no obligation whatsoever to provide any bug fixes, patches, or
	 upgrades to the features, functionality or performance of the source code
	 ("Enhancements") to anyone; however, if you choose to make your Enhancements
	 available either publicly, or directly to Lawrence Berkeley National
	 Laboratory, without imposing a separate written license agreement for such
	 Enhancements, then you hereby grant the following license: a non-exclusive,
	 royalty-free perpetual license to install, use, modify, prepare derivative
	 works, incorporate into other computer software, distribute, and sublicense
	 such enhancements or derivative works thereof, in binary and source code form.
*/
/// @file lobpcg++.cpp
/// @brief Interface to LOBPCG.
/// @date 2012-09-20
#ifndef _LOBPCG_HPP_
#define _LOBPCG_HPP_

#include "environment.hpp"

#include "fortran_matrix.h"
#include "fortran_interpreter.h"
#include "lobpcg.h"
#include "multivector.h"
#include "interpreter.h"

#ifndef BlopexInt
#define BlopexInt   Int  
#endif

/*--------------------------------------------------------------------------
 * Accessor functions for the Multi_Vector structure
 *--------------------------------------------------------------------------*/

#define serial_Multi_VectorData(vector)      ((vector) -> data)
#define serial_Multi_VectorSize(vector)      ((vector) -> size)
#define serial_Multi_VectorOwnsData(vector)  ((vector) -> owns_data)
#define serial_Multi_VectorNumVectors(vector) ((vector) -> num_vectors)


namespace dgdft{ 
namespace LOBPCG{
  //----------------------------------------------------------------
  // Definition of LAPACK functions used in LOBPCG (complex
  // version).  The komplex data structure used by LOBPCG is defined in
  // LOBPCG/fortran_matrix.h. Overloads the functions in lapack.h.
  extern "C"{
    void LAPACK(dpotrf) (char *uplo, BlopexInt *n,
                 double *a, BlopexInt * lda, BlopexInt *info);
    void LAPACK(dsygv)  (BlopexInt *itype, char *jobz, char *uplo, BlopexInt * n,
                 double *a, BlopexInt *lda, double *b, BlopexInt *ldb,
                 double *w, double *work, BlopexInt *lwork, BlopexInt *info);
    void LAPACK(zpotrf) (char *uplo, BlopexInt *n,
		  komplex *a, BlopexInt *lda, BlopexInt *info);
    void LAPACK(zhegv) (BlopexInt *itype, char *jobz, char *uplo, BlopexInt * n,
	       komplex *a, BlopexInt *lda, komplex *b, BlopexInt *ldb,
	       double *w, komplex *work, BlopexInt *lwork,
	       double * rwork,BlopexInt *info);
  }

  //----------------------------------------------------------------
  class serial_Multi_Vector {
    public:
      Scalar*        data;
      BlopexInt      size;
      BlopexInt      owns_data;
      BlopexInt      num_vectors;  /* the above "size" is size of one vector */

      BlopexInt      num_active_vectors;
      BlopexInt*     active_indices;   /* indices of active vectors; 0-based notation */
  };

  // From pcg_multi.h
  void*
    CreateCopyMultiVector( void*, BlopexInt copyValues );
  void
    DestroyMultiVector( void* );
  BlopexInt
    MultiVectorWidth( void* v );
  BlopexInt
    MultiVectorHeight( void* v );
  void
    MultiSetMask( void *vector, BlopexInt *mask );
  void
    CopyMultiVector( void* src, void* dest );
  void
    ClearMultiVector( void* );
  void
    MultiInnerProd( void*, void*,
		    BlopexInt gh, BlopexInt h, BlopexInt w, void* v );
  void
    MultiInnerProdDiag( void* x, void* y,
			BlopexInt* mask, BlopexInt n, void* diag );
  void
    MultiVectorByMatrix( void*,
			 BlopexInt gh, BlopexInt h, BlopexInt w, void* v,
			 void* );
  void MultiVectorByDiagonal( void* x,
			      BlopexInt* mask, BlopexInt n, void* diag,
			      void* y );
  void
    MultiVectorAxpy( double, void*, void* );

  BlopexInt
    SerialSetupInterpreter( mv_InterfaceInterpreter *i );
  void
    MultiVectorPrint(  void   *x, char* tag, BlopexInt limit);

  // From multi_vector.h  
  serial_Multi_Vector * serial_Multi_VectorCreate( BlopexInt size, BlopexInt num_vectors  );
  serial_Multi_Vector *serial_Multi_VectorRead( char *file_name );

  BlopexInt serial_Multi_VectorDestroy( serial_Multi_Vector *vector );
  BlopexInt serial_Multi_VectorInitialize( serial_Multi_Vector *vector );
  BlopexInt serial_Multi_VectorSetDataOwner(serial_Multi_Vector *vector , BlopexInt owns_data);
  /*
     BlopexInt serial_Multi_VectorPrint( serial_Multi_Vector *vector, const char *file_name );
     */
  BlopexInt serial_Multi_VectorPrint( serial_Multi_Vector *vector, char * tag, BlopexInt limit);

  BlopexInt serial_Multi_VectorSetConstantValues(serial_Multi_Vector *v,Scalar value);
  BlopexInt serial_Multi_VectorCopy( serial_Multi_Vector *x , serial_Multi_Vector *y);
  BlopexInt serial_Multi_VectorAxpy( double alpha , serial_Multi_Vector *x , serial_Multi_Vector *y);
  BlopexInt serial_Multi_VectorInnerProd( serial_Multi_Vector *x,
					  serial_Multi_Vector *y,
					  BlopexInt gh, BlopexInt h, BlopexInt w, Scalar* v);

  BlopexInt serial_Multi_VectorByDiag( serial_Multi_Vector *x,
				       BlopexInt                *mask,
				       BlopexInt                n,
				       Scalar                   *alpha,
				       serial_Multi_Vector      *y);

  BlopexInt serial_Multi_VectorInnerProdDiag( serial_Multi_Vector *x,
					      serial_Multi_Vector *y,
					      BlopexInt* mask, BlopexInt n, Scalar* diag);

  BlopexInt
    serial_Multi_VectorSetMask(serial_Multi_Vector *mvector, BlopexInt * mask);
  BlopexInt
    serial_Multi_VectorCopyWithoutMask(serial_Multi_Vector *x , serial_Multi_Vector *y);
  BlopexInt
    serial_Multi_VectorByMatrix(serial_Multi_Vector *x, BlopexInt rGHeight, BlopexInt rHeight,
				BlopexInt rWidth, Scalar* rVal, serial_Multi_Vector *y);
} // namespace LOBPCG 
} // namespace dgdft

#endif // _LOBPCG_HPP_

