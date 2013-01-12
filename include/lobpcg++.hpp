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

