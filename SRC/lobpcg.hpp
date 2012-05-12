#ifndef _LOBPCG_HPP_
#define _LOBPCG_HPP_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "comobject.hpp"
#include "vec3t.hpp"
#include "numtns.hpp"
#include "interpreter.h"


#ifndef BlopexInt
#define BlopexInt int
#endif

/*--------------------------------------------------------------------------
 * Accessor functions for the Multi_Vector structure
 *--------------------------------------------------------------------------*/

#define serial_Multi_VectorData(vector)      ((vector) -> data)
#define serial_Multi_VectorSize(vector)      ((vector) -> size)
#define serial_Multi_VectorOwnsData(vector)  ((vector) -> owns_data)
#define serial_Multi_VectorNumVectors(vector) ((vector) -> num_vectors)


//----------------------------------------------------------------
// Real version (double precision) of LOBPCG
namespace REAL{ namespace LOBPCG{
  //----------------------------------------------------------------
  class serial_Multi_Vector {
    public:
      double  *data;
      BlopexInt      size;
      BlopexInt      owns_data;
      BlopexInt      num_vectors;  /* the above "size" is size of one vector */

      BlopexInt      num_active_vectors;
      BlopexInt     *active_indices;   /* indices of active vectors; 0-based notation */
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
    SetMultiVectorRandomValues( void* v, BlopexInt seed );
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

  BlopexInt serial_Multi_VectorSetConstantValues(serial_Multi_Vector *v,double value);
  BlopexInt serial_Multi_VectorSetRandomValues(serial_Multi_Vector *v , BlopexInt seed);
  BlopexInt serial_Multi_VectorCopy( serial_Multi_Vector *x , serial_Multi_Vector *y);
  BlopexInt serial_Multi_VectorScale( double alpha , serial_Multi_Vector *y, BlopexInt *mask  );
  BlopexInt serial_Multi_VectorAxpy( double alpha , serial_Multi_Vector *x , serial_Multi_Vector *y);
  BlopexInt serial_Multi_VectorInnerProd( serial_Multi_Vector *x,
					  serial_Multi_Vector *y,
					  BlopexInt gh, BlopexInt h, BlopexInt w, double* v);
  BlopexInt serial_Multi_VectorMultiScale( double *alpha, serial_Multi_Vector *v, BlopexInt *mask );

  BlopexInt serial_Multi_VectorByDiag( serial_Multi_Vector *x,
				       BlopexInt                *mask,
				       BlopexInt                n,
				       double             *alpha,
				       serial_Multi_Vector *y);

  BlopexInt serial_Multi_VectorInnerProdDiag( serial_Multi_Vector *x,
					      serial_Multi_Vector *y,
					      BlopexInt* mask, BlopexInt n, double* diag);

  BlopexInt
    serial_Multi_VectorSetMask(serial_Multi_Vector *mvector, BlopexInt * mask);
  BlopexInt
    serial_Multi_VectorCopyWithoutMask(serial_Multi_Vector *x , serial_Multi_Vector *y);
  BlopexInt
    serial_Multi_VectorByMatrix(serial_Multi_Vector *x, BlopexInt rGHeight, BlopexInt rHeight,
				BlopexInt rWidth, double* rVal, serial_Multi_Vector *y);
  BlopexInt
    serial_Multi_VectorByMulti_Vector(serial_Multi_Vector *x,
				      serial_Multi_Vector *y,
				      serial_Multi_Vector *z);
} }



//----------------------------------------------------------------
// Complex version (double precision) of LOBPCG
namespace COMPLEX{ namespace LOBPCG{
  //----------------------------------------------------------------
  typedef struct {double real, imag;} komplex;
  komplex kzero = {0.0,0.0};
  void complex_multiply(komplex* A, komplex* B, komplex* C);
  void complex_add(komplex* A, komplex* B, komplex* C);
  void complex_subtract(komplex* A, komplex* B, komplex* C);
  void complex_divide(komplex* A, komplex* B, komplex* C);

  //----------------------------------------------------------------
  class serial_Multi_Vector {
    public:
      double  *data;
      BlopexInt      size;
      BlopexInt      owns_data;
      BlopexInt      num_vectors;  /* the above "size" is size of one vector */

      BlopexInt      num_active_vectors;
      BlopexInt     *active_indices;   /* indices of active vectors; 0-based notation */
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
    SetMultiVectorRandomValues( void* v, BlopexInt seed );
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
  serial_Multi_Vector * serial_Multi_VectorLoad( char fileName[] );

  BlopexInt serial_Multi_VectorDestroy( serial_Multi_Vector *vector );
  BlopexInt serial_Multi_VectorInitialize( serial_Multi_Vector *vector );
  BlopexInt serial_Multi_VectorSetDataOwner(serial_Multi_Vector *vector , BlopexInt owns_data);
  /*
     BlopexInt serial_Multi_VectorPrint( serial_Multi_Vector *vector, const char *file_name );
     */
  BlopexInt serial_Multi_VectorPrint( serial_Multi_Vector *vector, char * tag, BlopexInt limit);
  BlopexInt serial_Multi_VectorSetConstantValues(serial_Multi_Vector *v, komplex value);
  BlopexInt serial_Multi_VectorSetRandomValues(serial_Multi_Vector *v , BlopexInt seed);
  BlopexInt serial_Multi_VectorCopy( serial_Multi_Vector *x , serial_Multi_Vector *y);
  BlopexInt serial_Multi_VectorScale( double alpha , serial_Multi_Vector *y, BlopexInt *mask  );
  BlopexInt serial_Multi_VectorAxpy( double alpha , serial_Multi_Vector *x , serial_Multi_Vector *y);
  BlopexInt serial_Multi_VectorInnerProd( serial_Multi_Vector *x,
					  serial_Multi_Vector *y,
					  BlopexInt gh, BlopexInt h, BlopexInt w, komplex* v);
  BlopexInt serial_Multi_VectorMultiScale( double *alpha, serial_Multi_Vector *v, BlopexInt *mask );

  BlopexInt serial_Multi_VectorByDiag( serial_Multi_Vector *x,
				       BlopexInt                *mask,
				       BlopexInt                n,
				       komplex            *alpha,
				       serial_Multi_Vector *y);

  BlopexInt serial_Multi_VectorInnerProdDiag( serial_Multi_Vector *x,
					      serial_Multi_Vector *y,
					      BlopexInt* mask, BlopexInt n, komplex* diag);

  BlopexInt
    serial_Multi_VectorSetMask(serial_Multi_Vector *mvector, BlopexInt * mask);
  BlopexInt
    serial_Multi_VectorCopyWithoutMask(serial_Multi_Vector *x , serial_Multi_Vector *y);
  BlopexInt
    serial_Multi_VectorByMatrix(serial_Multi_Vector *x, BlopexInt rGHeight, BlopexInt rHeight,
				BlopexInt rWidth, komplex* rVal, serial_Multi_Vector *y);
  BlopexInt
    serial_Multi_VectorByMulti_Vector(serial_Multi_Vector *x,
				      serial_Multi_Vector *y,
				      serial_Multi_Vector *z);
} }


#endif _LOBPCG_HPP_

