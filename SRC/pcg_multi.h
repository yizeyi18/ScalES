/* @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ */
/* @@@ BLOPEX (version 1.1) LGPL Version 3 or above.  See www.gnu.org. */
/* @@@ Copyright 2010 BLOPEX team http://code.google.com/p/blopex/     */
/* @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ */
/* This code was developed by Merico Argentati, Andrew Knyazev, Ilya Lashuk and Evgueni Ovtchinnikov */

#ifndef MULTIVECTOR_VOID_FUNCTION_PROTOTYPES
#define MULTIVECTOR_VOID_FUNCTION_PROTOTYPES

#include "interpreter.h"

#ifdef __cplusplus
extern "C" {
#endif

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

/*
BlopexInt
MultiVectorPrint( void* x, const char* fileName );
void*
MultiVectorRead( MPI_Comm comm, void*, const char* fileName );
*/

/* Special subroutines */
//void MatMultiVecGlb (void * data, void * x, void * y);
//void ApplyPrecGlb (void * data, void * X, void * AX);
//void MatMultiVecBuf (void * data, void * x, void * y);
//void ApplyPrecBuf (void * data, void * X, void * AX);



#ifdef __cplusplus
}
#endif

#endif /* MULTIVECTOR_VOID_FUNCTION_PROTOTYPES */
