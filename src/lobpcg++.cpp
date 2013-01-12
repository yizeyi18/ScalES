#include "lobpcg++.hpp"
#include "blas.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "fortran_matrix.h"
#include "fortran_interpreter.h"
#include "lobpcg.h"
#include "multivector.h"
#include "interpreter.h"


namespace dgdft{ 
namespace LOBPCG{
/*--------------------------------------------------------------------------
	CreateCopyMultiVector                                              generic
	--------------------------------------------------------------------------*/
void*
	CreateCopyMultiVector( void* src_, BlopexInt copyValues )
	{
		serial_Multi_Vector *src = (serial_Multi_Vector *)src_;
		serial_Multi_Vector *dest;

		/* create vector with the same parameters as src */

		dest = serial_Multi_VectorCreate(src->size, src->num_vectors);
		serial_Multi_VectorInitialize(dest);

		/* copy values if necessary */

		if (copyValues)
			serial_Multi_VectorCopyWithoutMask(src, dest);

		return dest;
	}

/*--------------------------------------------------------------------------
	DestroyMultiVector                                                 generic
	--------------------------------------------------------------------------*/
void
	DestroyMultiVector( void *vvector )
	{
		BlopexInt dummy;
		serial_Multi_Vector *vector = (serial_Multi_Vector*)vvector;

		dummy=serial_Multi_VectorDestroy( vector );
	}

/*--------------------------------------------------------------------------
	MultiVectorWidth                                                   generic
	--------------------------------------------------------------------------*/
BlopexInt
	MultiVectorWidth( void* v )
	{
		return ((serial_Multi_Vector*)v)->num_vectors;
	}

/*--------------------------------------------------------------------------
	MultiSetMask                                                       generic
	--------------------------------------------------------------------------*/
void
	MultiSetMask( void *vector, BlopexInt *mask )
	{
		serial_Multi_VectorSetMask( ( serial_Multi_Vector *)vector, mask );
	}

/*--------------------------------------------------------------------------
	CopyMultiVector                                                    generic
	--------------------------------------------------------------------------*/
void
	CopyMultiVector( void *x, void *y)
	{
		BlopexInt dummy;

		dummy = serial_Multi_VectorCopy( (serial_Multi_Vector *) x,
				(serial_Multi_Vector *) y);
	}

/*--------------------------------------------------------------------------
	ClearMultiVector                                                   scalar
	--------------------------------------------------------------------------*/
void
	ClearMultiVector(void *x)
	{
		BlopexInt dummy;

		dummy=serial_Multi_VectorSetConstantValues( (serial_Multi_Vector *)x,
				static_cast<Scalar>(0.0) );
	}


/*--------------------------------------------------------------------------
	MultiInnerProd                                                      scalar
	--------------------------------------------------------------------------*/
void
	MultiInnerProd(void * x_, void * y_,
			BlopexInt gh, BlopexInt h, BlopexInt w, void* v )
	{
		serial_Multi_VectorInnerProd( (serial_Multi_Vector *)x_,
				(serial_Multi_Vector *)y_,
				gh, h, w, (Scalar *) v);
	}


/*--------------------------------------------------------------------------
	MultiInnerProdDiag                                                  scalar
	--------------------------------------------------------------------------*/
void
	MultiInnerProdDiag( void* x_, void* y_,
			BlopexInt* mask, BlopexInt n, void* diag )
	{
		serial_Multi_VectorInnerProdDiag( (serial_Multi_Vector *)x_,
				(serial_Multi_Vector *)y_,
				mask, n, (Scalar *) diag);
	}

/*--------------------------------------------------------------------------
	MultiVectorByDiagonal                                              scalar
	--------------------------------------------------------------------------*/
void
	MultiVectorByDiagonal( void* x,
			BlopexInt* mask, BlopexInt n, void* diag,
			void* y )
	{
		BlopexInt dummy;

		dummy = serial_Multi_VectorByDiag( (serial_Multi_Vector *) x, mask, n,
				(Scalar *) diag,
				(serial_Multi_Vector *) y );
	}

/*--------------------------------------------------------------------------
	MultiVectorByMatrix                                                scalar
	--------------------------------------------------------------------------*/
void
	MultiVectorByMatrix( void* x,
			BlopexInt gh, BlopexInt h, BlopexInt w, void* v,
			void* y )
	{
		serial_Multi_VectorByMatrix((serial_Multi_Vector *)x, gh, h,
				w, (Scalar *) v, (serial_Multi_Vector *)y);

	}
/*--------------------------------------------------------------------------
	MultiAxpy                                                          scalar
	--------------------------------------------------------------------------*/
void
	MultiVectorAxpy( double alpha, void   *x, void   *y)
	{
		serial_Multi_VectorAxpy(  alpha,
				(serial_Multi_Vector *) x,
				(serial_Multi_Vector *) y) ;
	}
/*--------------------------------------------------------------------------
	MultiVectorPrint                                                   scalar
	--------------------------------------------------------------------------*/
void
	MultiVectorPrint( void *x, char * tag, BlopexInt limit )
	{
		serial_Multi_VectorPrint( (serial_Multi_Vector *) x, tag, limit );
	}

BlopexInt
	/*--------------------------------------------------------------------------
		SerialSetupInterpreter                                             generic
		--------------------------------------------------------------------------*/
	SerialSetupInterpreter( mv_InterfaceInterpreter *i )
	{
		/* Vector part */

		i->CreateVector = NULL;
		i->DestroyVector = NULL;
		i->InnerProd = NULL;
		i->CopyVector = NULL;
		i->ClearVector = NULL;
		i->SetRandomValues = NULL;
		i->ScaleVector = NULL;
		i->Axpy = NULL;

		/* Multivector part. A minimal list */

		i->CreateMultiVector = NULL;
		i->CopyCreateMultiVector = CreateCopyMultiVector;
		i->DestroyMultiVector = DestroyMultiVector;

		i->Width = MultiVectorWidth;
		i->Height = NULL;
		i->SetMask = MultiSetMask;
		i->CopyMultiVector = CopyMultiVector;
		i->ClearMultiVector = ClearMultiVector;
		i->SetRandomVectors = NULL;
		i->MultiInnerProd = MultiInnerProd;
		i->MultiInnerProdDiag = MultiInnerProdDiag;
		i->MultiVecMat = MultiVectorByMatrix;
		i->MultiVecMatDiag = MultiVectorByDiagonal;
		i->MultiAxpy = MultiVectorAxpy;
		i->MultiXapy = NULL;
		i->Eval = NULL;
		i->MultiPrint = MultiVectorPrint;

		return 0;
	}

/*--------------------------------------------------------------------------
 * serial_Multi_VectorCreate                                         generic
 *--------------------------------------------------------------------------*/

serial_Multi_Vector *
	serial_Multi_VectorCreate( BlopexInt size, BlopexInt num_vectors  )
	{
		serial_Multi_Vector *mvector;

		mvector = (serial_Multi_Vector *) malloc (sizeof(serial_Multi_Vector));

		serial_Multi_VectorNumVectors(mvector) = num_vectors;
		serial_Multi_VectorSize(mvector) = size;

		serial_Multi_VectorOwnsData(mvector) = 1;
		serial_Multi_VectorData(mvector) = NULL;


		mvector->num_active_vectors=0;
		mvector->active_indices=NULL;

		return mvector;
	}
/*--------------------------------------------------------------------------
 * serial_Multi_VectorInitialize                                   scalar  
 *--------------------------------------------------------------------------*/
BlopexInt
	serial_Multi_VectorInitialize( serial_Multi_Vector *mvector )
	{
		BlopexInt    ierr = 0, i, size, num_vectors;

		size        = serial_Multi_VectorSize(mvector);
		num_vectors = serial_Multi_VectorNumVectors(mvector);

		if (NULL==serial_Multi_VectorData(mvector))
			serial_Multi_VectorData(mvector) = (Scalar*) malloc (sizeof(Scalar)*size*num_vectors);

		/* now we create a "mask" of "active" vectors; initially all vectors are active */
		if (NULL==mvector->active_indices)
		{
			mvector->active_indices=(BlopexInt *) malloc(sizeof(BlopexInt)*num_vectors);

			for (i=0; i<num_vectors; i++)
				mvector->active_indices[i]=i;

			mvector->num_active_vectors=num_vectors;
		}

		return ierr;
	}

/*--------------------------------------------------------------------------
 * serial_Multi_VectorSetDataOwner                                   generic
 *--------------------------------------------------------------------------*/
BlopexInt
	serial_Multi_VectorSetDataOwner( serial_Multi_Vector *mvector, BlopexInt owns_data )
	{
		BlopexInt    ierr=0;

		serial_Multi_VectorOwnsData(mvector) = owns_data;

		return ierr;
	}

/*--------------------------------------------------------------------------
 * serial_Multi_VectorDestroy                                        generic
 *--------------------------------------------------------------------------*/
BlopexInt
	serial_Multi_VectorDestroy( serial_Multi_Vector *mvector )
	{
		BlopexInt    ierr=0;

		if (NULL!=mvector)
		{
			if (serial_Multi_VectorOwnsData(mvector) && NULL!=serial_Multi_VectorData(mvector))
				free( serial_Multi_VectorData(mvector) );

			if (NULL!=mvector->active_indices)
				free(mvector->active_indices);

			free(mvector);
		}
		return ierr;
	}

/*--------------------------------------------------------------------------
 * serial_Multi_VectorSetMask                                        generic
 *--------------------------------------------------------------------------*/
BlopexInt
	serial_Multi_VectorSetMask(serial_Multi_Vector *mvector, BlopexInt * mask)
	{
		/* this routine accepts mask in "zeros and ones format, and converts it to the one used in
			 the structure "serial_Multi_Vector" */
		BlopexInt  num_vectors = mvector->num_vectors;
		BlopexInt i;


		/* may be it's better to just check if it is not null, and throw an error, if it is? */
		if (mvector->active_indices==NULL)
			mvector->active_indices=(BlopexInt *) malloc(sizeof(BlopexInt)*num_vectors);

		mvector->num_active_vectors=0;

		if (mask!=NULL)
			for (i=0; i<num_vectors; i++)
			{
				if ( mask[i] )
					mvector->active_indices[mvector->num_active_vectors++]=i;
			}
		else
			for (i=0; i<num_vectors; i++)
				mvector->active_indices[mvector->num_active_vectors++]=i;

		return 0;
	}


/*--------------------------------------------------------------------------
 * serial_Multi_VectorSetConstantValues                            scalar  
 *--------------------------------------------------------------------------*/
BlopexInt
	serial_Multi_VectorSetConstantValues( serial_Multi_Vector *v,
			Scalar value)
	{
		Scalar  *vector_data = (Scalar *) serial_Multi_VectorData(v);
		BlopexInt      size        = serial_Multi_VectorSize(v);
		BlopexInt      i, j, start_offset, end_offset;

		for (i = 0; i < v->num_active_vectors; i++)
		{
			start_offset = v->active_indices[i]*size;
			end_offset = start_offset+size;

			for (j=start_offset; j < end_offset; j++)
				vector_data[j]= value;
		}

		return 0;
	}


/*--------------------------------------------------------------------------
 * serial_Multi_VectorCopy   y=x  using indices                     scalar 
 *
 * y should have already been initialized at the same size as x
 *--------------------------------------------------------------------------*/
BlopexInt
	serial_Multi_VectorCopy( serial_Multi_Vector *x, serial_Multi_Vector *y)
	{
		Scalar *x_data;
		Scalar *y_data;
		BlopexInt i;
		BlopexInt size;
		BlopexInt num_bytes;
		BlopexInt num_active_vectors;
		Scalar * dest;
		Scalar * src;
		BlopexInt * x_active_ind;
		BlopexInt * y_active_ind;

		assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);

		num_active_vectors = x->num_active_vectors;
		size = x->size;
		num_bytes = size*sizeof(Scalar);
		x_data = (Scalar *) x->data;
		y_data = (Scalar *) y->data;
		x_active_ind=x->active_indices;
		y_active_ind=y->active_indices;

		for (i=0; i<num_active_vectors; i++)
		{
			src  = x_data + size * x_active_ind[i];
			dest = y_data + size * y_active_ind[i];

			memcpy(dest,src,num_bytes);
		}

		return 0;
	}

/*--------------------------------------------------------------------------
 * serial_Multi_VectorCopyWithoutMask              y=x               scalar
 * copies data from x to y without using indices
 * y should have already been initialized at the same size as x
 *--------------------------------------------------------------------------*/
BlopexInt
	serial_Multi_VectorCopyWithoutMask(serial_Multi_Vector *x , serial_Multi_Vector *y)
	{
		BlopexInt byte_count;

		assert (x->size == y->size && x->num_vectors == y->num_vectors);

		byte_count = sizeof(Scalar) * x->size * x->num_vectors;

		/* threading not done here since it's not done (reason?) in serial_VectorCopy
			 from vector.c */

		memcpy(y->data,x->data,byte_count);

		return 0;
	}

/*--------------------------------------------------------------------------
 * serial_Multi_VectorAxpy      y=alpha*x+y using indices            scalar
 *
 * call seq < MultiVectorAxpy < i->MultiAxpy < mv_MultiVectorAxpy
 * alpha is always 1 or -1 double
 *--------------------------------------------------------------------------*/
BlopexInt
	serial_Multi_VectorAxpy( double           alpha,
			serial_Multi_Vector *x,
			serial_Multi_Vector *y)
	{
		Scalar * x_data;
		Scalar * y_data;
		Scalar * src;
		Scalar * dest;
		BlopexInt * x_active_ind;
		BlopexInt * y_active_ind;
		BlopexInt i, j;
		BlopexInt size;
		BlopexInt num_active_vectors;

		assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);

		x_data = (Scalar *) x->data;
		y_data = (Scalar *) y->data;
		size = x->size;
		num_active_vectors = x->num_active_vectors;
		x_active_ind = x->active_indices;
		y_active_ind = y->active_indices;

		// Does not use full BLAS but allows deflation
		if(1){
			for(i=0; i<num_active_vectors; i++)
			{
				src = x_data + x_active_ind[i]*size;
				dest = y_data + y_active_ind[i]*size;

				blas::Axpy( size, static_cast<Scalar>(alpha), 
						src, 1, dest, 1 );
			}
		}

		// NEW Code, calculate everything. Make sure it works with deflation  
		if(0){
			blas::Axpy( (size * x->num_vectors), static_cast<Scalar>(alpha), 
					x_data, 1, y_data, 1 );
		}

		return 0;
	}

/*--------------------------------------------------------------------------
 * serial_Multi_VectorByDiag:    y=x*alpha using indices             scalar
 *
 * if y and x are mxn then alpha is nx1
 * call seq < MultiVectorByDiagonal < i->MultiVecMatDiag < mv_MultiVectorByDiagonal
 *--------------------------------------------------------------------------*/
BlopexInt
	serial_Multi_VectorByDiag( serial_Multi_Vector *x,
			BlopexInt           *mask,
			BlopexInt           n,
			Scalar              *alpha,
			serial_Multi_Vector *y)
	{
		Scalar  *x_data;
		Scalar  *y_data;
		BlopexInt      size;
		BlopexInt      num_active_vectors;
		BlopexInt      i,j;
		Scalar  *dest;
		Scalar  *src;
		BlopexInt * x_active_ind;
		BlopexInt * y_active_ind;
		BlopexInt * al_active_ind;
		BlopexInt num_active_als;
		Scalar current_alpha;

		assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);

		/* build list of active indices in alpha */

		al_active_ind = (BlopexInt *) malloc(sizeof(BlopexInt)*n);
		num_active_als = 0;

		if (mask!=NULL)
			for (i=0; i<n; i++)
			{
				if (mask[i])
					al_active_ind[num_active_als++]=i;
			}
		else
			for (i=0; i<n; i++)
				al_active_ind[num_active_als++]=i;

		assert (num_active_als==x->num_active_vectors);

		x_data = (Scalar *) x->data;
		y_data = (Scalar *) y->data;
		size = x->size;
		num_active_vectors = x->num_active_vectors;
		x_active_ind = x->active_indices;
		y_active_ind = y->active_indices;

		for(i=0; i<num_active_vectors; i++)
		{
			src = x_data + x_active_ind[i]*size;
			dest = y_data + y_active_ind[i]*size;
			current_alpha=alpha[ al_active_ind[i] ];
				
			for (j=0; j<size; j++){
				*(dest++) = (current_alpha) * (*(src++));
			}
		}

		free(al_active_ind);
		return 0;
	}

/*--------------------------------------------------------------------------
 * serial_Multi_VectorInnerProd        v=x'*y  using indices          scalar
 *
 * call seq < MultiInnerProd < i->MultiInnerProd < mv_MultiVectorByMultiVector
 *--------------------------------------------------------------------------*/
BlopexInt serial_Multi_VectorInnerProd( serial_Multi_Vector *x,
		serial_Multi_Vector *y,
		BlopexInt gh, BlopexInt h, BlopexInt w, Scalar* v)
{
	Scalar *x_data;
	Scalar *y_data;
	BlopexInt      size;
	BlopexInt      x_num_active_vectors;
	BlopexInt      y_num_active_vectors;
	BlopexInt      i,j,k;
	Scalar *y_ptr;
	Scalar *x_ptr;
	BlopexInt * x_active_ind;
	BlopexInt * y_active_ind;
	BlopexInt gap;

	assert (x->size==y->size);

	x_data = (Scalar *) x->data;
	y_data = (Scalar *) y->data;
	size = x->size;
	x_num_active_vectors = x->num_active_vectors;
	y_num_active_vectors = y->num_active_vectors;

	assert (x_num_active_vectors==h && y_num_active_vectors==w);

	x_active_ind = x->active_indices;
	y_active_ind = y->active_indices;

	gap = gh-h;

	// Does not use full BLAS but allows deflation
	if(1){
		for(j=0; j<y_num_active_vectors; j++)
		{
			y_ptr = y_data + y_active_ind[j]*size;

			for (i=0; i<x_num_active_vectors; i++) {
				x_ptr = x_data + x_active_ind[i]*size;
				*v++  = blas::Dot( size, x_ptr, 1, y_ptr, 1 );
			}
			v+=gap;
		}
	}

	// LLIN: New version This assumes that deflation is not used and uses BLAS3
	// Another disadvantage of this method is that it is not so easy to
	// generalize to parallel setup.
	if(0){
		assert( x->num_vectors == x->num_active_vectors &&
				y->num_vectors == y->num_active_vectors); 
		blas::Gemm('C', 'N', x_num_active_vectors, y_num_active_vectors,
				 size, SCALAR_ONE, x_data, size, y_data, size,
				 SCALAR_ZERO, v, gh);
	}


	return 0;
}

/*--------------------------------------------------------------------------
 * serial_Multi_VectorInnerProdDiag                                 scalar 
 *
 * diag=diagonal(x'*y) using indices
 * mask is index of diag
 *
 * call seq < MultiInnerProdDiag < i->MultiInnerProdDiag < mv_MultiVectorByMultiVectorDiag
 *--------------------------------------------------------------------------*/
BlopexInt serial_Multi_VectorInnerProdDiag( serial_Multi_Vector *x,
		serial_Multi_Vector *y,
		BlopexInt* mask, BlopexInt n, Scalar* diag)
{
	Scalar  *x_data;
	Scalar  *y_data;
	BlopexInt      size;
	BlopexInt      num_active_vectors;
	BlopexInt      * x_active_ind;
	BlopexInt      * y_active_ind;
	Scalar  *y_ptr;
	Scalar  *x_ptr;
	BlopexInt      i, k;
	BlopexInt      * al_active_ind;
	BlopexInt      num_active_als;

	assert(x->size==y->size && x->num_active_vectors == y->num_active_vectors);

	/* build list of active indices in alpha */

	al_active_ind = (BlopexInt *) malloc(sizeof(BlopexInt)*n);
	num_active_als = 0;

	if (mask!=NULL)
		for (i=0; i<n; i++)
		{
			if (mask[i])
				al_active_ind[num_active_als++]=i;
		}
	else
		for (i=0; i<n; i++)
			al_active_ind[num_active_als++]=i;

	assert (num_active_als==x->num_active_vectors);

	x_data = (Scalar *) x->data;
	y_data = (Scalar *) y->data;
	size = x->size;
	num_active_vectors = x->num_active_vectors;
	x_active_ind = x->active_indices;
	y_active_ind = y->active_indices;

	for (i=0; i<num_active_vectors; i++)
	{
		x_ptr = x_data + x_active_ind[i]*size;
		y_ptr = y_data + y_active_ind[i]*size;
		diag[al_active_ind[i]] = blas::Dot( size, x_ptr, 1, y_ptr, 1 );	
	}

	free(al_active_ind);
	return 0;
}

/*--------------------------------------------------------------------------
 * serial_Multi_VectorByMatrix      y=x*rVal using indices          scalar 
 *
 * call seq < MultiVectorByMatrix < i->MultiVecMat < mv_MultiVectorByMatrix
 *--------------------------------------------------------------------------*/
BlopexInt
serial_Multi_VectorByMatrix(serial_Multi_Vector *x, BlopexInt rGHeight, BlopexInt rHeight,
		BlopexInt rWidth, Scalar* rVal, serial_Multi_Vector *y)
{
	Scalar *x_data;
	Scalar *y_data;
	BlopexInt      size;
	BlopexInt     * x_active_ind;
	BlopexInt     * y_active_ind;
	Scalar *y_ptr;
	Scalar *x_ptr;
	Scalar  current_coef;
	BlopexInt      i,j,k;
	BlopexInt      gap;

	assert(rHeight>0);
	assert (rHeight==x->num_active_vectors && rWidth==y->num_active_vectors);

	x_data = (Scalar *) x->data;
	y_data = (Scalar *) y->data;
	size = x->size;
	x_active_ind = x->active_indices;
	y_active_ind = y->active_indices;
	gap = rGHeight - rHeight;

	//LLIN: Not using BLAS, but supports deflation
	if(1){
		for (j=0; j<rWidth; j++)
		{
			y_ptr = y_data + y_active_ind[j]*size;

			/* ------ set current "y" to first member in a sum ------ */
			x_ptr = x_data + x_active_ind[0]*size;
			current_coef = *rVal++;

			for (k=0; k<size; k++)
				y_ptr[k] = current_coef * x_ptr[k];

			/* ------ now add all other members of a sum to "y" ----- */
			for (i=1; i<rHeight; i++)
			{
				x_ptr = x_data + x_active_ind[i]*size;
				current_coef = *rVal++;

				for (k=0; k<size; k++) {
					y_ptr[k] += current_coef * x_ptr[k];
				}
			}

			rVal += gap;
		}
	}

	// LLIN: This assumes that deflation is not used and uses BLAS3
	if(0){
		assert( x->num_vectors == x->num_active_vectors &&
				y->num_vectors == y->num_active_vectors); 
		blas::Gemm( 'N', 'N', size, rWidth, rHeight, SCALAR_ONE, 
				x_data, size, rVal, rGHeight, SCALAR_ZERO, y_data,
				size );
	}


	return 0;
}

/*--------------------------------------------------------------------------
 * serial_Multi_VectorPrint                                           scalar
 *--------------------------------------------------------------------------*/
BlopexInt serial_Multi_VectorPrint(serial_Multi_Vector * x,char * tag, BlopexInt limit)
{
	Scalar * p;
	BlopexInt     * pact;
	BlopexInt       i, j;
	BlopexInt     rows,cols;
	printf("======= %s =========\n",tag);
	printf("size %d\n", x->size);
	printf("owns data %d\n",x->owns_data);
	printf("num vectors %d\n",x->num_vectors);
	printf("num active vectors  %d\n",x->num_active_vectors);

	rows = x->size;
	cols = x->num_active_vectors;
	if (limit != 0) {
		if (rows > limit) rows = limit;
		if (cols > limit) cols = limit;
	}

	pact=x->active_indices;
	for (i=0; i<cols; i++, pact++)
		printf("index %d active %d\n", i, *pact);

	for (i=0; i<cols; i++) {  
		p=(Scalar *)x->data;
		p = &p[x->active_indices[i]*x->size];
#ifndef _USE_COMPLEX_
		for (j = 0; j < rows; j++,p++)
			printf("%d %d  %22.16e\n",j,i,p);
#else
		for (j = 0; j < rows; j++,p++)
			printf("%d %d  %22.16e  %22.16e \n",j,i,p->real(),p->imag());
#endif
	}

	return 0;
}

/*--------------------------------------------------------------------------
 * serial_Multi_VectorPrintShort                                      scalar
 *--------------------------------------------------------------------------*/
BlopexInt serial_Multi_VectorPrintShort(serial_Multi_Vector * x)
{
	printf("size %d\n", x->size);
	printf("owns data %d\n",x->owns_data);
	printf("num vectors %d\n",x->num_vectors);
	printf("num active vectors  %d\n",x->num_active_vectors);
	return(0);
}

} // namespace LOBPCG 
} // namespace dgdft
