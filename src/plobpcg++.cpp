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
/// @file plobpcg++.cpp
/// @brief Implementation of the interface to LOBPCG.
/// @date 2012-09-20
#include "plobpcg++.hpp"
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
namespace PLOBPCG{
/*--------------------------------------------------------------------------
	CreateCopyMultiVector                                              generic
	--------------------------------------------------------------------------*/
void*
	CreateCopyMultiVector( void* src_, BlopexInt copyValues )
	{
		parallel_Multi_Vector *src = (parallel_Multi_Vector *)src_;
		parallel_Multi_Vector *dest;

		/* create vector with the same parameters as src */

		dest = parallel_Multi_VectorCreate(src->size, src->num_vectors);
		parallel_Multi_VectorInitialize(dest);

		/* copy values if necessary */

		if (copyValues)
			parallel_Multi_VectorCopyWithoutMask(src, dest);

		return dest;
	}

/*--------------------------------------------------------------------------
	DestroyMultiVector                                                 generic
	--------------------------------------------------------------------------*/
void
	DestroyMultiVector( void *vvector )
	{
		BlopexInt dummy;
		parallel_Multi_Vector *vector = (parallel_Multi_Vector*)vvector;

		dummy=parallel_Multi_VectorDestroy( vector );
	}

/*--------------------------------------------------------------------------
	MultiVectorWidth                                                   generic
	--------------------------------------------------------------------------*/
BlopexInt
	MultiVectorWidth( void* v )
	{
		return ((parallel_Multi_Vector*)v)->numTotal_vectors;
	}

/*--------------------------------------------------------------------------
	MultiSetMask                                                       generic
	--------------------------------------------------------------------------*/
void
	MultiSetMask( void *vector, BlopexInt *mask )
	{
		parallel_Multi_VectorSetMask( ( parallel_Multi_Vector *)vector, mask );
	}

/*--------------------------------------------------------------------------
	CopyMultiVector                                                    generic
	--------------------------------------------------------------------------*/
void
	CopyMultiVector( void *x, void *y)
	{
		BlopexInt dummy;

		dummy = parallel_Multi_VectorCopy( (parallel_Multi_Vector *) x,
				(parallel_Multi_Vector *) y);
	}

/*--------------------------------------------------------------------------
	ClearMultiVector                                                   scalar
	--------------------------------------------------------------------------*/
void
	ClearMultiVector(void *x)
	{
		BlopexInt dummy;

		dummy=parallel_Multi_VectorSetConstantValues( (parallel_Multi_Vector *)x,
				static_cast<Scalar>(0.0) );
	}


/*--------------------------------------------------------------------------
	MultiInnerProd                                                      scalar
	--------------------------------------------------------------------------*/
void
	MultiInnerProd(void * x_, void * y_,
			BlopexInt gh, BlopexInt h, BlopexInt w, void* v )
	{
		parallel_Multi_VectorInnerProd( (parallel_Multi_Vector *)x_,
				(parallel_Multi_Vector *)y_,
				gh, h, w, (Scalar *) v);
	}


/*--------------------------------------------------------------------------
	MultiInnerProdDiag                                                  scalar
	--------------------------------------------------------------------------*/
void
	MultiInnerProdDiag( void* x_, void* y_,
			BlopexInt* mask, BlopexInt n, void* diag )
	{
		parallel_Multi_VectorInnerProdDiag( (parallel_Multi_Vector *)x_,
				(parallel_Multi_Vector *)y_,
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

		dummy = parallel_Multi_VectorByDiag( (parallel_Multi_Vector *) x, mask, n,
				(Scalar *) diag,
				(parallel_Multi_Vector *) y );
	}

/*--------------------------------------------------------------------------
	MultiVectorByMatrix                                                scalar
	--------------------------------------------------------------------------*/
void
	MultiVectorByMatrix( void* x,
			BlopexInt gh, BlopexInt h, BlopexInt w, void* v,
			void* y )
	{
		parallel_Multi_VectorByMatrix((parallel_Multi_Vector *)x, gh, h,
				w, (Scalar *) v, (parallel_Multi_Vector *)y);

	}
/*--------------------------------------------------------------------------
	MultiAxpy                                                          scalar
	--------------------------------------------------------------------------*/
void
	MultiVectorAxpy( double alpha, void   *x, void   *y)
	{
		parallel_Multi_VectorAxpy(  alpha,
				(parallel_Multi_Vector *) x,
				(parallel_Multi_Vector *) y) ;
	}
/*--------------------------------------------------------------------------
	MultiVectorPrint                                                   scalar
	--------------------------------------------------------------------------*/
void
	MultiVectorPrint( void *x, char * tag, BlopexInt limit )
	{
		parallel_Multi_VectorPrint( (parallel_Multi_Vector *) x, tag, limit );
	}

BlopexInt
	/*--------------------------------------------------------------------------
		parallelSetupInterpreter                                             generic
		--------------------------------------------------------------------------*/
	parallelSetupInterpreter( mv_InterfaceInterpreter *i )
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
 * parallel_Multi_VectorCreate                                         generic
 *--------------------------------------------------------------------------*/

// FIXME Not enough input information. Can add more arguments.
parallel_Multi_Vector *
	parallel_Multi_VectorCreate( BlopexInt size, BlopexInt num_vectors  )
	{
    parallel_Multi_Vector *mvector;

		mvector = (parallel_Multi_Vector *) malloc (sizeof(parallel_Multi_Vector));

		parallel_Multi_VectorNumVectors(mvector) = num_vectors;
		parallel_Multi_VectorSize(mvector) = size;

		parallel_Multi_VectorOwnsData(mvector) = 1;
		parallel_Multi_VectorData(mvector) = NULL;

		mvector->num_active_vectors = 0;
		mvector->active_indices = NULL;

    mvector->comm = MPI_COMM_WORLD;

//huwei
  
//    MPI_Barrier(comm_);
//    Int mpirank;  MPI_Comm_rank(comm_, &mpirank);
//    Int mpisize;  MPI_Comm_size(comm_, &mpisize);
   
//    BlopexInt numTotal_vectors;

//    mpi::Allreduce( &num_vectors, &numTotal_vectors, 1, MPI_SUM, domain_.comm );

//    parallel_Multi_VectorNumTotalVectors(mvector) = numTotal_vectors;

    mvector->numTotal_vectors = 0;    
    mvector->global_indices = NULL;

//huwei

		return mvector;
	}
/*--------------------------------------------------------------------------
 * parallel_Multi_VectorInitialize                                   scalar  
 *--------------------------------------------------------------------------*/
// FIXME Add more attributes
BlopexInt
	parallel_Multi_VectorInitialize( parallel_Multi_Vector *mvector )
	{
		BlopexInt    ierr = 0, i, size, num_vectors;

		size        = parallel_Multi_VectorSize(mvector);
		num_vectors = parallel_Multi_VectorNumVectors(mvector);

		if (NULL==parallel_Multi_VectorData(mvector))
			parallel_Multi_VectorData(mvector) = (Scalar*) malloc (sizeof(Scalar)*size*num_vectors);

		/* now we create a "mask" of "active" vectors; initially all vectors are active */
		if (NULL==mvector->active_indices)
		{
			mvector->active_indices=(BlopexInt *) malloc(sizeof(BlopexInt)*num_vectors);

			for (i=0; i<num_vectors; i++)
				mvector->active_indices[i]=i;

			mvector->num_active_vectors=num_vectors;
		}


//huwei
  
		/* now we create a global_indices */
		if (NULL==mvector->global_indices)
		{

      MPI_Comm mpi_comm = mvector->comm;

      MPI_Barrier(mpi_comm);
      Int mpirank;  MPI_Comm_rank(mpi_comm, &mpirank);
      Int mpisize;  MPI_Comm_size(mpi_comm, &mpisize);
    
      BlopexInt numTotal_vectors; 

      mpi::Allreduce( &num_vectors, &numTotal_vectors, 1, MPI_SUM, mpi_comm );

      mvector->numTotal_vectors = numTotal_vectors;
      
			
      Int blocksize;
          
      if ( numTotal_vectors <=  mpisize ) {
        blocksize = 1;
        }
            
      else {
                
        if ( numTotal_vectors % mpisize == 0 ){
          blocksize = numTotal_vectors / mpisize;
        }
        else {
          blocksize = ((numTotal_vectors - 1) / mpisize) + 1;
        }
      }

      mvector->global_indices=(BlopexInt *) malloc(sizeof(BlopexInt)*num_vectors);
      
      for (Int i = 0; i < num_vectors; i++){
        if(blocksize * mpirank < numTotal_vectors){
          mvector->global_indices[i] = i + blocksize * mpirank;
        }
      }
		}

//huwei

		return ierr;
	}

/*--------------------------------------------------------------------------
 * parallel_Multi_VectorSetDataOwner                                   generic
 *--------------------------------------------------------------------------*/
BlopexInt
	parallel_Multi_VectorSetDataOwner( parallel_Multi_Vector *mvector, BlopexInt owns_data )
	{
		BlopexInt    ierr=0;

		parallel_Multi_VectorOwnsData(mvector) = owns_data;

		return ierr;
	}

/*--------------------------------------------------------------------------
 * parallel_Multi_VectorDestroy                                        generic
 *--------------------------------------------------------------------------*/
// FIXME Free global_indices
BlopexInt
	parallel_Multi_VectorDestroy( parallel_Multi_Vector *mvector )
	{
		BlopexInt    ierr=0;

		if (NULL!=mvector)
		{
			if (parallel_Multi_VectorOwnsData(mvector) && NULL!=parallel_Multi_VectorData(mvector))
				free( parallel_Multi_VectorData(mvector) );

			if (NULL!=mvector->active_indices)
				free(mvector->active_indices);

			free(mvector);
		}
		return ierr;
	}

/*--------------------------------------------------------------------------
 * parallel_Multi_VectorSetMask                                        generic
 *--------------------------------------------------------------------------*/
BlopexInt
	parallel_Multi_VectorSetMask(parallel_Multi_Vector *mvector, BlopexInt * mask)
	{
		/* this routine accepts mask in "zeros and ones format, and converts it to the one used in
			 the structure "parallel_Multi_Vector" */
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
 * parallel_Multi_VectorSetConstantValues                            scalar  
 *--------------------------------------------------------------------------*/
BlopexInt
	parallel_Multi_VectorSetConstantValues( parallel_Multi_Vector *v,
			Scalar value)
	{
		Scalar  *vector_data = (Scalar *) parallel_Multi_VectorData(v);
		BlopexInt      size        = parallel_Multi_VectorSize(v);
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
 * parallel_Multi_VectorCopy   y=x  using indices                     scalar 
 *
 * y should have already been initialized at the same size as x
 *--------------------------------------------------------------------------*/
// FIXME Add more attributes
BlopexInt
	parallel_Multi_VectorCopy( parallel_Multi_Vector *x, parallel_Multi_Vector *y)
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


		// OLD code 
		if(0){
			for (i=0; i<num_active_vectors; i++)
			{
				src  = x_data + size * x_active_ind[i];
				dest = y_data + size * y_active_ind[i];

				memcpy(dest,src,num_bytes);
			}
		}

		// NEW Code, calculate everything like BLAS. Make sure it works with deflation  
		if(1){
			blas::Copy( (size * x->num_vectors), x_data, 1, y_data, 1 );
		}



		return 0;
	}

/*--------------------------------------------------------------------------
 * parallel_Multi_VectorCopyWithoutMask              y=x               scalar
 * copies data from x to y without using indices
 * y should have already been initialized at the same size as x
 *--------------------------------------------------------------------------*/
BlopexInt
	parallel_Multi_VectorCopyWithoutMask(parallel_Multi_Vector *x , parallel_Multi_Vector *y)
	{
		BlopexInt byte_count;

		assert (x->size == y->size && x->num_vectors == y->num_vectors);

		byte_count = sizeof(Scalar) * x->size * x->num_vectors;

		/* threading not done here since it's not done (reason?) in parallel_VectorCopy
			 from vector.c */

		memcpy(y->data,x->data,byte_count);

		return 0;
	}

/*--------------------------------------------------------------------------
 * parallel_Multi_VectorAxpy      y=alpha*x+y using indices            scalar
 *
 * call seq < MultiVectorAxpy < i->MultiAxpy < mv_MultiVectorAxpy
 * alpha is always 1 or -1 double
 *--------------------------------------------------------------------------*/
// FIXME Local operation. Not change much.
BlopexInt
	parallel_Multi_VectorAxpy( double           alpha,
			parallel_Multi_Vector *x,
			parallel_Multi_Vector *y)
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
		if(0){
			for(i=0; i<num_active_vectors; i++)
			{
				src = x_data + x_active_ind[i]*size;
				dest = y_data + y_active_ind[i]*size;

				blas::Axpy( size, static_cast<Scalar>(alpha), 
						src, 1, dest, 1 );
			}
		}

		// NEW Code, calculate everything like BLAS3. Make sure it works with deflation  
		if(1){
			blas::Axpy( (size * x->num_vectors), static_cast<Scalar>(alpha), 
					x_data, 1, y_data, 1 );
		}

		return 0;
	}

/*--------------------------------------------------------------------------
 * parallel_Multi_VectorByDiag:    y=x*alpha using indices             scalar
 *
 * if y and x are mxn then alpha is nx1
 * call seq < MultiVectorByDiagonal < i->MultiVecMatDiag < mv_MultiVectorByDiagonal
 *--------------------------------------------------------------------------*/
// FIXME Change.  alpha should be multiplied to correct vectors,
// according to global_indices
BlopexInt parallel_Multi_VectorByDiag( parallel_Multi_Vector *x,
		BlopexInt           *mask,
		BlopexInt           n,
		Scalar              *alpha,
		parallel_Multi_Vector *y)
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

//#pragma omp parallel for
	for(i=0; i<num_active_vectors; i++)
	{
		src = x_data + x_active_ind[i]*size;
		dest = y_data + y_active_ind[i]*size;
		current_alpha=alpha[ al_active_ind[i] ];

//		for (j=0; j<size; j++){
//			*(dest++) = (current_alpha) * (*(src++));
//		}

		blas::Copy( size, src, 1, dest, 1 );
		blas::Scal( size, current_alpha, dest, 1 );
	}

	free(al_active_ind);
	return 0;
}

/*--------------------------------------------------------------------------
 * parallel_Multi_VectorInnerProd        v=x'*y  using indices          scalar
 *
 * call seq < MultiInnerProd < i->MultiInnerProd < mv_MultiVectorByMultiVector
 *--------------------------------------------------------------------------*/
// FIXME Major change.  call ScaLAPACK
BlopexInt parallel_Multi_VectorInnerProd( parallel_Multi_Vector *x,
		parallel_Multi_Vector *y,
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
	if(0){
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
	if(1){
		assert( x->num_vectors == x->num_active_vectors &&
				y->num_vectors == y->num_active_vectors); 
#ifndef _USE_COMPLEX_
		blas::Gemm('T', 'N', x_num_active_vectors, y_num_active_vectors,
				 size, SCALAR_ONE, x_data, size, y_data, size,
				 SCALAR_ZERO, v, gh);
#else
		blas::Gemm('C', 'N', x_num_active_vectors, y_num_active_vectors,
				 size, SCALAR_ONE, x_data, size, y_data, size,
				 SCALAR_ZERO, v, gh);
#endif
	}


	return 0;
}

/*--------------------------------------------------------------------------
 * parallel_Multi_VectorInnerProdDiag                                 scalar 
 *
 * diag=diagonal(x'*y) using indices
 * mask is index of diag
 *
 * call seq < MultiInnerProdDiag < i->MultiInnerProdDiag < mv_MultiVectorByMultiVectorDiag
 *--------------------------------------------------------------------------*/
// FIXME Not much change.  Local operation. Allreduce to obtain the
// global diag vector.
BlopexInt parallel_Multi_VectorInnerProdDiag( parallel_Multi_Vector *x,
		parallel_Multi_Vector *y,
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

//#pragma omp parallel for
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
 * parallel_Multi_VectorByMatrix      y=x*rVal using indices          scalar 
 *
 * call seq < MultiVectorByMatrix < i->MultiVecMat < mv_MultiVectorByMatrix
 *--------------------------------------------------------------------------*/
// FIXME Major change. ScaLAPACK.  
BlopexInt
parallel_Multi_VectorByMatrix(parallel_Multi_Vector *x, BlopexInt rGHeight, BlopexInt rHeight,
		BlopexInt rWidth, Scalar* rVal, parallel_Multi_Vector *y)
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
	if(0){
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
	if(1){
		assert( x->num_vectors == x->num_active_vectors &&
				y->num_vectors == y->num_active_vectors); 
		blas::Gemm( 'N', 'N', size, rWidth, rHeight, SCALAR_ONE, 
				x_data, size, rVal, rGHeight, SCALAR_ZERO, y_data,
				size );
	}


	return 0;
}

/*--------------------------------------------------------------------------
 * parallel_Multi_VectorPrint                                           scalar
 *--------------------------------------------------------------------------*/
BlopexInt parallel_Multi_VectorPrint(parallel_Multi_Vector * x,char * tag, BlopexInt limit)
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
 * parallel_Multi_VectorPrintShort                                      scalar
 *--------------------------------------------------------------------------*/
BlopexInt parallel_Multi_VectorPrintShort(parallel_Multi_Vector * x)
{
	printf("size %d\n", x->size);
	printf("owns data %d\n",x->owns_data);
	printf("num vectors %d\n",x->num_vectors);
	printf("num active vectors  %d\n",x->num_active_vectors);
	return(0);
}

} // namespace LOBPCG 
} // namespace dgdft