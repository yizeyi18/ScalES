#include "lobpcg.hpp"

//----------------------------------------------------------------
// Real version (double precision) of LOBPCG
namespace REAL{ namespace LOBPCG{
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
    CopyMultiVector                                                     double
    --------------------------------------------------------------------------*/
  void
    CopyMultiVector( void *x, void *y)
    {
      BlopexInt dummy;

      dummy = serial_Multi_VectorCopy( (serial_Multi_Vector *) x,
				       (serial_Multi_Vector *) y);
    }

  /*--------------------------------------------------------------------------
    ClearMultiVector                                                    double
    --------------------------------------------------------------------------*/
  void
    ClearMultiVector(void *x)
    {
      BlopexInt dummy;

      dummy=serial_Multi_VectorSetConstantValues((serial_Multi_Vector *)x,0.0);
    }

  /*--------------------------------------------------------------------------
    MultiVectorSetRandomValues                                         double
    --------------------------------------------------------------------------*/
  void
    SetMultiVectorRandomValues(void *x, BlopexInt seed)
    {
      BlopexInt dummy;

      dummy= serial_Multi_VectorSetRandomValues((serial_Multi_Vector *) x, seed) ;
    }


  /*--------------------------------------------------------------------------
    MultiInnerProd                                                      double
    --------------------------------------------------------------------------*/
  void
    MultiInnerProd(void * x_, void * y_,
		   BlopexInt gh, BlopexInt h, BlopexInt w, void* v )
    {
      serial_Multi_VectorInnerProd( (serial_Multi_Vector *)x_,
				    (serial_Multi_Vector *)y_,
				    gh, h, w, (double *) v);
    }


  /*--------------------------------------------------------------------------
    MultiInnerProdDiag                                                  double
    --------------------------------------------------------------------------*/
  void
    MultiInnerProdDiag( void* x_, void* y_,
			BlopexInt* mask, BlopexInt n, void* diag )
    {
      serial_Multi_VectorInnerProdDiag( (serial_Multi_Vector *)x_,
					(serial_Multi_Vector *)y_,
					mask, n, (double *) diag);
    }

  /*--------------------------------------------------------------------------
    MultiVectorByDiagonal                                               double
    --------------------------------------------------------------------------*/
  void
    MultiVectorByDiagonal( void* x,
			   BlopexInt* mask, BlopexInt n, void* diag,
			   void* y )
    {
      BlopexInt dummy;

      dummy = serial_Multi_VectorByDiag( (serial_Multi_Vector *) x, mask, n,
					 (double *) diag,
					 (serial_Multi_Vector *) y );
    }

  /*--------------------------------------------------------------------------
    MultiVectorByMatrix                                                 double
    --------------------------------------------------------------------------*/
  void
    MultiVectorByMatrix( void* x,
			 BlopexInt gh, BlopexInt h, BlopexInt w, void* v,
			 void* y )
    {
      serial_Multi_VectorByMatrix((serial_Multi_Vector *)x, gh, h,
				  w, (double *) v, (serial_Multi_Vector *)y);

    }
  /*--------------------------------------------------------------------------
    MultiAxpy                                                           double
    --------------------------------------------------------------------------*/
  void
    MultiVectorAxpy( double alpha, void   *x, void   *y)
    {
      serial_Multi_VectorAxpy(  alpha,
				(serial_Multi_Vector *) x,
				(serial_Multi_Vector *) y) ;
    }

  /*--------------------------------------------------------------------------
    MultiVectorPrint                                                   double
    --------------------------------------------------------------------------*/
  void
    MultiVectorPrint( void *x, char * tag, BlopexInt limit )
    {
      serial_Multi_VectorPrint( (serial_Multi_Vector *) x, tag, limit );
    }

  /*--------------------------------------------------------------------------
    SerialSetupInterpreter                                              double
    --------------------------------------------------------------------------*/
  BlopexInt
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

      /* Multivector part */

      i->CreateMultiVector = NULL;
      i->CopyCreateMultiVector = CreateCopyMultiVector;
      i->DestroyMultiVector = DestroyMultiVector;

      i->Width = MultiVectorWidth;
      i->Height = NULL;
      i->SetMask = MultiSetMask;
      i->CopyMultiVector = CopyMultiVector;
      i->ClearMultiVector = ClearMultiVector;
      i->SetRandomVectors = SetMultiVectorRandomValues;
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
   * serial_Multi_VectorInitialize                                      double
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorInitialize( serial_Multi_Vector *mvector )
    {
      BlopexInt    ierr = 0, i, size, num_vectors;

      size        = serial_Multi_VectorSize(mvector);
      num_vectors = serial_Multi_VectorNumVectors(mvector);

      if (NULL==serial_Multi_VectorData(mvector))
	serial_Multi_VectorData(mvector) =
	  (double *) malloc (sizeof(double)*size*num_vectors);

      /* now we create a "mask" of "active" vectors; initially all vectors are active */
      if (NULL==mvector->active_indices)
      {
	mvector->active_indices=(BlopexInt*)malloc(sizeof(BlopexInt)*num_vectors);

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
	mvector->active_indices=(BlopexInt*)malloc(sizeof(BlopexInt)*num_vectors);

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
   * serial_Multi_VectorSetConstantValues                               double
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorSetConstantValues( serial_Multi_Vector *v,
					  double             value)
    {
      double  *vector_data = serial_Multi_VectorData(v);
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
   * serial_Multi_VectorSetRandomValues                                double
   *
   *     returns vector of values randomly distributed between -1.0 and +1.0
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorSetRandomValues( serial_Multi_Vector *v, BlopexInt seed)
    {
      double  *vector_data = serial_Multi_VectorData(v);
      BlopexInt      size        = serial_Multi_VectorSize(v);
      BlopexInt      i, j, start_offset, end_offset;

      //   FIXME
      //   srand48(seed);
      srand48(1);

      for (i = 0; i < v->num_active_vectors; i++)
      {
	start_offset = v->active_indices[i]*size;
	end_offset = start_offset+size;

	for (j=start_offset; j < end_offset; j++)
	  vector_data[j]= 2.0 * drand48() - 1.0;
      }

      return 0;
    }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorCopy                                           double
   * copies data from x to y
   * y should have already been initialized at the same size as x
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorCopy( serial_Multi_Vector *x, serial_Multi_Vector *y)
    {
      double  *x_data;
      double  *y_data;
      BlopexInt i;
      BlopexInt size;
      BlopexInt num_bytes;
      BlopexInt num_active_vectors;
      double * dest;
      double * src;
      BlopexInt * x_active_ind;
      BlopexInt * y_active_ind;

      assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);

      num_active_vectors = x->num_active_vectors;
      size = x->size;
      num_bytes = size*sizeof(double);
      x_data = x->data;
      y_data = y->data;
      x_active_ind=x->active_indices;
      y_active_ind=y->active_indices;

      for (i=0; i<num_active_vectors; i++)
      {
	src=x_data + size * x_active_ind[i];
	dest = y_data + size * y_active_ind[i];

	memcpy(dest,src,num_bytes);
      }

      return 0;
    }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorCopyWithoutMask                                 double
   * copies data from x to y without using indices
   * y should have already been initialized at the same size as x
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorCopyWithoutMask(serial_Multi_Vector *x , serial_Multi_Vector *y)
    {
      BlopexInt byte_count;

      assert (x->size == y->size && x->num_vectors == y->num_vectors);

      byte_count = sizeof(double) * x->size * x->num_vectors;

      /* threading not done here since it's not done (reason?) in serial_VectorCopy
	 from vector.c */

      memcpy(y->data,x->data,byte_count);

      return 0;
    }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorAxpy                                           double
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorAxpy( double            alpha,
			     serial_Multi_Vector *x,
			     serial_Multi_Vector *y)
    {
      double  *x_data;
      double  *y_data;
      double * src;
      double * dest;
      BlopexInt * x_active_ind;
      BlopexInt * y_active_ind;
      BlopexInt i, j;
      BlopexInt size;
      BlopexInt num_active_vectors;

      assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);

      x_data = x->data;
      y_data = y->data;
      size = x->size;
      num_active_vectors = x->num_active_vectors;
      x_active_ind = x->active_indices;
      y_active_ind = y->active_indices;

      /* OLD Code */
      if(1){
	for(i=0; i<num_active_vectors; i++)
	{
	  src = x_data + x_active_ind[i]*size;
	  dest = y_data + y_active_ind[i]*size;

	  for (j=0; j<size; j++)
	    dest[j] += alpha*src[j];
	}
      }

      /* NEW Code, calculate everything  */
      if(0){
	int totalsize = size * x->num_vectors;
	int I_ONE = 1;
	daxpy_(&totalsize, &alpha, x_data, &I_ONE, y_data, &I_ONE);
      }

      return 0;
    }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorByDiag:                                         double
   " y(<y_mask>) = alpha(<mask>) .* x(<x_mask>) "
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorByDiag( serial_Multi_Vector *x,
			       BlopexInt                *mask,
			       BlopexInt                n,
			       double             *alpha,
			       serial_Multi_Vector *y)
    {
      double  *x_data;
      double  *y_data;
      BlopexInt      size;
      BlopexInt      num_active_vectors;
      BlopexInt      i,j;
      double  *dest;
      double  *src;
      BlopexInt * x_active_ind;
      BlopexInt * y_active_ind;
      BlopexInt * al_active_ind;
      BlopexInt num_active_als;
      double current_alpha;

      assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);

      /* build list of active indices in alpha */

      al_active_ind = (BlopexInt*)malloc(sizeof(BlopexInt)*n);
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

      x_data = x->data;
      y_data = y->data;
      size = x->size;
      num_active_vectors = x->num_active_vectors;
      x_active_ind = x->active_indices;
      y_active_ind = y->active_indices;

      for(i=0; i<num_active_vectors; i++)
      {
	src = x_data + x_active_ind[i]*size;
	dest = y_data + y_active_ind[i]*size;
	current_alpha=alpha[ al_active_ind[i] ];

	for (j=0; j<size; j++)
	  dest[j] = current_alpha*src[j];
      }

      free(al_active_ind);
      return 0;
    }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorInnerProd                                      double
   *--------------------------------------------------------------------------*/
  BlopexInt 
    serial_Multi_VectorInnerProd( serial_Multi_Vector *x,
				  serial_Multi_Vector *y,
				  BlopexInt gh, BlopexInt h, BlopexInt w, double* v)
    {
      /* to be reworked! */
      double  *x_data;
      double  *y_data;
      BlopexInt      size;
      BlopexInt      x_num_active_vectors;
      BlopexInt      y_num_active_vectors;
      BlopexInt      i,j,k;
      double  *y_ptr;
      double  *x_ptr;
      BlopexInt * x_active_ind;
      BlopexInt * y_active_ind;
      double current_product;
      BlopexInt gap;

      assert (x->size==y->size);

      x_data = x->data;
      y_data = y->data;
      size = x->size;
      x_num_active_vectors = x->num_active_vectors;
      y_num_active_vectors = y->num_active_vectors;

      //   assert (x_num_active_vectors ==  y_num_active_vectors); 
      if(!(x_num_active_vectors==h && y_num_active_vectors==w))
	fprintf(stderr, "active:\n %d %d %d %d\n", x_num_active_vectors, h, y_num_active_vectors, w);

      x_active_ind = x->active_indices;
      y_active_ind = y->active_indices;

      gap = gh-h;

      int I_ONE = 1;

      // LLIN: Old code using BLAS1, allows deflation
      if(0){ 
	for(j=0; j<y_num_active_vectors; j++)
	{
	  y_ptr = y_data + y_active_ind[j]*size;

	  for (i=0; i<x_num_active_vectors; i++)
	  {
	    x_ptr = x_data + x_active_ind[i]*size;

	    current_product = ddot_(&size, x_ptr, &I_ONE, y_ptr, &I_ONE); 
	    //	 current_product = cblas_ddot(size, x_ptr, I_ONE, y_ptr, I_ONE); 

	    //         current_product = 0.0;
	    //         for(k=0; k<size; k++)
	    //            current_product += x_ptr[k]*y_ptr[k];

	    /* fortran column-wise storage for results */
	    *v++ = current_product;
	  }
	  v+=gap;
	}
      }

      // LLIN: New code using BLAS3 does not allow deflation

      if(0){
	double *xmat, *ymat, *xmat_ptr, *ymat_ptr, *vmat, *vmat_ptr;
	xmat = (double*)malloc(sizeof(double)*size*x_num_active_vectors);
	ymat = (double*)malloc(sizeof(double)*size*y_num_active_vectors);
	vmat = (double*)malloc(sizeof(double)*y_num_active_vectors*x_num_active_vectors);
	for(j=0; j<y_num_active_vectors*x_num_active_vectors;j++)
	  vmat[j] = 0.0;
	for(j=0; j<y_num_active_vectors; j++)
	{
	  y_ptr = y_data + y_active_ind[j]*size;
	  ymat_ptr = ymat + j*size;
	  dcopy_(&size, y_ptr, &I_ONE, ymat_ptr, &I_ONE);
	}
	for(j=0; j<x_num_active_vectors; j++)
	{
	  x_ptr = x_data + x_active_ind[j]*size;
	  xmat_ptr = xmat + j*size;
	  dcopy_(&size, x_ptr, &I_ONE, xmat_ptr, &I_ONE);
	}
	char Ytrans[] = "T";
	char Ntrans[] = "N";
	double D_ONE = 1.0;
	double D_ZERO = 0.0;
	dgemm_(Ytrans, Ntrans, &x_num_active_vectors, 
	       &y_num_active_vectors, &size, &D_ONE, xmat, &size,
	       ymat, &size, &D_ZERO, vmat, &x_num_active_vectors);
	for(j=0; j<y_num_active_vectors; j++){
	  vmat_ptr = vmat + j*x_num_active_vectors;
	  dcopy_(&x_num_active_vectors, vmat_ptr, &I_ONE, v, &I_ONE);
	  v += gh;
	}

	free(xmat);
	free(ymat);
	free(vmat);
	printf("gh = %6d, x->num_vectors = %6d, x->num_active_vectors = %6d\n", 
	       gh, x->num_vectors, x->num_active_vectors);
      }


      // LLIN: New version This assumes that deflation is not used and uses BLAS3
      if(1){
	assert( x->num_vectors == x->num_active_vectors &&
		y->num_vectors == y->num_active_vectors); 
	char Ytrans = 'T';
	char Ntrans = 'N';
	double D_ONE = 1.0;
	double D_ZERO = 0.0;
	dgemm_(&Ytrans, &Ntrans, &x_num_active_vectors, 
	       &y_num_active_vectors, &size, &D_ONE, x_data, &size,
	       y_data, &size, &D_ZERO, v, &gh);
      }


      return 0;
    }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorInnerProdDiag                                   double
   *--------------------------------------------------------------------------*/
  BlopexInt 
    serial_Multi_VectorInnerProdDiag( serial_Multi_Vector *x,
				      serial_Multi_Vector *y,
				      BlopexInt* mask, BlopexInt n, double* diag)
    {
      /* to be reworked! */
      double   *x_data;
      double   *y_data;
      BlopexInt      size;
      BlopexInt      num_active_vectors;
      BlopexInt      * x_active_ind;
      BlopexInt      * y_active_ind;
      double   *y_ptr;
      double   *x_ptr;
      double   current_product;
      BlopexInt      i, k;
      BlopexInt      * al_active_ind;
      BlopexInt      num_active_als;

      assert(x->size==y->size && x->num_active_vectors == y->num_active_vectors);

      /* build list of active indices in alpha */

      al_active_ind = (BlopexInt*)malloc(sizeof(BlopexInt)*n);
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

      x_data = x->data;
      y_data = y->data;
      size = x->size;
      num_active_vectors = x->num_active_vectors;
      x_active_ind = x->active_indices;
      y_active_ind = y->active_indices;

      for (i=0; i<num_active_vectors; i++)
      {
	x_ptr = x_data + x_active_ind[i]*size;
	y_ptr = y_data + y_active_ind[i]*size;
	current_product = 0.0;

	for(k=0; k<size; k++)
	  current_product += x_ptr[k]*y_ptr[k];

	diag[al_active_ind[i]] = current_product;
      }

      free(al_active_ind);
      return 0;
    }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorByMatrix                                        double
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorByMatrix(serial_Multi_Vector *x, 
				BlopexInt rGHeight, BlopexInt rHeight,
				BlopexInt rWidth, double* rVal, 
				serial_Multi_Vector *y)
    {
      double   *x_data;
      double   *y_data;
      BlopexInt      size;
      BlopexInt      * x_active_ind;
      BlopexInt      * y_active_ind;
      double   *y_ptr;
      double   *x_ptr;
      double   current_coef;
      BlopexInt      i,j,k;
      BlopexInt      gap;

      assert(rHeight>0);
      assert (rHeight==x->num_active_vectors && rWidth==y->num_active_vectors);

      x_data = x->data;
      y_data = y->data;
      size = x->size;
      x_active_ind = x->active_indices;
      y_active_ind = y->active_indices;
      gap = rGHeight - rHeight;

      int I_ONE = 1;

      //LLIN: Tuning between BLAS1 and BLAS3. With deflation
      if(0){
	for (j=0; j<rWidth; j++)
	{
	  y_ptr = y_data + y_active_ind[j]*size;
	  for(k = 0; k < size; k++)
	    y_ptr[k] = 0.0;


	  /* NEW code using BLAS 1 */

	  if(1){
	    for (i=0; i<rHeight; i++){
	      current_coef = *rVal++;
	      x_ptr = x_data + x_active_ind[i]*size;
	      daxpy_(&size, &current_coef, x_ptr, &I_ONE, y_ptr, &I_ONE);
	      //	   cblas_daxpy(size, current_coef, x_ptr, I_ONE, y_ptr, I_ONE);
	    }	
	    rVal += gap;
	  }
	  /* OLD code */

	  /* ------ set current "y" to first member in a sum ------ */
	  if(0){
	    x_ptr = x_data + x_active_ind[0]*size;
	    current_coef = *rVal++;

	    for (k=0; k<size; k++)
	      y_ptr[k] = current_coef * x_ptr[k];

	    /* ------ now add all other members of a sum to "y" ----- */
	    for (i=1; i<rHeight; i++)
	    {
	      x_ptr = x_data + x_active_ind[i]*size;
	      current_coef = *rVal++;

	      for (k=0; k<size; k++)
		y_ptr[k] += current_coef * x_ptr[k];
	    }
	    rVal += gap;
	  }

	}
      }


      /* LLIN: NEW code using BLAS 3. Without deflation*/

      if(0){
	double *xmat, *ymat, *xmat_ptr, *ymat_ptr;
	xmat = (double*)malloc(sizeof(double)*size*rHeight);
	ymat = (double*)malloc(sizeof(double)*size*rWidth);
	for(j=0; j<rHeight; j++)
	{
	  x_ptr = x_data + x_active_ind[j]*size;
	  xmat_ptr = xmat + j*size;
	  dcopy_(&size, x_ptr, &I_ONE, xmat_ptr, &I_ONE);
	}
	char Ntrans[] = "N";
	double D_ONE = 1.0;
	double D_ZERO = 0.0;
	dgemm_(Ntrans, Ntrans, &size, &rWidth, 
	       &rHeight, &D_ONE, xmat, &size,
	       rVal, &rGHeight, &D_ZERO, ymat, &size);
	for(j=0; j<rWidth; j++)
	{
	  y_ptr = y_data + y_active_ind[j]*size;
	  ymat_ptr = ymat + j*size;
	  dcopy_(&size, ymat_ptr, &I_ONE, y_ptr, &I_ONE);
	}
	free(xmat);
	free(ymat);
      }


      // LLIN: This assumes that deflation is not used and uses BLAS3
      if(1){
	assert( x->num_vectors == x->num_active_vectors &&
		y->num_vectors == y->num_active_vectors); 
	char Ntrans[] = "N";
	double D_ONE = 1.0;
	double D_ZERO = 0.0;
	dgemm_(Ntrans, Ntrans, &size, &rWidth, 
	       &rHeight, &D_ONE, x_data, &size,
	       rVal, &rGHeight, &D_ZERO, y_data, &size);
      }


      return 0;
    }
  /*--------------------------------------------------------------------------
   * serial_Multi_VectorByMulti_Vector     z=x*y   with indices        double
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorByMulti_Vector(serial_Multi_Vector *x,
				      serial_Multi_Vector *y,
				      serial_Multi_Vector *z)
    {
      double *x_data;
      double *y_data;
      double *z_data;

      BlopexInt    * x_index;
      BlopexInt    * y_index;
      BlopexInt    * z_index;

      double * pzc;
      double * pyc;
      double * pxr;
      double * py;

      BlopexInt      i,j,k;

      assert (x->num_active_vectors == y->size);
      assert (z->size == x->size);
      assert (z->num_active_vectors == y->num_active_vectors);

      x_data = (double *) x->data;
      y_data = (double *) y->data;
      z_data = (double *) z->data;

      x_index = x->active_indices;
      y_index = y->active_indices;
      z_index = z->active_indices;

      for (j=0; j<y->num_active_vectors; j++) {
	pzc = z_data + z->size*z_index[j];
	pyc = y_data + y->size*y_index[j];
	for (i=0;i<x->size;i++) {
	  pxr = x_data + i + x->size*x_index[0];
	  *pzc = 0.0;
	  for (k=0, py=pyc; k<y->size; k++) {
	    *pzc += *pxr * *py ;
	    pxr += x->size;
	    py++;
	  }
	  pzc++;
	}
      }

      return 0;
    }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorPrint                                          double
   *--------------------------------------------------------------------------*/
  BlopexInt 
    serial_Multi_VectorPrint(serial_Multi_Vector * x,
			     char * tag, BlopexInt limit)
    {
      double * p;
      BlopexInt     * pact;
      BlopexInt       i, j;
      BlopexInt     rows,cols;
      printf("======= %s =========\n",tag);
      printf("size %d\n", x->size);
      printf("owns data %d\n",x->owns_data);
      printf("num vectors %d\n",x->num_vectors);
      printf("num active vectors  %d\n",x->num_active_vectors);

      rows = x->size;
      cols = x->num_vectors;
      if (limit != 0) {
	if (rows > limit) rows = limit;
	if (cols > limit) cols = limit;
      }

      pact=x->active_indices;
      for (i=0; i<cols; i++, pact++)
	printf("index %d active %d\n", i, *pact);

      for (i=0; i<cols; i++)
      {  p=(double *)x->data;
	p = &p[x->active_indices[i]*x->size];
	for (j = 0; j < rows; j++,p++)
	  printf("%d %d  %22.16e  \n",j,i,*p);
      }

      return 0;
    }

} }


//----------------------------------------------------------------
// Complex version (double precision) of LOBPCG
namespace COMPLEX{ namespace LOBPCG{
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
    CopyMultiVector                                                    complex
    --------------------------------------------------------------------------*/
  void
    CopyMultiVector( void *x, void *y)
    {
      BlopexInt dummy;

      dummy = serial_Multi_VectorCopy( (serial_Multi_Vector *) x,
				       (serial_Multi_Vector *) y);
    }

  /*--------------------------------------------------------------------------
    ClearMultiVector                                                   complex
    --------------------------------------------------------------------------*/
  void
    ClearMultiVector(void *x)
    {
      komplex kzero = {0.0,0.0};
      BlopexInt dummy;

      dummy=serial_Multi_VectorSetConstantValues((serial_Multi_Vector *)x,kzero);
    }

  /*--------------------------------------------------------------------------
    MultiVectorSetRandomValues                                         complex
    --------------------------------------------------------------------------*/
  void
    SetMultiVectorRandomValues(void *x, BlopexInt seed)
    {
      BlopexInt dummy;

      dummy= serial_Multi_VectorSetRandomValues((serial_Multi_Vector *) x, seed) ;
    }


  /*--------------------------------------------------------------------------
    MultiInnerProd                                                      complex
    --------------------------------------------------------------------------*/
  void
    MultiInnerProd(void * x_, void * y_,
		   BlopexInt gh, BlopexInt h, BlopexInt w, void* v )
    {
      serial_Multi_VectorInnerProd( (serial_Multi_Vector *)x_,
				    (serial_Multi_Vector *)y_,
				    gh, h, w, (komplex *) v);
    }


  /*--------------------------------------------------------------------------
    MultiInnerProdDiag                                                  complex
    --------------------------------------------------------------------------*/
  void
    MultiInnerProdDiag( void* x_, void* y_,
			BlopexInt* mask, BlopexInt n, void* diag )
    {
      serial_Multi_VectorInnerProdDiag( (serial_Multi_Vector *)x_,
					(serial_Multi_Vector *)y_,
					mask, n, (komplex *) diag);
    }

  /*--------------------------------------------------------------------------
    MultiVectorByDiagonal                                              complex
    --------------------------------------------------------------------------*/
  void
    MultiVectorByDiagonal( void* x,
			   BlopexInt* mask, BlopexInt n, void* diag,
			   void* y )
    {
      BlopexInt dummy;

      dummy = serial_Multi_VectorByDiag( (serial_Multi_Vector *) x, mask, n,
					 (komplex *) diag,
					 (serial_Multi_Vector *) y );
    }

  /*--------------------------------------------------------------------------
    MultiVectorByMatrix                                                complex
    --------------------------------------------------------------------------*/
  void
    MultiVectorByMatrix( void* x,
			 BlopexInt gh, BlopexInt h, BlopexInt w, void* v,
			 void* y )
    {
      serial_Multi_VectorByMatrix((serial_Multi_Vector *)x, gh, h,
				  w, (komplex *) v, (serial_Multi_Vector *)y);

    }
  /*--------------------------------------------------------------------------
    MultiAxpy                                                          complex
    --------------------------------------------------------------------------*/
  void
    MultiVectorAxpy( double alpha, void   *x, void   *y)
    {
      serial_Multi_VectorAxpy(  alpha,
				(serial_Multi_Vector *) x,
				(serial_Multi_Vector *) y) ;
    }
  /*--------------------------------------------------------------------------
    MultiVectorPrint                                                   complex
    --------------------------------------------------------------------------*/
  void
    MultiVectorPrint( void *x, char * tag, BlopexInt limit )
    {
      serial_Multi_VectorPrint( (serial_Multi_Vector *) x, tag, limit );
    }

  BlopexInt
    /*--------------------------------------------------------------------------
      SerialSetupInterpreter                                             complex
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

      /* Multivector part */

      i->CreateMultiVector = NULL;
      i->CopyCreateMultiVector = CreateCopyMultiVector;
      i->DestroyMultiVector = DestroyMultiVector;

      i->Width = MultiVectorWidth;
      i->Height = NULL;
      i->SetMask = MultiSetMask;
      i->CopyMultiVector = CopyMultiVector;
      i->ClearMultiVector = ClearMultiVector;
      i->SetRandomVectors = SetMultiVectorRandomValues;
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
   * serial_Multi_VectorInitialize                                    complex
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorInitialize( serial_Multi_Vector *mvector )
    {
      BlopexInt    ierr = 0, i, size, num_vectors;

      size        = serial_Multi_VectorSize(mvector);
      num_vectors = serial_Multi_VectorNumVectors(mvector);

      if (NULL==serial_Multi_VectorData(mvector))
	serial_Multi_VectorData(mvector) = (double*) malloc (sizeof(komplex)*size*num_vectors);

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
   * serial_Multi_VectorSetConstantValues                             complex
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorSetConstantValues( serial_Multi_Vector *v,
					  komplex value)
    {
      komplex  *vector_data = (komplex *) serial_Multi_VectorData(v);
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
   * serial_Multi_VectorSetRandomValues                               complex
   *
   *     returns vector of values randomly distributed between -1.0 and +1.0
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorSetRandomValues( serial_Multi_Vector *v, BlopexInt seed)
    {
      komplex  *vector_data = (komplex *) serial_Multi_VectorData(v);
      BlopexInt      size        = serial_Multi_VectorSize(v);
      BlopexInt      i, j, start_offset, end_offset;

      srand48(seed);
      printf("random size %d\n",size);
      printf("random active vectors %d\n",v->num_active_vectors);
      for (i = 0; i < v->num_active_vectors; i++)
      {
	start_offset = v->active_indices[i]*size;
	end_offset = start_offset+size;

	for (j=start_offset; j < end_offset; j++) {
	  vector_data[j].real= 2.0 * drand48() - 1.0;
	  vector_data[j].imag= 2.0 * drand48() - 1.0;;
	}
      }

      return 0;
    }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorCopy   y=x  using indices                     complex
   *
   * y should have already been initialized at the same size as x
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorCopy( serial_Multi_Vector *x, serial_Multi_Vector *y)
    {
      komplex *x_data;
      komplex *y_data;
      BlopexInt i;
      BlopexInt size;
      BlopexInt num_bytes;
      BlopexInt num_active_vectors;
      komplex * dest;
      komplex * src;
      BlopexInt * x_active_ind;
      BlopexInt * y_active_ind;

      assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);

      num_active_vectors = x->num_active_vectors;
      size = x->size;
      num_bytes = size*sizeof(komplex);
      x_data = (komplex *) x->data;
      y_data = (komplex *) y->data;
      x_active_ind=x->active_indices;
      y_active_ind=y->active_indices;

      for (i=0; i<num_active_vectors; i++)
      {
	src=x_data + size * x_active_ind[i];
	dest = y_data + size * y_active_ind[i];

	memcpy(dest,src,num_bytes);
      }

      return 0;
    }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorCopyWithoutMask              y=x              complex
   * copies data from x to y without using indices
   * y should have already been initialized at the same size as x
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorCopyWithoutMask(serial_Multi_Vector *x , serial_Multi_Vector *y)
    {
      BlopexInt byte_count;

      assert (x->size == y->size && x->num_vectors == y->num_vectors);

      byte_count = sizeof(komplex) * x->size * x->num_vectors;

      /* threading not done here since it's not done (reason?) in serial_VectorCopy
	 from vector.c */

      memcpy(y->data,x->data,byte_count);

      return 0;
    }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorAxpy      y=alpha*x+y using indices           complex
   *
   * call seq < MultiVectorAxpy < i->MultiAxpy < mv_MultiVectorAxpy
   * alpha is always 1 or -1 double
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorAxpy( double           alpha,
			     serial_Multi_Vector *x,
			     serial_Multi_Vector *y)
    {
      komplex  *x_data;
      komplex  *y_data;
      komplex * src;
      komplex * dest;
      BlopexInt * x_active_ind;
      BlopexInt * y_active_ind;
      BlopexInt i, j;
      BlopexInt size;
      BlopexInt num_active_vectors;

      assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);

      x_data = (komplex *) x->data;
      y_data = (komplex *) y->data;
      size = x->size;
      num_active_vectors = x->num_active_vectors;
      x_active_ind = x->active_indices;
      y_active_ind = y->active_indices;

      for(i=0; i<num_active_vectors; i++)
      {
	src = x_data + x_active_ind[i]*size;
	dest = y_data + y_active_ind[i]*size;

	for (j=0; j<size; j++) {
	  dest[j].real += alpha*src[j].real;
	  dest[j].imag += alpha*src[j].imag;
	}
      }

      return 0;
    }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorByDiag:    y=x*alpha using indices            complex
   *
   * if y and x are mxn then alpha is nx1
   * call seq < MultiVectorByDiagonal < i->MultiVecMatDiag < mv_MultiVectorByDiagonal
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorByDiag( serial_Multi_Vector *x,
			       BlopexInt                 *mask,
			       BlopexInt                 n,
			       komplex             *alpha,
			       serial_Multi_Vector *y)
    {
      komplex  *x_data;
      komplex  *y_data;
      BlopexInt      size;
      BlopexInt      num_active_vectors;
      BlopexInt      i,j;
      komplex  *dest;
      komplex  *src;
      BlopexInt * x_active_ind;
      BlopexInt * y_active_ind;
      BlopexInt * al_active_ind;
      BlopexInt num_active_als;
      komplex * current_alpha;

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

      x_data = (komplex *) x->data;
      y_data = (komplex *) y->data;
      size = x->size;
      num_active_vectors = x->num_active_vectors;
      x_active_ind = x->active_indices;
      y_active_ind = y->active_indices;

      for(i=0; i<num_active_vectors; i++)
      {
	src = x_data + x_active_ind[i]*size;
	dest = y_data + y_active_ind[i]*size;
	current_alpha=&alpha[ al_active_ind[i] ];

	for (j=0; j<size; j++)
	  complex_multiply(&dest[j],current_alpha,&src[j]);
      }

      free(al_active_ind);
      return 0;
    }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorInnerProd        v=x'*y  using indices        complex
   *
   * call seq < MultiInnerProd < i->MultiInnerProd < mv_MultiVectorByMultiVector
   *--------------------------------------------------------------------------*/
  BlopexInt serial_Multi_VectorInnerProd( serial_Multi_Vector *x,
					  serial_Multi_Vector *y,
					  BlopexInt gh, BlopexInt h, BlopexInt w, komplex* v)
  {
    /* to be reworked! */
    komplex *x_data;
    komplex *y_data;
    BlopexInt      size;
    BlopexInt      x_num_active_vectors;
    BlopexInt      y_num_active_vectors;
    BlopexInt      i,j,k;
    komplex *y_ptr;
    komplex *x_ptr;
    BlopexInt * x_active_ind;
    BlopexInt * y_active_ind;
    komplex current_product;
    komplex temp;
    komplex conj;
    BlopexInt gap;

    assert (x->size==y->size);

    x_data = (komplex *) x->data;
    y_data = (komplex *) y->data;
    size = x->size;
    x_num_active_vectors = x->num_active_vectors;
    y_num_active_vectors = y->num_active_vectors;

    assert (x_num_active_vectors==h && y_num_active_vectors==w);

    x_active_ind = x->active_indices;
    y_active_ind = y->active_indices;

    gap = gh-h;

    for(j=0; j<y_num_active_vectors; j++)
    {
      y_ptr = y_data + y_active_ind[j]*size;

      for (i=0; i<x_num_active_vectors; i++) {

	x_ptr = x_data + x_active_ind[i]*size;
	current_product.real = 0.0;
	current_product.imag = 0.0;

	for(k=0; k<size; k++) {
	  conj.real=x_ptr[k].real;
	  conj.imag=-x_ptr[k].imag;
	  complex_multiply(&temp,&conj,&y_ptr[k]);
	  complex_add(&current_product,&current_product,&temp);
	}

	/* fortran column-wise storage for results */
	*v++ = current_product;
      }
      v+=gap;
    }

    return 0;
  }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorInnerProdDiag                                 complex
   *
   * diag=diagonal(x'*y) using indices
   * mask is index of diag
   *
   * call seq < MultiInnerProdDiag < i->MultiInnerProdDiag < mv_MultiVectorByMultiVectorDiag
   *--------------------------------------------------------------------------*/
  BlopexInt serial_Multi_VectorInnerProdDiag( serial_Multi_Vector *x,
					      serial_Multi_Vector *y,
					      BlopexInt* mask, BlopexInt n, komplex* diag)
  {
    /* to be reworked! */
    komplex  *x_data;
    komplex  *y_data;
    BlopexInt      size;
    BlopexInt      num_active_vectors;
    BlopexInt      * x_active_ind;
    BlopexInt      * y_active_ind;
    komplex  *y_ptr;
    komplex  *x_ptr;
    komplex  current_product;
    komplex  temp;
    komplex  conj;
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

    x_data = (komplex *) x->data;
    y_data = (komplex *) y->data;
    size = x->size;
    num_active_vectors = x->num_active_vectors;
    x_active_ind = x->active_indices;
    y_active_ind = y->active_indices;

    for (i=0; i<num_active_vectors; i++)
    {
      x_ptr = x_data + x_active_ind[i]*size;
      y_ptr = y_data + y_active_ind[i]*size;
      current_product.real = 0.0;
      current_product.imag = 0.0;

      for(k=0; k<size; k++) {
	conj.real = x_ptr[k].real;
	conj.imag = -x_ptr[k].imag;
	complex_multiply(&temp,&conj,&y_ptr[k]);
	complex_add(&current_product,&current_product,&temp);
      }
      diag[al_active_ind[i]] = current_product;
    }

    free(al_active_ind);
    return 0;
  }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorByMatrix      y=x*rVal using indices          complex
   *
   * call seq < MultiVectorByMatrix < i->MultiVecMat < mv_MultiVectorByMatrix
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorByMatrix(serial_Multi_Vector *x, BlopexInt rGHeight, BlopexInt rHeight,
				BlopexInt rWidth, komplex* rVal, serial_Multi_Vector *y)
    {
      komplex *x_data;
      komplex *y_data;
      BlopexInt      size;
      BlopexInt     * x_active_ind;
      BlopexInt     * y_active_ind;
      komplex *y_ptr;
      komplex *x_ptr;
      komplex  current_coef;
      komplex  temp;
      BlopexInt      i,j,k;
      BlopexInt      gap;

      assert(rHeight>0);
      assert (rHeight==x->num_active_vectors && rWidth==y->num_active_vectors);

      x_data = (komplex *) x->data;
      y_data = (komplex *) y->data;
      size = x->size;
      x_active_ind = x->active_indices;
      y_active_ind = y->active_indices;
      gap = rGHeight - rHeight;

      for (j=0; j<rWidth; j++)
      {
	y_ptr = y_data + y_active_ind[j]*size;

	/* ------ set current "y" to first member in a sum ------ */
	x_ptr = x_data + x_active_ind[0]*size;
	current_coef = *rVal++;

	for (k=0; k<size; k++)
	  complex_multiply(&y_ptr[k],&current_coef,&x_ptr[k]);

	/* ------ now add all other members of a sum to "y" ----- */
	for (i=1; i<rHeight; i++)
	{
	  x_ptr = x_data + x_active_ind[i]*size;
	  current_coef = *rVal++;

	  for (k=0; k<size; k++) {
	    complex_multiply(&temp,&current_coef,&x_ptr[k]);
	    complex_add(&y_ptr[k],&y_ptr[k],&temp);
	  }
	}

	rVal += gap;
      }

      return 0;
    }
  /*--------------------------------------------------------------------------
   * serial_Multi_VectorByMulti_Vector     z=x*y   with indices        complex
   *--------------------------------------------------------------------------*/
  BlopexInt
    serial_Multi_VectorByMulti_Vector(serial_Multi_Vector *x,
				      serial_Multi_Vector *y,
				      serial_Multi_Vector *z)
    {
      komplex *x_data;
      komplex *y_data;
      komplex *z_data;

      BlopexInt     * x_index;
      BlopexInt     * y_index;
      BlopexInt     * z_index;
      komplex * pzc;
      komplex * pyc;
      komplex * pxr;
      komplex * py;

      komplex  temp;
      BlopexInt      i,j,k;

      assert (x->num_active_vectors == y->size);
      assert (z->size == x->size);
      assert (z->num_active_vectors == y->num_active_vectors);

      x_data = (komplex *) x->data;
      y_data = (komplex *) y->data;
      z_data = (komplex *) z->data;

      x_index = x->active_indices;
      y_index = y->active_indices;
      z_index = z->active_indices;

      for (j=0; j<y->num_active_vectors; j++) {
	pzc = z_data + z->size*z_index[j];
	pyc = y_data + y->size*y_index[j];
	for (i=0;i<x->size;i++) {
	  pxr = x_data + i + x->size*x_index[0];
	  pzc->real = 0.0;
	  pzc->imag = 0.0;
	  for (k=0, py=pyc; k<y->size; k++) {
	    complex_multiply(&temp, pxr, py);
	    complex_add(pzc, pzc, &temp);
	    pxr += x->size;
	    py++;
	  }
	  pzc++;
	}
      }

      return 0;
    }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorPrint                                          complex
   *--------------------------------------------------------------------------*/
  BlopexInt serial_Multi_VectorPrint(serial_Multi_Vector * x,char * tag, BlopexInt limit)
  {
    komplex * p;
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

    for (i=0; i<cols; i++)
    {  p=(komplex *)x->data;
      p = &p[x->active_indices[i]*x->size];
      for (j = 0; j < rows; j++,p++)
	printf("%d %d  %22.16e  %22.16e \n",j,i,p->real,p->imag);
    }

    return 0;
  }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorPrintShort                                     complex
   *--------------------------------------------------------------------------*/
  BlopexInt serial_Multi_VectorPrintShort(serial_Multi_Vector * x)
  {
    printf("size %d\n", x->size);
    printf("owns data %d\n",x->owns_data);
    printf("num vectors %d\n",x->num_vectors);
    printf("num active vectors  %d\n",x->num_active_vectors);
    return(0);
  }

  /*--------------------------------------------------------------------------
   * serial_Multi_VectorLoad                                          complex
   *--------------------------------------------------------------------------*/
  serial_Multi_Vector *
    serial_Multi_VectorLoad( char fileName[] ) {

      BlopexInt size, num_vectors, num_elements, j;
      serial_Multi_Vector *mvector;

      FILE* fp;
      komplex* p;
      float freal, fimag;

      if ( (fp=fopen(fileName,"r") )==NULL) {
	printf("file open failed\n");
	return NULL;
      }
      fscanf(fp,"%d %d",&size, &num_vectors);
      num_elements = size*num_vectors;
      mvector = serial_Multi_VectorCreate(size, num_vectors);
      serial_Multi_VectorInitialize( mvector);

      p = (komplex *)mvector->data;
      j = 0;
      while ( fscanf(fp,"%e %e",&freal,&fimag) != EOF && j<num_elements ) {
	p->real = freal;
	p->imag = fimag;
	j++;
	p++;
      }
      fclose(fp);
      return mvector;
    }
  /* ------------------------------------------------------------
     serial_Multi_VectorSymmetrize                        complex
     mtx=(mtx+transpose(mtx))/2
     ------------------------------------------------------------ */
  void
    serial_Multi_VectorSymmetrize( serial_Multi_Vector * mtx ) {

      long i, j, g, h, w, jump;
      komplex* p;
      komplex* q;
      komplex conj_q;

      assert( mtx != NULL );

      g = mtx->size;
      h = mtx->size;
      w = mtx->num_vectors;

      assert( h == w );

      jump = 0;

      for ( j = 0, p = (komplex *) mtx->data; j < w; j++ ) {
	q = p;
	for ( i = j; i < h; i++, p++, q += g ) {
	  conj_q.real = q->real;
	  conj_q.imag = - (q->imag);
	  complex_add(p,p,&conj_q);
	  p->real = p->real/2;
	  p->imag = p->imag/2;
	  q->real = p->real;
	  q->imag = -(p->imag);
	}
	p += ++jump;
      }
    }


} }
