#ifndef _CBLACS_H_
#define _CBLACS_H_

/* For descriptor use */
enum{
  CTXT_ = 2,
  M_    = 3,
  N_    = 4,
  MB_   = 5,
  NB_   = 6,
  LLD_  = 9,
  DLEN_ = 9,
};

#ifdef __cplusplus
extern "C"{
#endif

void Cblacs_get(int icontxt, int what, int* val);
void Cblacs_gridinit(int *icontxt, char *order, int nprow, int npcol);
void Cblacs_gridmap(int *icontxt, int *pmap, int ldpmap, int nprow, int npcol);
void Cblacs_gridinfo(int icontxt,  int* nprow, int* npcol, int* myprow, int* mypcol);
void descinit_(int* desc, int* m, int * n, int* mb, int* nb, int* irsrc, int* icsrc, 
	       int* ictxt, int* lld, int* info);
void pdlaprnt_(int* m, int *n, double* a, int* ia, int* ja, int* desca, int* irprnt, 
	       int* jrprnt, char* cmatnm, int* nout, double* work, int len);
/* pdsyevd is supported in MKL but is not included in mkl_scalapack.h.
 * This is a documentation bug */
void pdsyevd_(char *jobz, char *uplo, int *n, double *a, int *ia, 
	      int *ja, int *desca, double *w, double *z, int *iz, 
	      int *jz, int *descz, double *work, int *lwork, int* iwork,
	      int* liwork, int *info);
void pdsyev_(char *jobz, char *uplo, int *n, double *a, int *ia,
	     int *ja, int *desca, double *w, double *z, int *iz, 
	     int *jz, int *descz, double *work, int *lwork, int *info);

void pdsygst_(int* ibtype, char* uplo, int* n, double* a, int* ia, 
              int* ja, int* desca, double* B, int* ib, int* jb,
              int* descb, double* scale, int* info);

void pdsyngst_(int* ibtype, char* uplo, int* n, double* a, int* ia, 
              int* ja, int* desca, double* B, int* ib, int* jb,
              int* descb, double* scale, double* work, int* lwork, int* info);


void pdtrsm_(char* side, char* uplo, char* transa, char* diag, 
             int* m, int* n, double* alpha, double* a,
             int* ia, int* ja, int* desca, double* b,
             int* ib, int* jb, int* descb);
 
void pdpotrf_(char* uplo, int* n, double* a, int* ia,
              int* ja, int* desca, int* info);

void pdpocon_(char* uplo, int* n, double* a, int* ia,
	      int* ja, int* desca, double* anorm,
	      double* rcond, double* work, int* lwork,
	      int* iwork, int* liwork, int* info);


void pdgeqpf_(int* M, int* N, double* A, int* IA, int* JA,
              int* DESCA, int* IPIV, double* TAU, double* WORK,
              int* LWORK, int* INFO);

void pdormqr_(char* side, char* trans, int* M, int* N, int* K,
              double* A, int* IA, int* JA, int* DESCA, double* TAU,
              double* C, int* IC, int* JC, int* DESCC, double* WORK,
              int* LWORK, int* INFO);

void pdsygvx_(int* IBTYPE, char* JOBZ, char* RANGE, char* UPLO, 
	      int* N, double* A, int* IA, int* JA, int* DESCA, 
              double* B, int* IB, int* JB, int* DESCB, double* VL, 
              double* VU, int* IL, int* IU, double* ABSTOL, 
              int* M, int* NZ, double* W, double* ORFAC, double* Z, 
              int* IZ, int* JZ, int* DESCZ, double* WORK, int* LWORK, 
              int* IWORK, int* LIWORK, int* IFAIL, int* ICLUSTR,
              double* GAP, int* INFO );

int indxl2g_(int* indxloc, int* nb, int* iproc, int* isrcproc, int* nprocs);

int numroc_(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);

void pdgemm_(char* transa, char* transb, int* m, int* n, int* k,
             double* alpha, double* A, int* ia, int* ja, int* desca,
             double* B, int* ib, int* jb, int* descb, double* beta,
             double* C, int* ic, int* jc, int* descc);

#ifdef __cplusplus
}
#endif

#endif
