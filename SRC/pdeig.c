#include  <mpi.h>
#include  <stdio.h>
#include  <stdlib.h>
#include  <math.h>

#define eigvecs(i,j)  eigvecs[nloc*( (j) -1 )+ (i) - 1]
#define eigvals(i)    eigvals[(i)-1]

double dnrm2_(int *, double *, int *);

void pdeig(MPI_Comm comm, int nloc, double *pmat, int *nev, 
           double *eigvals, double *eigvecs)

/*  Arguement list:
 
    comm     communicator 
    nloc     the number of elements in a local vector
    pmat     matrix information assumed to be diagonal now
    nev      on entry, the number of eigenvalues requested
             on return, the number of converged eigenvector
    eigvals  (double*) dimension nev.
                       Converged eigenvalues.
    eigvecs  (double*) dimension nloc by nev.
                       Local eigenvector matrix.
*/

{
    int    i, j, ncv, mypid, ione = 1;
    int    lworkl, ldv;
    double reserr;
    double  tol=0.0, sigma=0.0, one = 1.0, zero = 0.0;
    double  *workl, *workd, *resid, *v;
    int    *select;
    int    ido, info, ishfts, maxitr, mode, rvec, ierr;
    int    iparam[11], ipntr[11], mycoords[2];
    char   *which="SA", *bmat="I", *all="A", trans='N';
    int    Fcomm;

    MPI_Comm_rank(comm,&mypid);

    /* set parameters and allocate temp space for ARPACK*/
    ncv = *nev + 20; /* use a 20-th degree polynomial for implicit restart */
    lworkl = ncv*(ncv+8);
    ido    = 0;
    info   = 0;
    ishfts = 1;
    maxitr = 300;
    mode   = 1;
    ldv    = nloc;

    iparam[0] = ishfts;
    iparam[2] = maxitr;
    iparam[6] = mode;
    printf("nloc = %d\n", nloc);

    resid = (double*) calloc(nloc,sizeof(double));
    if (!resid) fprintf(stderr," Fail to allocate resid\n");
    workl = (double*) calloc(lworkl,sizeof(double));
    if (!workl) fprintf(stderr," Fail to allocate workl\n");
    v     = (double*) calloc(ldv*ncv,sizeof(double));
    if (!v) fprintf(stderr," Fail to allocate v\n");
    workd = (double*) calloc(nloc*3,sizeof(double));
    if (!workd) fprintf(stderr, " Fail to allocate workd\n");
    select= (int*) calloc(ncv,sizeof(int));
    if (!select) fprintf(stderr, " Fail to allocate select\n");

    /* PARPACK reverse comm to compute eigenvalues and eigenvectors */
    Fcomm = MPI_Comm_c2f(comm);

    while (ido != 99 ) {
       pdsaupd_(&Fcomm, &ido , bmat,  &nloc  , which, nev,  
                &tol  , resid, &ncv,  v      , &ldv , iparam,
                ipntr , workd, workl, &lworkl, &info);
       if (ido == -1 || ido == 1) {
          /* do matvec here use workd[ipntr[0]-1] as the input
             and workd[ipntr[1]-1] as the output */
          for (i = 0; i < nloc; i++)
             workd[ipntr[1]-1+i] = pmat[i]*workd[ipntr[0]-1+i];
       }
    }

    /* ARPACK postprocessing */
    if (info < 0) {
       fprintf(stderr, " Error with _naupd, info = %d\n", info);
    }
    else {
       rvec = 1;
       *nev = iparam[4];

       pdseupd_(&Fcomm , &rvec, all  , select, eigvals, eigvecs, &nloc, 
                &sigma , bmat , &nloc, which , nev    , &tol   , resid,
                &ncv   , v    , &ldv , iparam, ipntr  , workd  , workl,
                &lworkl, &ierr);

       if (ierr != 0) fprintf(stderr," Error with _neupd, ierr = %d\n",ierr);

       if (mypid == 0)  { 
          /* show converged ritz values and residual error */
          for (j = 0; j < *nev; j++)  {
             for (i = 0; i < nloc; i++) {
                resid[i] = pmat[i]*eigvecs(i,j) - eigvals[j]*eigvecs(i,j);
                reserr = dnrm2_(&nloc, resid, &ione);
             }
             printf("eig[%3d]: %15.6e   %15.6e\n", 
                    j, eigvals[j], reserr); 
          }
       } 
    } 
 

    free(resid);
    free(workl);
    free(v);
    free(workd); 
    free(select);
}
#undef z
#undef d
