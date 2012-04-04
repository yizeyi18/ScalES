#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>

void pdeig(MPI_Comm comm, int nloc, double *pmat, int *nev,
           double *d, double *z);

int main(int argc, char ** argv) 
{
   MPI_Comm comm = MPI_COMM_WORLD;
   int ncpus, nloc, mypid, ierr, i, nev;
   double *pmat, *d, *z;

   MPI_Init(&argc,&argv);
   MPI_Comm_size(comm,&ncpus);
   MPI_Comm_rank(comm,&mypid);

   nloc = 100;
   nev = 4;
   pmat = (double*)malloc(nloc*sizeof(double));
   d    = (double*)malloc(nev*sizeof(double));
   z    = (double*)malloc(nloc*nev*sizeof(double));
   for (i = 0; i < nloc; i++) pmat[i] = (double) (i+1);

   pdeig(comm, nloc, pmat, &nev, d, z);

   free(pmat);
   free(d);
   free(z);
   ierr =  MPI_Finalize();
   return 0;
}
