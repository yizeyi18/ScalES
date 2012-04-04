// TESTEIG2 tests the block sparse eigenvalue solver. 
// The sparse matrix Avec is not given in the ParVec form, but in the
// type map<SpBlckKey, DblNumMat>
#include "scfdg.hpp"
#include "parallel.hpp"
#include <unistd.h>

extern "C"{
  void pdsaupd_(int* Fcomm, int* ido, char* bmat, int* nloc, char*  which, 
		int* nev, double* tol, double* resid, int* ncv, double* v, 
		int* ldv, int* iparam, int* ipntr, double* workd, double* workl, 
		int* lworkl, int* info);
  void pdseupd_(int* Fcomm , int* rvec, char* all, int*select, double *eigvals, 
		double* eigvecs, int* nloc, double* sigma , char* bmat , 
		int* nloc, char* which , int* nev, double* tol, 
		double* resid, int* ncv, double* v, int* ldv , 
		int* iparam, int* ipntr, double* workd, double* workl,
	        int* lworkl, int* ierr);
}

class VecPtn
{
public:
  IntNumVec _ownerinfo;
public:
  VecPtn() {;}
  ~VecPtn() {;}
  IntNumVec& ownerinfo() { return _ownerinfo; }
  int owner(int key){ return _ownerinfo(key); }
};



typedef Vec2T<int> SpBlckKey;
typedef map<SpBlckKey, DblNumMat> SpMat;

class SpBlckPtn{
  public:
    map<SpBlckKey, int> _ownerinfo;
  public:
    SpBlckPtn() {;}
    ~SpBlckPtn() {;}
    map<SpBlckKey, int>& ownerinfo(){ return _ownerinfo;}
    int owner(SpBlckKey key) { return _ownerinfo[key];}
};
int main(int argc, char ** argv) 
{
  MPI_Comm comm = MPI_COMM_WORLD;
  int ncpus, nloc, mypid, ierr, i, nev;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(comm,&ncpus);
  MPI_Comm_rank(comm,&mypid);

  int mpirank = mypid;
  
  SpMat Avec;


  int szblck = 2;
  int nbblck = 4;
  DblNumMat tt1(szblck, szblck), tt2(szblck, szblck), tt3(szblck, szblck);
  {
    DblNumMat blank(szblck, szblck);
    setvalue(blank, 0.0);
    tt1 = blank;  tt2 = blank;  tt3 = blank;
    tt1(0,0) = 2.0;  tt1(0,1) = -1.0; tt1(1,0) = -1.0; tt1(1,1) = 2.0;
    tt2(1,0) = -1.0;
    tt3(0,1) = -1.0;
  }

  if(mpirank == 0){
    Avec[SpBlckKey(0,0)] = tt1;
    Avec[SpBlckKey(0,1)] = tt2;
    Avec[SpBlckKey(1,0)] = tt3;
  }
  if(mpirank == 1){
    Avec[SpBlckKey(1,1)] = tt1;
    Avec[SpBlckKey(1,2)] = tt2;
  }
  if(mpirank == 2){
    Avec[SpBlckKey(2,1)] = tt3;
    Avec[SpBlckKey(2,2)] = tt1;
    Avec[SpBlckKey(2,3)] = tt2;
  }
  if(mpirank == 3){
    Avec[SpBlckKey(3,2)] = tt3;
    Avec[SpBlckKey(3,3)] = tt1;
  }

  if( mpirank == 0 ){
    cout << Avec[SpBlckKey(0,0)] << endl; 
  }

  nloc = 2;
  nev = 2;
  DblNumVec eigvals(nev);
  DblNumMat eigvecs(nloc, nev);
  map<int, Index2> lclpos;
  IntNumVec    ownerinfo(nbblck);

  for(int i = 0; i < nbblck; i++){
    lclpos[i]    = Index2(0,nloc);
    ownerinfo[i] = i;
  } 
  
  /* This is adapted from pdeig.c */

  {
    int    i, j, ncv, mypid, ione = 1;
    int    lworkl, ldv;
    double reserr;
    double  tol=0.0, sigma=0.0, one = 1.0, zero = 0.0;
    double  *workl, *workd, *resid, *v;
    int    *select;
    int    ido, info, ishfts, maxitr, mode, rvec, ierr;
    int    iparam[11], ipntr[11], mycoords[2];
    char   *which="SA", *bmat="I", *all="A", Ntrans='N';
    int    Fcomm;

    MPI_Comm_rank(comm,&mypid);

    /* set parameters and allocate temp space for ARPACK*/
    ncv = nev + 1; /* use a 20-th degree polynomial for implicit restart */
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

    /* Prepare for the vectors */
    ParVec<int, DblNumVec, VecPtn> Xvec;
    ParVec<int, DblNumVec, VecPtn> Yvec;
    VecPtn vprtn;
    vprtn.ownerinfo() = ownerinfo;

    Xvec.prtn() = vprtn;
    Yvec.prtn() = vprtn;

    /* Communication pattern and memory allocation */
    vector<int> col_keyvec, row_keyvec;
    {
      set<int> col_keyset;
      for(map<SpBlckKey, DblNumMat>::iterator mi = Avec.begin(); mi != Avec.end(); mi++){
	int curkey = (*mi).first[1];
	DblNumVec blank((*mi).second.n());   setvalue(blank,0.0);
	Xvec.lclmap()[curkey] = blank;
	if(Xvec.prtn().owner(curkey) != mpirank){
	  col_keyset.insert(curkey);
	}
      }
      col_keyvec.insert(col_keyvec.begin(), 
			col_keyset.begin(), col_keyset.end());
    
      set<int> row_keyset;
      for(map<SpBlckKey, DblNumMat>::iterator mi = Avec.begin(); mi != Avec.end(); mi++){
	int curkey = (*mi).first[0];
	DblNumVec blank((*mi).second.m());   setvalue(blank,0.0);
	Yvec.lclmap()[curkey] = blank;
	if(Yvec.prtn().owner(curkey) != mpirank){
	  row_keyset.insert(curkey);
	}
      }
      row_keyvec.insert(row_keyvec.begin(), 
			row_keyset.begin(), row_keyset.end());
   
    }

//    if(mpirank == 0){
//      cout << IntNumVec(row_keyvec.size(), false, &row_keyvec[0]) << endl;
//      cout << IntNumVec(col_keyvec.size(), false, &col_keyvec[0]) << endl;
//    }

    /* PARPACK reverse comm to compute eigenvalues and eigenvectors */
    Fcomm = MPI_Comm_c2f(comm);

    while (ido != 99 ) {
      pdsaupd_(&Fcomm, &ido , bmat,  &nloc  , which, &nev,  
	       &tol  , resid, &ncv,  v      , &ldv , iparam,
	       ipntr , workd, workl, &lworkl, &info);
      if (ido == -1 || ido == 1) {
	/* do matvec here use workd[ipntr[0]-1] as the input
	   and workd[ipntr[1]-1] as the output */

        /* Copy the data from workd */
	double *workx = &workd[ipntr[0]-1];
	for(map<int, DblNumVec>::iterator mi = Xvec.lclmap().begin(); mi!=Xvec.lclmap().end(); mi++){
	  int curkey = (*mi).first;
	  if( Xvec.prtn().owner(curkey) == mpirank )
	    copy(&workx[lclpos[curkey][0]], &workx[lclpos[curkey][1]], Xvec.lclmap()[curkey].data());
	}

	{
	  vector<int> all(1,1);
	  iC( Xvec.getBegin(col_keyvec, all) ); 
	  iC( Xvec.getEnd(all) );
	}


//	/* DEBUG */
//	{
//	  if( mpirank == 0 ){
//	    cout << DblNumVec(szblck, false, &workd[ipntr[0]-1]) << endl;
//	    cout << Xvec.lclmap()[0] << endl;
//	  }
//	  break;
//	}

	/* Clear up Yvec */
	for(map<SpBlckKey, DblNumMat>::iterator mi = Avec.begin(); mi != Avec.end(); mi++){
	  int curkey = (*mi).first[0];
	  setvalue(Yvec.lclmap()[curkey], 0.0) ;
	}


	/* Matrix-vector multiplication */
	for(map<SpBlckKey, DblNumMat>::iterator mi=Avec.begin(); mi!=Avec.end(); mi++){
	  int idx = (*mi).first[0];
	  SpBlckKey  curidx = (*mi).first;
	  DblNumMat& curmat = (*mi).second;
	  DblNumVec& curxvec = Xvec.lclmap()[curidx[1]];
	  DblNumVec& curyvec = Yvec.lclmap()[curidx[0]];
	  { 
	    int m = curmat.m(), n = curmat.n();
	    dgemv_(&Ntrans, &m, &n, &one, curmat.data(), &m,
		   curxvec.data(), &ione, &one, curyvec.data(), &ione);
	  }
	}

	{
	  vector<int> all(1,1);
	  iC( Yvec.putBegin(row_keyvec, all) ); 
	  iC( Yvec.putEnd(all, PARVEC_CMB) );
	}


        /* Copy the data back to workd */
	double *worky = &workd[ipntr[1]-1];
	for(map<int, DblNumVec>::iterator mi = Yvec.lclmap().begin(); mi!=Yvec.lclmap().end(); mi++){
	  int curkey = (*mi).first;
	  DblNumVec& curvec = (*mi).second;
	  if( Yvec.prtn().owner(curkey) == mpirank )
	    copy(curvec.data(), curvec.data()+curvec.m(), &worky[lclpos[curkey][0]]);
	}


	/* DEBUG */
	/* Direct matrix-vector multiplication passed 9/10/2011 */
	if(0)
	{
	  MPI_Barrier(comm);
	  sleep(0.2*(double)mpirank);
	  cout << "mpirank = " << mpirank << endl;
	  cout << "X = " << Xvec.lclmap()[mpirank] << endl;
	  MPI_Barrier(comm);
	  sleep(0.2*(double)mpirank);
	  cout << "mpirank = " << mpirank << endl;
	  cout << "Y = " << Yvec.lclmap()[mpirank] << endl;
	  MPI_Barrier(comm);
	}

      }
    }

    /* ARPACK postprocessing */
    if (info < 0) {
      fprintf(stderr, " Error with _naupd, info = %d\n", info);
    }
    else {
      rvec = 1;
      nev = iparam[4];

      pdseupd_(&Fcomm , &rvec, all  , select, eigvals.data(), eigvecs.data(), &nloc, 
	       &sigma , bmat , &nloc, which , &nev    , &tol   , resid,
	       &ncv   , v    , &ldv , iparam, ipntr  , workd  , workl,
	       &lworkl, &ierr);

      if (ierr != 0) fprintf(stderr," Error with _neupd, ierr = %d\n",ierr);

      if (mypid == 0)  { 
	/* show converged ritz values and residual error */
	for (j = 0; j < nev; j++)  {
//	  for (i = 0; i < nloc; i++) {
//	    resid[i] = pmat[i]*eigvecs(i,j) - eigvals[j]*eigvecs(i,j);
//	    reserr = dnrm2_(&nloc, resid, &ione);
//	  }
//	  printf("eig[%3d]: %15.6e   %15.6e\n", 
//		 j, eigvals[j], reserr); 
	  cout << "eig[" << j << "] = " << eigvals[j] << endl;
	}
      } 
    } 


    free(resid);
    free(workl);
    free(v);
    free(workd); 
    free(select);
  }

  ierr =  MPI_Finalize();
  return 0;
}
