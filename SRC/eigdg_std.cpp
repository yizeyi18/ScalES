#include "eigdg.hpp"
#include "interp.hpp"
#include "vecmatop.hpp"

extern FILE* fhstat;
extern int CONTXT;
  
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
int EigDG::solve_A_Aloc(ParVec<EmatKey,DblNumMat,EmatPtn>& A,int& AM,int& AN,NumTns< vector<int> >& indexvec,
			int* desca, DblNumMat& Aloc, int& AlocM, int& AlocN)
{
  //----------------------------------------------------------------------------------------------------------------  
  /* generate processor grid from CONTXT */
  //
  int mpirank; MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize; MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  time_t t0, t1;
  //
  t0 = time(0);
  
  int nprow, npcol, myprow, mypcol;
  Cblacs_gridinfo(CONTXT,  &nprow, &npcol, &myprow, &mypcol);
  IntNumMat proctmp(nprow, npcol);
  IntNumMat procgrid(nprow,npcol);
  setvalue(proctmp, 0);    setvalue(procgrid, 0);
  proctmp(myprow, mypcol) = mpirank;
  MPI_Allreduce(proctmp.data(), procgrid.data(), nprow*npcol, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  int szblck = _MB;   //LY: FINALLY USE _MB HERE
  
  int mbblck = ((AM+(szblck-1))/szblck);
  int nbblck = ((AN+(szblck-1))/szblck);
  int NA = mbblck * szblck; //the leading dimension of the matrix A (used for scalapack)
  //if(mpirank==0) {    cerr<<"BLOCK DATA "<<Ndof<<" "<<szblck<<" "<<nbblck<<" "<<NA<<endl;  }
  IntNumMat procdist(mbblck,nbblck);
  for(int j=0; j<nbblck; j++)
    for(int i=0; i<mbblck; i++)
      procdist(i,j) = procgrid(i%nprow, j%npcol);
  //
  BlckPtn bp;  bp.ownerinfo() = procdist;
  //
  //create Avec (the matrix) with bp (partition) and zero blocks for owned entries
  ParVec<Index2,DblNumMat,BlckPtn> Avec;
  Avec.prtn() = bp;
  DblNumMat blank(szblck,szblck);  setvalue(blank,0.0);
  for(int jb=0; jb<nbblck; jb++)
    for(int ib=0; ib<mbblck; ib++) {
      Index2 curkey(ib,jb); 
      if(Avec.prtn().owner(curkey)==mpirank)
	Avec.lclmap()[curkey] = blank;
    }
  //prepare data into Avec
  for(map<EmatKey,DblNumMat>::iterator mi=A.lclmap().begin(); mi!=A.lclmap().end(); mi++) {
    EmatKey curkey = (*mi).first;
    if(A.prtn().owner(curkey)==mpirank) { //ADD ONLY THE THINGS OWNED BY ME
      Index3 uk = curkey.first;
      Index3 vk = curkey.second;
      vector<int>& uindex = indexvec(uk(0),uk(1),uk(2));
      vector<int>& vindex = indexvec(vk(0),vk(1),vk(2));
      DblNumMat& S = (*mi).second;
      iA(S.m()==uindex.size() && S.n()==vindex.size());
      for(int a=0; a<S.m(); a++)
	for(int b=0; b<S.n(); b++) {
	  int i = uindex[a];
	  int j = vindex[b];
	  int ib = i/szblck;    int io = i%szblck;
	  int jb = j/szblck;    int jo = j%szblck;
	  map<Index2,DblNumMat>::iterator mi=Avec.lclmap().find(Index2(ib,jb));
	  if(mi==Avec.lclmap().end()) { //if does not exist, create an empty block
	    Avec.lclmap()[Index2(ib,jb)] = blank;
	    mi = Avec.lclmap().find(Index2(ib,jb));
	  }
	  ((*mi).second)(io,jo) += S(a,b);
	}
    }
  }
  if(mpirank==0) { 
    fprintf(stderr, "finish generating avec, start communication\n");
  }
  {
    //get the keys for Avec
    vector<Index2> keyvec;
    for(map<Index2,DblNumMat>::iterator mi=Avec.lclmap().begin(); mi!=Avec.lclmap().end(); mi++) {
      Index2 curkey = (*mi).first;
      if(Avec.prtn().owner(curkey)!=mpirank) //NOT OWNED BY ME
	keyvec.push_back((*mi).first);
    }
    //communicate
    vector<int> all(1,1);
    iC( Avec.putBegin(keyvec,all) );
    iC( Avec.putEnd(all,PARVEC_CMB) ); //LEXING: combine everything together
    //clear to save some space
    //
    vector<Index2> tmpvec;
    for(map<Index2,DblNumMat>::iterator mi=Avec.lclmap().begin(); mi!=Avec.lclmap().end(); mi++) {
      Index2 curkey = (*mi).first;
      if(Avec.prtn().owner(curkey)!=mpirank) //NOT OWNED BY ME
	tmpvec.push_back(curkey);
    }
    for(int a=0; a<tmpvec.size(); a++)      Avec.lclmap().erase(tmpvec[a]); //THEN REMOVE
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = time(0);  
  if(mpirank==0) { 
    fprintf(fhstat, "finish preparing Avec %15.3f secs\n", difftime(t1,t0));   
    fprintf(stderr, "finish preparing Avec %15.3f secs\n", difftime(t1,t0));   
  }
  
  //----------------------------------------------------------------------------------------------------------------  
  t0 = time(0);
  
  //int desca[DLEN_]; //descz[DLEN_];
  AlocM = AM;
  AlocN = AN;
  
  //LY: SET UP THE Aloc matrix here
  int IZERO = 0, IONE = 1;
  double DZERO = 0.0;
  int lda = NA, ldz = lda;
  int loc_blkm = (mbblck + nprow - 1 ) / nprow;
  int loc_blkn = (nbblck + npcol - 1 ) / npcol;
  int loc_m = loc_blkm * _MB;
  int loc_n = loc_blkn * _MB;
  int info; 
  descinit_(desca, &AM, &AN, &_MB, &_MB, &IZERO, &IZERO, &CONTXT, &loc_m, &info);  //descinit_(descz, &Ndof, &Ndof, &_MB, &_MB, &IZERO, &IZERO, &CONTXT, &loc_m, &info);
  Aloc.resize(loc_m, loc_n);  //DblNumMat locA(loc_m, loc_n);  //DblNumMat locZ(loc_m, loc_n);
  setvalue(Aloc, 0.0); //LLIN: IMPORTANT
  
  int cnt_row = 0, cnt_col = 0;
  bool flag_cnt;
  for(int jb=0; jb<nbblck; jb++){
    flag_cnt = false;
    for(int ib=0; ib<mbblck; ib++) {
      Index2 curkey(ib,jb); 
      if(Avec.prtn().owner(curkey)==mpirank){
	flag_cnt = true;
	map<Index2,DblNumMat>::iterator mi=Avec.lclmap().find(Index2(ib,jb));
	double* locAptr = &Aloc(cnt_row*_MB, cnt_col*_MB);
	double* Avecptr = ((*mi).second).data();
	/* Matrix copy */
	for(int j = 0; j < _MB; j++){
	  for(int i = 0; i < _MB; i++){
	    *locAptr = *Avecptr;
	    locAptr++; Avecptr++;
	  }
	  locAptr += loc_m - _MB;
	}
	cnt_row++;
      }
    }
    if( flag_cnt ){
      cnt_row = 0;
      cnt_col++;
    }
  }
  
  t1 = time(0);  
  if(mpirank==0) {
    fprintf(fhstat, "finish preparing Aloc %15.3f secs\n", difftime(t1,t0));   
    fprintf(stderr, "finish preparing Aloc %15.3f secs\n", difftime(t1,t0));   
  }
  
  return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
int EigDG::solve_Aloc_Zloc(int* desca, DblNumMat& Aloc, int& AlocM, int& AlocN,
			   int* descz, DblNumMat& Zloc, int& ZlocM, int& ZlocN, DblNumVec& W)
{
  //
  int mpirank; MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize; MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  time_t t0, t1;
  //
  ZlocM = AlocM;
  ZlocN = AlocN;
 
  int nprow, npcol, myprow, mypcol;
  Cblacs_gridinfo(CONTXT,  &nprow, &npcol, &myprow, &mypcol);
  
  int szblck = _MB;   //LY: FINALLY USE _MB HERE
  int mbblck = ((ZlocM+(szblck-1))/szblck);
  int nbblck = ((ZlocN+(szblck-1))/szblck);
  int NA = mbblck * szblck; //the leading dimension of the matrix A (used for scalapack)
  
  int IZERO = 0, IONE = 1;
  double DZERO = 0.0;
  int lda = NA, ldz = lda;
  int loc_blkm = (mbblck + nprow - 1 ) / nprow;
  int loc_blkn = (nbblck + npcol - 1 ) / npcol;
  int loc_m = loc_blkm * _MB;
  int loc_n = loc_blkn * _MB;
  int info; 
  descinit_(descz, &ZlocM, &ZlocN, &_MB, &_MB, &IZERO, &IZERO, &CONTXT, &loc_m, &info);
  Zloc.resize(loc_m, loc_n);
  
  /* The version using pdsyev which is based on QR algorithm */
  /*
  t0 = time(0);  
  char jobz     = 'V', 
       uplo     = 'U';
  int lwork = (20+loc_n)*Ndof;
  DblNumVec      W(Ndof);
  DblNumVec      work(lwork);

  pdsyev_(&jobz, &uplo, &Ndof, locA.data(), &IONE, &IONE, desca,
	  W.data(), locZ.data(), &IONE, &IONE, descz, work.data(),
	  &lwork, &info);
  iA(info == 0);
  work.resize(0);

  t1 = time(0);  
  if(mpirank==0) { 
    fprintf(fhstat, "pdsyev %15.3f secs\n", difftime(t1,t0));   
    fprintf(stderr, "pdsyev %15.3f secs\n", difftime(t1,t0));   
  }
  */
  
  int Ndof = AlocM;
  
  /* The version using pdsyevd which is based on Divide and Conquer */
  t0 = time(0);  
  char jobz     = 'V', 
    uplo     = 'U';
  int lwork  = 10*Ndof + 4 * loc_m * loc_n;
  int liwork = 10*Ndof + 10 * npcol;

  W.resize(Ndof);  //DblNumVec      W(Ndof); //eigenvecs
  IntNumVec      iwork(liwork);
  DblNumVec      work(lwork);

  // Print out A for test purpose
  if(0){
    int nout = 0;
    pdlaprnt_(&Ndof, &Ndof, Aloc.data(), &IONE, &IONE, desca, 
	      &IZERO, &IZERO, "A", &nout, work.data(), 1);
  }


  pdsyevd_(&jobz, &uplo, &Ndof, Aloc.data(), &IONE, &IONE, desca,
	   W.data(), Zloc.data(), &IONE, &IONE, descz, work.data(),
	   &lwork, iwork.data(), &liwork, &info);



  iA(info == 0);
  work.resize(0);

  t1 = time(0);  
  if(mpirank==0) { 
    fprintf(fhstat, "pdsyevd %15.3f secs\n", difftime(t1,t0));   
    fprintf(stderr, "pdsyevd %15.3f secs\n", difftime(t1,t0));   
  }
  
  return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
int EigDG::solve_Zloc_Eig(int* descz, DblNumMat& Zloc, int& ZlocM, int& ZlocN, DblNumVec& W,
			  int npsi, vector<double>& eigvals, ParVec<Index3, DblNumMat, ElemPtn>& eigvecsvec, NumTns< vector<int> >& indexvec)
{
  Point3 hs = _hs; //ELEMENT SIZE
  Index3 Ns = _Ns; //NUMBER OF ELEMENTS
  Index3 Nlbls = _Nlbls; //NUMBER OF LBL GRID POINTS
  //
  int N1 = Ns(0);  int N2 = Ns(1);  int N3 = Ns(2);
  double h1 = hs(0);  double h2 = hs(1);  double h3 = hs(2);
  int Nlbl1 = Nlbls(0);  int Nlbl2 = Nlbls(1);  int Nlbl3 = Nlbls(2);
  int Nlbltot = Nlbl1*Nlbl2*Nlbl3;
  //
  int mpirank; MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize; MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  time_t t0, t1;
  //----------------------------------------------------------------------------------------------------------------  
  //get the required blocks
  //
  t0 = time(0);  
  
  int nprow, npcol, myprow, mypcol;
  Cblacs_gridinfo(CONTXT,  &nprow, &npcol, &myprow, &mypcol);
  IntNumMat proctmp(nprow, npcol);
  IntNumMat procgrid(nprow,npcol);
  setvalue(proctmp, 0);    setvalue(procgrid, 0);
  proctmp(myprow, mypcol) = mpirank;
  MPI_Allreduce(proctmp.data(), procgrid.data(), nprow*npcol, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  int szblck = _MB;   //LY: FINALLY USE _MB HERE

  int mbblck = ((ZlocM+(szblck-1))/szblck);
  int nbblck = ((ZlocN+(szblck-1))/szblck);
  int NA = mbblck * szblck; //the leading dimension of the matrix A (used for scalapack)
  //if(mpirank==0) {    cerr<<"BLOCK DATA "<<Ndof<<" "<<szblck<<" "<<nbblck<<" "<<NA<<endl;  }
  IntNumMat procdist(mbblck,nbblck);
  for(int j=0; j<nbblck; j++)
    for(int i=0; i<mbblck; i++)
      procdist(i,j) = procgrid(i%nprow, j%npcol);
  //
  BlckPtn bp;  bp.ownerinfo() = procdist;
  //
  ParVec<Index2,DblNumMat,BlckPtn> Zvec;
  Zvec.prtn() = bp;
  DblNumMat blank(szblck,szblck);  setvalue(blank,0.0);
  for(int ib=0; ib<mbblck; ib++)
    for(int jb=0; jb<nbblck; jb++) {
      Index2 curkey(ib,jb); 
      if(Zvec.prtn().owner(curkey)==mpirank)
	Zvec.lclmap()[curkey] = blank;
    }

  int loc_blkm = (mbblck + nprow - 1 ) / nprow;
  int loc_blkn = (nbblck + npcol - 1 ) / npcol;
  int loc_m = loc_blkm * _MB;
  int loc_n = loc_blkn * _MB;
  
  int cnt_row = 0, cnt_col = 0;
  bool flag_cnt;  //cnt_row = 0; cnt_col = 0;
  for(int jb=0; jb<nbblck; jb++){
    flag_cnt = false;
    for(int ib=0; ib<mbblck; ib++) {
      Index2 curkey(ib,jb); 
      if(Zvec.prtn().owner(curkey)==mpirank){
	flag_cnt = true;
	map<Index2,DblNumMat>::iterator mi=Zvec.lclmap().find(Index2(ib,jb));
	double* locZptr = &Zloc(cnt_row*_MB, cnt_col*_MB);
	double* Zvecptr = ((*mi).second).data();
	/* Matrix copy */
	for(int j = 0; j < _MB; j++){
	  for(int i = 0; i < _MB; i++){
	    *Zvecptr = *locZptr;
	    locZptr++; Zvecptr++;
	  }
	  locZptr += loc_m - _MB;
	}
	cnt_row++;
      }
    }
    if( flag_cnt ){
      cnt_row = 0;
      cnt_col++;
    }
  }
  
  {
    //get the keys for Zvec, get all the blocks in the right row
    set<Index2> keyset;
    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if(_elemptn.owner(curkey)==mpirank) {
	    vector<int>& index = indexvec(i1,i2,i3);
	    for(int g=0; g<index.size(); g++) {
	      int ii = index[g];
	      int ib = ii/szblck;	  int io = ii%szblck;
	      for(int jb=0; jb<nbblck; jb++)
		keyset.insert( Index2(ib,jb) );
	    }
	  }
	}
    vector<Index2> keyvec;    keyvec.insert(keyvec.begin(), keyset.begin(), keyset.end());
    //communicate Zvec to the processors that need it
    vector<int> all(1,1);
    iC( Zvec.getBegin(keyvec,all) );
    iC( Zvec.getEnd(all) );
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = time(0);  
  if(mpirank==0) { 
    fprintf(fhstat, "finish preparing Zvec %15.3f secs\n", difftime(t1,t0));   
    fprintf(stderr, "finish preparing Zvec %15.3f secs\n", difftime(t1,t0));   
  }
  
  //----------------------------------------------------------------------------------------------------------------  
  //write back to eigenvalues and eigenvectors
  t0 = time(0);
  int Neigs = npsi;
  eigvals.resize(Neigs);
  for(int g=0; g<Neigs; g++)
    eigvals[g] = W(g);
  
  for(int i3=0; i3<N3; i3++)
    for(int i2=0; i2<N2; i2++)
      for(int i1=0; i1<N1; i1++) {
	Index3 curkey(i1,i2,i3);
	if(_elemptn.owner(curkey)==mpirank) {
	  vector<int>& index = indexvec(i1,i2,i3);
	  DblNumMat& eigvecs = eigvecsvec.lclmap()[curkey];
	  eigvecs.resize(index.size(),Neigs);
	  setvalue(eigvecs,0.0);
	  for(int g=0; g<index.size(); g++) {
	    int ii = index[g];
	    int ib = ii/szblck;	  int io = ii%szblck;
	    for(int jb=0; jb<nbblck; jb++) {
	      DblNumMat& Ztmp = Zvec.lclmap()[Index2(ib,jb)];
	      for(int jo=0; jo<szblck; jo++) {
		int h = jb*szblck+jo;
		if(h<Neigs)
		  eigvecs(g,h) = Ztmp(io,jo);
	      }
	    }
	  }
	}
      }
  MPI_Barrier(MPI_COMM_WORLD);
  
  t1 = time(0);
  if(mpirank==0) {
    fprintf(fhstat, "finish preparing eigvecs %15.3f secs\n", difftime(t1,t0));   
    fprintf(stderr, "finish preparing eigvecs %15.3f secs\n", difftime(t1,t0));   
  }
  
    
  /*
  //LEXING: send eigvecs to the ones who might need it
  {
    vector<Index3> keyvec;    keyvec.insert(keyvec.begin(), _keyset.begin(), _keyset.end());
    vector<int> mask(Elem_Number,0);
    mask[Elem_eigvecs] = 1;
    iC( elemvec.getBegin(keyvec, mask) );
    iC( elemvec.getEnd(mask) );
    MPI_Barrier(MPI_COMM_WORLD);
  }
  */
  
  
  //   t1 = time(0);  
  //   if(mpirank==0) { 
  //     fprintf(stderr, "post processing time %15.3f secs\n", difftime(t1,t0));   
  //     fprintf(fhstat, "post processing time %15.3f secs\n", difftime(t1,t0));   
  //   }
  return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
int EigDG::solve_C_Cloc(ParVec<Index3,DblNumMat,ElemPtn>& A, int&AM, int&AN, NumTns< vector<int> >& indexvec,
			int* desca, DblNumMat& Aloc, int& AlocM, int& AlocN)
{
  //----------------------------------------------------------------------------------------------------------------  
  /* generate processor grid from CONTXT */
  //
  int mpirank; MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize; MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  time_t t0, t1;
  //int AM = _Ndof;
  //int AN = N1*N2*N3*_Norbperelem;
  //
  t0 = time(0);
  
  int nprow, npcol, myprow, mypcol;
  Cblacs_gridinfo(CONTXT,  &nprow, &npcol, &myprow, &mypcol);
  IntNumMat proctmp(nprow, npcol);
  IntNumMat procgrid(nprow,npcol);
  setvalue(proctmp, 0);    setvalue(procgrid, 0);
  proctmp(myprow, mypcol) = mpirank;
  MPI_Allreduce(proctmp.data(), procgrid.data(), nprow*npcol, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  int szblck = _MB;   //LY: FINALLY USE _MB HERE
  
  int mbblck = ((AM+(szblck-1))/szblck);
  int nbblck = ((AN+(szblck-1))/szblck);
  int NA = mbblck * szblck; //the leading dimension of the matrix A (used for scalapack)
  //if(mpirank==0) {    cerr<<"BLOCK DATA "<<Ndof<<" "<<szblck<<" "<<nbblck<<" "<<NA<<endl;  }
  IntNumMat procdist(mbblck,nbblck);
  for(int j=0; j<nbblck; j++)
    for(int i=0; i<mbblck; i++)
      procdist(i,j) = procgrid(i%nprow, j%npcol);
  //
  BlckPtn bp;  bp.ownerinfo() = procdist;
  //
  //create Avec (the matrix) with bp (partition) and zero blocks for owned entries
  ParVec<Index2,DblNumMat,BlckPtn> Avec;
  Avec.prtn() = bp;
  DblNumMat blank(szblck,szblck);  setvalue(blank,0.0);
  for(int jb=0; jb<nbblck; jb++)
    for(int ib=0; ib<mbblck; ib++) {
      Index2 curkey(ib,jb); 
      if(Avec.prtn().owner(curkey)==mpirank)
	Avec.lclmap()[curkey] = blank;
    }
  //prepare data into Avec
  for(map<Index3,DblNumMat>::iterator mi=A.lclmap().begin(); mi!=A.lclmap().end(); mi++) {
    Index3 curkey = (*mi).first;
    if(A.prtn().owner(curkey)==mpirank) { //ADD ONLY THE THINGS OWNED BY ME
      DblNumMat& S = (*mi).second;
      vector<int>& vindex = indexvec(curkey(0),curkey(1),curkey(2));
      //iA(S.m()==_Ndof && S.n()==_Norbperelem);
      for(int a=0; a<S.m(); a++)
	for(int b=0; b<S.n(); b++) {
	  int i = a;
	  int j = vindex[b];
	  int ib = i/szblck;    int io = i%szblck;
	  int jb = j/szblck;    int jo = j%szblck;
	  map<Index2,DblNumMat>::iterator mi=Avec.lclmap().find(Index2(ib,jb));
	  if(mi==Avec.lclmap().end()) { //if does not exist, create an empty block
	    Avec.lclmap()[Index2(ib,jb)] = blank;
	    mi = Avec.lclmap().find(Index2(ib,jb));
	  }
	  ((*mi).second)(io,jo) += S(a,b);
	}
    }
  }
  if(mpirank==0) { 
    fprintf(stderr, "finish generating avec, start communication\n");
  }
  {
    //get the keys for Avec
    vector<Index2> keyvec;
    for(map<Index2,DblNumMat>::iterator mi=Avec.lclmap().begin(); mi!=Avec.lclmap().end(); mi++) {
      Index2 curkey = (*mi).first;
      if(Avec.prtn().owner(curkey)!=mpirank) //NOT OWNED BY ME
	keyvec.push_back((*mi).first);
    }
    //communicate
    vector<int> all(1,1);
    iC( Avec.putBegin(keyvec,all) );
    iC( Avec.putEnd(all,PARVEC_CMB) ); //LEXING: combine everything together
    //clear to save some space
    vector<Index2> tmpvec;
    for(map<Index2,DblNumMat>::iterator mi=Avec.lclmap().begin(); mi!=Avec.lclmap().end(); mi++) {
      Index2 curkey = (*mi).first;
      if(Avec.prtn().owner(curkey)!=mpirank) //NOT OWNED BY ME
	tmpvec.push_back(curkey);
    }
    for(int a=0; a<tmpvec.size(); a++)      Avec.lclmap().erase(tmpvec[a]); //THEN REMOVE
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = time(0);  
  if(mpirank==0) { 
    fprintf(fhstat, "finish preparing Avec %15.3f secs\n", difftime(t1,t0));   
    fprintf(stderr, "finish preparing Avec %15.3f secs\n", difftime(t1,t0));   
  }
  
  //----------------------------------------------------------------------------------------------------------------  
  t0 = time(0);
  
  //int desca[DLEN_]; //descz[DLEN_];
  AlocM = AM;
  AlocN = AN;
  
  //LY: SET UP THE Aloc matrix here
  int IZERO = 0, IONE = 1;
  double DZERO = 0.0;
  int lda = NA, ldz = lda;
  int loc_blkm = (mbblck + nprow - 1 ) / nprow;
  int loc_blkn = (nbblck + npcol - 1 ) / npcol;
  int loc_m = loc_blkm * _MB;
  int loc_n = loc_blkn * _MB;
  int info; 
  descinit_(desca, &AM, &AN, &_MB, &_MB, &IZERO, &IZERO, &CONTXT, &loc_m, &info);  //descinit_(descz, &Ndof, &Ndof, &_MB, &_MB, &IZERO, &IZERO, &CONTXT, &loc_m, &info);
  Aloc.resize(loc_m, loc_n);  //DblNumMat locA(loc_m, loc_n);  //DblNumMat locZ(loc_m, loc_n);
  setvalue(Aloc, 0.0); //LLIN: IMPORTANT
  
  int cnt_row = 0, cnt_col = 0;
  bool flag_cnt;
  for(int jb=0; jb<nbblck; jb++){
    flag_cnt = false;
    for(int ib=0; ib<mbblck; ib++) {
      Index2 curkey(ib,jb); 
      if(Avec.prtn().owner(curkey)==mpirank){
	flag_cnt = true;
	map<Index2,DblNumMat>::iterator mi=Avec.lclmap().find(Index2(ib,jb));
	double* locAptr = &Aloc(cnt_row*_MB, cnt_col*_MB);
	double* Avecptr = ((*mi).second).data();
	/* Matrix copy */
	for(int j = 0; j < _MB; j++){
	  for(int i = 0; i < _MB; i++){
	    *locAptr = *Avecptr;
	    locAptr++; Avecptr++;
	  }
	  locAptr += loc_m - _MB;
	}
	cnt_row++;
      }
    }
    if( flag_cnt ){
      cnt_row = 0;
      cnt_col++;
    }
  }
  
  t1 = time(0);  
  if(mpirank==0) {
    fprintf(fhstat, "finish preparing Cloc %15.3f secs\n", difftime(t1,t0));   
    fprintf(stderr, "finish preparing Cloc %15.3f secs\n", difftime(t1,t0));   
  }
  
  return 0;
}


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//LLIN: Perform pivot QR on the coefficient matrix Cloc to get the reduced
//coefficient matrix Crdcloc 
int EigDG::solve_Cloc_QR(DblNumMat& Cloc, int& ClocM, int& ClocN, int* descC, 
			 DblNumMat& Crdcloc, int& CrdclocM, int& CrdclocN, int* descCrdc){
  int mpirank; MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int ione = 1, izero = 0;
  int lwork, info;
  int nprow, npcol, myprow, mypcol;
  Cblacs_gridinfo(CONTXT,  &nprow, &npcol, &myprow, &mypcol);
  
  int szblck = _MB;   
  int mbblck = ((ClocM+(szblck-1))/szblck);
  int nbblck = ((ClocN+(szblck-1))/szblck);
  int loc_blkm = (mbblck + nprow - 1 ) / nprow;
  int loc_blkn = (nbblck + npcol - 1 ) / npcol;
  int loc_m = loc_blkm * _MB;
  int loc_n = loc_blkn * _MB;
  
  DblNumMat locQ(loc_m, loc_n);   setvalue(locQ, 0.0);
  DblNumMat locR(loc_m, loc_n);   setvalue(locR, 0.0);
  
  DblNumVec tau(loc_m);
  IntNumVec ipiv(loc_m);
  locQ = Cloc;
  
  time_t t0, t1;

  t0 = time(0);

  /* Pivot QR factorization by pdgeqpf. The Q-matrices are stored in the
   * form of elementary reflectors and R in the upper triangular part of
   * locQ */
  int descq[DLEN_];
  descinit_(descq, &ClocM, &ClocN, &_MB, &_MB, &izero, &izero, &CONTXT,
	    &loc_m, &info);
  
  /* LLIN: Workspace is allocated via the query subroutine to avoid bugs */
  {
    DblNumVec work;
    work.resize(1);
    lwork = -1;
    pdgeqpf_(&ClocM, &ClocN, locQ.data(), &ione, &ione, descq,
	     ipiv.data(), tau.data(), work.data(), &lwork, &info);
    
    lwork = (int)work(0); // LLIN: IMPORTANT
    work.resize(lwork);
    pdgeqpf_(&ClocM, &ClocN, locQ.data(), &ione, &ione, descq,
	     ipiv.data(), tau.data(), work.data(), &lwork, &info);
    iC(info);
  }

  /* Get the R matrix */
  locR = locQ;
  int gi, gj;
  double EPS = 1e-7; // LLIN: FIXME: Change this parameter to the input 
  int tCN;
  tCN = ClocN;
  
  if(mpirank == 0 ){
    cerr << "ClocM = " << ClocM << endl;
    cerr << "ClocN = " << ClocN << endl;
    cerr << "loc_m = " << loc_m << endl;
    cerr << "loc_n = " << loc_n << endl;
  }
 
  for(int j = 0; j < loc_n; j++){
    gj = npcol*_MB*(j/_MB) + mypcol*_MB + j%_MB ;  //LLIN: global index
    if( gj >= ClocN ) continue;
    for(int i = 0; i < loc_m; i++){
      gi = nprow*_MB*(i/_MB) + myprow*_MB + i%_MB; //LLIN: global index
      if( gi >= ClocM ) continue;
      /* Only get the upper triangular part */
      if(gi > gj){
	locR(i,j) = 0.0;
      } 
      /* Threshold the diagonal elements of R */
      if(gi == gj ){
	if( abs(locR(i,j)) < EPS ){  // LLIN: IMPORTANT: j < i
	  if(0){
	    cerr << "myid = " << mpirank << ", locj = " << j << 
	      ", glbj = " << gj << ", abs(locR) = " << abs(locR(j,j)) 
	      << ", tCN = " << tCN << endl;
	    sleep(0.2);
	  }
	  tCN = min(tCN, gj);
	}
      }
    }
  }

  /* Since pivot QR is used, CNrdc is the last column with 
   * abs(diag(R)) > EPS.  The rest of the columns are discarded
   * later */
  CrdclocM = ClocM;
  MPI_Allreduce(&tCN, &CrdclocN, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  /* Descriptor the Cloc matrix with reduced column size */
  descinit_(descCrdc, &CrdclocM, &CrdclocN, &_MB, &_MB, &izero, &izero,
	    &CONTXT, &loc_m, &info); 
  
  if(mpirank == 0 ){
    cerr << "CrdclocM = " << CrdclocM << endl;
    cerr << "CrdclocN = " << CrdclocN << endl;
  }
 


  {
    DblNumVec work;
    char side     = 'L';
    char trans    = 'N';  
    
    work.resize(1);
    lwork = -1;
    pdormqr_(&side, &trans, &CrdclocM, &CrdclocN, &ClocN, locQ.data(), 
	     &ione, &ione, descq, tau.data(), locR.data(),
	     &ione, &ione, descCrdc, work.data(), &lwork, &info);
    
    lwork = ((int)work(0)); // LLIN: IMPORTANT
    work.resize(lwork);
    pdormqr_(&side, &trans, &CrdclocM, &CrdclocN, &ClocN, locQ.data(), 
	     &ione, &ione, descq, tau.data(), locR.data(),
	     &ione, &ione, descCrdc, work.data(), &lwork, &info);
  }

  Crdcloc.resize(loc_m, loc_n);
  Crdcloc = locR;  
  
  t1 = time(0);  
  if(mpirank==0) {
    fprintf(fhstat, "finish pivot QR factorization of C %15.3f secs\n", 
	    difftime(t1,t0));   
    fprintf(stderr, "finish pivot QR factorization of C %15.3f secs\n", 
	    difftime(t1,t0));   
  }
  
  return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
/*LLIN: Solve the generalized eigenvalue problem
    A V = B V E
  Using ScaLAPACK.  It transforms the generalized eigenvalue problem
  into a standard eigenvalue problem, and solve the standard eigenvalue
  problem using pdsyevd.  The resulting algorithm is more efficient that
  the ScaLAPACK subroutine pdsygvx, which uses the same procedure but
  calls pdsyevx for solving the standard eigenvalue problem.
  
  IMPORTANT: The input matrix Aloc and Bloc will be overwritten!
*/
int EigDG::solve_GE(DblNumMat& Aloc, int& AlocM, int& AlocN, int* descA,
		    DblNumMat& Bloc, int& BlocM, int& BlocN, int* descB,
		    DblNumMat& Vloc, int& VlocM, int& VlocN, int* descV,
		    DblNumVec& EC){
  int mpirank; MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize; MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  int ione = 1, izero = 0;
  double one = 1.0, zero = 0.0;
  int info;
  int nprow, npcol, myprow, mypcol;
  char   uplo    = 'L'; //LLIN: Always use the lower triangular part
  
  time_t t0, t1;

  t0 = time(0);
  
  if(AlocM != BlocM || AlocN != BlocN || 
     AlocM != AlocN || BlocM != BlocN)
    return 1;
  
  Cblacs_gridinfo(CONTXT,  &nprow, &npcol, &myprow, &mypcol);

  VlocM = AlocM;
  VlocN = AlocN;
  
  int szblck = _MB;   
  int mbblck = ((VlocM+(szblck-1))/szblck);
  int nbblck = ((VlocN+(szblck-1))/szblck);
  int loc_blkm = (mbblck + nprow - 1 ) / nprow;
  int loc_blkn = (nbblck + npcol - 1 ) / npcol;
  int loc_m = loc_blkm * _MB;
  int loc_n = loc_blkn * _MB;
 
  EC.resize(AlocM);  // LLIN: Eigenvalues are distributed across all processors
  Vloc.resize(loc_m, loc_n);
  descinit_(descV, &VlocM, &VlocN, &_MB, &_MB, &izero, &izero,
	    &CONTXT, &loc_m, &info); 

  /* Factorization of matrix B */
  pdpotrf_(&uplo, &BlocM, Bloc.data(), &ione,
	   &ione, descB, &info);
  if(mpirank == 0){
    cerr << "pdpotrf info = " << info << endl;
  }
  iC(info);
  if(mpirank == 0){
    cerr << "Finish parallel factorization" << endl;
  }

  /* Form the standard eigenvalue problem, A is overwritten by
   * inv(L) * A * inv(L**T) */
  {
    int    itype   = 1;
    double scale;

    pdsygst_(&itype, &uplo, &AlocM, Aloc.data(), &ione, &ione, descA,  
	     Bloc.data(), &ione, &ione, descB, &scale, &info); 
    iC(info);
    
    if(mpirank == 0){
      cerr << "Finish forming standard eigenvalue problem" << endl;
    }
  }
  
  /* Solve the standard eigenvalue problem */
  {
    char     jobz   = 'V';
    int      lwork, liwork;
    DblNumVec work;
    IntNumVec iwork;

    /* Query for space. PDSYEVD has a bug! */ 
    lwork  = -1;
    liwork = 1;
    work.resize(1);
    iwork.resize(1);

    pdsyevd_(&jobz, &uplo, &AlocM, Aloc.data(), &ione, &ione, descA,
	     EC.data(), Vloc.data(), &ione, &ione, descV, 
	     work.data(), &lwork, iwork.data(), &liwork, &info);

    lwork = ((int)work(0))+2*AlocM+5*_MB*_MB; 
    // LLIN: PDSYEVD has a bug in the calculated LWORK. Even this is not
    // guaranteed to work ...
    liwork = (int)iwork(0);
    work.resize(lwork);
    iwork.resize(liwork);

    if(mpirank == 0){
      cerr << "Finish querying space for standard eigenvalue problem" << endl;
      cerr << "lwork_est = " << lwork << endl;
      cerr << "liwork_est = " << liwork << endl;
    }

    //LLIN: Somehow the query system does not work for pdsyevd
//    int trilwmin = 3*AlocM + (mpisize+3)*_MB;
//    lwork = 2*(max(1+6*AlocM+2*loc_m*loc_m, trilwmin) + 2 * AlocM); // Strange factor of 2
//    liwork = (7 * AlocM + 8 * npcol + 2);                         // Strange factor of 2
//    if(mpirank == 0){
//      cerr << "lwork = " << lwork << endl;
//      cerr << "liwork = " << liwork << endl;
//    }
//    work.resize(lwork);
//    iwork.resize(liwork);

    pdsyevd_(&jobz, &uplo, &AlocM, Aloc.data(), &ione, &ione, descA,
	     EC.data(), Vloc.data(), &ione, &ione, descV, 
	     work.data(), &lwork, iwork.data(), &liwork, &info);
    iC(info);
    
    if(mpirank == 0){
      cerr << "Finish solving standard eigenvalue problem" << endl;
    }
  }
  
  /* Obtain the correct eigenfunctions and save them in V */
  {
    char    side  = 'L';
    char    trans = 'T';
    char    diag  = 'N';
    pdtrsm_(&side, &uplo, &trans, &diag, &BlocM, &BlocN, &one,
	    Bloc.data(), &ione, &ione, descB, Vloc.data(),
	    &ione, &ione, descV);
  }

  t1 = time(0);  
  if(mpirank==0) {
    fprintf(fhstat, "finish solving generalized eigenvalue problem %15.3f secs\n", 
	    difftime(t1,t0));   
    fprintf(stderr, "finish solving generalized eigenvalue problem %15.3f secs\n", 
	    difftime(t1,t0));   
  }

  return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// LLIN: Generate the coefficient matrix for nonorthogonal adaptive
// local basis functions
int EigDG::solve_A_C(ParVec<EmatKey,DblNumMat,EmatPtn>& A, int& AM, int& AN,
		     NumTns< vector<int> >& Aindexvec,
		     ParVec<Index3,DblNumMat,ElemPtn>& C, int& CM, int& CN,
		     NumTns< vector<int> >& Cindexvec,
		     ParVec<Index3,vector<DblNumTns>,ElemPtn>& basesvec){
  Point3 hs = _hs; //ELEMENT SIZE
  Index3 Ns = _Ns; //NUMBER OF ELEMENTS
  Index3 Nlbls = _Nlbls; //NUMBER OF LBL GRID POINTS
  //
  int N1 = Ns(0);  int N2 = Ns(1);  int N3 = Ns(2);
  double h1 = hs(0);  double h2 = hs(1);  double h3 = hs(2);
  int Nlbl1 = Nlbls(0);  int Nlbl2 = Nlbls(1);  int Nlbl3 = Nlbls(2);
  int Nlbltot = Nlbl1*Nlbl2*Nlbl3;
  //
  int mpirank; MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize; MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

  time_t t0, t1, tt0, tt1;
  tt0 = time(0);

  // Determine the number of neighbours to be included for solving the
  // nonorthogonal adaptive local basis functions
  Index3 Nbrs;
  // Shift vectors for locating the neighbors (including itself)
  Index3 NbrSfts;
  for(int i = 0; i < 3; i++){
    Nbrs(i)    = (Ns(i)>=3) ? 3 : 1;
    NbrSfts(i) = (Ns(i)>=3) ? -1 : 0;
  }
  int Nbr1 = Nbrs(0), Nbr2 = Nbrs(1), Nbr3 = Nbrs(2);
  int NbrSft1 = NbrSfts(0), NbrSft2 = NbrSfts(1), NbrSft3 = NbrSfts(2);

  // LLIN: Number of eigenvectors to be computed in the buffer
  if(mpirank == 0){
    cerr << "Nbrs   : " << Nbrs << endl;
    cerr << "NbrSfts:" << NbrSfts << endl;
  }
  int Neigbuf = Nbr1*Nbr2*Nbr3*_Neigperele;

  //compute weights
  DblNumVec x1(Nlbl1);    
  lglnodes(x1.data(), Nlbl1-1);    for(int i=0; i<Nlbl1; i++)      x1(i) = x1(i)*h1/2;
  DblNumVec x2(Nlbl2);    
  lglnodes(x2.data(), Nlbl2-1);    for(int i=0; i<Nlbl2; i++)      x2(i) = x2(i)*h2/2;
  DblNumVec x3(Nlbl3);    
  lglnodes(x3.data(), Nlbl3-1);    for(int i=0; i<Nlbl3; i++)      x3(i) = x3(i)*h3/2;    

  //gi, left->right 0,1,2
  NumTns<DblNumTns> weight(Nbr1,Nbr2,Nbr3);

  //LLIN: FIXME Remember that the minh criterion may fail on very
  //non-isotropic elements
  double minh = min(min(h1, h2), h3);

  //LLIN: compute _basisradius if _delta is specified

  // Old version before input format v1.2
  if(_inputformat == "v1.1" || _inputformat == "v1.0"){
    if( _delta > 0.0 ){
      _basisradius = (1-_delta)*minh*1.5;
    }

    for(int g1=0; g1<Nbr1; g1++)
      for(int g2=0; g2<Nbr2; g2++)
	for(int g3=0; g3<Nbr3; g3++) {
	  DblNumVec tx1(Nlbl1);	  for(int k=0; k<Nlbl1; k++)	    tx1(k) = x1(k) + (g1+NbrSft1)*h1;
	  DblNumVec tx2(Nlbl2);	  for(int k=0; k<Nlbl2; k++)	    tx2(k) = x2(k) + (g2+NbrSft2)*h2;
	  DblNumVec tx3(Nlbl3);	  for(int k=0; k<Nlbl3; k++)	    tx3(k) = x3(k) + (g3+NbrSft3)*h3;
	  DblNumTns tmp(Nlbl1,Nlbl2,Nlbl3);	  setvalue(tmp,0.0);
	  for(int a1=0; a1<Nlbl1; a1++)
	    for(int a2=0; a2<Nlbl2; a2++)
	      for(int a3=0; a3<Nlbl3; a3++) {
		double gg = sqrt(tx1(a1)*tx1(a1) + tx2(a2)*tx2(a2) + tx3(a3)*tx3(a3));
		tmp(a1,a2,a3) = (gg>_basisradius); 
	      }
	  weight(g1,g2,g3) = tmp;
	}
  }

  //LLIN: New format. Rounded rectangular truncation.
  if(_inputformat == "v1.2"){
    //NOTE: The meaning of delta changes here
    if( _delta > 0.0 ){
      _basisradius = (1-_delta)*minh;
    }

    if( mpirank == 0 ){
      fprintf(fhstat, "Basis radius = %15.5f\n", 
	      _basisradius);   
      fprintf(stderr, "Basis radius = %15.5f\n", 
	      _basisradius);   
    }

    for(int g1=0; g1<Nbr1; g1++)
      for(int g2=0; g2<Nbr2; g2++)
	for(int g3=0; g3<Nbr3; g3++) {
	  DblNumVec tx1(Nlbl1);	  
	  for(int k=0; k<Nlbl1; k++){
	    tx1(k) = abs(x1(k) + (g1+NbrSft1)*h1) - h1 / 2.0;
	    tx1(k) = (tx1(k) > 0.0) ? tx1(k) : 0.0;
	  }
	  
	  DblNumVec tx2(Nlbl2);	  
	  for(int k=0; k<Nlbl2; k++){
	    tx2(k) = abs(x2(k) + (g2+NbrSft2)*h2) - h2 / 2.0;
	    tx2(k) = (tx2(k) > 0.0) ? tx2(k) : 0.0;
	  }

	  DblNumVec tx3(Nlbl3);	  
	  for(int k=0; k<Nlbl3; k++){
	    tx3(k) = abs(x3(k) + (g3+NbrSft3)*h3) - h3 / 2.0;
	    tx3(k) = (tx3(k) > 0.0) ? tx3(k) : 0.0;
	  }

	  DblNumTns tmp(Nlbl1,Nlbl2,Nlbl3);	  setvalue(tmp,0.0);
	  for(int a1=0; a1<Nlbl1; a1++)
	    for(int a2=0; a2<Nlbl2; a2++)
	      for(int a3=0; a3<Nlbl3; a3++) {
		// L^2 norm
		double gg = sqrt(tx1(a1)*tx1(a1) + tx2(a2)*tx2(a2) + tx3(a3)*tx3(a3));
		tmp(a1,a2,a3) = (gg>_basisradius); 
	      }
	  weight(g1,g2,g3) = tmp;
	}
  }

  //LEXING: FOR THE TIME BEING, ONE ELEMENT PER PROCESSOR, CURRENT ELEMENT is (cur1,cur2,cur3)
  //FIND THE CURRENT ELEMENT
  int cur1=0,cur2=0,cur3=0; // LLIN: Be careful with the name of the indices
  for(int t3=0; t3<N3; t3++)
    for(int t2=0; t2<N2; t2++)
      for(int t1=0; t1<N1; t1++) {
	Index3 curkey = Index3(t1,t2,t3);
	if(_elemptn.owner(curkey)==mpirank) {
	  cur1 = t1;	    cur2 = t2;	    cur3 = t3;
	}
      }

  //LEXING: aux1,2,3, stores the neighbor in three dimensions, for
  //example aux1 = (cur1-1,cur1,cur1+1) if Nbr=3
  vector<int> aux1;    for(int g=0; g<Nbr1; g++)      aux1.push_back( (cur1+g+NbrSft1 + N1)%N1 );
  vector<int> aux2;    for(int g=0; g<Nbr2; g++)      aux2.push_back( (cur2+g+NbrSft2 + N2)%N2 );
  vector<int> aux3;    for(int g=0; g<Nbr3; g++)      aux3.push_back( (cur3+g+NbrSft3 + N3)%N3 );

  //get collect the basis funcs for (cur1,cur2,cur3)'s neighbor
  vector<int> mask(1,1);
  iC( basesvec.getBegin(_nbhdvec, mask) );
  iC( basesvec.getEnd(mask) );
  if(mpirank == 0)
    cerr << "basesvec comminucatation finished" << endl;

  NumTns< vector<DblNumTns> > basis_aux(3,3,3);
  for(int g1=0; g1<Nbr1; g1++)
    for(int g2=0; g2<Nbr2; g2++)
      for(int g3=0; g3<Nbr3; g3++) {
	Index3 gidx(aux1[g1],aux2[g2],aux3[g3]);
	basis_aux(g1,g2,g3) = basesvec.lclmap()[gidx];
      }

  //get the submatrix with Dirichelt boundary condition, and form it
  //into a DblNumMat
  vector<EmatKey> keyvec;
  for(int g1=0; g1<Nbr1; g1++)
    for(int g2=0; g2<Nbr2; g2++)
      for(int g3=0; g3<Nbr3; g3++) {
	Index3 gidx(aux1[g1],aux2[g2],aux3[g3]);
	for(int t1=0; t1<Nbr1; t1++) {
	  for(int t2=0; t2<Nbr2; t2++)
	    for(int t3=0; t3<Nbr3; t3++) {
	      Index3 tidx(aux1[t1],aux2[t2],aux3[t3]);
	      keyvec.push_back( EmatKey(gidx,tidx) );
	    }
	}
      }
  iC( A.getBegin(keyvec, mask) );
  iC( A.getEnd(mask) );
  if(mpirank == 0)
    cerr << "A comminucatation finished" << endl;

  //LEXING:  ttl is the number of DG basis functions for the local problem
  int ttl = 0;
  IntNumTns sz_aux(Nbr1,Nbr2,Nbr3);
  IntNumTns of_aux(Nbr1,Nbr2,Nbr3);
  for(int g1=0; g1<Nbr1; g1++)
    for(int g2=0; g2<Nbr2; g2++)
      for(int g3=0; g3<Nbr3; g3++) {
	of_aux(g1,g2,g3) = ttl;
	sz_aux(g1,g2,g3) = Aindexvec(aux1[g1],aux2[g2],aux3[g3]).size();
	ttl = ttl + sz_aux(g1,g2,g3);
      }

  //copy matrix
  DblNumMat At(ttl,ttl);    setvalue(At,0.0);
  for(int g1=0; g1<Nbr1; g1++)
    for(int g2=0; g2<Nbr2; g2++)
      for(int g3=0; g3<Nbr3; g3++) {
	Index3 gidx(aux1[g1],aux2[g2],aux3[g3]);
	for(int t1=0; t1<Nbr1; t1++)
	  for(int t2=0; t2<Nbr2; t2++)
	    for(int t3=0; t3<Nbr3; t3++) {
	      Index3 tidx(aux1[t1],aux2[t2],aux3[t3]);
	      int gof = of_aux(g1,g2,g3);
	      int tof = of_aux(t1,t2,t3);
	      DblNumMat& S = A.lclmap()[EmatKey(gidx,tidx)];
	      for(int a=0; a<S.m(); a++)
		for(int b=0; b<S.n(); b++)
		  At(gof+a,tof+b) += S(a,b);
	    }
      }

  tt1 = time(0);
  if(mpirank==0) { 
    fprintf(fhstat, "Preparation for forming nonorthogonal basis %15.3f secs\n", 
	    difftime(tt1,tt0));   
    fprintf(stderr, "Preparation for forming nonorthogonal basis %15.3f secs\n", 
	    difftime(tt1,tt0));   
  }

  //PERFORM EIG on matrix At
  DblNumMat Vt(ttl,ttl);
  DblNumVec Et(ttl);
  //LLIN: Diagonalize  At to get Vt and Et using dsyevd
  {
    t0 = time(0);

    int     Nmat    = ttl;
    char    jobz    = 'V';
    char    uplo    = 'L';
    int     lda     = Nmat;
    int     lwork, liwork;
    int     info;
    DblNumVec      work;
    IntNumVec      iwork;

    Vt = At; // LLIN: IMPORTANT: dsyevd rewrites the input matrix by
    // the output eigenvectors

    /* Query for space */
    lwork = -1;
    liwork = -1;
    work.resize(1);
    iwork.resize(1);
    dsyevd_(&jobz, &uplo, &Nmat, Vt.data(), &lda, Et.data(), work.data(), 
	    &lwork, iwork.data(), &liwork, &info);

    lwork = (int)work(0); // LLIN: IMPORTANT
    liwork = (int)iwork(0);
    work.resize(lwork);
    iwork.resize(liwork);

    dsyevd_(&jobz, &uplo, &Nmat, Vt.data(), &lda, Et.data(), work.data(), 
	    &lwork, iwork.data(), &liwork, &info);
    iC(info);

    t1 = time(0);
    if(mpirank==0) { 
      fprintf(fhstat, "Standard eigenvalue problem first %15.3f secs\n", 
	      difftime(t1,t0));   
      fprintf(stderr, "Standard eigenvalue problem first %15.3f secs\n", 
	      difftime(t1,t0));   
    }
  }

  //LEXING: Ct contains first Neigbuf columns of Vt
  DblNumMat Ct(ttl,Neigbuf);
  for(int b=0; b<Neigbuf; b++)
    for(int a=0; a<ttl; a++)
      Ct(a,b) = Vt(a,b);

  //form the www weights
  DblNumTns www(Nlbl1,Nlbl2,Nlbl3);
  {
    DblNumVec x1(Nlbl1), w1(Nlbl1);
    lglnodes(x1.data(), w1.data(), Nlbl1-1);
    for(int g=0; g<x1.m(); g++)    x1(g) = x1(g)/2.0*h1;
    for(int g=0; g<w1.m(); g++)    w1(g) = w1(g)/2.0*h1;
    DblNumVec x2(Nlbl2), w2(Nlbl2);
    lglnodes(x2.data(), w2.data(), Nlbl2-1);
    for(int g=0; g<x2.m(); g++)    x2(g) = x2(g)/2.0*h2;
    for(int g=0; g<w2.m(); g++)    w2(g) = w2(g)/2.0*h2;
    DblNumVec x3(Nlbl3), w3(Nlbl3);
    lglnodes(x3.data(), w3.data(), Nlbl3-1);
    for(int g=0; g<x3.m(); g++)    x3(g) = x3(g)/2.0*h3;
    for(int g=0; g<w3.m(); g++)    w3(g) = w3(g)/2.0*h3;
    for(int g1=0; g1<Nlbl1; g1++)
      for(int g2=0; g2<Nlbl2; g2++)
	for(int g3=0; g3<Nlbl3; g3++) {
	  www(g1,g2,g3) = w1(g1)*w2(g2)*w3(g3);
	}
  }
  DblNumMat Wt(ttl,ttl);    setvalue(Wt,0.0);
  for(int g1=0; g1<Nbr1; g1++)
    for(int g2=0; g2<Nbr2; g2++)
      for(int g3=0; g3<Nbr3; g3++) {
	Index3 gidx(aux1[g1],aux2[g2],aux3[g3]);
	int sznow = sz_aux(g1,g2,g3);
	DblNumMat S(sznow,sznow);
	vector<DblNumTns>& basistmp = basis_aux(g1,g2,g3); iA(sznow==basistmp.size());
	for(int a=0; a<sznow; a++)
	  for(int b=0; b<sznow; b++) {
	    S(a,b) = FPS(basistmp[a].data(),basistmp[b].data(),weight(g1,g2,g3).data(),www.data(),Nlbltot);
	  }
	int gof = of_aux(g1,g2,g3);
	for(int a=0; a<S.m(); a++)
	  for(int b=0; b<S.n(); b++)
	    Wt(gof+a,gof+b) += S(a,b);
      }


  // if _gamma == 0.0 (default) then 
  // Estimate the eigenvalues of Wt and At to determine the value of
  // gamma (WeightRatio) automatically so that the change of Wt and the
  // change of At is in the same order of magnitude
  if( _gamma == 0.0 ){
    double sumWt, sumAt;
    
    // Wt
    {
      DblNumMat Bt(ttl,ttl);
      for(int a=0; a<ttl; a++)
	for(int b=0; b<ttl; b++)
	  Bt(a,b) = Wt(a,b);

      DblNumMat Cttran(Neigbuf,ttl);
      for(int a=0; a<Neigbuf; a++)
	for(int b=0; b<ttl; b++)
	  Cttran(a,b) = Ct(b,a);
      DblNumMat Aux(Neigbuf,ttl);         setvalue(Aux,0.0);
      iC( dgemm(1.0, Cttran, Bt, 0.0, Aux) );
      DblNumMat CWgAC(Neigbuf,Neigbuf);  setvalue(CWgAC, 0.0);
      iC( dgemm(1.0, Aux, Ct, 0.0, CWgAC) );

      DblNumMat Gt(Neigbuf,Neigbuf);
      DblNumVec Ft(Neigbuf);
      //LLIN: Diagonalize  CWgAC to get Gt and Ft using dsyevd
      {
	int     Nmat    = Neigbuf;
	char    jobz    = 'V';
	char    uplo    = 'L';
	int     lda     = Nmat;
	int     lwork, liwork;
	int     info;
	DblNumVec      work;
	IntNumVec      iwork;

	Gt = CWgAC; // LLIN: IMPORTANT: dsyevd rewrites the input matrix by
	// the output eigenvectors


	/* Query for space */
	lwork = -1;
	liwork = -1;
	work.resize(1);
	iwork.resize(1);
	dsyevd_(&jobz, &uplo, &Nmat, Gt.data(), &lda, Ft.data(), work.data(), 
		&lwork, iwork.data(), &liwork, &info);

	lwork = (int)work(0); // LLIN: IMPORTANT
	liwork = (int)iwork(0);
	work.resize(lwork);
	iwork.resize(liwork);

	dsyevd_(&jobz, &uplo, &Nmat, Gt.data(), &lda, Ft.data(), work.data(), 
		&lwork, iwork.data(), &liwork, &info);

	iC(info);

	sumWt = 0.0;
	for(int l = 0; l < _Norbperele; l++){
	  sumWt += Ft(l) - Ft(0);
	}

	if(mpirank==0) { 
	  fprintf(fhstat, "sum of _Norbperele eigvals for Wt - eig(Wt)[0] = %15.5e\n",
		  sumWt);
	  fprintf(stderr, "sum of _Norbperele eigvals for Wt - eig(Wt)[0] = %15.5e\n",
		  sumWt);
	}
      }
    } // Wt
    
    // At
    {
      DblNumMat Bt(ttl,ttl);
      for(int a=0; a<ttl; a++)
	for(int b=0; b<ttl; b++)
	  Bt(a,b) = At(a,b);

      DblNumMat Cttran(Neigbuf,ttl);
      for(int a=0; a<Neigbuf; a++)
	for(int b=0; b<ttl; b++)
	  Cttran(a,b) = Ct(b,a);
      DblNumMat Aux(Neigbuf,ttl);         setvalue(Aux,0.0);
      iC( dgemm(1.0, Cttran, Bt, 0.0, Aux) );
      DblNumMat CWgAC(Neigbuf,Neigbuf);  setvalue(CWgAC, 0.0);
      iC( dgemm(1.0, Aux, Ct, 0.0, CWgAC) );

      DblNumMat Gt(Neigbuf,Neigbuf);
      DblNumVec Ft(Neigbuf);
      //LLIN: Diagonalize  CWgAC to get Gt and Ft using dsyevd
      {
	int     Nmat    = Neigbuf;
	char    jobz    = 'V';
	char    uplo    = 'L';
	int     lda     = Nmat;
	int     lwork, liwork;
	int     info;
	DblNumVec      work;
	IntNumVec      iwork;

	Gt = CWgAC; // LLIN: IMPORTANT: dsyevd rewrites the input matrix by
	// the output eigenvectors


	/* Query for space */
	lwork = -1;
	liwork = -1;
	work.resize(1);
	iwork.resize(1);
	dsyevd_(&jobz, &uplo, &Nmat, Gt.data(), &lda, Ft.data(), work.data(), 
		&lwork, iwork.data(), &liwork, &info);

	lwork = (int)work(0); // LLIN: IMPORTANT
	liwork = (int)iwork(0);
	work.resize(lwork);
	iwork.resize(liwork);

	dsyevd_(&jobz, &uplo, &Nmat, Gt.data(), &lda, Ft.data(), work.data(), 
		&lwork, iwork.data(), &liwork, &info);

	iC(info);

	sumAt = 0.0;
	for(int l = 0; l < _Norbperele; l++){
	  sumAt += (Ft(l) - Ft(0));
	}

	if(mpirank==0) { 
	  fprintf(fhstat, "sum of _Norbperele eigvals for At - eig(At)[0]= %15.5e\n",
		  sumAt);
	  fprintf(stderr, "sum of _Norbperele eigvals for At - eig(At)[0]= %15.5e\n", 
		  sumAt); 
	} 
      }
    } // At

    double EPS = 1e-8; // LLIN:FIXME
    if( sumAt > EPS ){
      _gamma = sumWt / sumAt;
    }
    else{
      _gamma = 1.0;
    }
    
  }

  if(mpirank==0) { 
    fprintf(stderr, "gamma = %15.5e\n", _gamma);
    fprintf(fhstat, "gamma = %15.5e\n", _gamma);
  } 

  //form matrix Ct' * (Wt+gamma*At) * Ct;
  //double gamma = 0.1;
  //LEXING: Bt = Wt + gamma*At;
  DblNumMat Bt(ttl,ttl);
  for(int a=0; a<ttl; a++)
    for(int b=0; b<ttl; b++)
      Bt(a,b) = Wt(a,b) + _gamma * At(a,b);

  DblNumMat Cttran(Neigbuf,ttl);
  for(int a=0; a<Neigbuf; a++)
    for(int b=0; b<ttl; b++)
      Cttran(a,b) = Ct(b,a);
  DblNumMat Aux(Neigbuf,ttl);         setvalue(Aux,0.0);
  iC( dgemm(1.0, Cttran, Bt, 0.0, Aux) );
  DblNumMat CWgAC(Neigbuf,Neigbuf);  setvalue(CWgAC, 0.0);
  iC( dgemm(1.0, Aux, Ct, 0.0, CWgAC) );

  DblNumMat Gt(Neigbuf,Neigbuf);
  DblNumVec Ft(Neigbuf);
  //LLIN: Diagonalize  CWgAC to get Gt and Ft using dsyevd
  {
    time_t  t0, t1;
    t0 = time(0);

    int     Nmat    = Neigbuf;
    char    jobz    = 'V';
    char    uplo    = 'L';
    int     lda     = Nmat;
    int     lwork, liwork;
    int     info;
    DblNumVec      work;
    IntNumVec      iwork;

    Gt = CWgAC; // LLIN: IMPORTANT: dsyevd rewrites the input matrix by
    // the output eigenvectors


    /* Query for space */
    lwork = -1;
    liwork = -1;
    work.resize(1);
    iwork.resize(1);
    dsyevd_(&jobz, &uplo, &Nmat, Gt.data(), &lda, Ft.data(), work.data(), 
	    &lwork, iwork.data(), &liwork, &info);

    lwork = (int)work(0); // LLIN: IMPORTANT
    liwork = (int)iwork(0);
    work.resize(lwork);
    iwork.resize(liwork);

    dsyevd_(&jobz, &uplo, &Nmat, Gt.data(), &lda, Ft.data(), work.data(), 
	    &lwork, iwork.data(), &liwork, &info);

    iC(info);

    t1 = time(0);
    if(mpirank==0) { 
      fprintf(fhstat, "Standard eigenvalue problem second %15.3f secs\n", 
	      difftime(t1,t0));   
      fprintf(stderr, "Standard eigenvalue problem second %15.3f secs\n", 
	      difftime(t1,t0));   
    }
  }

  //LEXING: the first few columns of Gt form Ht
  DblNumMat Ht(Neigbuf,_Norbperele);
  for(int a=0; a<Neigbuf; a++)
    for(int b=0; b<_Norbperele; b++)
      Ht(a,b) = Gt(a,b);

  //form Tt of size Ndof by Norbperele
  DblNumMat CtHt(ttl,_Norbperele);    setvalue(CtHt,0.0);
  iC( dgemm(1.0, Ct, Ht, 0.0, CtHt) );

  int Ndof = AM;
  DblNumMat Tt(Ndof,_Norbperele);  setvalue(Tt, 0.0); //LLIN: IMPORTANT!
  for(int g1=0; g1<Nbr1; g1++)
    for(int g2=0; g2<Nbr2; g2++)
      for(int g3=0; g3<Nbr3; g3++) {
	vector<int>& indextmp = Aindexvec(aux1[g1],aux2[g2],aux3[g3]);
	int oftmp = of_aux(g1,g2,g3);
	int sztmp = sz_aux(g1,g2,g3);
	for(int a=0; a<sztmp; a++)
	  for(int b=0; b<_Norbperele; b++)
	    Tt(indextmp[a], b) = CtHt(oftmp+a, b);
      }

  //LEXING: Put Tt into a single matrix C (the whole compression matrix)
  //This should be changed later when _Norbperele is vector<Index3>
  //C is a parallelly stored matrix of size Ndof by Norbttl
  int Norbttl = N1*N2*N3 * _Norbperele;

  CM = Ndof;
  CN = Norbttl;
  C.prtn() = _elemptn;
  C.lclmap()[Index3(cur1,cur2,cur3)] = Tt;
  Cindexvec.resize(N1,N2,N3);
  {
    int cnt = 0;
    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  vector<int> vtmp;
	  for(int g=0; g<_Norbperele; g++) {
	    vtmp.push_back(cnt);
	    cnt++;
	  }
	  Cindexvec(i1,i2,i3) = vtmp;
	}
    iA(cnt == Norbttl);
  }

  tt1 = time(0);
  if(mpirank==0) { 
    fprintf(fhstat, "Form C %15.3f secs\n", 
	    difftime(tt1,tt0));   
    fprintf(stderr, "Form C %15.3f secs\n", 
	    difftime(tt1,tt0));   
  }

  return 0;
}


