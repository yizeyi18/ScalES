#include "eigdg.hpp"
#include "interp.hpp"

extern FILE* fhstat;

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
int EigDG::solve_Elem_A(ParVec<Index3,vector<double>,ElemPtn>& vtotvec, ParVec<Index3,vector<DblNumTns>,ElemPtn>& basesvec, ParVec<int,Psdo,PsdoPtn>& psdovec,
			ParVec<EmatKey,DblNumMat,EmatPtn>& A, int& AM, int& AN, NumTns< vector<int> >& Aindexvec)
{
  //LEXING: form the stiffness matrix A, and put the data into the right location
  //
  Point3 hs = _hs; //ELEMENT SIZE
  Index3 Ns = _Ns; //NUMBER OF ELEMENTS
  Index3 Nlbls = _Nlbls; //NUMBER OF LBL GRID POINTS
  int N1 = Ns(0);  int N2 = Ns(1);  int N3 = Ns(2);
  double h1 = hs(0);  double h2 = hs(1);  double h3 = hs(2);
  int Nlbl1 = Nlbls(0);  int Nlbl2 = Nlbls(1);  int Nlbl3 = Nlbls(2);
  int Nlbltot = Nlbl1*Nlbl2*Nlbl3;
  //
  int mpirank; MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize; MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  int myid = mpirank;
  //
  time_t t0, t1, tt0, tt1;
  //
  //--------------------------------------------------------------------------
  //setup the size of A, AM, AN, indexvec
  IntNumTns aux(N1,N2,N3);    setvalue(aux,0);
  for(int i3=0; i3<N3; i3++)
    for(int i2=0; i2<N2; i2++)
      for(int i1=0; i1<N1; i1++) {
	Index3 curidx(i1,i2,i3);
	if(_elemptn.owner(curidx)==myid) {
	  vector<DblNumTns>& curbases = basesvec.lclmap()[curidx];
	  aux(i1,i2,i3) = curbases.size();
	}
      }
  IntNumTns cmb(N1,N2,N3);    setvalue(cmb,0); //cmb stores the number of dofs in each element
  iC( MPI_Allreduce(aux.data(), cmb.data(), N1*N2*N3, MPI_INT, MPI_SUM, MPI_COMM_WORLD) );
  //every processor compute all index sets
  Aindexvec.resize(N1,N2,N3);
  vector<int> owneraux;  owneraux.resize(0);
  int cnt=0;
  for(int i3=0; i3<N3; i3++)
    for(int i2=0; i2<N2; i2++)
      for(int i1=0; i1<N1; i1++) {
	Index3 curkey = Index3(i1,i2,i3);
	int ownerproc = _elemptn.owner(curkey);
	int ntmp = cmb(i1,i2,i3);
	vector<int> vtmp;
	for(int g=0; g<ntmp; g++) {
	  vtmp.push_back(cnt);	  //LEXING: the following line set ownerproc
	  owneraux.push_back(ownerproc);	  //owneraux[ cnt ] = ownerproc;
	  cnt++;
	}
	//add index
	Aindexvec(i1,i2,i3) = vtmp;
      }
  int Ndof = cnt;
  //
  A.prtn() = _ematptn;  //AM = _Ndof;  //AN = _Ndof;
  //LEXING: MAKE IT EMPTY FOR THE TIME BEING
  for(int i3=0; i3<N3; i3++)
    for(int i2=0; i2<N2; i2++)
      for(int i1=0; i1<N1; i1++) {
	Index3 ikey = Index3(i1,i2,i3);
	int inum = Aindexvec(i1,i2,i3).size();
	for(int j3=0; j3<N3; j3++)
	  for(int j2=0; j2<N2; j2++)
	    for(int j1=0; j1<N1; j1++) {
	      Index3 jkey = Index3(j1,j2,j3);
	      int jnum = Aindexvec(j1,j2,j3).size();
	      EmatKey emk(ikey,jkey);
	      if(_ematptn.owner(emk)==mpirank) {
		DblNumMat zero(inum,jnum);
		setvalue(zero, 0.0);
		A.lclmap()[emk] = zero;
	      }
	    }
      }
  AM = Ndof;
  AN = Ndof;
  if(mpirank == 0){
    fprintf(stderr, "Finish setting size\n"); 
  }
  //
  //-------------------------------------------------------------------------------------------------------------------
  //SETUP STAGE
  //-------------------------------------------------------------------------------------------------------------------
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
  DblNumTns www(Nlbl1,Nlbl2,Nlbl3);
  for(int g1=0; g1<Nlbl1; g1++)
    for(int g2=0; g2<Nlbl2; g2++)
      for(int g3=0; g3<Nlbl3; g3++) {
	www(g1,g2,g3) = w1(g1)*w2(g2)*w3(g3);
      }
  DblNumTns sqrtwww(Nlbl1,Nlbl2,Nlbl3);
  for(int g1=0; g1<Nlbl1; g1++)
    for(int g2=0; g2<Nlbl2; g2++)
      for(int g3=0; g3<Nlbl3; g3++) {
	sqrtwww(g1,g2,g3) = sqrt(www(g1,g2,g3));
      }
  DblNumTns recsqrtwww(Nlbl1,Nlbl2,Nlbl3);
  for(int g1=0; g1<Nlbl1; g1++)
    for(int g2=0; g2<Nlbl2; g2++)
      for(int g3=0; g3<Nlbl3; g3++) {
	recsqrtwww(g1,g2,g3) = 1.0/sqrt(www(g1,g2,g3));
      }
  DblNumTns xx1(Nlbl1,Nlbl2,Nlbl3);  DblNumTns xx2(Nlbl1,Nlbl2,Nlbl3);  DblNumTns xx3(Nlbl1,Nlbl2,Nlbl3);
  for(int g1=0; g1<Nlbl1; g1++)
    for(int g2=0; g2<Nlbl2; g2++)
      for(int g3=0; g3<Nlbl3; g3++) {
	xx1(g1,g2,g3) = x1(g1);	  xx2(g1,g2,g3) = x2(g2);	  xx3(g1,g2,g3) = x3(g3);
      }
  if(mpirank == 0){
    fprintf(stderr, "Finish setting grid\n"); 
  }
  
  //----------------------------------------------------------------------------------------------------------------  
  //start communication
  {
    t0 = time(0);
    if(mpirank == 0){
      fprintf(stderr, "before union\n"); 
    }


    //LL: set_union needs pre-allocated space
    vector<Index3> tmpvec;
    tmpvec.resize(_jumpvec.size() + _psdovec.size());
    vector<Index3>::iterator tmpend = set_union(_jumpvec.begin(), _jumpvec.end(), _psdovec.begin(), _psdovec.end(), tmpvec.begin());
    vector<Index3> keyvec;
    keyvec.insert(keyvec.begin(), tmpvec.begin(), tmpend);
    
    vector<int> mask(1,1);
    iC( basesvec.getBegin(keyvec, mask) );
  if(mpirank == 0){
    fprintf(stderr, "end begin\n"); 
  }
    t1 = time(0);
    if(mpirank==0) { 
      fprintf(fhstat, "communication cost %15.3f secs\n", difftime(t1,t0));   
      fprintf(stderr, "communication cost %15.3f secs\n", difftime(t1,t0));   
    }
  }
  
  //----------------------------------------------------------------------------------------------------------------  
  //inner product
  t0 = time(0);
  if(1) {
    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if(_elemptn.owner(curkey)==mpirank) {
	    //
	    vector<DblNumTns>& bases = basesvec.lclmap()[curkey];
	    vector<DblNumTns>  derxs;	derxs.resize(bases.size());
	    vector<DblNumTns>  derys;	derys.resize(bases.size());
	    vector<DblNumTns>  derzs;	derzs.resize(bases.size());
	    for(int g=0; g<bases.size(); g++) {
	      derxs[g].resize(Nlbl1, Nlbl2, Nlbl3);
	      derys[g].resize(Nlbl1, Nlbl2, Nlbl3);
	      derzs[g].resize(Nlbl1, Nlbl2, Nlbl3);
	      DiffPsi(Nlbls, hs, bases[g].data(), derxs[g].data(), derys[g].data(), derzs[g].data() );
	    }
	    int nbases = bases.size();
	    DblNumMat S(nbases,nbases);
	    for(int a=0; a<nbases; a++)
	      for(int b=0; b<nbases; b++) {
		DblNumTns& basesaDX = derxs[a];
		DblNumTns& basesaDY = derys[a];
		DblNumTns& basesaDZ = derzs[a];
		DblNumTns& basesbDX = derxs[b];
		DblNumTns& basesbDY = derys[b];
		DblNumTns& basesbDZ = derzs[b];
		//LY:
		S(a,b) = 0.5*(TPS(basesaDX.data(),basesbDX.data(),www.data(),Nlbltot) +
			      TPS(basesaDY.data(),basesbDY.data(),www.data(),Nlbltot) +
			      TPS(basesaDZ.data(),basesbDZ.data(),www.data(),Nlbltot));
	      }
	    EmatKey emk(curkey,curkey);
	    map<EmatKey,DblNumMat>::iterator mi = A.lclmap().find(emk);
	    if(mi==A.lclmap().end()) {
	      A.lclmap()[emk] = S;
	    } else {
	      DblNumMat& tmp = (*mi).second;
	      for(int a=0; a<S.m(); a++)	      for(int b=0; b<S.n(); b++)		tmp(a,b)+=S(a,b);
	    }
	  }
	}
  }
  t1 = time(0);  
  if(mpirank==0) { 
    fprintf(fhstat, "gradient inner product %15.3f secs\n", difftime(t1,t0));   
    fprintf(stderr, "gradient inner product %15.3f secs\n", difftime(t1,t0));   
  }
  
  //----------------------------------------------------------------------------------------------------------------  
  //potential
  t0 = time(0);  
  if(1) {
    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if(_elemptn.owner(curkey)==mpirank) {
	    vector<DblNumTns>& bases = basesvec.lclmap()[curkey];
	    int nbases = bases.size();
	    DblNumTns vt(Nlbl1,Nlbl2,Nlbl3,false, &(vtotvec.lclmap()[curkey][0]));	  //DblNumTns vt(Nlbl1,Nlbl2,Nlbl3,false, &(curdat.vtot()[0]));
	    DblNumMat S(nbases,nbases);
	    for(int a=0; a<nbases; a++)
	      for(int b=0; b<nbases; b++) {
		DblNumTns& basesaVL = bases[a];
		DblNumTns& basesbVL = bases[b];
		//LY
		//memcpy(&tmpvec[0], basesaVL.data(), sizeof(double)*Nlbltot);
		//XScaleByY(&tmpvec[0], vt.data(), Nlbltot);
		//S(a,b) = ddot_(&Nlbltot, &tmpvec[0], &INTONE, basesbVL.data(), &INTONE);
		S(a,b) = FPS(basesaVL.data(),basesbVL.data(),vt.data(),www.data(),Nlbltot);
	      }
	    EmatKey emk(curkey,curkey);
	    map<EmatKey,DblNumMat>::iterator mi = A.lclmap().find(emk);
	    if(mi==A.lclmap().end()) {
	      A.lclmap()[emk] = S;
	    } else {
	      DblNumMat& tmp = (*mi).second;
	      for(int a=0; a<S.m(); a++)	      for(int b=0; b<S.n(); b++)		tmp(a,b)+=S(a,b);
	    }
	  }
	}
  }
  t1 = time(0);  
  if(mpirank==0) { 
    fprintf(fhstat, "local potential %15.3f secs\n", difftime(t1,t0));   
    fprintf(stderr, "local potential %15.3f secs\n", difftime(t1,t0));   
  }
  
  //----------------------------------------------------------------------------------------------------------------  
  //end communication
  {
    t0 = time(0);
    vector<int> mask(1,1);
    iC( basesvec.getEnd(mask) );
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = time(0);
    if(mpirank==0) { 
      fprintf(fhstat, "extra communication cost %15.3f secs\n", difftime(t1,t0));   
      fprintf(stderr, "extra communication cost %15.3f secs\n", difftime(t1,t0));   
    }
  }
  
  //----------------------------------------------------------------------------------------------------------------  
  //nonlocal pseudopotential
  t0 = time(0);
  if(1) {
    //vector<double> tmpvec;    tmpvec.resize(Nlbltot);
    for(map<int,Psdo>::iterator mi=psdovec.lclmap().begin(); mi!=psdovec.lclmap().end(); mi++) {
      int  curkey = (*mi).first;
      Psdo& curdat = (*mi).second;
      if(psdovec.prtn().owner(curkey)==mpirank) {
	vector< pair<NumTns<SparseVec>,double> >& vnls = curdat.vnls();
	for(int g=0; g<vnls.size(); g++) {
	  NumTns<SparseVec>& ppcur = vnls[g].first;
	  double ppsgn = vnls[g].second;	//LLIN: ppsgn changed to double type for consistency.
	  //compute inner product
	  NumTns< vector<double> > inner(N1,N2,N3);
	  for(int i3=0; i3<N3; i3++)
	    for(int i2=0; i2<N2; i2++)
	      for(int i1=0; i1<N1; i1++) {
		if( ppcur(i1,i2,i3).first.m()>0) {
		  //map<Index3,Elem>::iterator ei = basesvec.lclmap().find(Index3(i1,i2,i3)); iA(ei!=basesvec.lclmap().end());
		  vector<DblNumTns>& bases = basesvec.lclmap()[Index3(i1,i2,i3)];
		  int nbases = bases.size();
		  vector<double> mid(nbases, 0.0);
		  //LY:
		  for(int a=0; a<nbases; a++) {
		    //mid[a] = TPS(bases[a].data(),ppcur(i1,i2,i3).data(),www.data(),Nlbltot);
		    mid[a] = 0.0;
		    IntNumVec& tmpi = ppcur(i1,i2,i3).first;
		    DblNumMat& tmpv = ppcur(i1,i2,i3).second;
		    int VL=0, DX=1, DY=2, DZ=3;
		    for(int g=0; g<tmpi.m(); g++) {
		      mid[a] += (*(bases[a].data()+tmpi[g])) * tmpv(VL,g) * (*(www.data()+tmpi[g]));
		    }
		  }
		  inner(i1,i2,i3)=mid;
		}
	      }
	  //double loop
	  for(int u3=0; u3<N3; u3++)
	    for(int u2=0; u2<N2; u2++)
	      for(int u1=0; u1<N1; u1++) {
		if( ppcur(u1,u2,u3).first.m()>0) {		//if( !ppcur(i1,i2,i3).isempty() ) {
		  vector<int>& uindex = Aindexvec(u1,u2,u3);
		  int nubases = uindex.size();
		  vector<double>& miinner = inner(u1,u2,u3);
		  
		  for(int v3=0; v3<N3; v3++)
		    for(int v2=0; v2<N2; v2++)
		      for(int v1=0; v1<N1; v1++) {
			if( ppcur(v1,v2,v3).first.m()>0) {		//if( !ppcur(v1,v2,v3).isempty() ) {
			  vector<int>& vindex = Aindexvec(v1,v2,v3);
			  int nvbases = vindex.size();
			  vector<double>& niinner = inner(v1,v2,v3);
			  
			  DblNumMat S(nubases,nvbases);
			  for(int a=0; a<nubases; a++)
			    for(int b=0; b<nvbases; b++) {
			      double tmp1 = miinner[a];
			      double tmp2 = niinner[b];
			      S(a,b) = ppsgn*tmp1*tmp2;
			    }
			  //put back
			  Index3 ukey = Index3(u1,u2,u3);
			  Index3 vkey = Index3(v1,v2,v3);
			  EmatKey emk(ukey,vkey);
			  map<EmatKey,DblNumMat>::iterator mi = A.lclmap().find(emk);
			  if(mi==A.lclmap().end()) {
			    A.lclmap()[emk] = S;
			  } else {
			    DblNumMat& tmp = (*mi).second;
			    for(int a=0; a<S.m(); a++)	      for(int b=0; b<S.n(); b++)		tmp(a,b)+=S(a,b);
			  }
			}
		      }
		}
	      } //mpirank check
	}
      }
    }
  }
  t1 = time(0);  
  if(mpirank==0) { 
    fprintf(fhstat, "stiff matrix nonlocal pseudopotential %15.3f secs\n", difftime(t1,t0));   
    fprintf(stderr, "stiff matrix nonlocal pseudopotential %15.3f secs\n", difftime(t1,t0));   
  }
  
  //----------------------------------------------------------------------------------------------------------------  
  //Boundary terms
  t0 = time(0);  
  //yz face
  if(1) {
    DblNumMat ww(Nlbl2,Nlbl3);
    for(int g2=0; g2<Nlbl2; g2++)
      for(int g3=0; g3<Nlbl3; g3++) {
	ww(g2,g3) = w2(g2)*w3(g3);
      }
    int Nfacesize = Nlbl2*Nlbl3;
    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if(_elemptn.owner(curkey)==mpirank) {
	    int p1;	  if(i1==0)	    p1 = N1-1;	  else	    p1 = i1-1;
	    for(int uch=0; uch<=1; uch++)
	      for(int vch=0; vch<=1; vch++) {
		//-------------
		int u1,u2,u3;
		if(uch==0) {
		  u1=p1;		u2=i2;		u3=i3;
		} else {
		  u1=i1;		u2=i2;		u3=i3;
		}
		vector<DblNumTns>& ubases = basesvec.lclmap()[Index3(u1,u2,u3)];
		vector<DblNumTns>  uderxs;	uderxs.resize(ubases.size());
		vector<DblNumTns>  uderys;	uderys.resize(ubases.size());
		vector<DblNumTns>  uderzs;	uderzs.resize(ubases.size());
		for(int g=0; g<ubases.size(); g++) {
		  uderxs[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  uderys[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  uderzs[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  DiffPsi(Nlbls, hs, ubases[g].data(), uderxs[g].data(), uderys[g].data(), uderzs[g].data() );
		}
		vector<int>& uindex = Aindexvec(u1,u2,u3);
		int nubases = ubases.size();
		
		int v1,v2,v3;
		if(vch==0) {
		  v1=p1;		v2=i2;		v3=i3;
		} else {
		  v1=i1;		v2=i2;		v3=i3;
		}
		vector<DblNumTns>& vbases = basesvec.lclmap()[Index3(v1,v2,v3)];
		vector<DblNumTns>  vderxs;	vderxs.resize(vbases.size());
		vector<DblNumTns>  vderys;	vderys.resize(vbases.size());
		vector<DblNumTns>  vderzs;	vderzs.resize(vbases.size());
		for(int g=0; g<vbases.size(); g++) {
		  vderxs[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  vderys[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  vderzs[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  DiffPsi(Nlbls, hs, vbases[g].data(), vderxs[g].data(), vderys[g].data(), vderzs[g].data() );
		}
		vector<int>& vindex = Aindexvec(v1,v2,v3);
		int nvbases = vbases.size();
		
		DblNumMat S(nubases,nvbases);
		vector<DblNumMat> uDXave(nubases);
		vector<DblNumMat> uVLjmp(nubases);
		for(int a=0; a<nubases; a++) {
		  uDXave[a].resize(Nlbl2,Nlbl3);
		  uVLjmp[a].resize(Nlbl2,Nlbl3);
		  if(uch==0) {
		    for(int g2=0; g2<Nlbl2; g2++)
		      for(int g3=0; g3<Nlbl3; g3++) {
			uDXave[a](g2,g3) = (uderxs[a](Nlbl1-1,g2,g3) + 0)/2;
			uVLjmp[a](g2,g3) = (ubases[a](Nlbl1-1,g2,g3)*1 + 0);
		      }
		  } else {
		    for(int g2=0; g2<Nlbl2; g2++)
		      for(int g3=0; g3<Nlbl3; g3++) {
			uDXave[a](g2,g3) = (uderxs[a](0,g2,g3) + 0)/2;
			uVLjmp[a](g2,g3) = (ubases[a](0,g2,g3)*(-1) + 0);
		      }
		  }
		}
		
		vector<DblNumMat> vDXave(nvbases);
		vector<DblNumMat> vVLjmp(nvbases);
		for(int b=0; b<nvbases; b++) {
		  vDXave[b].resize(Nlbl2,Nlbl3);
		  vVLjmp[b].resize(Nlbl2,Nlbl3);
		  if(vch==0) {
		    for(int g2=0; g2<Nlbl2; g2++)
		      for(int g3=0; g3<Nlbl3; g3++) {
			vDXave[b](g2,g3) = (vderxs[b](Nlbl1-1,g2,g3) + 0)/2;
			vVLjmp[b](g2,g3) = (vbases[b](Nlbl1-1,g2,g3)*1 + 0);
		      }
		  } else {
		    for(int g2=0; g2<Nlbl2; g2++)
		      for(int g3=0; g3<Nlbl3; g3++) {
			vDXave[b](g2,g3) = (vderxs[b](0,g2,g3) + 0)/2;
			vVLjmp[b](g2,g3) = (vbases[b](0,g2,g3)*(-1) + 0);
		      }
		  }
		}
		
		for(int a=0; a<nubases; a++)
		  for(int b=0; b<nvbases; b++) {
		    double tmp1 = -0.5*TPS(uDXave[a].data(),vVLjmp[b].data(),ww.data(),Nfacesize);
		    double tmp2 = -0.5*TPS(vDXave[b].data(),uVLjmp[a].data(),ww.data(),Nfacesize);
		    double tmp3 = _alpha/h1*TPS(uVLjmp[a].data(),vVLjmp[b].data(),ww.data(),Nfacesize); //LY: HERE USE _alpha
		    S(a,b) = tmp1 + tmp2 + tmp3;
		  }
		
		Index3 ukey = Index3(u1,u2,u3);
		Index3 vkey = Index3(v1,v2,v3);
		EmatKey emk(ukey,vkey);
		map<EmatKey,DblNumMat>::iterator mi = A.lclmap().find(emk);
		if(mi==A.lclmap().end()) {
		  A.lclmap()[emk] = S;
		} else {
		  DblNumMat& tmp = (*mi).second;
		  for(int a=0; a<S.m(); a++)	      for(int b=0; b<S.n(); b++)		tmp(a,b)+=S(a,b);
		}
	      }
	  }
	}
  }
  //xz face
  if(1) {
    DblNumMat ww(Nlbl1,Nlbl3);
    for(int g1=0; g1<Nlbl1; g1++)
      for(int g3=0; g3<Nlbl3; g3++) {
	ww(g1,g3) = w1(g1)*w3(g3);
      }
    int Nfacesize = Nlbl1*Nlbl3;
    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if(_elemptn.owner(curkey)==mpirank) {
	    int p2;	  if(i2==0)	    p2 = N2-1;	  else	    p2 = i2-1;
	    for(int uch=0; uch<=1; uch++)
	      for(int vch=0; vch<=1; vch++) {
		//-------------
		int u1,u2,u3;
		if(uch==0) {
		  u1=i1;		u2=p2;		u3=i3;
		} else {
		  u1=i1;		u2=i2;		u3=i3;
		}
		vector<DblNumTns>& ubases = basesvec.lclmap()[Index3(u1,u2,u3)];
		vector<DblNumTns>  uderxs;	uderxs.resize(ubases.size());
		vector<DblNumTns>  uderys;	uderys.resize(ubases.size());
		vector<DblNumTns>  uderzs;	uderzs.resize(ubases.size());
		for(int g=0; g<ubases.size(); g++) {
		  uderxs[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  uderys[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  uderzs[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  DiffPsi(Nlbls, hs, ubases[g].data(), uderxs[g].data(), uderys[g].data(), uderzs[g].data() );
		}
		vector<int>& uindex = Aindexvec(u1,u2,u3);
		int nubases = ubases.size();
		
		int v1,v2,v3;
		if(vch==0) {
		  v1=i1;		v2=p2;		v3=i3;
		} else {
		  v1=i1;		v2=i2;		v3=i3;
		}
		vector<DblNumTns>& vbases = basesvec.lclmap()[Index3(v1,v2,v3)];
		vector<DblNumTns>  vderxs;	vderxs.resize(vbases.size());
		vector<DblNumTns>  vderys;	vderys.resize(vbases.size());
		vector<DblNumTns>  vderzs;	vderzs.resize(vbases.size());
		for(int g=0; g<vbases.size(); g++) {
		  vderxs[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  vderys[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  vderzs[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  DiffPsi(Nlbls, hs, vbases[g].data(), vderxs[g].data(), vderys[g].data(), vderzs[g].data() );
		}
		vector<int>& vindex = Aindexvec(v1,v2,v3);
		int nvbases = vbases.size();
		
		DblNumMat S(nubases,nvbases);
		vector<DblNumMat> uDYave(nubases);
		vector<DblNumMat> uVLjmp(nubases);
		for(int a=0; a<nubases; a++) {
		  uDYave[a].resize(Nlbl1,Nlbl3);
		  uVLjmp[a].resize(Nlbl1,Nlbl3);
		  if(uch==0) {
		    for(int g1=0; g1<Nlbl1; g1++)
		      for(int g3=0; g3<Nlbl3; g3++) {
			uDYave[a](g1,g3) = (uderys[a](g1,Nlbl2-1,g3) + 0)/2;
			uVLjmp[a](g1,g3) = (ubases[a](g1,Nlbl2-1,g3)*1 + 0);
		      }
		  } else {
		    for(int g1=0; g1<Nlbl1; g1++)
		      for(int g3=0; g3<Nlbl3; g3++) {
			uDYave[a](g1,g3) = (uderys[a](g1,0,g3) + 0)/2;
			uVLjmp[a](g1,g3) = (ubases[a](g1,0,g3)*(-1) + 0);
		      }
		  }
		}
		
		vector<DblNumMat> vDYave(nvbases);
		vector<DblNumMat> vVLjmp(nvbases);
		for(int b=0; b<nvbases; b++) {
		  vDYave[b].resize(Nlbl1,Nlbl3);
		  vVLjmp[b].resize(Nlbl1,Nlbl3);
		  if(vch==0) {
		    for(int g1=0; g1<Nlbl1; g1++)
		      for(int g3=0; g3<Nlbl3; g3++) {
			vDYave[b](g1,g3) = (vderys[b](g1,Nlbl2-1,g3) + 0)/2;
			vVLjmp[b](g1,g3) = (vbases[b](g1,Nlbl2-1,g3)*1 + 0);
		      }
		  } else {
		    for(int g1=0; g1<Nlbl1; g1++)
		      for(int g3=0; g3<Nlbl3; g3++) {
			vDYave[b](g1,g3) = (vderys[b](g1,0,g3) + 0)/2;
			vVLjmp[b](g1,g3) = (vbases[b](g1,0,g3)*(-1) + 0);
		      }
		  }
		}
		
		for(int a=0; a<nubases; a++)
		  for(int b=0; b<nvbases; b++) {
		    double tmp1 = -0.5*TPS(uDYave[a].data(),vVLjmp[b].data(),ww.data(),Nfacesize);
		    double tmp2 = -0.5*TPS(vDYave[b].data(),uVLjmp[a].data(),ww.data(),Nfacesize);
		    double tmp3 = _alpha/h2*TPS(uVLjmp[a].data(),vVLjmp[b].data(),ww.data(),Nfacesize);
		    S(a,b) = tmp1 + tmp2 + tmp3;
		  }

		Index3 ukey = Index3(u1,u2,u3);
		Index3 vkey = Index3(v1,v2,v3);
		EmatKey emk(ukey,vkey);
		map<EmatKey,DblNumMat>::iterator mi = A.lclmap().find(emk);
		if(mi==A.lclmap().end()) {
		  A.lclmap()[emk] = S;
		} else {
		  DblNumMat& tmp = (*mi).second;
		  for(int a=0; a<S.m(); a++)	      for(int b=0; b<S.n(); b++)		tmp(a,b)+=S(a,b);
		}
	      }
	  }
	}
  }
  //xy face
  if(1) {
    DblNumMat ww(Nlbl1,Nlbl2);
    for(int g1=0; g1<Nlbl1; g1++)
      for(int g2=0; g2<Nlbl2; g2++) {
	ww(g1,g2) = w1(g1)*w2(g2);
      }
    int Nfacesize = Nlbl1*Nlbl2;
    for(int i3=0; i3<N3; i3++)
      for(int i2=0; i2<N2; i2++)
	for(int i1=0; i1<N1; i1++) {
	  Index3 curkey = Index3(i1,i2,i3);
	  if(_elemptn.owner(curkey)==mpirank) {
	    int p3;	  if(i3==0)	    p3 = N3-1;	  else	    p3 = i3-1;
	    for(int uch=0; uch<=1; uch++)
	      for(int vch=0; vch<=1; vch++) {
		//-------------
		int u1,u2,u3;
		if(uch==0) {
		  u1=i1;		u2=i2;		u3=p3;
		} else {
		  u1=i1;		u2=i2;		u3=i3;
		}
		vector<DblNumTns>& ubases = basesvec.lclmap()[Index3(u1,u2,u3)];
		vector<DblNumTns>  uderxs;	uderxs.resize(ubases.size());
		vector<DblNumTns>  uderys;	uderys.resize(ubases.size());
		vector<DblNumTns>  uderzs;	uderzs.resize(ubases.size());
		for(int g=0; g<ubases.size(); g++) {
		  uderxs[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  uderys[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  uderzs[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  DiffPsi(Nlbls, hs, ubases[g].data(), uderxs[g].data(), uderys[g].data(), uderzs[g].data() );
		}	    
		vector<int>& uindex = Aindexvec(u1,u2,u3);
		int nubases = ubases.size();
		
		int v1,v2,v3;
		if(vch==0) {
		  v1=i1;		v2=i2;		v3=p3;
		} else {
		  v1=i1;		v2=i2;		v3=i3;
		}
		vector<DblNumTns>& vbases = basesvec.lclmap()[Index3(v1,v2,v3)];
		vector<DblNumTns>  vderxs;	vderxs.resize(vbases.size());
		vector<DblNumTns>  vderys;	vderys.resize(vbases.size());
		vector<DblNumTns>  vderzs;	vderzs.resize(vbases.size());
		for(int g=0; g<vbases.size(); g++) {
		  vderxs[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  vderys[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  vderzs[g].resize(Nlbl1, Nlbl2, Nlbl3);
		  DiffPsi(Nlbls, hs, vbases[g].data(), vderxs[g].data(), vderys[g].data(), vderzs[g].data() );
		}
		vector<int>& vindex = Aindexvec(v1,v2,v3);
		int nvbases = vbases.size();
		
		DblNumMat S(nubases,nvbases);
		vector<DblNumMat> uDZave(nubases);
		vector<DblNumMat> uVLjmp(nubases);
		for(int a=0; a<nubases; a++) {
		  uDZave[a].resize(Nlbl1,Nlbl2);
		  uVLjmp[a].resize(Nlbl1,Nlbl2);
		  if(uch==0) {
		    for(int g1=0; g1<Nlbl1; g1++)
		      for(int g2=0; g2<Nlbl2; g2++) {
			uDZave[a](g1,g2) = (uderzs[a](g1,g2,Nlbl3-1) + 0)/2;
			uVLjmp[a](g1,g2) = (ubases[a](g1,g2,Nlbl3-1)*1 + 0);
		      }
		  } else {
		    for(int g1=0; g1<Nlbl1; g1++)
		      for(int g2=0; g2<Nlbl2; g2++) {
			uDZave[a](g1,g2) = (uderzs[a](g1,g2,0) + 0)/2;
			uVLjmp[a](g1,g2) = (ubases[a](g1,g2,0)*(-1) + 0);
		      }
		  }
		}
		
		vector<DblNumMat> vDZave(nvbases);
		vector<DblNumMat> vVLjmp(nvbases);
		for(int b=0; b<nvbases; b++) {
		  vDZave[b].resize(Nlbl1,Nlbl2);
		  vVLjmp[b].resize(Nlbl1,Nlbl2);
		  if(vch==0) {
		    for(int g1=0; g1<Nlbl1; g1++)
		      for(int g2=0; g2<Nlbl2; g2++) {
		    vDZave[b](g1,g2) = (vderzs[b](g1,g2,Nlbl3-1) + 0)/2;
		    vVLjmp[b](g1,g2) = (vbases[b](g1,g2,Nlbl3-1)*1 + 0);
		      }
		  } else {
		    for(int g1=0; g1<Nlbl1; g1++)
		      for(int g2=0; g2<Nlbl2; g2++) {
			vDZave[b](g1,g2) = (vderzs[b](g1,g2,0) + 0)/2;
			vVLjmp[b](g1,g2) = (vbases[b](g1,g2,0)*(-1) + 0);
		      }
		  }
		}
		
		for(int a=0; a<nubases; a++)
		  for(int b=0; b<nvbases; b++) {
		    double tmp1 = -0.5*TPS(uDZave[a].data(),vVLjmp[b].data(),ww.data(),Nfacesize);
		    double tmp2 = -0.5*TPS(vDZave[b].data(),uVLjmp[a].data(),ww.data(),Nfacesize);
		    double tmp3 = _alpha/h3*TPS(uVLjmp[a].data(),vVLjmp[b].data(),ww.data(),Nfacesize);
		    S(a,b) = tmp1 + tmp2 + tmp3;
		  }
		
		Index3 ukey = Index3(u1,u2,u3);
		Index3 vkey = Index3(v1,v2,v3);
		EmatKey emk(ukey,vkey);
		map<EmatKey,DblNumMat>::iterator mi = A.lclmap().find(emk);
		if(mi==A.lclmap().end()) {
		  A.lclmap()[emk] = S;
		} else {
		  DblNumMat& tmp = (*mi).second;
		  for(int a=0; a<S.m(); a++)	      for(int b=0; b<S.n(); b++)		tmp(a,b)+=S(a,b);
		}
	      }
	  }
	}
  }
  
  
  t1 = time(0);  
  if(mpirank==0) { 
    fprintf(fhstat, "boundaries %15.3f secs\n", difftime(t1,t0));   
    fprintf(stderr, "boundaries %15.3f secs\n", difftime(t1,t0));   
  }
  
  //do a put to make A consistent
  if(1) {
    vector<EmatKey> keyvec;
    for(map<EmatKey,DblNumMat>::iterator mi=A.lclmap().begin(); mi!=A.lclmap().end(); mi++) {
      EmatKey curkey = (*mi).first;
      if(A.prtn().owner(curkey)!=mpirank) //NOT OWNED BY ME
	keyvec.push_back( (*mi).first );
    }
    //communicate
    vector<int> all(1,1);
    iC( A.putBegin(keyvec,all) );
    iC( A.putEnd(all,PARVEC_CMB) ); //COMBINE TOGETHER
    //save space
    vector<EmatKey> tmpvec;
    for(map<EmatKey,DblNumMat>::iterator mi=A.lclmap().begin(); mi!=A.lclmap().end(); mi++) {
      EmatKey curkey = (*mi).first;
      if(A.prtn().owner(curkey)!=mpirank) //NOT OWNED BY ME
	tmpvec.push_back(curkey); //STORE THE ONES TO BE DELETED
    }
    for(int a=0; a<tmpvec.size(); a++)      A.lclmap().erase(tmpvec[a]); //THEN REMOVE
  }
  
  //----------------------------------------------------------------------------------------------------------------  
  //Output Asp into the format of a series of sparse matricies
  /*
  if(0){
    iC( MPI_Barrier(MPI_COMM_WORLD) );
    //
    char filename[100];
    sprintf(filename, "%s_%d_%d", "Asp", mpirank, mpisize);
    char data[120];
    ofstream fout(filename);
    for(map<Index2,double>::iterator mi=Asp.lclmap().begin(); mi!=Asp.lclmap().end(); mi++) {
      Index2 idx = (*mi).first;
      double val = (*mi).second;
      sprintf(data, "%6d %6d %25.15e\n", idx[0], idx[1], val);
      fout << data; 
    }
    fout.close();
    }
  */
  return 0;
}
