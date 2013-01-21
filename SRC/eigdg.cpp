#include "eigdg.hpp"
#include "interp.hpp"
#include "vecmatop.hpp"
#include "parallel.hpp"

extern FILE* fhstat;
extern int CONTXT;


//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
//--------------------------------------------
EigDG::EigDG()
{
}

//--------------------------------------------
EigDG::~EigDG()
{
}

//--------------------------------------------
int EigDG::setup()
{
	return 0;
}

//--------------------------------------------
int EigDG::solve(ParVec<Index3, vector<double>, ElemPtn>& vtotvec, ParVec<Index3, vector<DblNumTns>, ElemPtn>& basesvec, ParVec<int,Psdo,PsdoPtn>& psdovec,
		int npsi, vector<double>& eigvals, ParVec<Index3, DblNumMat, ElemPtn>& eigvecsvec,
		ParVec<Index3,DblNumMat,ElemPtn> & EOcoef)
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
	time_t t0, t1, tt0, tt1;
	//
	int myid = mpirank;
	_elemptn = vtotvec.prtn();
	_psdoptn = psdovec.prtn();
	_ematptn.ownerinfo() = _elemptn.ownerinfo();

	MPI_Barrier(MPI_COMM_WORLD);

	if(mpirank == 0){
		fprintf(stderr, "Start eigdg::solve\n"); 
	}

	//compute the communication index sets for _elemptn
	set<Index3> jumpset;
	for(int i3=0; i3<N3; i3++)
		for(int i2=0; i2<N2; i2++)
			for(int i1=0; i1<N1; i1++) {
				Index3 curkey = Index3(i1,i2,i3);
				if(_elemptn.owner(curkey)==mpirank) {
					int p1;	  if(i1==0)	    p1 = N1-1;	  else	    p1 = i1-1;
					int p2;	  if(i2==0)	    p2 = N2-1;	  else	    p2 = i2-1;
					int p3;	  if(i3==0)	    p3 = N3-1;	  else	    p3 = i3-1;
					jumpset.insert(Index3(p1,i2,i3));
					jumpset.insert(Index3(i1,p2,i3));
					jumpset.insert(Index3(i1,i2,p3));
				}
			}
	_jumpvec.clear();  _jumpvec.insert(_jumpvec.begin(), jumpset.begin(), jumpset.end());

	set<Index3> psdoset;
	for(map<int,Psdo>::iterator mi=psdovec.lclmap().begin(); mi!=psdovec.lclmap().end(); mi++) {
		int  curkey = (*mi).first;
		Psdo& curdat = (*mi).second;
		if(_psdoptn.owner(curkey)==mpirank) {
			vector< pair<NumTns<SparseVec>,double> >& vnls = curdat.vnls();
			for(int g=0; g<vnls.size(); g++) {
				NumTns<SparseVec>& ppcur = vnls[g].first;
				for(int i3=0; i3<N3; i3++)
					for(int i2=0; i2<N2; i2++)
						for(int i1=0; i1<N1; i1++) {//if(!ppcur(i1,i2,i3).isempty())		  keyset.insert(Index3(i1,i2,i3));
							if(ppcur(i1,i2,i3).first.m()>0) //LEXING: IMPORTANT CHANGE
								psdoset.insert(Index3(i1,i2,i3));
						}
			}
		}
	}
	_psdovec.clear();  _psdovec.insert(_psdovec.begin(), psdoset.begin(), psdoset.end());

	if(0) {
		if(mpirank == 0){
			cerr << "jumpvec" << endl;
			for(vector<Index3>::iterator mi = _jumpvec.begin(); mi != _jumpvec.end(); mi++){
				cerr << (*mi) << endl;
			}
			cerr << "psdovec" << endl;
			for(vector<Index3>::iterator mi = _psdovec.begin(); mi != _psdovec.end(); mi++){
				cerr << (*mi) << endl;
			}
		}
	}

	set<Index3> nbhdset;
	for(int i3=0; i3<N3; i3++)
		for(int i2=0; i2<N2; i2++)
			for(int i1=0; i1<N1; i1++) {
				Index3 curkey = Index3(i1,i2,i3);
				if(_elemptn.owner(curkey)==mpirank) {
					for(int g1=-1; g1<=1; g1++)
						for(int g2=-1; g2<=1; g2++)
							for(int g3=-1; g3<=1; g3++) {
								int a1 = (i1+g1 + N1)%N1;
								int a2 = (i2+g2 + N2)%N2;
								int a3 = (i3+g3 + N3)%N3;
								nbhdset.insert(Index3(a1,a2,a3));
							}
				}
			}
	_nbhdvec.clear();  _nbhdvec.insert(_nbhdvec.begin(), nbhdset.begin(), nbhdset.end());

	if(mpirank == 0){
		fprintf(stderr, "Finish setting initial variables for eigdg::solve\n"); 
	}

	//----------------------------------------------------------------------------------------------------------------  
	//----------------------------------------------------------------------------------------------------------------  

	//-----------------------------
	//standard approach
	if(_dgsolver == "std" ) {

		//build stiffness matrix
		ParVec<EmatKey,DblNumMat,EmatPtn> A;
		int AM, AN;
		NumTns< vector<int> > Aindexvec;
		iC( solve_Elem_A(vtotvec,basesvec,psdovec,   A,AM,AN,Aindexvec) );

		// FIXME: Dump out the DG matrix Avec in local triplet form
#ifdef __DUMP_DGMAT
		{

			vector<int> noMask(1);
			ostringstream oss;

			vector<int> RowVec;
			vector<int> ColVec;
			vector<double> ValVec;
			RowVec.clear();
			ColVec.clear();
			ValVec.clear();

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
										DblNumMat& ValMat = A.lclmap()[emk];
										if( energy(ValMat) > 1e-10 ){
											for(int jc = 0; jc < jnum; jc++){
												for(int ic = 0; ic < inum; ic++){
													RowVec.push_back(Aindexvec(i1,i2,i3)[ic]);
													ColVec.push_back(Aindexvec(j1,j2,j3)[jc]);
													ValVec.push_back(ValMat(ic,jc));
												}
											}
										}
									}
								}
					}

			{
				int LocSize = RowVec.size();
				// Header
				serialize(AM, oss, noMask);
				serialize(LocSize, oss, noMask);
				// Triplet form
				serialize(RowVec, oss, noMask);
				serialize(ColVec, oss, noMask);
				serialize(ValVec, oss, noMask);

				Separate_Write("DGMAT", oss);
			}

		}
#endif
#ifdef __DUMP_ONCE
		fclose(fhstat);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
		iA(0);
#endif


		//transform A to scalapack form
		int descA[DLEN_];
		DblNumMat Aloc;
		int AlocM, AlocN; //the true matrix size
		iC( solve_A_Aloc(A,AM,AN,Aindexvec,   descA,Aloc,AlocM,AlocN) );

		//call scalapack to solve for Zloc
		int descZ[DLEN_];
		DblNumMat Zloc;
		int ZlocM, ZlocN;
		DblNumVec W; //eigvals 
		iC( solve_Aloc_Zloc(descA,Aloc,AlocM,AlocN,   descZ,Zloc,ZlocM,ZlocN,W) );

		//put Zloc back
		iC( solve_Zloc_Eig(descZ,Zloc,ZlocM,ZlocN,W, npsi,eigvals,eigvecsvec,Aindexvec) );

		//LEXING: get the eigenvecs for force calculation (pseudopotential)
		{
			vector<int> mask(1,1);
			iC( eigvecsvec.getBegin(_psdovec,mask) );
			iC( eigvecsvec.getEnd(mask) );
		}


	}

	//-----------------------------
	//nonorthogonal approach
	if( _dgsolver == "nonorth" ){
		//
		//build stiffness matrix
		ParVec<EmatKey,DblNumMat,EmatPtn> A;
		int AM, AN;
		NumTns< vector<int> > Aindexvec;
		iC( solve_Elem_A(vtotvec,basesvec,psdovec,   A,AM,AN,Aindexvec) );

		//LLIN: Generate the C matrix 
		int CM, CN;
		NumTns< vector<int> > Cindexvec;
		iC( solve_A_C(A, AM, AN, Aindexvec, EOcoef, CM, CN, Cindexvec,
					basesvec) );


		//transform A to scalapack form Aloc
		int descA[DLEN_];
		DblNumMat Aloc;
		int AlocM, AlocN; 
		//LLIN: AlocM and AlocN are the global matrix size of A, NOT the
		//loc_m and loc_n used by ScaLAPACK! The same notation follows in
		//the rest part of the code
		iC( solve_A_Aloc(A,AM,AN,Aindexvec,   descA,Aloc,AlocM,AlocN) );

		//DEBUG
		if(0){
			int   nout   = 0, ione = 1, izero = 0;
			DblNumVec work(_MB);
			if(mpirank == 0 ) cerr << endl;
			pdlaprnt_(&AlocM, &AlocN, Aloc.data(), &ione, &ione, descA, 
					& izero, &izero, "A", &nout, work.data(), 1);
		}


		//transform EOcoef to scalapack form Cloc
		int descC[DLEN_];
		DblNumMat Cloc;
		int ClocM, ClocN; 
		iC( solve_C_Cloc(EOcoef,CM,CN,Cindexvec, descC,Cloc,ClocM,ClocN) );

		//DEBUG
		if(0){
			int   nout   = 0, ione = 1, izero = 0;
			DblNumVec work(_MB);
			if(mpirank == 0 ) cerr << endl;
			pdlaprnt_(&ClocM, &ClocN, Cloc.data(), &ione, &ione, descC, 
					& izero, &izero, "C", &nout, work.data(), 1);
		}


		//LLIN: Perform pivot QR on the coefficient matrix (Cloc) to get the
		//reduced coefficient matrix (Crdcloc) 
		int descCrdc[DLEN_]; 
		DblNumMat Crdcloc;
		int CrdclocM, CrdclocN;

		if(0){
			if(mpirank == 0){
				fprintf(stderr, "With QR pruning!\n");
				fprintf(fhstat, "With QR pruning!\n");
			}
			iC( solve_Cloc_QR(Cloc, ClocM, ClocN, descC, Crdcloc, CrdclocM,
						CrdclocN, descCrdc) );
		}
		else{
			//LLIN: DEBUG purpose FIXME
			if(mpirank == 0){
				fprintf(stderr, "DEBUG purpose, no QR pruning!\n");
				fprintf(fhstat, "DEBUG purpose, no QR pruning!\n");
			}
			for(int i = 0; i < DLEN_; i++){
				descCrdc[i] = descC[i];
			}
			CrdclocM = ClocM;
			CrdclocN = ClocN;
			Crdcloc  = Cloc;
		}


		//DEBUG
		if(0){
			int   nout   = 0, ione = 1, izero = 0;
			DblNumVec work(_MB);
			if(mpirank == 0 ) cerr << endl;
			pdlaprnt_(&CrdclocM, &CrdclocN, Crdcloc.data(), &ione, &ione, descCrdc, 
					& izero, &izero, "Cr", &nout, work.data(), 2);
		}


		// For the purpose of matrix-matrix multiplication 
		char ntrans = 'N', trans = 'T';      

		// LLIN: Calculate Crdcloc' * Aloc * Crdcloc and Crdcloc'*Crdcloc
		// after pivot QR step
		int descCAC[DLEN_];
		DblNumMat CACloc;
		int CAClocM, CAClocN; //the true matrix size
		int descCC[DLEN_];
		DblNumMat CCloc;
		int CClocM, CClocN; //the true matrix size
		{
			time_t t0, t1;
			t0 = time(0);

			int descAC[DLEN_];
			DblNumMat ACloc;
			int AClocM, AClocN;
			iC(pdgemm(&ntrans, &ntrans, 
						Aloc, AlocM, AlocN, descA, 
						Crdcloc, CrdclocM, CrdclocN, descCrdc, 
						ACloc, AClocM, AClocN, descAC,
						_MB, _MB, CONTXT));

			iC(pdgemm(&trans, &ntrans, 
						Crdcloc, CrdclocM, CrdclocN, descCrdc, 
						ACloc, AClocM, AClocN, descAC,
						CACloc, CAClocM, CAClocN, descCAC,
						_MB, _MB, CONTXT));


			iC( pdgemm(&trans, &ntrans,
						Crdcloc, CrdclocM, CrdclocN, descCrdc,
						Crdcloc, CrdclocM, CrdclocN, descCrdc,
						CCloc,   CClocM,   CClocN,   descCC,
						_MB, _MB, CONTXT) );

			t1 = time(0);
			if(mpirank==0) { 
				fprintf(fhstat, "Form CAC and CC %15.3f secs\n", 
						difftime(t1,t0));   
				fprintf(stderr, "Form CAC and CC %15.3f secs\n", 
						difftime(t1,t0));   
			}
			//DEBUG
			if(0){
				int   nout   = 0, ione = 1, izero = 0;
				DblNumVec work(_MB);
				if(mpirank == 0 ) cerr << endl;
				pdlaprnt_(&CClocM, &CClocN, CCloc.data(), &ione, &ione, descCC, 
						& izero, &izero, "CC", &nout, work.data(), 2);
			}
		}

		//LLIN: solved generalized eigenvalue problem, get matrix VCloc and
		//eigenvals EC.  NOTE: CACloc and CCloc will be overwritten!
		int Norbttl = N1*N2*N3 * _Norbperele; //LLIN: FIXME 

		DblNumVec EC(Norbttl);
		int descVC[DLEN_];
		DblNumMat VCloc;
		int VClocM, VClocN; //the true matrix size
		iC( solve_GE(CACloc, CAClocM, CAClocN, descCAC,
					CCloc,  CClocM,  CClocN,  descCC,
					VCloc,  VClocM,  VClocN,  descVC, 
					EC) );

		//LLIN: Calculate the eigenvectors for Aloc (adaptive local basis
		//set).  The eigenvectors are Zloc = Crdcloc * VCloc
		int descZ[DLEN_];
		DblNumMat Zloc;
		int ZlocM, ZlocN;
		{
			time_t t0, t1;
			t0 = time(0);
			iC( pdgemm(&ntrans, &ntrans, 
						Crdcloc, CrdclocM, CrdclocN, descCrdc,
						VCloc,   VClocM,   VClocN,   descVC,
						Zloc,    ZlocM,    ZlocN,    descZ,
						_MB,     _MB,      CONTXT) );

			t1 = time(0);
			if(mpirank==0) { 
				fprintf(fhstat, "Forming Zloc %15.3f secs\n", 
						difftime(t1,t0));   
				fprintf(stderr, "Forming Zloc %15.3f secs\n", 
						difftime(t1,t0));   
			}

			if(0)
			{
				if(mpirank == 0){
					cerr << "ZlocM = " << ZlocM << endl;
					cerr << "ZlocN = " << ZlocN << endl;
					cerr << "Eigenvalues:" << EC << endl;
				}
				int   nout   = 0, ione = 1, izero = 0;
				DblNumVec work(_MB);
				pdlaprnt_(&ZlocM, &ZlocN, Zloc.data(), &ione, &ione, descZ, 
						& izero, &izero, "Z", &nout, work.data(), 1);
			}
		}


		//put Zloc back to Eig
		iC( solve_Zloc_Eig(descZ,Zloc,ZlocM,ZlocN, EC, npsi,eigvals,eigvecsvec,Aindexvec) );

		//LEXING: get the eigenvecs for force calculation (pseudopotential)
		{
			vector<int> mask(1,1);
			iC( eigvecsvec.getBegin(_psdovec,mask) );
			iC( eigvecsvec.getEnd(mask) );
		}
	}

	return 0;
}

int EigDG::DumpNALB(Index3 cur, ParVec<Index3,DblNumMat,ElemPtn>&  C){
	// Dump the basis functions for the (1,1,1) element
	//
	//  DblNumMat& coef = C.lclmap()[cur];
	//  int nbases = coef.m(), Nb = coef.n();
	//  int N1=_Ns[0];    int N2=_Ns[1];    int N3=_Ns[2];
	//  int Nlbl1 = _Nlbls[0], Nlbl2 = _Nlbls[1], Nlbl3 = _Nlbls[2];
	//  int I_ONE = 1;
	//  
	//  int Nsglbtot1, Nsglbtot2, Nstotglb3;
	//  {
	//    Elem& curdat = _elemvec(1,1,1); 
	//    Index3 Nsglb = curdat._Nsglb;
	//    Nsglbtot1 = Nsglb[0] * N1;
	//    Nsglbtot2 = Nsglb[1] * N2;
	//    Nsglbtot3 = Nsglb[2] * N3;
	//  }
	//
	//  DblNumVec psielemtmp(Nlbl1 * Nlbl2 * Nlbl3);
	//  DblNumTns fun_all(Nsglbtot1, Nsglbtot2, Nsglbtot3),
	//	    fun_tmp(Nsglbtot1, Nsglbtot2, Nsglbtot3);
	//
	//  ofstream funid;
	//  string strtmp = string("fun_all.idat");
	//  funid.open(strtmp.c_str(), ios::out | ios::trunc);
	//  
	//  for(int g=0; g<Nb; g++){
	//    setvalue(fun_all, 0.0);
	//    setvalue(fun_tmp, 0.0);
	//    
	//    NumTns<DblNumTns> fun_glb;	fun_glb.resize(N1,N2,N3);
	//    //fun_glb.resize(Nsglb[0],Nsglb[1],Nsglb[2]);
	//    for(int i3=0; i3<N3; i3++)
	//      for(int i2=0; i2<N2; i2++)
	//	for(int i1=0; i1<N1; i1++) {
	//	  Index3 curkey = Index3(i1,i2,i3);
	//	  if( _elemptn.owner(curkey)==mpirank ) {
	//	    //map<Index3,Elem>::iterator mi=_elemvec.lclmap().find(curkey);
	//	    //iA(mi!=_elemvec.lclmap().end());
	//	    Elem& curdat = _elemvec(i1,i2,i3);
	//	    vector<DblNumTns>& bases = _basesvec.lclmap()[curkey];		//curdat.bases();
	//	    setvalue(psielemtmp, 0.0);
	//	    int I_ONE = 1;
	//	    for(int a=0; a<nbases; a++){
	//	      daxpy_(&ntotelem, &coef(a,g), bases[a].data(), &I_ONE, psielemtmp.data(), &I_ONE);
	//	    }
	//	    /* the data in fun_glb(i1,i2,i3) will be rewritten by lxinterp_local for next orbital */
	//	    Index3 Nsglb = curdat._Nsglb;
	//	    fun_glb(i1,i2,i3).resize(Nsglb[0],Nsglb[1],Nsglb[2]);
	//	    lxinterp_local(fun_glb(i1,i2,i3).data(), psielemtmp.data(), curdat);
	//	  
	//	    Index3 posidx = curdat.posidx();
	//	    for(int k = 0; k < Nsglb[2]; k++){		  int ksh = posidx[2]+ k; 
	//	      for(int j = 0; j < Nsglb[1]; j++){		    int jsh = posidx[1] + j; 
	//		for(int i = 0; i < Nsglb[0]; i++){		      int ish = posidx[0] + i;
	//		  fun_tmp(ish, jsh, ksh) += fun_glb(i1,i2,i3)(i,j,k);
	//		}
	//	      }
	//	    }
	//	  
	//	  }
	//	}
	//    iC( MPI_Allreduce(fun_tmp.data(), fun_all.data(), Nsglbtot1*Nsglbtot2*Nsglbtot3, 
	//		      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );
	//
	//    funid << g << endl;
	//    funid << fun_all <, endl;
	//  } // for(g)
	//  
	//  funid.close();
	//
	//
	//
	return 1;
}
