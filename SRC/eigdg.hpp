#ifndef _EIGDG_HPP_
#define _EIGDG_HPP_

#include "comobject.hpp"
#include "vec3t.hpp"
#include "numtns.hpp"
#include "parvec.hpp"
#include "scfdg.hpp"

using std::vector;
using std::pair;
using std::map;
using std::set;
using std::cerr;
using std::cout;
using std::ostream;
using std::istream;
using std::istringstream;
using std::ifstream;
using std::ofstream;

//----------------------------------------------------------------
//typedef Index2 Index2;

class BlckPtn
{
public:
  IntNumMat _ownerinfo;
public:
  BlckPtn() {;}
  ~BlckPtn() {;}
  IntNumMat& ownerinfo() { return _ownerinfo; }
  int owner(Index2 key) {
    return _ownerinfo(key(0), key(1));
  }
};

//----------------------------------------------------------------
typedef pair<Index3,Index3> EmatKey;

class EmatPtn //element matrix, partition
{
public:
  IntNumTns _ownerinfo;
public:
  EmatPtn() {;}
  ~EmatPtn() {;}
  IntNumTns& ownerinfo() { return _ownerinfo; }
  int owner(EmatKey key) {
    Index3 a = key.first;
    return _ownerinfo(a(0),a(1),a(2)); //partition according to rows
  }
};

//----------------------------------------------------------------
class EigDG
{
public:
  //--------------------------------
  //Control parameters
  int _output_bases; // dump the basis functions
  string _inputformat; 

  //--------------------------------

  //Domain _dm;
  Point3 _hs;
  Index3 _Ns; //NUMBER OF ELEMENTS IN EACH DIRECTION
  Index3 _Nlbls; //NUMBER OF LBL POINTS IN EACH DIRECITON
  double _alpha; //LLIN: Interior penalty in the DG formulation, should
                 // not be confused with _gamma 
  int _MB;

  //LEXING: contracted solve needs to specify the following values VERY IMPORTANT
  string _dgsolver; //LLIN: Standard solver (std) or Nonorthogonal (nonorth)
  double _delta;    //LLIN: Wall width for the Frobenius penalty  
  double _basisradius; //LLIN: Radius of nonlocal basis function that is not penalized. 
  double _gamma;    //LLIN: Weight ratio between eigenvalue and penalty
  int    _Neigperele; //LLIN: Number of eigenvalues to be solved per element in the buffer
  int    _Norbperele; //LLIN: Number of nonorthogonal oritals in the element
  double _DeltaFermi; //LLIN: The increase of the Fermi energy to control the number of candidate functions

  DblNumTns _EcutCnddt;  //LLIN: The cutoff of the energy for the candidate functions

  //
  //local stuff
  ElemPtn _elemptn;
  PsdoPtn _psdoptn;
  //int _Ndof;
  //NumTns< vector<int> > _indexvec;
  EmatPtn _ematptn;
  vector<Index3> _jumpvec; //LEXING: index of other element required for jump calculation
  vector<Index3> _psdovec; //LEXING: index of other element required for pseudo potential calculation
  vector<Index3> _nbhdvec; //LEXING: index of neighbor element required (for nonlocal computation)
  
public:
  EigDG();
  ~EigDG();
  int setup(); //for what?
  int solve(ParVec<Index3,vector<double>,ElemPtn>& vtotvec, ParVec<Index3,vector<DblNumTns>,ElemPtn>& basesvec, 
	    ParVec<int,Psdo,PsdoPtn>& psdovec,
	    int npsi, vector<double>& eigvals, ParVec<Index3, DblNumMat, ElemPtn>& eigvecsvec, 
	    ParVec<Index3,DblNumMat,ElemPtn>& EOcoef);
  int solve_Elem_A(ParVec<Index3,vector<double>,ElemPtn>& vtotvec, ParVec<Index3,vector<DblNumTns>,ElemPtn>& basesvec,ParVec<int,Psdo,PsdoPtn>& psdovec,
		   ParVec<EmatKey,DblNumMat,EmatPtn>& A,int& AM,int& AN,NumTns< vector<int> >& indexvec);
  int solve_A_Aloc(ParVec<EmatKey,DblNumMat,EmatPtn>& A,int& AM,int& AN,NumTns< vector<int> >& indexvec,
		   int* desca, DblNumMat& Aloc, int& AlocM, int& AlocN);
  int solve_Aloc_Zloc(int* desca, DblNumMat& Aloc, int& AlocM, int& AlocN,
		      int* descz, DblNumMat& Zloc, int& ZlocM, int& ZlocN, DblNumVec& W);
  int solve_Zloc_Eig(int* descz, DblNumMat& Zloc, int& ZlocM, int& ZlocN, DblNumVec& W,
		     int npsi, vector<double>& eigvals, ParVec<Index3, DblNumMat, ElemPtn>& eigvecsvec, NumTns< vector<int> >& indexvec);
  int solve_C_Cloc(ParVec<Index3,DblNumMat,ElemPtn>&  C,int& CM,int& CN,   NumTns< vector<int> >& indexvec,
		   int* descc, DblNumMat& Cloc, int& ClocM, int& ClocN);
  int solve_Cloc_QR(DblNumMat& Cloc, int& ClocM, int& ClocN, int* descC, 
		   DblNumMat& Crdcloc, int& CrdclocM, int& CrdclocN, int* descCrdc);
  int solve_GE(DblNumMat& Aloc, int& AlocM, int& AlocN, int* descA,
	       DblNumMat& Bloc, int& BlocM, int& BlocN, int* descB,
	       DblNumMat& Vloc, int& VlocM, int& VlocN, int* descV,
	       DblNumVec& EC);
  int solve_A_C(ParVec<EmatKey,DblNumMat,EmatPtn>& A, int& AM, int& AN,
		NumTns< vector<int> >& Aindexvec,
		ParVec<Index3,DblNumMat,ElemPtn>& C, int& CM, int& CN,
		NumTns< vector<int> >& Cindexvec,
		ParVec<Index3,vector<DblNumTns>,ElemPtn>& basesvec);

  // DEBUG subroutines
  // Dump the nonorthogonal basis functions in element cur
  int DumpNALB(Index3 cur, ParVec<Index3,DblNumMat,ElemPtn>&  C);  
};

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

inline double TPS(double* x, double* y, double* z, int ntot) {
  double sum =0;
  for(int i=0; i<ntot; i++) {
    sum += (*x)*(*y)*(*z);
    x++;    y++;    z++;
  }
  return sum;
}

inline double FPS(double* w, double* x, double* y, double* z, int ntot) {
  double sum =0;
  for(int i=0; i<ntot; i++) {
    sum += (*w)*(*x)*(*y)*(*z);
    w++;    x++;    y++;    z++;
  }
  return sum;
}



#endif
