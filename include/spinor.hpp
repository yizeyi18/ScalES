#ifndef _SPINOR_HPP_
#define _SPINOR_HPP_

#include  "environment_decl.hpp"
#include  "environment_impl.hpp"

//#include "periodtable.hpp"
//#include "numtns.hpp"
//#include "nummat.hpp"
//#include "numvec.hpp"
//#include "fftprepare.hpp"
//#include "serialize.hpp"
//
//using std::vector;
//using std::pair;
//using std::map;
//using std::set;
//using std::cerr;
//using std::cout;
//using std::istream; 
//using std::ostream;
//using std::istringstream;
//using std::ostringstream;
//using std::ifstream;
//using std::ofstream;

namespace dgdft{
class Spinor {
private:
  Domain            domain_;                    // mesh should be used here for general cases 
	std::vector<Vec>  wavefun_;                   // PETSc format
	NumTns<Scalar>    localWavefun_;              // local wavefunction for wavefun_
  bool              isNormalized_;

public:
  Spinor();
  Spinor( const Domain &dm, const Index3 &val );
  Spinor( const Domain &dm, Int n, Int p );
  Spinor( const Domain &dm, Int n, Int p, Bool owndata, Scalar *data );
  ~Spinor();

  void Normalize();

	// Inquiries
  Int numComponent() const { return localWavefun_.n();}
	Int numStates()    const { return localWavefun_.p();}

  // operations     
//	void AddScalarDiag ( Vec  
//  int add_scalardiag (DblNumVec &val, CpxNumTns &a3);
//  int add_nonlocalPS (vector< vector< pair<SparseVec,double> > > &val, CpxNumTns &a3);
//  int add_D2_c2c     (CpxNumTns &a3, FFTPrepare &fp);
//  int get_D2_c2c     (CpxNumTns &a3, FFTPrepare &fp);

  // Spin related operations
//  int add_sigma_x    (DblNumVec &a1, CpxNumTns &a3);
//  int add_sigma_y    (DblNumVec &a1, CpxNumTns &a3);
//  int add_sigma_z    (DblNumVec &a1, CpxNumTns &a3);
//  int add_nonlocalPS_SOC 
//    (vector< vector< pair<SparseVec,double> > > &val, 
//     vector<Atom> &atomvec,
//     vector<DblNumVec> &grid, CpxNumTns &a3, FFTPrepare &fp);
//  int add_matrix_ij (int ir, int jc, DblNumVec &a1, CpxNumTns &a3);
//  int add_matrix_ij (int ir, int jc, double  *ptr1, CpxNumTns &a3);
//  int get_DKS       (DblNumVec &vtot, DblNumMat &vxc,
//		     vector< vector< pair<SparseVec,double> > > &vnlss,
//		     vector< vector< pair<SparseVec,double> > > &vnlso,
//		     vector<Atom> &atomvec, CpxNumTns &a3, 
//		     FFTPrepare &fp, vector<DblNumVec> &grid);
};  // 

}



////////////

//LLIN: IMPORTANT: inline function must be defined OUT of the scope of
//the namespace.
inline int serialize(const REAL::Spinor &val, ostream &os, 
  const vector<int> &mask) {
  serialize(val._is_normalized, os, mask);
  serialize(val._domain._Ls,    os, mask);
  serialize(val._domain._Ns,    os, mask);
  serialize(val._domain._pos,   os, mask);
  serialize(val._wavefun,       os, mask);
  return 0;
};

inline int serialize(const COMPLEX::Spinor &val, ostream &os, 
  const vector<int> &mask) {
  serialize(val._is_normalized, os, mask);
  serialize(val._domain._Ls,    os, mask);
  serialize(val._domain._Ns,    os, mask);
  serialize(val._domain._pos,   os, mask);
  serialize(val._wavefun,       os, mask);
  return 0;
};

inline int deserialize(REAL::Spinor &val, istream &is, 
  const vector<int> &mask) {
  deserialize(val._is_normalized, is, mask);
  deserialize(val._domain._Ls,    is, mask);
  deserialize(val._domain._Ns,    is, mask);
  deserialize(val._domain._pos,   is, mask);
  deserialize(val._wavefun,       is, mask);
  return 0;
};

inline int deserialize(COMPLEX::Spinor &val, istream &is, 
  const vector<int> &mask) {
  deserialize(val._is_normalized, is, mask);
  deserialize(val._domain._Ls,    is, mask);
  deserialize(val._domain._Ns,    is, mask);
  deserialize(val._domain._pos,   is, mask);
  deserialize(val._wavefun,       is, mask);
  return 0;
};

inline int deserialize(COMPLEX::Spinor &val, istream &is, 
  const vector<int> &mask, int &ntot_in, int &ncom_in, int &nocc_in) {
  deserialize(val._is_normalized, is, mask);
  deserialize(val._domain._Ls,    is, mask);
  deserialize(val._domain._Ns,    is, mask);
  deserialize(val._domain._pos,   is, mask);

  int ntot = val._wavefun._m;
  int ncom = val._wavefun._n;
  int nocc = val._wavefun._p;

  is.read((char*)&ntot_in, sizeof(int));
  is.read((char*)&ncom_in, sizeof(int));
  is.read((char*)&nocc_in, sizeof(int));
  iA (ntot_in == ntot);

  int size = ntot_in * ncom_in * nocc_in;

  if( (ncom_in == ncom) && (nocc_in == nocc) ) {
    is.read((char*)val._wavefun.data(), sizeof(cpx)*size);
  }
  else {
    cpx *ptr = (cpx*) malloc(sizeof(cpx)*size);
    is.read((char*)ptr, sizeof(cpx)*size);
    CpxNumTns tns(ntot_in, ncom_in, nocc_in, false, ptr);
    for (int k=0; k<nocc_in; k++) {
      for (int j=0; j<ncom_in; j++) {
	cpx *ptr0 = tns.clmdata(j,k);
	cpx *ptr1 = val._wavefun.clmdata(j,k);
	for (int i=0; i<ntot_in; i++) *(ptr1++) = *(ptr0++);
      }
    }
    free(ptr);
  }
  return 0;
};


#endif

