#ifndef _SPINOR_HPP_
#define _SPINOR_HPP_

#include  "environment.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"
#include  "numtns_impl.hpp"
#include  "fourier.hpp"
#include  "utility.hpp"

namespace dgdft{

class Spinor {
private:
	Domain            domain_;                // mesh should be used here for general cases 
	NumTns<Scalar>    wavefun_;               // Local data of the wavefunction 

public:
	// *********************************************************************
	// Constructor and destructor
	// *********************************************************************
	Spinor(); 
	
	Spinor( const Domain &dm, const Int numComponent, const Int numState,
			const Scalar val = static_cast<Scalar>(0) );

	Spinor( const Domain &dm, const Int numComponent, const Int numState,
			const bool owndata, Scalar* data );

	~Spinor();

	// *********************************************************************
	// Inquiries
	// *********************************************************************
	Int NumGridTotal() const { return wavefun_.m(); }
	Int NumComponent() const { return wavefun_.n(); }
	Int NumState()     const { return wavefun_.p(); }

	NumTns<Scalar>& Wavefun() { return wavefun_; } 
	const NumTns<Scalar>& Wavefun() const { return wavefun_; } 
	Scalar& Wavefun(const Int i, const Int j, const Int k) {return wavefun_(i,j,k); }
	const Scalar& Wavefun(const Int i, const Int j, const Int k) const {return wavefun_(i,j,k); }


	// *********************************************************************
	// Access
	// *********************************************************************

	// *********************************************************************
	// Operators
	// *********************************************************************
	Spinor& operator = (const Spinor& psi); 

	// *********************************************************************
	// Operations
	// *********************************************************************
	void Normalize();

  void AddScalarDiag (const Int iocc, const DblNumVec &val, NumTns<Scalar>& a3);
  void AddScalarDiag (const DblNumVec &val, NumTns<Scalar> &a3);
//  void AddScalarDiag (const IntNumVec &activeIndex, DblNumVec &val, NumTns<Scalar> &a3);
//  void AddNonlocalPseudo(Int iocc, vector< vector< pair<SparseVec,double> > > &val, DblNumTns &a3);
//  int add_nonlocalPS (vector< vector< pair<SparseVec,double> > > &val, DblNumTns &a3);
//  int add_nonlocalPS (IntNumVec &active_ind, vector< vector< pair<SparseVec,double> > > &val, DblNumTns &a3);

	void AddLaplacian (NumTns<Scalar>& a3, Fourier* fftPtr);
	void AddNonlocalPP (const std::vector<std::vector<NonlocalPP> >& vnlDoubleList, NumTns<Scalar> &a3);

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


};  // Spinor

////////////

// IMPORTANT: inline function must be defined OUT of the scope of
// the namespace.
//inline int serialize(const Spinor &val, std::ostream &os, 
//		const vector<int> &mask) {
//	serialize(val.isNormalized_, os, mask);
//	serialize(val._domain._Ls,    os, mask);
//	serialize(val._domain._Ns,    os, mask);
//	serialize(val._domain._pos,   os, mask);
//	serialize(val._wavefun,       os, mask);
//	return 0;
//};
//
//inline int serialize(const COMPLEX::Spinor &val, ostream &os, 
//		const vector<int> &mask) {
//	serialize(val._is_normalized, os, mask);
//	serialize(val._domain._Ls,    os, mask);
//	serialize(val._domain._Ns,    os, mask);
//	serialize(val._domain._pos,   os, mask);
//	serialize(val._wavefun,       os, mask);
//	return 0;
//};
//
//inline int deserialize(REAL::Spinor &val, istream &is, 
//		const vector<int> &mask) {
//	deserialize(val._is_normalized, is, mask);
//	deserialize(val._domain._Ls,    is, mask);
//	deserialize(val._domain._Ns,    is, mask);
//	deserialize(val._domain._pos,   is, mask);
//	deserialize(val._wavefun,       is, mask);
//	return 0;
//};
//
//inline int deserialize(COMPLEX::Spinor &val, istream &is, 
//		const vector<int> &mask) {
//	deserialize(val._is_normalized, is, mask);
//	deserialize(val._domain._Ls,    is, mask);
//	deserialize(val._domain._Ns,    is, mask);
//	deserialize(val._domain._pos,   is, mask);
//	deserialize(val._wavefun,       is, mask);
//	return 0;
//};
//
//inline int deserialize(COMPLEX::Spinor &val, istream &is, 
//		const vector<int> &mask, int &ntot_in, int &ncom_in, int &nocc_in) {
//	deserialize(val._is_normalized, is, mask);
//	deserialize(val._domain._Ls,    is, mask);
//	deserialize(val._domain._Ns,    is, mask);
//	deserialize(val._domain._pos,   is, mask);
//
//	int ntot = val._wavefun._m;
//	int ncom = val._wavefun._n;
//	int nocc = val._wavefun._p;
//
//	is.read((char*)&ntot_in, sizeof(int));
//	is.read((char*)&ncom_in, sizeof(int));
//	is.read((char*)&nocc_in, sizeof(int));
//	iA (ntot_in == ntot);
//
//	int size = ntot_in * ncom_in * nocc_in;
//
//	if( (ncom_in == ncom) && (nocc_in == nocc) ) {
//		is.read((char*)val._wavefun.data(), sizeof(cpx)*size);
//	}
//	else {
//		cpx *ptr = (cpx*) malloc(sizeof(cpx)*size);
//		is.read((char*)ptr, sizeof(cpx)*size);
//		CpxNumTns tns(ntot_in, ncom_in, nocc_in, false, ptr);
//		for (int k=0; k<nocc_in; k++) {
//			for (int j=0; j<ncom_in; j++) {
//				cpx *ptr0 = tns.clmdata(j,k);
//				cpx *ptr1 = val._wavefun.clmdata(j,k);
//				for (int i=0; i<ntot_in; i++) *(ptr1++) = *(ptr0++);
//			}
//		}
//		free(ptr);
//	}
//	return 0;
//};

} // namespace dgdft




#endif // _SPINOR_HPP_
