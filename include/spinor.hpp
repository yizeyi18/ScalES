/*
	 Copyright (c) 2012 The Regents of the University of California,
	 through Lawrence Berkeley National Laboratory.  

   Author: Lin Lin
	 
   This file is part of DGDFT. All rights reserved.

	 Redistribution and use in source and binary forms, with or without
	 modification, are permitted provided that the following conditions are met:

	 (1) Redistributions of source code must retain the above copyright notice, this
	 list of conditions and the following disclaimer.
	 (2) Redistributions in binary form must reproduce the above copyright notice,
	 this list of conditions and the following disclaimer in the documentation
	 and/or other materials provided with the distribution.
	 (3) Neither the name of the University of California, Lawrence Berkeley
	 National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
	 be used to endorse or promote products derived from this software without
	 specific prior written permission.

	 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
	 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
	 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	 You are under no obligation whatsoever to provide any bug fixes, patches, or
	 upgrades to the features, functionality or performance of the source code
	 ("Enhancements") to anyone; however, if you choose to make your Enhancements
	 available either publicly, or directly to Lawrence Berkeley National
	 Laboratory, without imposing a separate written license agreement for such
	 Enhancements, then you hereby grant the following license: a non-exclusive,
	 royalty-free perpetual license to install, use, modify, prepare derivative
	 works, incorporate into other computer software, distribute, and sublicense
	 such enhancements or derivative works thereof, in binary and source code form.
*/
/// @file spinor.hpp
/// @brief Spinor (wavefunction) for the global domain or extended
/// element.
/// @date 2012-10-06
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
  IntNumVec         wavefunIdx_;
  Int               numStateTotal_;
  Int               blocksize_;

public:
	// *********************************************************************
	// Constructor and destructor
	// *********************************************************************
	Spinor(); 
	
	Spinor( const Domain &dm, const Int numComponent, const Int numStateTotal, Int numStateLocal,
      const Scalar val = static_cast<Scalar>(0) );

	Spinor( const Domain &dm, const Int numComponent, const Int numStateTotal, Int numStateLocal,
      const bool owndata, Scalar* data );

	~Spinor();

	void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal, const Int numStateLocal,
      const Scalar val = static_cast<Scalar>(0) ); 

	void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal, const Int numStateLocal,
      const bool owndata, Scalar* data );

	// *********************************************************************
	// Inquiries
	// *********************************************************************
	Int NumGridTotal()  const { return wavefun_.m(); }
	Int NumComponent()  const { return wavefun_.n(); }
	Int NumState()      const { return wavefun_.p(); }
  Int NumStateTotal() const { return numStateTotal_; }
  Int Blocksize()     const { return blocksize_; }
  
  IntNumVec&  WavefunIdx() { return wavefunIdx_; }
  const IntNumVec&  WavefunIdx() const { return wavefunIdx_; }
  Int&  WavefunIdx(const Int k) { return wavefunIdx_(k); }
  const Int&  WavefunIdx(const Int k) const { return wavefunIdx_(k); }

	NumTns<Scalar>& Wavefun() { return wavefun_; } 
	const NumTns<Scalar>& Wavefun() const { return wavefun_; } 
	Scalar& Wavefun(const Int i, const Int j, const Int k) {return wavefun_(i,j,k); }
	const Scalar& Wavefun(const Int i, const Int j, const Int k) const {return wavefun_(i,j,k); }


	// *********************************************************************
	// Access
	// *********************************************************************

	// *********************************************************************
	// Operations
	// *********************************************************************
	void Normalize();

  // Perform all operations of matrix vector multiplication on a fine grid.
  void AddMultSpinorFine( Fourier& fft, const DblNumVec& vtot, 
      const std::vector<PseudoPot>& pseudo, NumTns<Scalar>& a3 );
  void AddMultSpinorFineR2C( Fourier& fft, const DblNumVec& vtot, 
      const std::vector<PseudoPot>& pseudo, NumTns<Scalar>& a3 );

  void AddScalarDiag (Int iocc, const DblNumVec &val, NumMat<Scalar>& y);
  void AddScalarDiag (const DblNumVec &val, NumTns<Scalar> &a3);

	void AddLaplacian (Int iocc, Fourier* fftPtr, NumMat<Scalar>& y);
	void AddLaplacian (Fourier* fftPtr, NumTns<Scalar>& a3);

	void AddNonlocalPP (Int iocc, const std::vector<PseudoPot>& pseudo, NumMat<Scalar>& y);
	void AddNonlocalPP (const std::vector<PseudoPot>& pseudo, NumTns<Scalar> &a3);

//	void AddNonlocalPPFine (Fourier* fftPtr, const std::vector<PseudoPot>& pseudo, NumTns<Scalar> &a3);


  void AddTeterPrecond( Int iocc, Fourier* fftPtr, NumTns<Scalar>& a3 );
  void AddTeterPrecond( Fourier* fftPtr, NumTns<Scalar>& a3 );

  void AddMultSpinorEXX ( Fourier& fft, const DblNumVec& vtot, 
    const std::vector<PseudoPot>& pseudo, 
    const NumTns<Scalar>& phi,
    const bool isEXXActive,
    NumTns<Scalar>& a3 );


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
