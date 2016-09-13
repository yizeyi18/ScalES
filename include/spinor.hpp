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
#include  "lapack.hpp"

namespace dgdft{

class Spinor {
private:
  Domain            domain_;                // mesh should be used here for general cases 
  NumTns<Real>      wavefun_;               // Local data of the wavefunction 
  IntNumVec         wavefunIdx_;
  Int               numStateTotal_;
  Int               blocksize_;

  // For density fitting
  Int               numMu_;
  IntNumVec         pivQR_;

public:
  // *********************************************************************
  // Constructor and destructor
  // *********************************************************************
  Spinor(); 

  Spinor( const Domain &dm, const Int numComponent, const Int numStateTotal, Int numStateLocal,
      const Real val = static_cast<Real>(0) );

  Spinor( const Domain &dm, const Int numComponent, const Int numStateTotal, Int numStateLocal,
      const bool owndata, Real* data );

  ~Spinor();

  void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal, const Int numStateLocal,
      const Real val = static_cast<Real>(0) ); 

  void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal, const Int numStateLocal,
      const bool owndata, Real* data );

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

  NumTns<Real>& Wavefun() { return wavefun_; } 
  const NumTns<Real>& Wavefun() const { return wavefun_; } 
  Real& Wavefun(const Int i, const Int j, const Int k) {return wavefun_(i,j,k); }
  const Real& Wavefun(const Int i, const Int j, const Int k) const {return wavefun_(i,j,k); }


  // *********************************************************************
  // Access
  // *********************************************************************

  // *********************************************************************
  // Operations
  // *********************************************************************
  void Normalize();

  // Perform all operations of matrix vector multiplication on a fine grid.
  void AddMultSpinorFine( Fourier& fft, const DblNumVec& vtot, 
      const std::vector<PseudoPot>& pseudo, NumTns<Real>& a3 );
  void AddMultSpinorFineR2C( Fourier& fft, const DblNumVec& vtot, 
      const std::vector<PseudoPot>& pseudo, NumTns<Real>& a3 );

  void AddTeterPrecond( Fourier* fftPtr, NumTns<Real>& a3 );

  /// @brief Apply the exchange operator to the spinor by solving
  /// Poisson like equations
  /// EXX: Spinor with exact exchange. 
  /// Keeping the names separate is good for now, since the new
  /// algorithm requires a different set of input parameters for AddMultSpinor
  void AddMultSpinorEXX ( Fourier& fft,
      const NumTns<Real>& phi,
      const DblNumVec& exxgkkR2CFine,
      Real  exxFraction,
      Real  numSpin,
      const DblNumVec& occupationRate,
      NumTns<Real>& a3 );

  /// @brief Spinor with exact exchange, and the cost is reduced using density fitting schemes.
  /// The density fitting uses the interpolative separable density fitting method
  ///
  /// J. Lu, L. Ying, Compression of the electron repulsion integral tensor
  /// in tensor hypercontraction format with cubic scaling cost, J.
  /// Comput. Phys. 302 (2015) 329–335. 
  ///
  /// Only sequential version is implemented
  void AddMultSpinorEXXDF ( Fourier& fft, 
      const NumTns<Real>& phi,
      const DblNumVec& exxgkkR2C,
      Real  exxFraction,
      Real  numSpin,
      const DblNumVec& occupationRate,
      const Real numMuFac,
      NumTns<Real>& a3,
      NumMat<Real>& VxMat, 
      bool isFixColumnDF );


};  // Spinor


} // namespace dgdft




#endif // _SPINOR_HPP_
