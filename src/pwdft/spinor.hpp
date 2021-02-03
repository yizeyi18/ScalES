//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin, Wei Hu, Weile Jia

/// @file spinor.hpp
/// @brief Spinor (wavefunction) for the global domain or extended
/// element.
/// @date 2012-10-06
#ifndef _SPINOR_HPP_
#define _SPINOR_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "numtns_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "utility.hpp"
#include  "esdf.hpp"

#ifdef DEVICE
#include  "device_numvec_impl.hpp"
#include  "device_numtns_impl.hpp"
#include  "device_blas.hpp"
#include  "device_fft.hpp"
#include  "device_utility.hpp"
#endif

namespace scales{

class Spinor {
private:
  //Domain            domain_;                // mesh should be used here for general cases 
  std::shared_ptr<Domain> domain_ = nullptr;
  std::shared_ptr<Fourier> fft_   = nullptr;


  NumTns<Real>      wavefun_;               // Local data of the wavefunction 
  IntNumVec         wavefunIdx_;
  Int               numStateTotal_;
  Int               blocksize_;
#ifdef DEVICE
  // not use wavefun_ in the DEVICE implementation.
  deviceNumTns<Real>   cu_wavefun_;
#endif

  // For density fitting
  Int               numMu_;
  IntNumVec         pivQR_;
  DblNumMat         G_;
  IntNumVec         numProcPotrf_;

public:
  // *********************************************************************
  // Constructor and destructor
  // *********************************************************************
  Spinor() = delete; 
  ~Spinor();

  Spinor( std::shared_ptr<Fourier> fft, const Int numComponent, const Int numStateTotal, Int numStateLocal,
      const Real val = static_cast<Real>(0) );

  Spinor( std::shared_ptr<Fourier> fft, const Int numComponent, const Int numStateTotal, Int numStateLocal,
      const bool owndata, Real* data );
#ifdef DEVICE
  // Weile, needs further consideration.
  Spinor( const Domain &dm, const Int numComponent, const Int numStateTotal, Int numStateLocal,
      const bool owndata, Real* data, bool isGPU);
  void SetupGPU( const Domain &dm, const Int numComponent, const Int numStateTotal, const Int numStateLocal,
      const bool owndata, Real* data );
#endif

#if 0
  void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal, const Int numStateLocal,
      const Real val = static_cast<Real>(0) ); 

  void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal, const Int numStateLocal,
      const bool owndata, Real* data );
#endif

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
#ifdef DEVICE
  deviceNumTns<Real>& cuWavefun() { return cu_wavefun_; } 
  const deviceNumTns<Real>& cuWavefun() const { return cu_wavefun_; } 
#endif
  Real& Wavefun(const Int i, const Int j, const Int k) {return wavefun_(i,j,k); }
  const Real& Wavefun(const Int i, const Int j, const Int k) const {return wavefun_(i,j,k); }

  // *********************************************************************
  // Access
  // *********************************************************************

  // *********************************************************************
  // Operations
  // *********************************************************************
  void Normalize();

  // Perform all operations of matrix vector multiplication.
  void AddMultSpinor( const DblNumVec& vtot, 
      const std::vector<PseudoPot>& pseudo, NumTns<Real>& Hpsi );
  
  // LL: 1/3/2021
  // This function requires the coarse grid to be an odd number along each
  // direction, and therefore should be deprecated in the future.
  void AddMultSpinorR2C( const DblNumVec& vtot, 
      const std::vector<PseudoPot>& pseudo, NumTns<Real>& Hpsi );

#ifdef DEVICE 
  void AddMultSpinorR2C( const DblNumVec& vtot, 
      const std::vector<PseudoPot>& pseudo, deviceNumTns<Real>& Hpsi );
  void AddTeterPrecond( deviceNumTns<Real>& Hpsi );
#endif

  void AddTeterPrecond( NumTns<Real>& Hpsi );

  /// @brief Apply the exchange operator to the spinor by solving
  /// Poisson like equations
  /// EXX: Spinor with exact exchange. 
  /// Keeping the names separate is good for now, since the new
  /// algorithm requires a different set of input parameters for AddMultSpinor
  void AddMultSpinorEXX ( 
      const NumTns<Real>& phi,
      const DblNumVec& exxgkkR2CFine,
      Real  exxFraction,
      Real  numSpin,
      const DblNumVec& occupationRate,
      NumTns<Real>& Hpsi );

#ifdef DEVICE
  /// @brief Apply the exchange operator to the spinor by solving
  /// Poisson like equations
  /// EXX: Spinor with exact exchange. 
  /// Keeping the names separate is good for now, since the new
  /// algorithm requires a different set of input parameters for AddMultSpinor
  void AddMultSpinorEXX ( 
      const NumTns<Real>& phi,
      const DblNumVec& exxgkkR2CFine,
      Real  exxFraction,
      Real  numSpin,
      const DblNumVec& occupationRate,
      deviceNumTns<Real>& Hpsi );
#endif

  /// @brief Spinor with exact exchange, and the cost is reduced using density fitting schemes.
  /// The density fitting uses the interpolative separable density fitting method
  ///
  /// J. Lu, L. Ying, Compression of the electron repulsion integral tensor
  /// in tensor hypercontraction format with cubic scaling cost, J.
  /// Comput. Phys. 302 (2015) 329–335. 
  ///
  /// Only sequential version is implemented
  void AddMultSpinorEXXDF ( 
      const NumTns<Real>& phi,
      const DblNumVec& exxgkkR2C,
      Real  exxFraction,
      Real  numSpin,
      const DblNumVec& occupationRate,
      const Real numMuFac,
      const Real numGaussianRandomFac,
      const Int numProcScaLAPACKPotrf, 
      const Int scaPotrfBlockSize, 
      NumTns<Real>& Hpsi,
      NumMat<Real>& VxMat, 
      bool isFixColumnDF );


//  void AddMultSpinorEXXDF2 ( 
//      const NumTns<Real>& phi,
//      const DblNumVec& exxgkkR2C,
//      Real  exxFraction,
//      Real  numSpin,
//      const DblNumVec& occupationRate,
//      const Real numMuFac,
//      const Real numGaussianRandomFac,
//      const Int numProcScaLAPACKPotrf, 
//      const Int scaPotrfBlockSize, 
//      NumTns<Real>& Hpsi,
//      NumMat<Real>& VxMat, 
//      bool isFixColumnDF );
//
//  void AddMultSpinorEXXDF3 ( 
//      const NumTns<Real>& phi,
//      const DblNumVec& exxgkkR2C,
//      Real  exxFraction,
//      Real  numSpin,
//      const DblNumVec& occupationRate,
//      const Real numMuFac,
//      const Real numGaussianRandomFac,
//      const Int numProcScaLAPACKPotrf, 
//      const Int scaPotrfBlockSize, 
//      NumTns<Real>& Hpsi,
//      NumMat<Real>& VxMat, 
//      bool isFixColumnDF );
//
//  void AddMultSpinorEXXDF4 ( 
//      const NumTns<Real>& phi,
//      const DblNumVec& exxgkkR2C,
//      Real  exxFraction,
//      Real  numSpin,
//      const DblNumVec& occupationRate,
//      const Real numMuFac,
//      const Real numGaussianRandomFac,
//      const Int numProcScaLAPACKPotrf, 
//      const Int scaPotrfBlockSize, 
//      NumTns<Real>& Hpsi,
//      NumMat<Real>& VxMat, 
//      bool isFixColumnDF );
//
//  void AddMultSpinorEXXDF5 ( 
//      const NumTns<Real>& phi,
//      const DblNumVec& exxgkkR2C,
//      Real  exxFraction,
//      Real  numSpin,
//      const DblNumVec& occupationRate,
//      const Real numMuFac,
//      const Real numGaussianRandomFac,
//      const Int numProcScaLAPACKPotrf, 
//      const Int scaPotrfBlockSize, 
//      NumTns<Real>& Hpsi,
//      NumMat<Real>& VxMat, 
//      bool isFixColumnDF );
  
  void AddMultSpinorEXXDF6 ( 
      const NumTns<Real>& phi,
      const DblNumVec& exxgkkR2C,
      Real  exxFraction,
      Real  numSpin,
      const DblNumVec& occupationRate,
      const Real numMuFac,
      const Real numGaussianRandomFac,
      const Int numProcScaLAPACKPotrf, 
      const Real hybridDFTolerance,
      const Int scaPotrfBlockSize, 
      NumTns<Real>& Hpsi,
      NumMat<Real>& VxMat, 
      bool isFixColumnDF );

  void AddMultSpinorEXXDF7 ( 
      const NumTns<Real>& phi,
      const DblNumVec& exxgkkR2C,
      Real  exxFraction,
      Real  numSpin,
      const DblNumVec& occupationRate,
      std::string hybridDFType_,
      Real  hybridDFKmeansTolerance, 
      Int   hybridDFKmeansMaxIter, 
      const Real numMuFac,
      const Real numGaussianRandomFac,
      const Int numProcScaLAPACKPotrf, 
      const Real hybridDFTolerance,
      const Int scaPotrfBlockSize, 
      NumTns<Real>& Hpsi,
      NumMat<Real>& VxMat, 
      bool isFixColumnDF );
#ifdef DEVICE
  void AddMultSpinorEXXDF3_GPU ( 
      const NumTns<Real>& phi,
      const DblNumVec& exxgkkR2C,
      Real  exxFraction,
      Real  numSpin,
      const DblNumVec& occupationRate,
      const Real numMuFac,
      const Real numGaussianRandomFac,
      const Int numProcScaLAPACKPotrf, 
      const Int scaPotrfBlockSize, 
      deviceDblNumMat & cu_Hpsi,
      NumMat<Real>& VxMat, 
      bool isFixColumnDF );

#endif


};  // Spinor


} // namespace scales




#endif // _SPINOR_HPP_
