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
/// @file hamiltonian.hpp
/// @brief Hamiltonian class for planewave basis diagonalization method.
/// @date 2012-09-16
#ifndef _HAMILTONIAN_HPP_
#define _HAMILTONIAN_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "periodtable.hpp"
#include  "spinor.hpp"
#include  "utility.hpp"
#include  "esdf.hpp"
#include  <xc.h>

namespace dgdft{

// *********************************************************************
// Base Hamiltonian class 
// *********************************************************************




/// @brief Pure virtual class for handling different types of
/// Hamiltonian.
class Hamiltonian {
protected:
  Domain                      domain_;
  // List of atoms
  std::vector<Atom>           atomList_;
  Int                         numSpin_;
  Int                         numExtraState_;
  Int                         numOccupiedState_;
  Int                         numDensityComponent_;
  // Type of pseudopotential, default HGH
  std::string                 pseudoType_;
  // Id of the exchange-correlation potential
  std::string                 XCType_;
  Int                         XCId_;
  Int                         XId_;
  Int                         CId_;
  // Exchange-correlation potential using libxc package.
  xc_func_type                XCFuncType_; 
  xc_func_type                XFuncType_; 
  xc_func_type                CFuncType_; 
  bool                        XCInitialized_;

  // Pseudocharge to represent the local pseudopotential
  DblNumVec                   pseudoCharge_;
  // density_(:,1)    electron density
  // density_(:,2-4)  magnetization along x,y,z directions
  DblNumMat                   density_;         
  std::vector<DblNumMat>      gradDensity_;
  // External potential. TODO not implemented
  DblNumVec                   vext_;            
  // Hartree potential
  DblNumVec                   vhart_;           
  // vxc_(:,1)        exchange-correlation potential
  // vxc_(:,2-4)      vector exchange-correlation potential along x,y,z
  // directions
  // The same dimension as the dimension of density_
  DblNumMat                   vxc_;             
  // Total potential
  DblNumVec                   vtot_;        
  // the exchange-correlation energy density
  DblNumVec                   epsxc_; 

  // Pseudopotential for each atom
  std::vector<PseudoPot>      pseudo_;

  // Eigenvalues
  DblNumVec                   eigVal_;
  // Occupation rate
  DblNumVec                   occupationRate_;

  // EXX variables
  bool                        isHybrid_;
  bool                        isEXXActive_;

  /// @brief Screening parameter mu for range separated hybrid functional. Currently hard coded
  const Real                  screenMu_ = 0.106;

  /// @brief Mixing parameter for hybrid functional calculation. Currently hard coded
  const Real                  exxFraction_ = 0.25;

  bool                        isHybridACE_;
  bool                        isHybridDF_;
  Real                        numMuHybridDF_;
  Real                        numGaussianRandomHybridDF_;
  Int                         numProcScaLAPACKPotrfHybridDF_;

  Int                         exxDivergenceType_;

  Real                        exxDiv_;

  // ~~~ * ~~~
  // Internal variables related to wavefunction filter in CheFSI for PWDFT 
  bool apply_filter_;
  bool apply_first_;                           
  Real wfn_cutoff_;

public:

  // *********************************************************************
  // Lifecycle
  // *********************************************************************
  Hamiltonian() {}
  virtual ~Hamiltonian() {}

  virtual void Setup (
      const esdf::ESDFInputParam& esdfParam,
      const Domain&              dm,
      const std::vector<Atom>&   atomList ) = 0;


  // *********************************************************************
  // Operations
  // *********************************************************************

  virtual void CalculatePseudoPotential( PeriodTable &ptable ) = 0;

  virtual void CalculateDensity( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier &fft ) = 0;

  virtual void CalculateGradDensity( Fourier &fft ) = 0;

  virtual void CalculateXC (Real &val, Fourier& fft) = 0;

  virtual void CalculateHartree( Fourier& fft ) = 0;

  virtual void CalculateVtot( DblNumVec& vtot ) = 0;

  /// @brief Calculate the Hellmann-Feynman force for each atom.
  virtual void CalculateForce ( Spinor& psi, Fourier& fft ) = 0;

  /// @brief Calculate the Hellmann-Feynman force for each atom.
  /// LL 2015/05/02:
  /// This is a clean version for computing the force
  virtual void CalculateForce2 ( Spinor& psi, Fourier& fft ) = 0;

  // Matrix vector multiplication
  virtual void MultSpinor(Spinor& psi, NumTns<Real>& a3, Fourier& fft) = 0;

  virtual NumTns<Real>& PhiEXX() = 0;

  virtual void InitializeEXX( Real ecutWavefunction, Fourier& fft ) = 0;

  virtual void SetPhiEXX(const Spinor& psi, Fourier& fft) = 0;

  virtual void CalculateVexxACE( Spinor& psi, Fourier& fft ) = 0;

  virtual void CalculateVexxACEDF( Spinor& psi, Fourier& fft, bool isFixColumnDF ) = 0;

  virtual Real CalculateEXXEnergy( Spinor& psi, Fourier& fft ) = 0;

  //  virtual void UpdateHybrid ( Int phiIter, const Spinor& psi, Fourier& fft, Real Efock ) = 0;

  void UpdateHamiltonian ( std::vector<Atom>&  atomList ) { atomList_ = atomList; }


  // *********************************************************************
  // Access
  // *********************************************************************
  DblNumVec&  Vtot() { return vtot_; }
  DblNumVec&  Vext() { return vext_; }
  DblNumMat&  Vxc()  { return vxc_; }
  DblNumVec&  Vhart() { return vhart_; }

  DblNumMat&  Density() { return density_; }
  std::vector<DblNumMat>  GradDensity() { return gradDensity_; }
  DblNumVec&  PseudoCharge() { return pseudoCharge_; }
  std::vector<PseudoPot>& Pseudo() {return pseudo_; };
  DblNumVec&  EigVal() { return eigVal_; }
  DblNumVec&  OccupationRate() { return occupationRate_; }

  std::vector<Atom>&  AtomList() { return atomList_; }

  bool        IsEXXActive() { return isEXXActive_; }
  bool        IsHybrid() { return isHybrid_; }

  void        SetEXXActive(bool flag) { isEXXActive_ = flag; }

  Real        ScreenMu() { return screenMu_;}
  Real        EXXFraction() { return exxFraction_;}
  bool        IsHybridACE() { return isHybridACE_; }
  bool        IsHybridDF()  { return isHybridDF_; }


  // Functions to set and toggle state of filter application
  void set_wfn_filter(int apply_filter, int apply_first, Real wfn_cutoff)
  {
    apply_filter_ = apply_filter;
    apply_first_ = apply_first;
    wfn_cutoff_ = wfn_cutoff;
  }
  void set_wfn_filter(int apply_first)
  {
    apply_first_ = apply_first;
  }


  // *********************************************************************
  // Inquiry
  // *********************************************************************
  Int  NumSpin() const { return numSpin_; }
  Int  NumStateTotal() const { return numExtraState_ + numOccupiedState_; }
  Int  NumOccupiedState() const { return numOccupiedState_; }
  Int  NumExtraState() const { return numExtraState_; }
  Int  NumDensityComponent() const { return density_.n(); }

};


// *********************************************************************
// One-component Kohn-Sham class
// *********************************************************************
/// @brief Detailed implementation of one-component (spin-restricted)
/// Kohn-Sham calculations.
class KohnSham: public Hamiltonian {
private: 

  /// @brief Store all the orbitals for exact exchange calculation
  /// NOTE: This might impose serious memory constraint for relatively
  /// large systems.
  NumTns<Real>                phiEXX_; 
  DblNumMat                   vexxProj_; 
  DblNumVec                   exxgkkR2C_;

public:

  // *********************************************************************
  // Lifecycle
  // *********************************************************************
  KohnSham();
  ~KohnSham();

  virtual void Setup (
      const esdf::ESDFInputParam& esdfParam,
      const Domain&              dm,
      const std::vector<Atom>&   atomList );

  // *********************************************************************
  // Operations
  // *********************************************************************

  virtual void CalculatePseudoPotential( PeriodTable &ptable );

  virtual void CalculateDensity( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier& fft );

  virtual void CalculateGradDensity( Fourier& fft );

  virtual void CalculateXC ( Real &val, Fourier& fft );

  virtual void CalculateHartree( Fourier& fft );

  virtual void CalculateVtot( DblNumVec& vtot );

  /// @brief Calculate the Hellmann-Feynman force for each atom.
  virtual void CalculateForce ( Spinor& psi, Fourier& fft );

  /// @brief Calculate the Hellmann-Feynman force for each atom.
  /// LL 2015/05/02:
  /// This is a clean version for computing the force
  virtual void CalculateForce2 ( Spinor& psi, Fourier& fft );

  // Matrix vector multiplication
  virtual void MultSpinor(Spinor& psi, NumTns<Real>& a3, Fourier& fft);


  /// @brief Update phiEXX by the spinor psi. The Phi are normalized in
  /// the real space as
  ///
  /// \int |\phi(x)|^2 dx = 1
  ///
  /// while the wavefunction satisfies the normalization
  ///
  /// \sum |\psi(x)|^2 = 1, differing by a normalization constant. FIXME
  virtual void SetPhiEXX(const Spinor& psi, Fourier& fft);

  virtual NumTns<Real>& PhiEXX() {return phiEXX_;}

  /// @brief Construct the ACE operator
  virtual void CalculateVexxACE( Spinor& psi, Fourier& fft );

  /// @brief onstruct the ACE operator in the density fitting format.
  virtual void CalculateVexxACEDF( Spinor& psi, Fourier& fft, bool isFixColumnDF );

  virtual Real CalculateEXXEnergy( Spinor& psi, Fourier& fft );

  virtual void InitializeEXX( Real ecutWavefunction, Fourier& fft );

  // Not implemented yet
  //  virtual void UpdateHybrid ( Int phiIter, const Spinor& psi, Fourier& fft, Real Efock );

};



} // namespace dgdft


#endif // _HAMILTONIAN_HPP_
