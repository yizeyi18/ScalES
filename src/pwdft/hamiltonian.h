//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin, Wei Hu, Weile Jia, and David Williams-Young

/// @file hamiltonian.hpp
/// @brief Hamiltonian class for planewave basis diagonalization method.
/// @date 2012-09-16
#ifndef _HAMILTONIAN_HPP_
#define _HAMILTONIAN_HPP_

#include  "environment.h"
#include  "numvec_impl.hpp"
#include  "domain.h"
#include  "periodtable.h"
#include  "spinor.h"
#include  "utility.h"
#include  "esdf.h"
#include  <xc.h>
#ifdef DEVICE
#include "device_blas.h"
#include "device_fft.h"
#include "device_utility.h"
#ifdef USE_MAGMA
#include "magma.h"
#else
#include "device_solver.h"
#endif
#endif
namespace scales{

// *********************************************************************
// Base Hamiltonian class 
// *********************************************************************

/// @brief Hamiltonian class for Gamma-point Kohn-Sham calculations. 
/// 
/// So far only the restricted Kohn-Sham calculations are supported.
class Hamiltonian {
private:
  Domain                      domain_;
  // List of atoms
  std::vector<Atom>           atomList_;
  Int                         numSpin_;
  Int                         numExtraState_;
  Int                         numOccupiedState_;
  Int                         numDensityComponent_;
  // Type of pseudopotential, default HGH
  std::string                 pseudoType_;
  // Info of the exchange-correlation potential
  std::string                 XCType_;
  Int                         XCId_;
  Int                         XId_;
  Int                         CId_;
  std::string                 XCFamily_;
  bool                        isXCSeparate_;
  // Exchange-correlation potential using libxc package.
  xc_func_type                XCFuncType_; 
  xc_func_type                XFuncType_; 
  xc_func_type                CFuncType_; 
  bool                        XCInitialized_;
 
  // MPI communication 
  // LL: FIXME 01/04/2021 Rethink whether the comms should be here
  MPI_Comm rowComm_, colComm_;

  // Now pseudocharge represent the Gaussian compensation
  // pseudocharge
  DblNumVec                   pseudoCharge_;
  // Short range part of the the local pseudopotential
  DblNumVec                   vLocalSR_;

  DblNumMat                   density_;         
  // Gradient of the density
  std::vector<DblNumMat>      gradDensity_;
  // atomic charge densities
  DblNumVec                   atomDensity_;
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

  // Below are related to hybrid functionals
  bool                        isEXXActive_;


  /// @brief Store all the orbitals for exact exchange calculation
  /// NOTE: This might impose serious memory constraint for relatively
  /// large systems.
  NumTns<Real>                phiEXX_; 
  DblNumMat                   vexxProj_; 
#ifdef DEVICE
  deviceDblNumMat             cu_vexxProj_; 
#endif
  DblNumVec                   exxgkkR2C_;


  /// @brief Screening parameter mu for range separated hybrid functional. Currently hard coded
  const Real                  screenMu_ = 0.106;

  /// @brief Mixing parameter for hybrid functional calculation. Currently hard coded
  const Real                  exxFraction_ = 0.25;

  std::string                 hybridDFType_;
  Real                        hybridDFKmeansTolerance_;
  Int                         hybridDFKmeansMaxIter_;
  Real                        hybridDFNumMu_;
  Real                        hybridDFNumGaussianRandom_;
  Int                         hybridDFNumProcScaLAPACK_;
  Int                         BlockSizeScaLAPACK_;
  Real                        hybridDFTolerance_;

  Int                         exxDivergenceType_;

  Real                        exxDiv_;

  Real                EVdw_;                     // Van der Waals energy
  Real                Eself_;                    // Self energy due to the pseudopotential
  Real                EIonSR_;                   // Short range repulsion energy for Gaussian charge
  Real                Eext_;                     // Energy due to external potential

  // Van der Waals force
  DblNumMat           forceVdw_;                 

  // Ion short range repulsion energy due to the use of // Gaussian
  // compensation charge formulation 
  DblNumMat           forceIonSR_;
  
  // Force due to external potential
  DblNumMat           forceext_;

  // ~~~ * ~~~
  // Internal variables related to wavefunction filter in CheFSI for PWDFT 
  bool apply_filter_;
  bool apply_first_;                           
  Real wfn_cutoff_;

public:

  // *********************************************************************
  // Publicly accessible variables
  // *********************************************************************

  // LL: FIXME 01/04/2021 In the progress of moving setters and getters and clean up the code

  // External potential. TODO not implemented
  DblNumVec                   vext;            


  // *********************************************************************
  // Lifecycle
  // *********************************************************************
  Hamiltonian();
  ~Hamiltonian();

  void Setup (
      const Domain&              dm,
      const std::vector<Atom>&   atomList );


  // *********************************************************************
  // Operations
  // *********************************************************************

  /// @brief Calculate pseudopotential, as well as other atomic related
  /// energies and forces, such as self energy, short range repulsion
  /// energy and VdW energies.
  void CalculatePseudoPotential( PeriodTable &ptable );

  /// @brief Atomic density is implemented using the structure factor
  /// method due to the large cutoff radius
  void CalculateAtomDensity( PeriodTable &ptable, Fourier &fft );

  void CalculateDensity( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier &fft );

  void CalculateGradDensity( Fourier &fft );
  void CalculateGradDensity( Fourier &fft, bool isMPIFFTW );

  void CalculateXC (Real &val, Fourier& fft);

  void CalculateXC (Real &val, Fourier& fft, bool extra);

  void CalculateHartree( Fourier& fft );
  void CalculateHartree( Fourier& fft , bool extra);

  void CalculateVtot( DblNumVec& vtot );

//  /// @brief Calculate the Hellmann-Feynman force for each atom.
//  void CalculateForce ( Spinor& psi, Fourier& fft );

  /// @brief Calculate the Hellmann-Feynman force for each atom.
  /// This is a clean version for computing the force.
  ///
  /// In particular it is very important to calculate the nonlocal
  /// contribution of the force on the fine grid.
  void CalculateForce ( Spinor& psi, Fourier& fft );

  void MultSpinor(Spinor& psi, NumTns<Real>& Hpsi, Fourier& fft);

#ifdef DEVICE
  void MultSpinor(Spinor& psi, deviceNumTns<Real>& Hpsi, Fourier& fft);
  void MultSpinor_old(Spinor& psi, deviceNumTns<Real>& Hpsi, Fourier& fft);
  void ACEOperator( deviceDblNumMat& cu_psi, Fourier& fft, deviceDblNumMat& cu_Hpsi);
#endif
  
  NumTns<Real>& PhiEXX() {return phiEXX_;}

  /// @brief Update phiEXX by the spinor psi. The Phi are normalized in
  /// the real space as
  ///
  /// \int |\phi(x)|^2 dx = 1
  ///
  /// while the wavefunction satisfies the normalization
  ///
  /// \sum |\psi(x)|^2 = 1, differing by a normalization constant. FIXME
  void SetPhiEXX(const Spinor& psi, Fourier& fft);

#ifdef DEVICE
  void CalculateVexxACEGPU( Spinor& psi, Fourier& fft );
  void CalculateVexxACEDFGPU( Spinor& psi, Fourier& fft, bool isFixColumnDF );
#endif 

  /// @brief Construct the ACE operator
  void CalculateVexxACE( Spinor& psi, Fourier& fft );

  /// @brief Construct the ACE operator in the density fitting format.
  void CalculateVexxACEDF( Spinor& psi, Fourier& fft, bool isFixColumnDF );

  Real CalculateEXXEnergy( Spinor& psi, Fourier& fft );

  
  void InitializeEXX( Real ecutWavefunction, Fourier& fft );

  //  void UpdateHybrid ( Int phiIter, const Spinor& psi, Fourier& fft, Real Efock );

  void UpdateHamiltonian ( std::vector<Atom>&  atomList ) { atomList_ = atomList; }

  void SetupXC( std::string XCType );
  
  void DestroyXC();


  /// @brief Calculate ionic self energy and short range repulsion
  /// energy and force
  void  CalculateIonSelfEnergyAndForce( PeriodTable &ptable );

  /// @brief Calculate Van der Waals energy and force (which only depends on the
  /// atomic position)
  void  CalculateVdwEnergyAndForce();


  // *********************************************************************
  // Access
  // *********************************************************************
  DblNumVec&  Vtot() { return vtot_; }
  DblNumMat&  Vxc()  { return vxc_; }
  DblNumVec&  Vhart() { return vhart_; }

  DblNumMat&  Density() { return density_; }
  std::vector<DblNumMat>  GradDensity() { return gradDensity_; }
  DblNumVec&  AtomDensity() { return atomDensity_; }
  DblNumVec&  PseudoCharge() { return pseudoCharge_; }
  std::vector<PseudoPot>& Pseudo() {return pseudo_; };
  DblNumVec&  EigVal() { return eigVal_; }
  DblNumVec&  OccupationRate() { return occupationRate_; }

  std::vector<Atom>&  AtomList() { return atomList_; }

  bool        IsEXXActive() { return isEXXActive_; }
  bool        IsHybrid() { return (XCFamily_ == "Hybrid"); }

  void        SetEXXActive(bool flag) { isEXXActive_ = flag; }

  Real        ScreenMu() { return screenMu_;}
  Real        EXXFraction() { return exxFraction_;}

  Real        EVdw() { return EVdw_; }
  Real        Eself() {return Eself_;}
  Real        EIonSR() {return EIonSR_;}
  Real        Eext() {return Eext_;}
  DblNumMat&  ForceVdw() { return forceVdw_;}
  DblNumMat&  ForceIonSR() {return forceIonSR_;}
  DblNumMat&  ForceExt() {return forceext_;}

  void        SetForceExt(const DblNumMat& forceext) {forceext_ = forceext;}

  bool        XCRequireGradDensity()  {return XCFamily_ == "GGA" or XCFamily_ == "Hybrid"; }
  bool        XCRequireIterateDensity() {return XCFamily_ != "Hybrid" or ( not isEXXActive_ ); }
  bool        XCRequireIterateWavefun()    {return XCFamily_ == "Hybrid" and isEXXActive_; }

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



} // namespace scales


#endif // _HAMILTONIAN_HPP_
