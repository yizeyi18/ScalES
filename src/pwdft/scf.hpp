//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin, Wei Hu, Amartya Banerjee

/// @file scf.hpp
/// @brief SCF class for the global domain or extended element.
/// @date 2012-10-25 Initial version
/// @date 2014-02-01 Dual grid implementation
/// @date 2014-08-07 Parallelization for PWDFT
/// @date 2016-01-19 Add hybrid functional
/// @date 2016-04-08 Update mixing

#ifndef _SCF_HPP_ 
#define _SCF_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "hamiltonian.hpp"
#include  "spinor.hpp"
#include  "periodtable.hpp"
#include  "esdf.hpp"
#include  "eigensolver.hpp"

// TODO In the case of K-point, should the occupation rate, the energy
// and the potential subroutines be moved to SCF class rather than
// stay in the Hamiltonian class?

namespace scales{

class SCF
{
private:
  // Control parameters
  Int                 mixMaxDim_;
  std::string         mixType_;
  Real                mixStepLength_;            
  Real                eigMinTolerance_;
  Real                eigTolerance_;
  Int                 eigMaxIter_;
  Real                scfTolerance_;
  Int                 scfMaxIter_;
  Int                 scfPhiMaxIter_;
  Real                scfPhiTolerance_;
  Int                 numUnusedState_;
  bool                isEigToleranceDynamic_;
  Int                 BlockSizeScaLAPACK_;

  std::string         restartDensityFileName_;
  std::string         restartPotentialFileName_;
  std::string         restartWfnFileName_;

  // Physical parameters
  Real                Tbeta_;                    // Inverse of temperature in atomic unit
  Real                Efree_;                    // Helmholtz free energy
  Real                EfreeHarris_;              // Free energy from Harris functional
  Real                Etot_;                     // Total energy
  Real                Ekin_;                     // Kinetic energy
  Real                Ehart_;                    // Hartree (correction) energy
  Real                Ecor_;                     // Nonlinear correction energy
  Real                Exc_;                      // Exchange-correlation energy
  Real                EVdw_;                     // Van der Waals energy
  Real                EVxc_;                     // Exchange-correlation potential energy
  Real                Eself_;                    // Self energy due to the pseudopotential
  Real                EIonSR_;                   // Short range repulsion energy for Gaussian charge
  Real                Eext_;                     // External energy
  Real                fermi_;                    // Fermi energy
  Real                Efock_;                    // Hartree-Fock energy

  Real                totalCharge_;              // Total number of computed electron charge

  EigenSolver*        eigSolPtr_;
  PeriodTable*        ptablePtr_;


  // SCF variables
  DblNumVec           vtotNew_;
  Real                scfNorm_;                 // ||V_{new} - V_{old}|| / ||V_{old}||
  Real                efreeDifPerAtom_;         // Difference between free energy and 
                                                // Harris free energy per atom
  // for Anderson iteration
  DblNumMat           dfMat_;
  DblNumMat           dvMat_;

  /// @brief Work array for the mixing variable in the inner iteration.
  DblNumVec           mixSave_;


  Index3  numGridWavefunctionElem_;
  Index3  numGridDensityElem_;



  // Chebyshev Filtering variables
  bool Diag_SCF_PWDFT_by_Cheby_;
  Int First_SCF_PWDFT_ChebyFilterOrder_;
  Int First_SCF_PWDFT_ChebyCycleNum_;
  Int General_SCF_PWDFT_ChebyFilterOrder_;
  bool PWDFT_Cheby_use_scala_;
  bool PWDFT_Cheby_apply_wfn_ecut_filt_;


  // Do the usual Chebyshev filtering schedule or work in ionic movement mode
  Int Cheby_iondynamics_schedule_flag_;


public:

  // *********************************************************************
  // Life-cycle
  // *********************************************************************
  SCF();
  ~SCF();

  // *********************************************************************
  // Operations
  // *********************************************************************
  // Basic parameters. Density and wavefunction
  void  Setup( EigenSolver& eigSol, PeriodTable& ptable ); 

  void  Execute();

  void  IterateDensity();
  
  void  IterateWavefun();
  
  // Solve the linear eigenvalue problem and compute potential etc
  void  InnerSolve( Int iter ); 
 
  void  Update();

  void  CalculateOccupationRate ( DblNumVec& eigVal, DblNumVec& occupationRate );
  void  CalculateEnergy();
  void  CalculateHarrisEnergy();



  void  PrintState( const Int iter );

  // Mixing
  void  AndersonMix( 
      Int iter,
      Real            mixStepLength,
      std::string     mixType,
      DblNumVec&      vMix,
      DblNumVec&      vOld,
      DblNumVec&      vNew,
      DblNumMat&      dfMat,
      DblNumMat&      dvMat );

  void  KerkerPrecond(
      DblNumVec&  precResidual,
      const DblNumVec&  residual );

  /// @brief Update the parameters for SCF during the MD simulation
  /// This is done by modifying the global esdfParam parameters
  void UpdateMDParameters( );

  /// @brief Update the parameters for SCF during the TDDFT simulation
  /// This is done by modifying the global esdfParam parameters
  void UpdateTDDFTParameters( );

  // *********************************************************************
  // Inquiry
  // *********************************************************************
  // Energy etc.
  Real Etot() const  {return Etot_;};    
  Real Efree() const {return Efree_;};    
  Real Efock() const {return Efock_;};    

  Real Fermi() const {return fermi_;};    

  void UpdateEfock( Real Efock ) {Efock_ = Efock; Etot_ -= Efock; Efree_ -= Efock;}

  // Setup the Cheby-iondynamics flag
  void set_Cheby_iondynamics_schedule_flag(int flag){Cheby_iondynamics_schedule_flag_ = flag;}


}; // -----  end of class  SCF ----- 


} // namespace scales
#endif // _SCF_HPP_

