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
/// @file scf.hpp
/// @brief SCF class for the global domain or extended element.
/// @date 2012-10-25
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

namespace dgdft{

class SCF
{
private:
	// Control parameters
	Int                 mixMaxDim_;
	std::string         mixType_;
	Real                mixStepLength_;            
  Real                eigTolerance_;
  Int                 eigMaxIter_;
	Real                scfTolerance_;
	Int                 scfMaxIter_;
	Int                 scfPhiMaxIter_;
	Real                scfPhiTolerance_;
  Int                 numUnusedState_;
  bool                isEigToleranceDynamic_;
	bool                isRestartDensity_;
	bool                isRestartWfn_;
	bool                isOutputDensity_;
	bool                isOutputWfn_;

  bool                isCalculateForceEachSCF_;
  
	std::string         restartDensityFileName_;
  std::string         restartWfnFileName_;

	// Physical parameters
	Real                Tbeta_;                    // Inverse of temperature in atomic unit
	Real                Efree_;                    // Helmholtz free energy
	Real                Etot_;                     // Total energy
	Real                Ekin_;                     // Kinetic energy
	Real                Ehart_;                    // Hartree energy
	Real                Ecor_;                     // Nonlinear correction energy
	Real                Exc_;                      // Exchange-correlation energy
	Real                Evdw_;                     // Van der Waals energy
	Real                EVxc_;                     // Exchange-correlation potential energy
	Real                Eself_;                    // Self energy due to the pseudopotential
	Real                fermi_;                    // Fermi energy
  Real                Efock_;                    // Hartree-Fock energy

	Real                totalCharge_;              // Total number of computed electron charge
	
  EigenSolver*        eigSolPtr_;
  PeriodTable*        ptablePtr_;

  std::string         XCType_;
  std::string         VDWType_;
  
  /// @brief Needed for GGA, meta-GGA and hybrid functional calculations
  bool                isCalculateGradRho_; 

  // SCF variables
  DblNumVec           vtotNew_;
	Real                scfNorm_;                 // ||V_{new} - V_{old}|| / ||V_{old}||
	// for Anderson iteration
	DblNumMat           dfMat_;
	DblNumMat           dvMat_;
	// TODO Elliptic preconditioner


  Index3  numGridWavefunctionElem_;
  Index3  numGridDensityElem_;
	
  DblNumMat           forceVdw_;
	
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
	void  Setup( const esdf::ESDFInputParam& esdfParam, EigenSolver& eigSol, PeriodTable& ptable ); 
	void  Iterate();
	void  Update();

	void  CalculateOccupationRate ( DblNumVec& eigVal, DblNumVec& occupationRate );
	void  CalculateEnergy();
	void  CalculateVDW ( Real& VDWEnergy, DblNumMat& VDWForce );
  // EXX
	void  IterateHybrid();
  void  CalculateEXXEnergy( Real& fockEnergy );



	void  PrintState( const Int iter );
	void  OutputState();
	void  LastSCF( Real& etot, Real& efree, Real& ekin, Real& ehart,
    Real& eVxc, Real& exc, Real& evdw, Real& eself, Real& ecor,
    Real& fermi, Real& totalCharge, Real& scfNorm );

	// Mixing
	void  AndersonMix( const Int iter );
	void  KerkerMix();

  // 

//	void  EllipticMix();

	// *********************************************************************
	// Inquiry
	// *********************************************************************
	// Energy etc.
	

}; // -----  end of class  SCF ----- 


} // namespace dgdft
#endif // _SCF_HPP_

