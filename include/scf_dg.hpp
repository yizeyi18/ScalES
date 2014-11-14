/*
	 Copyright (c) 2012 The Regents of the University of California,
	 through Lawrence Berkeley National Laboratory.  

   Author: Lin Lin and Wei Hu
	 
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
/// @file scf_dg.hpp
/// @brief Self consistent iteration using the DG method.
/// @date 2013-02-05
/// @date 2014-08-06 Intra-element parallelization
#ifndef _SCF_DG_HPP_ 
#define _SCF_DG_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "hamiltonian.hpp"
#include  "spinor.hpp"
#include  "periodtable.hpp"
#include  "esdf.hpp"
#include  "eigensolver.hpp"
#include  "utility.hpp"
#include  "hamiltonian_dg.hpp"
#include  "hamiltonian_dg_conversion.hpp"

namespace dgdft{

class SCFDG
{
private:
	// Control parameters
	Int                 mixMaxDim_;
	std::string         mixType_;
	Real                mixStepLength_;            
  Real                eigTolerance_;
  Int                 eigMaxIter_;
  Real                scfInnerTolerance_;
	Int                 scfInnerMaxIter_;
  /// @brief Criterion for convergence using Efree rather than the
  /// potential difference.
	Real                scfOuterEnergyTolerance_;
	Real                scfOuterTolerance_;
	Int                 scfOuterMaxIter_;
	Real                scfNorm_;                 // ||V_{new} - V_{old}|| / ||V_{old}||
  Int                 numUnusedState_;
	Real                SVDBasisTolerance_;
  bool                isEigToleranceDynamic_;
	bool                isRestartDensity_;
	bool                isRestartWfn_;
	bool                isOutputDensity_;
	bool                isOutputWfnElem_;
	bool                isOutputWfnExtElem_;
	bool                isOutputPotExtElem_; 
	bool                isCalculateAPosterioriEachSCF_;
	bool                isCalculateForceEachSCF_;
	bool                isOutputHMatrix_;
	Real                ecutWavefunction_;
	Real                densityGridFactor_;        
	Real                LGLGridFactor_;

	bool                isPeriodizePotential_;
  Point3              distancePeriodize_;

  std::string         restartDensityFileName_;
  std::string         restartWfnFileName_;
  std::string         XCType_;
  /// @brief Same as @ref esdf::ESDFInputParam::solutionMethod
  std::string         solutionMethod_;

  // PEXSI parameters
#ifdef _USE_PEXSI_
  PPEXSIPlan          pexsiPlan_;
  PPEXSIOptions       pexsiOptions_;
#endif


  /// @brief The total number of processors used by PEXSI.
  /// 
  /// Let npPerPole_ = numProcRowPEXSI_ * numProcColPEXSI_, then
  /// The size of pexsiComm_ is 
  ///
  /// min( int(mpisize/npPerPole_)*npPerPole_, 
  /// npPerPole_ * numPole ),
  ///
  /// which is the maximum number of processors that can be used by
  /// PEXSI.
  ///
  /// This assumes that the number of processors used by PEXSI is
  /// smaller than the number of processors used by DG.
  Int                 numProcTotalPEXSI_;
  /// @brief Communicator used only by PEXSI.  
  ///
  MPI_Comm            pexsiComm_;
  /// @brief Whether PEXSI has been initialized.
  bool                isPEXSIInitialized_;
  /// @brief The number of row processors used by PEXSI for each pole.
  Int                 numProcRowPEXSI_;
  /// @brief The number of column processors used by PEXSI for each pole.
  Int                 numProcColPEXSI_;

  Int                 inertiaCountSteps_;
  // Minimum of the tolerance for the inertia counting in the
  // dynamically adjustment strategy
  Real                muInertiaToleranceTarget_; 
  // Minimum of the tolerance for the PEXSI solve in the
  // dynamically adjustment strategy
  Real                numElectronPEXSIToleranceTarget_;

	// Physical parameters
	Real                Tbeta_;                    // Inverse of temperature in atomic unit
	Real                EfreeHarris_;              // Helmholtz free energy defined through Harris energy functional
	Real                EfreeSecondOrder_;         // Second order accurate Helmholtz free energy 
	Real                Efree_;                    // Helmholtz free energy (KS energy functional)
	Real                Etot_;                     // Total energy (KSenergy functional)
	Real                Ekin_;                     // Kinetic energy
	Real                Ehart_;                    // Hartree energy
	Real                Ecor_;                     // Nonlinear correction energy
	Real                Exc_;                      // Exchange-correlation energy
	Real                EVxc_;                     // Exchange-correlation potential energy
	Real                Eself_;                    // Self energy due to the pseudopotential
	Real                fermi_;                    // Fermi energy

  /// @brief Number of processor rows and columns for ScaLAPACK
  Int                 dmRow_;
  Int                 dmCol_;

  Int                 numProcScaLAPACK_;

  // Density matrices

  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distDMMat_;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distEDMMat_;
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      distFDMMat_;

	PeriodTable*        ptablePtr_;

	HamiltonianDG*      hamDGPtr_;

	DistVec<Index3, EigenSolver, ElemPrtn>*   distEigSolPtr_;

	DistFourier*        distfftPtr_;

	// SCF variables
	std::string         mixVariable_;

	/// @brief Work array for the old mixing variable in the outer iteration.
	DistDblNumVec       mixOuterSave_;
	/// @brief Work array for the old mixing variable in the inner iteration.
	DistDblNumVec       mixInnerSave_;
	// TODO Remove dfOuterMat_, dvOuterMat_
	/// @brief Work array for the Anderson mixing in the outer iteration.
	DistDblNumMat       dfOuterMat_;
	/// @brief Work array for the Anderson mixing in the outer iteration.
	DistDblNumMat       dvOuterMat_;
	/// @brief Work array for the Anderson mixing in the inner iteration.
	DistDblNumMat       dfInnerMat_;
	/// @brief Work array for the Anderson mixing in the inner iteration.
	DistDblNumMat       dvInnerMat_;
	/// @brief Work array for updating the local potential on the LGL
	/// grid.
	DistDblNumVec       vtotLGLSave_;
	
	Int                 scfTotalInnerIter_;       // For the purpose of Anderson mixing
	Real                scfInnerNorm_;            // ||V_{new} - V_{old}|| / ||V_{old}||
	Real                scfOuterNorm_;            // ||V_{new} - V_{old}|| / ||V_{old}||

	/// @brief Global domain.
	Domain              domain_;

	Index3              numElem_;

	Index3              extElemRatio_;

	/// @brief Partition of element.
	ElemPrtn            elemPrtn_;

	Int                 scaBlockSize_;

	/// @brief Interpolation matrix from uniform grid in the extended
	/// element with periodic boundary condition to LGL grid in each
	/// element (assuming all the elements are the same).
	std::vector<DblNumMat>    PeriodicUniformToLGLMat_;
  std::vector<DblNumMat>    PeriodicUniformFineToLGLMat_;

	/// @brief Context for BLACS.
	Int                 contxt_;

public:
	
	
	// *********************************************************************
	// Life-cycle
	// *********************************************************************
	SCFDG();
	~SCFDG();
  
	// *********************************************************************
	// Operations
	// *********************************************************************
	
	/// @brief Setup the basic parameters for initial SCF iteration.
	void  Setup( 
			const esdf::ESDFInputParam& esdfParam, 
			HamiltonianDG& hamDG,
			DistVec<Index3, EigenSolver, ElemPrtn>&  distEigSol,
			DistFourier&   distfft,
			PeriodTable&   ptable,
		  Int            contxt ); 

  /// @brief Update the basic parameters for SCF interation for MD and
  /// geometry optimization.
  void  Update( ); 


	/// @brief Main self consistent iteration subroutine.
	void  Iterate();

	/// @brief Inner self consistent iteration subroutine without
	/// correcting the basis functions.
	void  InnerIterate( Int outerIter );

	/// @brief Update the local potential in the extended element and the element.
	void  UpdateElemLocalPotential();

	/// @brief Calculate the occupation rate and the Fermi energy given
	/// the eigenvalues
	void  CalculateOccupationRate ( DblNumVec& eigVal, DblNumVec&
			occupationRate );

	/// @brief Interpolate the uniform grid in the periodic extended
	/// element domain to LGL grid in each element.
	void InterpPeriodicUniformToLGL( const Index3& numUniformGrid,
			const Index3& numLGLGrid, const Real* psiUniform, Real* psiLGL );
  
	void InterpPeriodicUniformFineToLGL( const Index3& numUniformGridFine,
			const Index3& numLGLGrid, const Real* rhoUniform, Real* rhoLGL );

	/// @brief Calculate the Kohn-Sham energy and other related energies.
	void  CalculateKSEnergy();

	/// @brief Calculate the Kohn-Sham energy and other related energies
  /// using the energy density matrix and the free energy density matrix.
	void  CalculateKSEnergyDM(
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distEDMMat,
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distFDMMat );


	/// @brief Calculate the Harris (free) energy.  
	///
	/// The difference between the Kohn-Sham energy and the Harris energy
	/// is that the nonlinear correction term in the Harris energy
	/// functional must be computed via the input electron density, rather
	/// than the output electron density or the mixed electron density.
	///
	/// Reference:
	///
	/// [Soler et al. "The SIESTA method for ab initio order-N
	/// materials", J. Phys. Condens. Matter. 14, 2745 (2002) pp 18]
	void  CalculateHarrisEnergy();


	/// @brief Calculate the Harris (free) energy using density matrix and
  /// free energy density matrix.  
  ///
  /// @see CalculateHarrisEnergy
	void  CalculateHarrisEnergyDM(
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distFDMMat );


	/// @brief Calculate the second order accurate energy that is
	/// applicable to both density and potential mixing.
	///
	/// Reference:
	///
	/// Research note, "On the understanding and generalization of Harris
	/// energy functional", 08/26/2013.
	void  CalculateSecondOrderEnergy();

	/// @brief Print out the state variables at each SCF iteration.
	void  PrintState(  );

	/// @brief Parallel preconditioned Anderson mixing. Can be used for
	/// potential mixing or density mixing.
	void  AndersonMix( 
			Int             iter, 
			Real            mixStepLength,
			std::string     mixType,
			DistDblNumVec&  distvMix,
			DistDblNumVec&  distvOld,
			DistDblNumVec&  distvNew,
			DistDblNumMat&  dfMat,
			DistDblNumMat&  dvMat);
	
	/// @brief Parallel Kerker preconditioner. Can be used for
	/// potential mixing or density mixing.
	void  KerkerPrecond(
		DistDblNumVec&  distPrecResidual,
		const DistDblNumVec&  distResidual );



	// *********************************************************************
	// Inquiry
	// *********************************************************************
  Real Efree() const {return Efree_;};	
  
  Real Fermi() const {return fermi_;};	
	
  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& DMMat() {return distDMMat_;};

}; // -----  end of class  SCFDG ----- 



} // namespace dgdft
#endif // _SCF_HPP_

