/// @file scf_dg.hpp
/// @brief Self consistent iteration using the DF method.
/// @author Lin Lin
/// @date 2013-02-05
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

namespace dgdft{

class SCFDG
{
private:
	// Control parameters
	Int                 mixMaxDim_;
	std::string         mixType_;
	Real                mixStepLength_;            
	Real                scfTolerance_;
	Int                 scfMaxIter_;
	bool                isRestartDensity_;
	bool                isRestartWfn_;
	bool                isOutputDensity_;
	bool                isOutputWfn_;
  
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
	Real                EVxc_;                     // Exchange-correlation potential energy
	Real                Eself_;                    // Self energy due to the pseudopotential
	Real                fermi_;                    // Fermi energy

	Real                totalCharge_;              // Total number of computed electron charge
	
	PeriodTable*        ptablePtr_;

	HamiltonianDG*      hamDGPtr_;

	DistVec<Index3, EigenSolver, ElemPrtn>*   distEigSolPtr_;

	DistFourier*        distfftPtr_;

	// SCF variables
	DistDblNumVec       vtotNew_;
	
	Real                scfNorm_;                 // ||V_{new} - V_{old}|| / ||V_{old}||

	/// @brief Global domain.
	Domain                      domain_;

	Index3                      numElem_;

	/// @brief Partition of element.
	ElemPrtn                    elemPrtn_;

	Int                 scaBlockSize_;

	Int                 numALB_;

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
			PeriodTable&   ptable ); 

	void  Iterate();
//
//	void  CalculateOccupationRate ( DblNumVec& eigVal, DblNumVec& occupationRate );

	/// @brief Calculate all energies for output
	void  CalculateEnergy();
	
	/// @brief Print out the state variables at each SCF iteration.
	void  PrintState( const Int iter );

//	void  OutputState();

	// Mixing
//	void  AndersonMix( const Int iter );
//	void  KerkerMix();

	// *********************************************************************
	// Inquiry
	// *********************************************************************
	// Energy etc.
	

}; // -----  end of class  SCFDG ----- 



} // namespace dgdft
#endif // _SCF_HPP_

