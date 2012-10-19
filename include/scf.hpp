#ifndef _SCF_HPP_ 
#define _SCF_HPP_

#include  "environment_impl.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "hamiltonian.hpp"
#include  "spinor.hpp"
#include  "periodtable.hpp"
#include  "esdf.hpp"
#include  "eigensolver.hpp"
#include  "utility.hpp"
#include	"blas.hpp"
#include	"lapack.hpp"

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
	Real                Ecor_;                     // Nonlinear correction energy
	Real                Exc_;                      // Exchange-correlation energy
	Real                fermi_;                    // Fermi energy

	Real                totalCharge_;              // Total number of computed electron charge
	
	EigenSolver*        eigSolPtr_;
	PeriodTable*        ptablePtr_;

	// SCF variables
  DblNumVec           vtotNew_;
	Real                scfNorm_;                 // ||V_{new} - V_{old}|| / ||V_{old}||
	// for Anderson iteration
	DblNumMat           dfMat_;
	DblNumMat           dvMat_;
	// TODO Elliptic preconditioner

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

	void  CalculateOccupationRate ( DblNumVec& eigVal, DblNumVec& occupationRate );
	void  CalculateEnergy();
	void  PrintState( const Int iter );
	void  OutputState();

	// Mixing
	// TODO
	void  AndersonMix( const Int iter );
	void  KerkerMix();
	// TODO
//	void  EllipticMix();

	// *********************************************************************
	// Inquiry
	// *********************************************************************
	// Energy etc.
	

}; // -----  end of class  SCF ----- 



// *********************************************************************
// Other subroutines
// *********************************************************************
void PrintInitialState( const esdf::ESDFInputParam& esdfParam );


} // namespace dgdft
#endif // _SCF_HPP_

