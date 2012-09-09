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


namespace dgdft{


// *********************************************************************
// The data used by SCF
// *********************************************************************
struct  SCFData{
	Int                 mixMaxDim;
	std::string         mixType;
	Real                mixStepLength;            
	Real                scfTolerance;
	Int                 scfMaxIter;
	bool                isRestartDensity;
	bool                isRestartWfn;
	bool                isOutputDensity;
	bool                isOutputWfn;

	Real                Tbeta;                    // Inverse of temperature in atomic unit


	Real                Efree;
	Real                Etot;
	Real                Ekin;
	Real                Ecor;
	Real                Exc;
	Real                fermi;                    // Fermi level (mu)

	Real                totalCharge;


	// TODO XC
};

class SCF
{
private:
	SCFData             scfData_;
	EigenSolver*        eigSolPtr_;
	const PeriodTable*  ptablePtr_;

	// SCF variables
	Vec                 vtotNew_;
	// Use for anderson iteration
	std::vector<Vec*>   dfMat_;
	std::vector<Vec*>   dvMat_;
	// TODO Elliptic preconditioner

public:
	// *********************************************************************
	// Life-cycle
	// *********************************************************************
	SCF();
	SCF( EigenSolver &eigSol, const PeriodTable& ptable );
	~SCF();
  
	// *********************************************************************
	// Operations
	// *********************************************************************
	// Basic parameters. Density and wavefunction
	void  Setup( const esdf::ESDFInputParam& esdfparam ); 
	void  Iterate();

	void  CalEnergy();
	void  Print( std::ostream &os );
	void  PrintState( std::ostream &os );

	// Mixing
	void  AndersonMix( Int iter );
	// TODO
	void  KerkerMix();
	// TODO
	void  EllipticMix();

	// *********************************************************************
	// Inquiry
	// *********************************************************************
	// Energy etc.
	

}; // -----  end of class  SCF ----- 

} // namespace dgdft
#endif // _SCF_HPP_

