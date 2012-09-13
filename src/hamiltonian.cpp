#include  "hamiltonian.hpp"

namespace dgdft{

Hamiltonian::Hamiltonian() {};
Hamiltonian::~Hamiltonian() {};

Hamiltonian::Hamiltonian	( 
			const Domain                   &dm, 
			const std::vector<Atom>        &atomList, 
			const std::string               pseudoType,
			const xc_func_type             &XCFuncType; 
			const Int                       numExtraState, 
      const Int                       numDensityComponent):
  domain_(dm), atomList_(atomList), pseudoType_(pseudoType),
	XCFuncType_(XCFuncType), numExtraState_(numExtraState)
{
#ifndef _RELEASE_
	PushCallStack("Hamiltonian::Hamiltonian");
#endif
  Int ntot = domain_.NumGridTotal();

	density_.Resize( ntot, numDensityComponent );   
	SetValue( density_, 0.0 );

	pseudoCharge_.Resize( ntot );
	SetValue( pseudoCharge_, 0.0 );
	
	vext_.Resize( ntot );
	SetValue( vext_, 0.0 );

	vhart_.Resize( ntot );
	SetValue( vhart_, 0.0 );

	vtot_.Resize( ntot );
	SetValue( vtot_, 0.0 );

	epsxc_.Resize( ntot );
	SetValue( epsxc_, 0.0 );

	vxc_.Resize( ntot, numDensityComponent );
	SetValue( vxc_, 0.0 );

#ifndef _RELEASE_
	PopCallStack();
#endif
	return ;
} 		// -----  end of method Hamiltonian::Hamiltonian  ----- 


// Virtual functions
void Hamiltonian::CalculateDensity () { return; } 	
void Hamiltonian::CalculateXC() { return; } 	
void Hamiltonian::CalculatePseudoCharge() { return; } 	
void CalculateNonlocalPS(PeriodTable &ptable, int cnt);

} // namespace dgdft
