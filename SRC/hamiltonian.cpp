#include  "hamiltonian.hpp"

namespace  dgdft{

void
Hamiltonian::Hamiltonian	(const Domain &dm, 
			const std::vector<Atom> &atvec, 
			const Int nexstate, 
			const Int numDensityComponent)
{
#ifndef _RELEASE_
	PushCallStack("Hamiltonian::Hamiltonian");
#endif

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method Hamiltonian::Hamiltonian  ----- 


void
Hamiltonian::SetDensity	( 
		const Spinor& psi,
	 	)
{
#ifndef _RELEASE_
	PushCallStack("Hamiltonian::SetDensity");
#endif

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method Hamiltonian::SetDensity  ----- 

} // namespace dgdft
