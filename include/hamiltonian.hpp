#ifndef _HAMILTONIAN_HPP_
#define _HAMILTONIAN_HPP_

#include  "environment_impl.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"
#include  "spinor.hpp"
//#include  <xc.h>

namespace dgdft{

// *********************************************************************
// Base Hamiltonian class 
// *********************************************************************

class Hamiltonian {
private:
	Domain                   domain_;
	std::vector<Atom>        atomList_;
	Int                      numExtraState_;
	Int                      numOccupiedState_;
	Int                      numTotalState_;

	std::string              pseudoType_;
	std::string              PWSolver_;                 // Type of exchange-correlation functional
	std::string              XCType_;
	std::vector<SparseVec>   pseudoChargeList_;   // The list of pseudocharge for each atom.

	std::vector<std::vector<NonlocalPP> >  vnlList_;  // The list of nonlocal pseudopotential for each atom (first argument) for each nonlocal pseudopotential (second argument)


	Vec                      pseudoCharge_;       // total pseudocharge
	Vec                      density_;            // total electron density
	std::vector<Vec>         magnetization_;      // magnitization (x,y,z)
	Vec                      vxc_;                // scalar part of the exchange-correlation potential
	std::vector<Vec>         Bxc_;                // magnetization part of the exchange-correlation potential
	Vec                      vext_;
	Vec                      vhart_;
	Vec                      vtot_;
	Vec                      epsxc_;    // the exchange-correlation density vector
	Int                      numDensityComponent_; // 1, 2, or 4
public:
  Hamiltonian();
  virtual ~Hamiltonian();
	// TODO
  Hamiltonian(const Domain &dm, 
			const std::vector<Atom> &atvec, 
			const Int nexstate, 
			const Int numDensityComponent);
  
  virtual void SetDensity(Spinor &psi, DblNumVec &occrate, double &val);

  // density dependent potentials
  void SetHartree(FFTPrepare &fp);
  virtual void SetXC(xc_func_type& XCFunc, double &val);
  virtual void SetPseudoCharge(PeriodTable &ptable);

  // density independet potentials
	void SetExternal();
  virtual void SetNonLocalPP(PeriodTable &ptable, int cnt);
  virtual void SetVtot();
  virtual void SetVtot(DblNumVec &vtot);
    
  // H*Psi
  virtual void MultSpinor(Spinor &psi0, CpxNumTns &a3, FFTPrepare &fp);
};

int generate_uniformmesh(Domain &dm, vector<DblNumVec> &gridpos);
} // namespace dgdft


#endif // _HAMILTONIAN_HPP_
