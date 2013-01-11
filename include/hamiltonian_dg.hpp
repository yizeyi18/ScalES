/// @file hamiltonian_dg.hpp
/// @brief The Hamiltonian class for DG calculation.
/// @author Lin Lin
/// @date 2013-01-09
#ifndef _HAMILTONIAN_DG_HPP_
#define _HAMILTONIAN_DG_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "numtns_impl.hpp"
#include  "distvec_impl.hpp"
#include  "domain.hpp"
#include  "periodtable.hpp"
#include  "utility.hpp"
#include  "esdf.hpp"
#include  "fourier.hpp"
#include  <xc.h>

namespace dgdft{

// *********************************************************************
// Partitions
// *********************************************************************

/// @struct ElemPrtn
/// @brief Partition class (used by DistVec) according to the element
/// index.
struct ElemPrtn
{
	IntNumTns                 ownerInfo;

	Int Owner(Index3 key) {
		return ownerInfo(key(0), key(1), key(2));
	}
};


/// @struct AtomPrtn
/// @brief Partition class (used by DistVec) according to the atom
/// index.
struct AtomPrtn 
{
	std::vector<Int> ownerInfo; 

	Int Owner(Int key) {
		return ownerInfo[key];
	}
};

struct PseudoPotential 
{
	/// @brief Pseudocharge of an atom, defined on the uniform grid.
  NumTns<SparseVec>                                    pseudoCharge; 
	/// @brief Nonlocal projectors of an atom, defined on the LGL grid.
	std::vector<std::pair<NumTns<SparseVec>, Real> >     vnlList;
};


// *********************************************************************
// Typedefs
// *********************************************************************

typedef DistVec<Index3, DblNumVec, ElemPrtn>   DistDblNumVec;

typedef DistVec<Index3, CpxNumVec, ElemPrtn>   DistCpxNumVec;

typedef DistVec<Index3, DblNumMat, ElemPrtn>   DistDblNumMat;

typedef DistVec<Index3, CpxNumMat, ElemPrtn>   DistCpxNumMat;

typedef DistVec<Index3, DblNumTns, ElemPrtn>   DistDblNumTns;

typedef DistVec<Index3, CpxNumTns, ElemPrtn>   DistCpxNumTns;

typedef DistVec<Int, PseudoPotential, AtomPrtn>  DistPseudoPotential;

// *********************************************************************
// Main class
// *********************************************************************

/// @class HamiltonianDG 
/// @brief Main class of DG for storing and assembling the DG matrix.
class HamiltonianDG {
private:
	
	// *********************************************************************
	// Physical variables
	// *********************************************************************
	/// @brief Global domain.
	Domain                      domain_;

	/// @brief Element subdomains.
	NumTns<Domain>              domainElem_;

	/// @brief Uniform grid in the global domain
	std::vector<DblNumVec>      uniformGrid_;

	/// @brief Number of uniform grids in each element.  
	///
	/// Note: It must be satisifed that
	///
	/// domain_.numGrid[d] = numUniformGridElem_[d] * numElem_[d]
	Index3                      numUniformGridElem_;

	/// @brief Number of LGL grids in each element.
	Index3                      numLGLGridElem_;

	/// @brief Uniform grid in the elements, each has size 
	/// numUniformGridElem_
	NumTns<std::vector<DblNumVec> >   uniformGridElem_;

	/// @brief Legendre-Gauss-Lobatto grid in the elements, each has size
	/// numLGLGridElem_
	NumTns<std::vector<DblNumVec> >   LGLGridElem_;

	/// @brief List of atoms.
	std::vector<Atom>           atomList_;
	/// @brief Number of extra states for fractional occupation number.
	Int                         numExtraState_;
	/// @brief Number of occupied states.
	Int                         numOccupiedState_;
	/// @brief Type of pseudopotential, default is HGH
	std::string                 pseudoType_;
	/// @brief Id of the exchange-correlation potential
	Int                         XCId_;
	/// @brief Exchange-correlation potential using libxc package.
	xc_func_type                XCFuncType_; 




	// *********************************************************************
	// Computational variables
	// *********************************************************************

	/// @brief The number of elements.
	Index3                      numElem_;

	/// @brief Partition of element.
	ElemPrtn                    elemPrtn_;

	/// @brief Partition of atom.
	AtomPrtn                    atomPrtn_;

	/// @brief Interior penalty parameter.
	Real                        penaltyAlpha_;

	/// @brief 1D distribution of the electron density which is compatible
	/// with the input DistFourier structure.
	DblNumVec                   densityLocal_;
	/// @brief 1D distribution of the pseudo charge which is compatible
	/// with the input DistFourier structure.
	DblNumVec                   pseudoChargeLocal_;
	/// @brief 1D distribution of the Hartree potential which is compatible
	/// with the input DistFourier structure.
	DblNumVec                   vhartLocal_;

	/// @brief Pseudocharge in the global domain. 
	DistDblNumVec    pseudoCharge_;

	/// @brief Electron density in the global domain. No magnitization for
	/// DG calculation.
	DistDblNumVec    density_;

	/// @brief External potential in the global domain. (not implemented)
	DistDblNumVec    vext_;

	/// @brief Hartree potential in the global domain.
	DistDblNumVec    vhart_;

	/// @brief Exchange-correlation potential in the global domain. No
	/// magnization calculation in the DG code.
	DistDblNumVec    vxc_;

	/// @brief Exchange-correlation energy density in the global domain.
	DistDblNumVec    epsxc_;

	/// @brief Total potential in the global domain.
	DistDblNumVec    vtot_;


	/// @brief Total potential on the local LGL grid.
	DistDblNumVec    vtotLGL_;

	/// @brief Basis functions on the local LGL grid.
	DistDblNumMat    basisLGL_;

	/// @brief Pseudopotential and nonlocal projectors associated with
	/// each atom.
	DistPseudoPotential  pseudo_;

public:

	// *********************************************************************
	// Lifecycle
	// *********************************************************************
	HamiltonianDG() {}
	~HamiltonianDG() {}
	HamiltonianDG( const esdf::ESDFInputParam& esdfParam );


	// *********************************************************************
	// Operations
	// *********************************************************************

	void CalculatePseudoPotential( PeriodTable &ptable );
	//
	//	virtual void CalculateNonlocalPP( PeriodTable &ptable ) = 0;
	//
	//	virtual void CalculateDensity( const Spinor &psi, const DblNumVec &occrate, Real &val ) = 0;
	//
	//	virtual void CalculateXC (Real &val) = 0;
	//
	void CalculateHartree( DistFourier& fft );
	//
	//	virtual void CalculateVtot( DblNumVec& vtot ) = 0;
	//

	// *********************************************************************
	// Access
	// *********************************************************************

	/// @brief Total potential in the global domain.
	DistDblNumVec&  Vtot() { return vtot_; }

	/// @brief Exchange-correlation potential in the global domain. No
	/// magnization calculation in the DG code.
	DistDblNumVec&  Vxc()  { return vxc_; }

	/// @brief Hartree potential in the global domain.
	DistDblNumVec&  Vhart() { return vhart_; }

	DistDblNumVec&  Density() { return density_; }

	DistDblNumVec&  PseudoCharge() { return pseudoCharge_; }
	
	std::vector<Atom>&  AtomList() { return atomList_; }

	// *********************************************************************
	// Inquiry
	// *********************************************************************

	Int NumStateTotal() const { return numExtraState_ + numOccupiedState_; }

	Int NumOccupiedState() const { return numOccupiedState_; }

	Int NumExtraState() const { return numExtraState_; }

};



} // namespace dgdft


#endif // _HAMILTONIAN_HPP_
