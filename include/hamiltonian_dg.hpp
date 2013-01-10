/// @file hamiltonian_dg.hpp
/// @brief The Hamiltonian class for DG calculation.
/// @author Lin Lin
/// @date 2013-01-09
#ifndef _HAMILTONIAN_DG_HPP_
#define _HAMILTONIAN_DG_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "numtns_impl.hpp"
#include  "dist_numvec_impl.hpp"
#include  "domain.hpp"
#include  "periodtable.hpp"
#include  "utility.hpp"
#include  "esdf.hpp"
#include  "fourier.hpp"
#include  <xc.h>

namespace dgdft{




/// @class HamiltonianDG 
/// @brief Main class of DG for storing and assembling the DG matrix.
class HamiltonianDG {
private:
	/// @brief Global domain.
	Domain                      domain_;
	NumTns<Domain>              elementTns_;
	NumTns<Domain>              extendedElementTns_;

	/// @brief 1D distribution of the electron density which is compatible
	/// with the input DistFourier structure.
	DblNumVec                   densityLocal_;
	/// @brief 1D distribution of the pseudo charge which is compatible
	/// with the input DistFourier structure.
	DblNumVec                   pseudoChargeLocal_;
	/// @brief 1D distribution of the Hartree potential which is compatible
	/// with the input DistFourier structure.
	DblNumVec                   vhartLocal_;

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

	/// @brief Pseudo charge in the global domain.
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

	/// @brief Pseudocharge list associated to each atom.
	std::vector<SparseVec>      pseudoChargeList_; 

	/// @brief Nonlocal pseudopotential list.
	///
	/// First index: atom
	/// Second index: nonlocal pseudopotential
	std::vector<std::vector<NonlocalPP> >    vnlDoubleList_;

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

	void CalculatePseudoCharge( PeriodTable &ptable );
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
