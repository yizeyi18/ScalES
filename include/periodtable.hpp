/// @file periodtable.hpp
/// @brief Periodic table and its entries.
/// @author Lin Lin
/// @date 2012-08-10
#ifndef _PERIODTABLE_HPP_
#define _PERIODTABLE_HPP_

#include "environment.hpp"
#include "tinyvec_impl.hpp"
#include "numvec_impl.hpp"
#include "nummat_impl.hpp"
#include "numtns_impl.hpp"
#include "utility.hpp"

namespace dgdft{


// *********************************************************************
// Atomic information
// *********************************************************************
/// @struct Atom
/// @brief Atom structure.
struct Atom
{
	/// @brief Atomic number.
  Int          type;                            
	/// @brief Position.
  Point3       pos;  
	/// @brief Velocity.
  Point3       vel;  
	/// @brief Force.
  Point3       force; 

	Atom( const Int t, const Point3 p, const Point3 v, const Point3 f): 
		type(t), pos(p), vel(v), force(f) {}
  ~Atom() {;}
};

/// @namespace PTParam
/// @brief Index of the parameters in each entry of the periodic table.
namespace PTParam{
enum {
	/// @brief Atomic number.
	ZNUC   = 0,
	/// @brief Nuclear mass.
	MASS   = 1,
	/// @brief Nuclear charge (valence).
	ZION   = 2,
	/// @brief Self-interaction energy.
	ESELF  = 3
};
}

/// @namespace PTSample
/// @brief Index of the radial grid, the pseudocharge and derivative
/// of the pseudocharge in samples and cuts.
namespace PTSample{
enum {
	RADIAL_GRID       = 0,
	PSEUDO_CHARGE     = 1,
	DRV_PSEUDO_CHARGE = 2,
	NONLOCAL          = 3,
};
}

/// @namespace PTType
/// @brief Type of each sample, including radial grid, pseudocharge, and
/// nonlocal projectors of different angular momentum, and the
/// spin-orbit contribution. 
namespace PTType{
enum{
	RADIAL            = 9,
	PSEUDO_CHARGE     = 99,
	L0                = 0,
	L1                = 1,
	L2                = 2,
  L3                = 3,
  SPINORBIT_L1 	    = -1,
	SPINORBIT_L2      = -2,
	SPINORBIT_L3      = -3
};
};

/// @brief The start of the radial grid.
const Real MIN_RADIAL = 1e-8;

/// @struct PTEntry
/// @brief Each entry for the periodic table structure.
struct PTEntry
{
	/// @brief Parameters following the order in PTParam.
	DblNumVec params; 
	/// @brief Radial grid, pseudocharge and nonlocal projectors and
	/// their derivatives.
	DblNumMat samples; 
	/// @brief Weight of each sample. Only used for the nonlocal projectors. 
	DblNumVec weights; 
	/// @brief Type of each sample, following PTType.
	IntNumVec types; 
	/// @brief Cutoff value for different sample, following PTSample.
	DblNumVec cutoffs; 
};

Int serialize(const PTEntry&, std::ostream&, const std::vector<Int>&);
Int deserialize(PTEntry&, std::istream&, const std::vector<Int>&);
Int combine(PTEntry&, PTEntry&);


/// @class PeriodTable
/// @brief Periodic table for pseudopotentials.
class PeriodTable
{
private:
	/// @brief Map from atomic number to PTEntry
	std::map<Int, PTEntry> ptemap_; 
	/// @brief Map from atomic number to splines for pseudopotentials.
	std::map<Int, std::map< Int,std::vector<DblNumVec> > > splmap_;
public:
	PeriodTable() {;}
	~PeriodTable() {;}
	
	/// @brief Map from atomic number to PTEntry
	std::map<Int, PTEntry>& ptemap() { return ptemap_; }

	/// @brief Map from atomic number to splines for pseudopotentials.
	std::map<Int, std::map< Int,std::vector<DblNumVec> > > splmap() { return splmap_; }

	/// @brief Read the information of the periodic table from file.  
	///
	/// All processors can call this routine at the same time
	void Setup( const std::string );
	
	/// @brief Generate the pseudo-charge and its derivatives.
	///
	/// The data are saved in the sparse veector res
	///   res[0]         : pseudo-charge values
	///   res[1]--res[3] : x,y,z components of the derivatives of the
	///		     pseudo-charge
	void CalculatePseudoCharge( const Atom& atom, const Domain& dm, 
			const std::vector<DblNumVec>& gridpos,
			SparseVec& res );

	/// @brief Generate the pseudo-charge and its derivatives on a set of
	/// elements.
	///
	/// The data are saved in the sparse veector res
	///   res[0]         : pseudo-charge values
	///   res[1]--res[3] : x,y,z components of the derivatives of the
	///		     pseudo-charge
	void CalculatePseudoCharge	(
			const Atom& atom, 
			const Domain& dm,
			const NumTns<std::vector<DblNumVec> >& gridposElem,
			NumTns<SparseVec>& res );

	/// @brief Generate the nonlocal pseudopotential projectors.
	void CalculateNonlocalPP( const Atom& atom, const Domain& dm, 
			const std::vector<DblNumVec>& gridpos,
			std::vector<NonlocalPP>& vnlList );

	/// @brief Generate the nonlocal pseudopotential projectors over a set
	/// of elements.
	void CalculateNonlocalPP( const Atom& atom, const Domain& dm, 
			const NumTns<std::vector<DblNumVec> >&   gridposElem,
			std::vector<std::pair<NumTns<SparseVec>, Real> >& vnlList );



	//---------------------------------------------
	// TODO SpinOrbit from RelDFT

	//---------------------------------------------
	// TODO: DG pseudopotential from DGDFT

};

} // namespace dgdft


#endif // _PERIODTABLE_HPP_
