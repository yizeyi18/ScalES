#ifndef _PERIODTABLE_HPP_
#define _PERIODTABLE_HPP_

#include "environment_impl.hpp"
#include "tinyvec_impl.hpp"
#include "numvec_impl.hpp"
#include "nummat_impl.hpp"
#include "utility.hpp"

// TODO PTEntry and PeriodTable are relatively independent.  Deal with
// the names later.

namespace dgdft{


// *********************************************************************
// Atomic information
// *********************************************************************
struct Atom
{
  Int          type;                            // Atomic number
  Point3       pos;                             // Position
  Point3       vel;                             // Velocity
  Point3       force;                           // Atomic force

	Atom( const Int t, const Point3 p, const Point3 v, const Point3 f): 
		type(t), pos(p), vel(v), force(f) {}
  ~Atom() {;}
};


//---------------------------------------------
class PTEntry
{
public:
	DblNumVec _params; //size 5
	DblNumMat _samples; //ns by nb
	DblNumVec _wgts; //nb
	IntNumVec _typs; //nb
	DblNumVec _cuts; //cutoff value for different mode
	//std::map<Int, std::vector<DblNumVec> > _spldata; //data to be generated
public:
	DblNumVec& params() { return _params; }
	DblNumMat& samples() { return _samples; }
	DblNumVec& wgts() { return _wgts; }
	IntNumVec& typs() { return _typs; }
	DblNumVec& cuts() { return _cuts; }
	//std::map<Int, std::vector<DblNumVec> >& spldata() { return _spldata; } //data to be generated
};

Int serialize(const PTEntry&, std::ostream&, const std::vector<Int>&);
Int deserialize(PTEntry&, std::istream&, const std::vector<Int>&);
Int combine(PTEntry&, PTEntry&);


//---------------------------------------------
class PeriodTable
{
public:
	enum {
		i_Znuc = 0,
		i_mass = 1,
		i_Zion = 2,
		i_Es = 3,
	};
	enum {
		i_rad = 0,
		i_rho0 = 1,
		i_drho0 = 2,
		//the following ones are all pseudopotentials
	};
public:
	std::map<Int, PTEntry> _ptemap; //std::map from atom_id to PTEntry
	std::map<Int, std::map< Int,std::vector<DblNumVec> > > _splmap;
public:
	PeriodTable() {;}
	~PeriodTable() {;}
	std::map<Int, PTEntry>& ptemap() { return _ptemap; }
	std::map<Int, std::map< Int,std::vector<DblNumVec> > > splmap() { return _splmap; }

	// Read the information of the periodic table from file.  All
	// processors can call this routine at the same time
	Int Setup( const std::string );
	
	//---------------------------------------------
	//Generate the pseudo-charge and its derivatives, and saved in the
	//sparse veector res
	//  res[0]         : pseudo-charge values
	//  res[1]--res[3] : x,y,z components of the derivatives of the
	//		     pseudo-charge
	Int SetPseudoCharge( const Atom& atom, const Point3 Ls, const Point3 pos, 
			const Index3 Ns, SparseVec& res );

	//----------------------
	Int SetNonlocalPseudoPotential(  const Atom& atom, const Point3 Ls, const Point3 pos, 
			const std::vector<DblNumVec>& gridpos,
			std::vector< std::pair<SparseVec,Real> >& vnls );

	//----------------------
//	Int pseudoNLSpinOrbit(  Atom, Point3 Ls, Point3 pos, 
//			std::vector<DblNumVec> gridpos,
//			std::vector< pair<SparseVec,Real> >& vnls);


};


//// *********************************************************************
//// An entry for periodic table
//// *********************************************************************
//struct PeriodTableEntry
//{
//	// TODO See HGH.m and remark the following
//	
//  // size 5
//	DblNumVec                 param;  
//	// number of grids by number of number of basis	
//	DblNumMat                 sample; 
//  // number of 
//  DblNumVec                 weight; 
//	// 
//  IntNumVec                 type;   
//	//cutoff value for different mode	
//	DblNumVec                 cut;    
//};

//
//Int serialize(const PTEntry&, ostream&, const std::vector<Int>&);
//Int deserialize(PTEntry&, istream&, const std::vector<Int>&);
//Int combine(PTEntry&, PTEntry&);



// *********************************************************************
// Periodic table
// *********************************************************************

//struct PeriodTable{
//  std::map<Int, PeriodTableEntry>   PeriodTableEMap; //map from atom_id to PTEntry
//  std::map<Int, std::map< Int,std::vector<DblNumVec> > > _splmap;
//}
//
} // namespace dgdft


#endif // _PERIODTABLE_HPP_
