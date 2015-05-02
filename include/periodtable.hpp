/*
	 Copyright (c) 2012 The Regents of the University of California,
	 through Lawrence Berkeley National Laboratory.  

   Author: Lin Lin
	 
   This file is part of DGDFT. All rights reserved.

	 Redistribution and use in source and binary forms, with or without
	 modification, are permitted provided that the following conditions are met:

	 (1) Redistributions of source code must retain the above copyright notice, this
	 list of conditions and the following disclaimer.
	 (2) Redistributions in binary form must reproduce the above copyright notice,
	 this list of conditions and the following disclaimer in the documentation
	 and/or other materials provided with the distribution.
	 (3) Neither the name of the University of California, Lawrence Berkeley
	 National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
	 be used to endorse or promote products derived from this software without
	 specific prior written permission.

	 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
	 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
	 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	 You are under no obligation whatsoever to provide any bug fixes, patches, or
	 upgrades to the features, functionality or performance of the source code
	 ("Enhancements") to anyone; however, if you choose to make your Enhancements
	 available either publicly, or directly to Lawrence Berkeley National
	 Laboratory, without imposing a separate written license agreement for such
	 Enhancements, then you hereby grant the following license: a non-exclusive,
	 royalty-free perpetual license to install, use, modify, prepare derivative
	 works, incorporate into other computer software, distribute, and sublicense
	 such enhancements or derivative works thereof, in binary and source code form.
*/
/// @file periodtable.hpp
/// @brief Periodic table and its entries.
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

  Atom() {;}
  ~Atom() {;}
	Atom( const Int t, const Point3 p, const Point3 v, const Point3 f): 
		type(t), pos(p), vel(v), force(f) {}
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
const Real MIN_RADIAL = 1e-3;

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

// Serialization / Deserialization
inline Int serialize(const Atom& val, std::ostream& os, const std::vector<Int>& mask)
{
  serialize(val.type, os, mask);
  serialize(val.pos,  os, mask);
  serialize(val.vel,  os, mask);
  serialize(val.force,  os, mask);
  return 0;
}

inline Int deserialize(Atom& val, std::istream& is, const std::vector<Int>& mask)
{
  deserialize(val.type, is, mask);
  deserialize(val.pos,  is, mask);
  deserialize(val.vel,  is, mask);
  deserialize(val.force,  is, mask);
  return 0;
}


} // namespace dgdft


#endif // _PERIODTABLE_HPP_
