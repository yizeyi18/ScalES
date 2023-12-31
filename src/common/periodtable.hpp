//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin and Weile Jia

/// @file periodtable.hpp
/// @brief Periodic table and its entries.
/// @date 2012-08-10
/// @date 2017-12-26 Add support for UPF format
#ifndef _PERIODTABLE_HPP_
#define _PERIODTABLE_HPP_

#include "environment.hpp"
#include "domain.hpp"
#include "tinyvec_impl.hpp"
#include "numvec_impl.hpp"
#include "nummat_impl.hpp"
#include "numtns_impl.hpp"

namespace scales{

// The following typedefs must be IDENTIAL to that in utility.hpp 
typedef std::pair<IntNumVec, DblNumMat > SparseVec; 
typedef std::pair<SparseVec, Real> NonlocalPP; 


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

/// @struct PTParam
/// @brief Index of the parameters in each entry of the periodic table.
/// However, the detailed value may depend on the pseudopotential.
///
struct PTParam{
  /// @brief Atomic number.
  Int ZNUC;
  /// @brief Nuclear mass.
  Int MASS;
  /// @brief Nuclear valence charge.
  Int ZVAL;
  /// @brief Effective radius of the Gaussian charge
  /// In the future this could be a table storing the default value for
  /// each type of species.
  Int RGAUSSIAN;
};


/// @struct PTType
/// @brief Type of each sample, including radial grid, pseudocharge, and
/// nonlocal projectors of different angular momentum, and the
/// spin-orbit contribution. 
/// However, the detailed value may depend on the pseudopotential.
///
struct PTType{
  Int RADIAL;
  Int PSEUDO_CHARGE;
  Int VLOCAL;
  Int RHOATOM;
  Int L0;
  Int L1;
  Int L2;
  Int L3;
  Int SPINORBIT_L1;
  Int SPINORBIT_L2;
  Int SPINORBIT_L3;
};


/// @struct PTSample
/// @brief Index of the radial grid, the pseudocharge, derivative
/// of the pseudocharge, model core charge density, derivative of model
/// core charge density, nonlocal pseudopotentials in samples and cuts.
/// However, the detailed value may depend on the pseudopotential.
///
/// LL: FIXME: Make PSEUDO_CHARGE and DRV_PSEUDO_CHARGE deprecated
/// PSEUDO_CHARGE = VLOCAL, DRV_PSEUDO_CHARGE = DRV_VLOCAL.
struct PTSample{
  Int RADIAL_GRID;
  Int PSEUDO_CHARGE;
  Int DRV_PSEUDO_CHARGE;
  Int VLOCAL;
  Int DRV_VLOCAL;
  Int RHOATOM;
  Int DRV_RHOATOM;
  Int NONLOCAL;
};


/// @brief The start of the radial grid.
const Real MIN_RADIAL = 1e-10;

/// @struct PTEntry
/// @brief Each entry for the periodic table structure.
struct PTEntry
{
  /// @brief Parameters following the order in PTParam.
  DblNumVec params; 
  /// @brief Radial grid, pseudocharge and nonlocal projectors and
  /// their derivatives.
  DblNumMat samples; 
  /// @brief The weight for the nonlocal pseudopotential, obtained as
  /// the eigenvalues by diagonlzing the DIJ matrix
  DblNumVec weights; 
  /// @brief Type of each sample, following PTType.
  IntNumVec types; 
  /// @brief Cutoff value for different samples, following PTSample
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
  PTParam  ptparam_;
  PTType   pttype_;
  PTSample ptsample_;

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
  /// All processors can call this routine at the same time. 
  void Setup( );



  /// @brief Generate the pseudo-charge and its derivatives.
  ///
  /// The data are saved in the sparse veector res
  ///   res[0]         : pseudo-charge values
  ///   res[1]--res[3] : x,y,z components of the derivatives of the
  ///             pseudo-charge
  void CalculatePseudoCharge( const Atom& atom, const Domain& dm, 
      const std::vector<DblNumVec>& gridpos,
      SparseVec& res );

  /// @brief Generate the pseudo-charge and its derivatives on a set of
  /// elements.
  ///
  /// The data are saved in the sparse veector res
  ///   res[0]         : pseudo-charge values
  ///   res[1]--res[3] : x,y,z components of the derivatives of the
  ///             pseudo-charge
  ///
  /// LL: FIXME 01/12/2021 This should be moved to DG 
  void CalculatePseudoCharge    (
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
  /// LL: FIXME 01/12/2021 This should be moved to DG 
  void CalculateNonlocalPP( const Atom& atom, const Domain& dm, 
      const NumTns<std::vector<DblNumVec> >&   gridposElem,
      std::vector<std::pair<NumTns<SparseVec>, Real> >& vnlList );



  /// @brief Generate the atomic density. This is to be used with
  /// structure factor to generate the initial atomic density.
  ///
  /// This can be used both for PWDFT and ScalES.
  /// 
  /// NOTE: This assumes atom.pos should agree with posStart in domain
  /// (PWDFT) and global domain (ScalES).
  ///
  void CalculateAtomDensity( const Atom& atom, const Domain& dm, 
      const std::vector<DblNumVec>& gridpos, 
      DblNumVec& atomDensity );


  /// @brief Generate the short range local pseudopotential and its
  /// derivatives, as well as the Gaussian compensation charge and its derivatives.
  ///
  /// The data are saved in the sparse veector res
  ///   res[0]         : values
  ///   res[1]--res[3] : x,y,z components of the derivatives of the
  void CalculateVLocal( const Atom& atom, const Domain& dm, 
      const std::vector<DblNumVec>& gridpos,
      SparseVec& resVLocalSR, 
      SparseVec& resGaussianPseudoCharge );

  /// @brief Read PTEntry from UPF file.
  ///
  /// This is obtained and modified from the UPF2QSO subroutine in Qbox.
  void ReadUPF( std::string file_name, PTEntry& pt, Int& atom);

  /// @brief Whether the atom type has nonlocal pseudopotential
  bool IsNonlocal(Int type) {return ptemap_[type].cutoffs.m()>ptsample_.NONLOCAL;}
  
  /// @brief Cutoff radius for the pseudocharge in the real space
  Real RcutPseudoCharge(Int type)   {return ptemap_[type].cutoffs(ptsample_.PSEUDO_CHARGE);}
 
  /// @brief Cutoff radius for the pseudocharge in the real space
  Real RcutVLocal(Int type)   {return ptemap_[type].cutoffs(ptsample_.VLOCAL);}

  /// @brief Cutoff radius for model atomic density in the real space.
  /// This is only used for constructing initial charge density, and
  /// does not need to be very accurate
  Real RcutRhoAtom(Int type)   {return ptemap_[type].cutoffs(ptsample_.RHOATOM);}
  
  /// @brief Cutoff radius for the nonlocal pseudpotental in the real
  /// space. If there are multiple pseudopotentials, Rcut should be the
  /// maximum radius so that ALL nonlocal pseudopotentials are accurate.
  Real RcutNonlocal(Int type)   {return (this->IsNonlocal(type)) ? 
    ptemap_[type].cutoffs(ptsample_.NONLOCAL) : 0.0;}


  /// @brief Atomic mass
  Real Mass(Int type) {return ptemap_[type].params(ptparam_.MASS);}

  /// @brief Valence charge of the ion
  Real Zval(Int type) {return ptemap_[type].params(ptparam_.ZVAL);}

  /// @brief Self ionic interaction energy.
  ///
  Real SelfIonInteraction (Int type); 
  
  /// @brief Cutoff radius for the pseudocharge in the real space
  Real RGaussian(Int type)   {return ptemap_[type].params(ptparam_.RGAUSSIAN);}
};



// Serialization / Deserialization
Int serialize(const Atom& val, std::ostream& os, const std::vector<Int>& mask);

Int deserialize(Atom& val, std::istream& is, const std::vector<Int>& mask);

Real MaxForce( const std::vector<Atom>& atomList );



} // namespace scales


#endif // _PERIODTABLE_HPP_
