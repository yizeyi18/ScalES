/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

   Author: Lin Lin and Wei Hu
	 
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
/// @file hamiltonian_dg.hpp
/// @brief The Hamiltonian class for DG calculation.
/// @date 2013-01-09
/// @date 2014-08-06 Intra-element parallelization
#ifndef _HAMILTONIAN_DG_HPP_
#define _HAMILTONIAN_DG_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "numtns_impl.hpp"
#include  "distvec_impl.hpp"
#include  "sparse_matrix_impl.hpp"
#include  "domain.hpp"
#include  "periodtable.hpp"
#include  "utility.hpp"
#include  "esdf.hpp"
#include  "fourier.hpp"
#include  <xc.h>
#include  "mpi_interf.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"


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

	Int Owner (const Index3& key) const {
		return ownerInfo(key(0), key(1), key(2));
	}
};

typedef std::pair<Index3, Index3>  ElemMatKey;

/// @struct ElemMatPrtn
/// @brief Partition class of a matrix according to the element index
/// (row index).  This is used to represent the stiffness matrix and
/// mass matrix.
struct ElemMatPrtn
{
	IntNumTns                 ownerInfo;

	Int Owner (const ElemMatKey& key) const {
		Index3 keyRow = key.first;
		return ownerInfo(keyRow(0), keyRow(1), keyRow(2));
	}
};

/// @struct AtomPrtn
/// @brief Partition class (used by DistVec) according to the atom
/// index.
struct AtomPrtn 
{
	std::vector<Int> ownerInfo; 

	Int Owner(Int key) const {
		return ownerInfo[key];
	}
};

/// @struct BlockMatPrtn
/// @brief Partition class of a matrix according to the 2D block cyclic
/// distribution.  This is used for ScaLAPACK calculation.
struct BlockMatPrtn
{
	IntNumMat                 ownerInfo;

	Int Owner (const Index2& key) const {
		return ownerInfo(key(0), key(1));
	}
};

/// @struct VecPrtn
/// @brief Partition class for a structure indexed by a one dimensional
/// integer vector.  For instance, this can be used to partition the
/// row (column) index of a matrix.
struct VecPrtn
{
	IntNumVec                 ownerInfo;

	Int Owner (const Int & key) const {
		return ownerInfo(key);
	}
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
  std::vector<DblNumVec>      uniformGridFine_;

	/// @brief Number of uniform grids in each element.  
	///
	/// Note: It must be satisifed that
	///
	/// domain_.numGrid[d] = numUniformGridElem_[d] * numElem_[d]
	Index3                      numUniformGridElem_;
  Index3                      numUniformGridElemFine_;

	/// @brief Number of LGL grids in each element.
	Index3                      numLGLGridElem_;

	/// @brief Number of element in externd element.
  Index3 numExtElem_;

	/// @brief Uniform grid in the elements, each has size 
	/// numUniformGridElem_
	NumTns<std::vector<DblNumVec> >   uniformGridElem_;
  NumTns<std::vector<DblNumVec> >   uniformGridElemFine_;

	/// @brief Legendre-Gauss-Lobatto grid in the elements, each has size
	/// numLGLGridElem_
	NumTns<std::vector<DblNumVec> >   LGLGridElem_;

  /// @brief The 1D LGL weight along the x,y,z dimensions of each element.
  std::vector<DblNumVec>            LGLWeight1D_;

  /// @brief The 2D LGL weight for the surface perpendicular to the x,y,z
  /// axis of each element.
  std::vector<DblNumMat>            LGLWeight2D_;

  /// @brief The 3D LGL weight for each element.
  DblNumTns                         LGLWeight3D_;


	/// @brief List of atoms.
	std::vector<Atom>           atomList_;
	/// @brief Number of spin-degeneracy, can be 1 or 2.
	Int                         numSpin_;
	/// @brief Number of extra states for fractional occupation number.
	Int                         numExtraState_;
	/// @brief Number of occupied states.
	Int                         numOccupiedState_;
  /// @brief Type of pseudopotential, default is HGH
  std::string                 pseudoType_;
  /// @brief Id of the exchange-correlation potential
  Int                         XCId_;
  Int                         XId_;
  Int                         CId_;
  /// @brief Exchange-correlation potential using libxc package.
  xc_func_type                XCFuncType_; 
  xc_func_type                XFuncType_; 
  xc_func_type                CFuncType_; 
  /// @brief Whether libXC has been initialized.
  bool                        XCInitialized_;




	// *********************************************************************
	// Computational variables
	// *********************************************************************

	/// @brief The number of elements.
	Index3                      numElem_;

	/// @brief Partition of element.
	ElemPrtn                    elemPrtn_;

  /// @brief Number of processor rows and columns
  Int                         dmRow_;
  Int                         dmCol_;
  //IntNumVec                   groupRank_;

  /// @brief Partition of a matrix defined through elements.
	ElemMatPrtn                 elemMatPrtn_;

	/// @brief Partition of atom.
	AtomPrtn                    atomPrtn_;

	/// @brief Interior penalty parameter.
	Real                        penaltyAlpha_;

	/// @brief Pseudocharge in the global domain. 
	DistDblNumVec    pseudoCharge_;

	/// @brief Electron density in the global domain. No magnitization for
	/// DG calculation.
  DistDblNumVec    density_;

  std::vector<DistDblNumVec>      gradDensity_;

  /// @brief Electron density in the global domain defined on the LGL
  /// grid. No magnitization for DG calculation.  
  /// FIXME This is only a temporary variable and MAY NOT BE the same as
  /// the density_ variable.
	DistDblNumVec    densityLGL_;


	/// @brief External potential in the global domain. This is usually
	/// not used.
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
  IntNumVec        basisLGLIdx_;
	DistDblNumMat    basisLGL_;

	/// @brief Eigenvalues
	DblNumVec        eigVal_;

	/// @brief Occupation number
	DblNumVec        occupationRate_;

	/// @brief Coefficients of the eigenfunctions
	DistDblNumMat    eigvecCoef_;

	/// @brief Pseudopotential and nonlocal projectors in each element for
	/// each atom. The value is given on a dense LGL grid.
	DistVec<Index3, std::map<Int, PseudoPot>, ElemPrtn>  pseudo_;

	// FIXME
	DistVec<Index3, std::map<Int, DblNumMat>, ElemPrtn>  vnlCoef_;
	
	// FIXME
	std::vector<DistVec<Index3, std::map<Int, DblNumMat>, ElemPrtn> >  vnlDrvCoef_;

	std::map<Int, DblNumVec>  vnlWeightMap_;

	/// @brief Differentiation matrix on the LGL grid.
	std::vector<DblNumMat>    DMat_;

	/// @brief Interpolation matrix from LGL to uniform grid in each
	/// element (assuming all the elements are the same).
  std::vector<DblNumMat>    LGLToUniformMat_;
  std::vector<DblNumMat>    LGLToUniformMatFine_;

  /// @brief Gaussian convolution interpolation matrix from LGL to 
  /// uniform grid in each element.
  NumTns< std::vector<DblNumMat> >    LGLToUniformGaussMatFine_;

  /// @brief DG Hamiltonian matrix.
	DistVec<ElemMatKey, DblNumMat, ElemMatPrtn>  HMat_;

	/// @brief The size of the H matrix.
	Int    sizeHMat_;

	/// @brief Indices of all the basis functions.
  NumTns< std::vector<Int> >  elemBasisIdx_;

  /// @brief Inverse mapping of elemBasisIdx_
  std::vector<Index3>         elemBasisInvIdx_;

public:

	// *********************************************************************
	// Lifecycle
	// *********************************************************************
	HamiltonianDG();

	~HamiltonianDG();

	HamiltonianDG( const esdf::ESDFInputParam& esdfParam );

  /// @brief Setup the Hamiltonian DG class from the input parameter.
	void Setup ( const esdf::ESDFInputParam& esdfParam );

  /// @brief Update hamiltonianDG for Geometry optimization and MD  
	void UpdateHamiltonianDG( const std::vector<Atom>& atomList );

	// *********************************************************************
	// Operations
	// *********************************************************************

	/// @brief Differentiate the basis functions on a certain element
	/// along the dimension d.
  void DiffPsi(const Index3& numGrid, const Real* psi, Real* Dpsi, Int d);

  /// @brief Interpolation matrix from LGL to uniform grid in each element.
  void InterpLGLToUniform( const Index3& numLGLGrid, const Index3& numUniformGridFine, 
      const Real* rhoLGL, Real* rhoUniform );

  /// @brief Gaussian convolution interpolation matrix from LGL to 
  /// uniform grid in each element.
  void GaussConvInterpLGLToUniform( const Index3& numLGLGrid, const Index3& numUniform, 
      const Real* rhoLGL, Real* rhoUniform, std::vector<DblNumMat> LGLToUniformGaussMatFine );

  /// @brief Initialize the pseudopotential used on the LGL grid for
  /// each element.
  void CalculatePseudoPotential( PeriodTable &ptable );

  /// @brief Compute the electron density after the diagonalization
	/// of the DG Hamiltonian matrix.
	void CalculateDensity( 
      DistDblNumVec& rho, 
      DistDblNumVec& rhoLGL );
	
	/// @brief Compute the electron density using the density matrix. This
  /// is used after obtaining the density matrix using PEXSI.
	void CalculateDensityDM( 
      DistDblNumVec& rho, 
      DistDblNumVec& rhoLGL, 
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distDMMat );

	/// @brief Compute the electron density using the density matrix and
  /// with intra-element parallelization. This is used after obtaining
  /// the density matrix using PEXSI.
	void CalculateDensityDM2( 
      DistDblNumVec& rho, 
      DistDblNumVec& rhoLGL, 
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distDMMat );

  void CalculateGradDensity( DistFourier&   fft );

  /// @brief Compute the exchange-correlation potential and energy.
  void CalculateXC ( 
      Real &Exc, 
      DistDblNumVec&   epsxc,
      DistDblNumVec&   vxc,
      DistFourier&   fft );
  /// @brief Compute the Hartree potential.
  void CalculateHartree( 
      DistDblNumVec& vhart, 
      DistFourier&   fft );

  /// @brief Compute the total potential
  void CalculateVtot( DistDblNumVec& vtot );

  /// @brief Assemble the DG Hamiltonian matrix. The mass matrix is
  /// identity in the framework of adaptive local basis functions.
  void CalculateDGMatrix( ); 

  /// @brief Update the DG Hamiltonian matrix with the same set of
  /// adaptive local basis functions, but different local potential. 
  ///
  /// This subroutine is used in the inner SCF loop when only the local
  /// pseudopotential is updated.
  ///
  /// @param vtotLGLDiff Difference of vtot defined on each LGL grid.
  /// The contribution of this difference is to be added to the
  /// Hamiltonian matrix HMat_.
  void UpdateDGMatrix( DistDblNumVec&  vtotLGLDiff );

  /// @brief Calculate the Hellmann-Feynman force for each atom.
  void CalculateForce ( DistFourier& fft );

  /// @brief Calculate the Hellmann-Feynman force for each atom using
  /// the density matrix formulation.
  void CalculateForceDM ( DistFourier& fft, 
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distDMMat );


  /// @brief Calculate the residual type a posteriori error estimator
  /// for the solution. 
  ///
  /// Currently only the residual term is computed, and it is assumed
  /// that the eigenvalues and eigenfunctions have been computed and
  /// saved in eigVal_ and eigvecCoef_.
  ///
  /// Currently the nonlocal pseudopotential is not implemented in this
  /// subroutine.
  ///
  /// @param[out] eta2Total Total residual-based error estimator.
  /// Reduced among all processors.
  /// @param[out] eta2Residual Residual term.
  /// @param[out] eta2GradJump Jump of the gradient of the
  /// eigenfunction, or "face" term.  
  /// @param[out] eta2Jump Jump of the value of the eigenfunction, or
  /// "jump" term.
  void CalculateAPosterioriError( 
      DblNumTns&       eta2Total,
      DblNumTns&       eta2Residual,
      DblNumTns&       eta2GradJump,
      DblNumTns&       eta2Jump	);

  // *********************************************************************
  // Access
  // *********************************************************************

  /// @brief Total potential in the global domain.
  DistDblNumVec&  Vtot( ) { return vtot_; }

  /// @brief External potential in the global domain.
  DistDblNumVec&  Vext( ) { return vext_; }

  /// @brief Exchange-correlation potential in the global domain. No
  /// magnization calculation in the DG code.
  DistDblNumVec&  Vxc()  { return vxc_; }

  /// @brief Exchange-correlation energy density in the global domain.
  DistDblNumVec&  Epsxc()  { return epsxc_; }

  /// @brief Hartree potential in the global domain.
	DistDblNumVec&  Vhart() { return vhart_; }

	/// @brief Electron density in the global domain. No magnitization for
  /// DG calculation.
  DistDblNumVec&  Density() { return density_; }

  std::vector<DistDblNumVec>  GradDensity() { return gradDensity_; }

  DistDblNumVec&  DensityLGL() { return densityLGL_; }

  DistDblNumVec&  PseudoCharge() { return pseudoCharge_; }

  std::vector<Atom>&  AtomList() { return atomList_; }

	Int NumSpin () { return numSpin_; }

	DblNumVec&  EigVal() { return eigVal_; }
	
	DblNumVec&  OccupationRate() { return occupationRate_; }


	DistDblNumVec&  VtotLGL() { return vtotLGL_; }

  IntNumVec&  BasisLGLIdx() { return basisLGLIdx_; }
  const IntNumVec&  BasisLGLIdx() const { return basisLGLIdx_; }
  Int&  BasisLGLIdx(const Int k) { return basisLGLIdx_(k); }
  const Int&  BasisLGLIdx(const Int k) const { return basisLGLIdx_(k); }

  DistDblNumMat&  BasisLGL() { return basisLGL_; }

	DistDblNumMat&  EigvecCoef() { return eigvecCoef_; }

	/// @brief DG Hamiltonian matrix.
	DistVec<ElemMatKey, DblNumMat, ElemMatPrtn>&  
		HMat() { return HMat_; } 

	Int NumBasisTotal() const { return sizeHMat_; }

  NumTns< std::vector<Int> >&  ElemBasisIdx() { return elemBasisIdx_; }

  std::vector<Index3>&  ElemBasisInvIdx() { return elemBasisInvIdx_; }

	/// domain_.numGrid[d] = numUniformGridElem_[d] * numElem_[d]
	Index3 NumUniformGridElem() const { return numUniformGridElem_; }
  Index3 NumUniformGridElemFine() const { return numUniformGridElemFine_; }

	/// @brief Number of LGL grids in each element.
	Index3 NumLGLGridElem() const { return numLGLGridElem_; }

	/// @brief Return the uniform grid on each element
	NumTns<std::vector<DblNumVec> >& UniformGridElem(){ return  uniformGridElem_; }
  NumTns<std::vector<DblNumVec> >& UniformGridElemFine(){ return  uniformGridElemFine_; }

	/// @brief Return the LGL grid on each element
	NumTns<std::vector<DblNumVec> >& LGLGridElem(){ return  LGLGridElem_; }

	/// @brief Return the element domain information
	NumTns<Domain>&  DomainElem(){ return domainElem_; }

	/// @brief Return the 1D LGL weights
	std::vector<DblNumVec>&  LGLWeight1D(){ return LGLWeight1D_; }

	/// @brief Return the 2D LGL weights
	std::vector<DblNumMat>&  LGLWeight2D(){ return LGLWeight2D_; }

	/// @brief Return the 3D LGL weights
	DblNumTns&  LGLWeight3D(){ return LGLWeight3D_; }

	// *********************************************************************
	// Inquiry
	// *********************************************************************

	Int NumStateTotal() const { return numExtraState_ + numOccupiedState_; }

	Int NumOccupiedState() const { return numOccupiedState_; }

	Int NumExtraState() const { return numExtraState_; }

};



/// @brief Computes the inner product of three terms.
inline Real ThreeDotProduct(Real* x, Real* y, Real* z, Int ntot) {
  Real sum =0;
  for(Int i=0; i<ntot; i++) {
    sum += (*x++)*(*y++)*(*z++);
  }
  return sum;
}

/// @brief Computes the inner product of four terms.
inline Real FourDotProduct(Real* w, Real* x, Real* y, Real* z, Int ntot) {
  Real sum =0;
  for(Int i=0; i<ntot; i++) {
    sum += (*w++)*(*x++)*(*y++)*(*z++);
  }
  return sum;
}

} // namespace dgdft


#endif // _HAMILTONIAN_DG_HPP_
