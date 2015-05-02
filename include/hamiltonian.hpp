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
/// @file hamiltonian.hpp
/// @brief Hamiltonian class for planewave basis diagonalization method.
/// @date 2012-09-16
#ifndef _HAMILTONIAN_HPP_
#define _HAMILTONIAN_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "periodtable.hpp"
#include  "spinor.hpp"
#include  "utility.hpp"
#include  "esdf.hpp"
#include  <xc.h>

namespace dgdft{

// *********************************************************************
// Base Hamiltonian class 
// *********************************************************************



class Hamiltonian {
protected:
	Domain                      domain_;
	// List of atoms
	std::vector<Atom>           atomList_;
	Int                         numSpin_;
	Int                         numExtraState_;
	Int                         numOccupiedState_;
	// Type of pseudopotential, default HGH
	std::string                 pseudoType_;
	// Id of the exchange-correlation potential
	Int                         XCId_;
	Int                         XId_;
	Int                         CId_;
	// Exchange-correlation potential using libxc package.
	xc_func_type                XCFuncType_; 
	xc_func_type                XFuncType_; 
	xc_func_type                CFuncType_; 
	bool                        XCInitialized_;

	// Pseudocharge to represent the local pseudopotential
  DblNumVec                   pseudoCharge_;
	// density_(:,1)    electron density
	// density_(:,2-4)  magnetization along x,y,z directions
	DblNumMat                   density_;         
	std::vector<DblNumMat>      gradDensity_;
	// External potential. TODO not implemented
	DblNumVec                   vext_;            
	// Hartree potential
	DblNumVec                   vhart_;           
	// vxc_(:,1)        exchange-correlation potential
	// vxc_(:,2-4)      vector exchange-correlation potential along x,y,z
	// directions
	// The same dimension as the dimension of density_
	DblNumMat                   vxc_;             
	// Total potential
	DblNumVec                   vtot_;        
  DblNumVec                   vtotCoarse_;  // Coarse 
	// the exchange-correlation energy density
	DblNumVec                   epsxc_; 

	// Pseudopotential for each atom
	std::vector<PseudoPot>      pseudo_;

	// Eigenvalues
	DblNumVec                   eigVal_;
	// Occupation rate
	DblNumVec                   occupationRate_;

public:

	// *********************************************************************
	// Lifecycle
	// *********************************************************************
	Hamiltonian() {}
	virtual ~Hamiltonian() {}
	Hamiltonian( 
			const esdf::ESDFInputParam& esdfParam,
			const Int                   numDensityComponent );

	virtual void Setup (
		const Domain&              dm,
		const std::vector<Atom>&   atomList,
		std::string                pseudoType,
		std::string                XCType,
		Int                        numExtraState,
    Int                        numDensityComponent );
 

	// *********************************************************************
	// Operations
	// *********************************************************************

	virtual void CalculatePseudoPotential( PeriodTable &ptable ) = 0;

	virtual void CalculateDensity( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier &fft ) = 0;

	virtual void CalculateGradDensity( Fourier &fft ) = 0;

	virtual void CalculateXC (Real &val, Fourier& fft) = 0;

	virtual void CalculateHartree( Fourier& fft ) = 0;

	virtual void CalculateVtot( DblNumVec& vtot ) = 0;

	/// @brief Calculate the Hellmann-Feynman force for each atom.
	virtual void CalculateForce ( Spinor& psi, Fourier& fft ) = 0;

	/// @brief Calculate the Hellmann-Feynman force for each atom.
  /// LL 2015/05/02:
  /// This is a clean version for computing the force
	virtual void CalculateForce2 ( Spinor& psi, Fourier& fft ) = 0;

	// Matrix vector multiplication
	virtual void MultSpinor(Spinor& psi, NumTns<Scalar>& a3, Fourier& fft) = 0;

	virtual void MultSpinor(Int iocc, Spinor& psi, NumMat<Scalar>& y, Fourier& fft) = 0;


	// *********************************************************************
	// Access
	// *********************************************************************
  DblNumVec&  Vtot() { return vtot_; }
  DblNumVec&  VtotCoarse() { return vtotCoarse_; }
  DblNumVec&  Vext() { return vext_; }
	DblNumMat&  Vxc()  { return vxc_; }
	DblNumVec&  Vhart() { return vhart_; }

	DblNumMat&  Density() { return density_; }
  std::vector<DblNumMat>  GradDensity() { return gradDensity_; }
	DblNumVec&  PseudoCharge() { return pseudoCharge_; }
	std::vector<PseudoPot>& Pseudo() {return pseudo_; };
  DblNumVec&  EigVal() { return eigVal_; }
	DblNumVec&  OccupationRate() { return occupationRate_; }

	std::vector<Atom>&  AtomList() { return atomList_; }

	// *********************************************************************
	// Inquiry
	// *********************************************************************
	Int  NumSpin() const { return numSpin_; }
	Int  NumStateTotal() const { return numExtraState_ + numOccupiedState_; }
	Int  NumOccupiedState() const { return numOccupiedState_; }
	Int  NumExtraState() const { return numExtraState_; }
	Int  NumDensityComponent() const { return density_.n(); }
	
};


// *********************************************************************
// One-component Kohn-Sham class
// *********************************************************************
class KohnSham: public Hamiltonian {
public:

	// *********************************************************************
	// Lifecycle
	// *********************************************************************
  KohnSham();
  ~KohnSham();

	KohnSham( 
			const esdf::ESDFInputParam& esdfParam,
      const Int                   numDensityComponent );

	void Setup (
		const Domain&              dm,
		const std::vector<Atom>&   atomList,
		std::string                pseudoType,
		std::string                XCType,
		Int                        numExtraState = 0,
    Int                        numDensityComponent = 1 );

	void Update ( std::vector<Atom>&  atomList );

	// *********************************************************************
	// Operations
	// *********************************************************************

	virtual void CalculatePseudoPotential( PeriodTable &ptable );

	virtual void CalculateDensity( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier& fft );

  virtual void CalculateGradDensity( Fourier& fft );

  virtual void CalculateXC ( Real &val, Fourier& fft );

	virtual void CalculateHartree( Fourier& fft );
	
	virtual void CalculateVtot( DblNumVec& vtot );

	/// @brief Calculate the Hellmann-Feynman force for each atom.
	virtual void CalculateForce ( Spinor& psi, Fourier& fft );

	/// @brief Calculate the Hellmann-Feynman force for each atom.
  /// LL 2015/05/02:
  /// This is a clean version for computing the force
	virtual void CalculateForce2 ( Spinor& psi, Fourier& fft );

	// Matrix vector multiplication
	virtual void MultSpinor(Spinor& psi, NumTns<Scalar>& a3, Fourier& fft);

	virtual void MultSpinor(Int iocc, Spinor& psi, NumMat<Scalar>& y, Fourier& fft);

};



// Two-component Kohn-Sham with spin orbit coupling.
//class KohnSham2C: public Hamiltonian {
//  public:
//    // Total number of projectors for spin-orbit coupling
//    int _ntotalPSSpinOrbit;
//    // nonlocal potential for spin-orbit coupling
//    vector< vector< pair<SparseVec,double> > > _vnlSO;
//
//  public:
//    KohnSham2C();
//    ~KohnSham2C();
//    KohnSham2C(Domain &dm, Index2 &val);
//    KohnSham2C(Domain &dm, int val);
//    KohnSham2C(Domain &dm, vector<Atom> &atvec, int nexstate, string PStype, int n);
//    KohnSham2C(Domain &dm, vector<Atom> &atvec, int nexstate, int n);
//
//    int get_density(Spinor &psi, DblNumVec &occrate, double &val); 
//    int set_XC(xc_func_type& XCFunc, double &val);
//    int set_total();
//    int set_total(DblNumVec &vtot);
//    int set_nonlocalPS(PeriodTable &ptable, int &cnt);
//    int set_nonlocalPSSpinOrbit(PeriodTable &ptable, int &cnt);
//    int set_atomPSden(PeriodTable &ptable);
//    int set_PS(PeriodTable &ptable);
//    int act_spinor(Spinor &psi0, CpxNumTns &a3, FFTPrepare &fp);
//};

// Four-component Dirac-Kohn-Sham
//class DiracKohnSham: public Hamiltonian {
//  public:
//    // Total number of projectors for spin-orbit coupling
//    int _ntotalPSSpinOrbit;
//    // nonlocal potential for spin-orbit coupling
//    vector< vector< pair<SparseVec,double> > > _vnlSO;
//
//  public:
//    DiracKohnSham();
//    ~DiracKohnSham();
//    DiracKohnSham(Domain &dm, Index2 &val);
//    DiracKohnSham(Domain &dm, int val);
//    DiracKohnSham(Domain &dm, vector<Atom> &atvec, int nexstate, string PStype, int n);
//    DiracKohnSham(Domain &dm, vector<Atom> &atvec, int nexstate, int n);
//
//    int get_density(Spinor &psi, DblNumVec &occrate, double &val); 
//    int set_XC(xc_func_type& XCFunc, double &val);
//    int set_total();
//    int set_total(DblNumVec &vtot);
//    int set_nonlocalPS(PeriodTable &ptable, int &cnt);
//    int set_nonlocalPSSpinOrbit(PeriodTable &ptable, int &cnt);
//    int set_atomPSden(PeriodTable &ptable);
//    int set_PS(PeriodTable &ptable);
//    int act_spinor(Spinor &psi0, CpxNumTns &a3, FFTPrepare &fp);
//};

} // namespace dgdft


#endif // _HAMILTONIAN_HPP_
