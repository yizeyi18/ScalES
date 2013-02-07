/// @file hamiltonian.hpp
/// @brief Hamiltonian class for planewave basis diagonalization method.
/// @author Lin Lin
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
	// Exchange-correlation potential using libxc package.
	xc_func_type                XCFuncType_; 
	bool                        XCInitialized_;

	// Pseudocharge to represent the local pseudopotential
  DblNumVec                   pseudoCharge_;
	// density_(:,1)    electron density
	// density_(:,2-4)  magnetization along x,y,z directions
	DblNumMat                   density_;         
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
		Int                        XCId,
		Int                        numExtraState,
    Int                        numDensityComponent );
 

	// *********************************************************************
	// Operations
	// *********************************************************************

	virtual void CalculatePseudoPotential( PeriodTable &ptable ) = 0;

	virtual void CalculateDensity( const Spinor &psi, const DblNumVec &occrate, Real &val ) = 0;

	virtual void CalculateXC (Real &val) = 0;

	virtual void CalculateHartree( Fourier& fft ) = 0;

	virtual void CalculateVtot( DblNumVec& vtot ) = 0;

	// Matrix vector multiplication
	virtual void MultSpinor(Spinor& psi, NumTns<Scalar>& a3, Fourier& fft) = 0;

	// *********************************************************************
	// Access
	// *********************************************************************
  DblNumVec&  Vtot() { return vtot_; }
	DblNumMat&  Vxc()  { return vxc_; }
	DblNumVec&  Vhart() { return vhart_; }

	DblNumMat&  Density() { return density_; }
	DblNumVec&  PseudoCharge() { return pseudoCharge_; }
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
		Int                        XCId,
		Int                        numExtraState = 0,
    Int                        numDensityComponent = 1 );


	// *********************************************************************
	// Operations
	// *********************************************************************

	virtual void CalculatePseudoPotential( PeriodTable &ptable );

	virtual void CalculateDensity( const Spinor &psi, const DblNumVec &occrate, Real &val );

	virtual void CalculateXC ( Real &val );

	virtual void CalculateHartree( Fourier& fft );
	
	virtual void CalculateVtot( DblNumVec& vtot );

	// Matrix vector multiplication
	virtual void MultSpinor(Spinor& psi, NumTns<Scalar>& a3, Fourier& fft);
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
