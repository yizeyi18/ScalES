#ifndef _HAMILTONIAN_HPP_
#define _HAMILTONIAN_HPP_

#include  "environment_impl.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "periodtable.hpp"
#include  "spinor.hpp"
#include  "utility.hpp"
#include  <xc.h>

namespace dgdft{

// *********************************************************************
// Base Hamiltonian class 
// *********************************************************************
//class Hamiltonian{
//public:
//	void MultSpinor(Spinor& psi, NumTns<Scalar>& a3, Fourier* fftPtr)
//	{
//		psi.AddLaplacian(a3, fftPtr);
//	}
//};

class Hamiltonian {
protected:
	Domain                      domain_;
	// List of atoms
	std::vector<Atom>           atomList_;
	Int                         numExtraState_;
	Int                         numOccupiedState_;
	// Type of pseudopotential, default HGH
	std::string                 pseudoType_;
	// Type of exchange-correlation potential using libxc package.
	xc_func_type                XCFuncType_; 

	// Pseudocharge to represent the local pseudopotential
  DblNumVec                   pseudoCharge_;
	// Pseudocharge associated to each atom
	std::vector<SparseVec>      pseudoChargeList_; 
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
	// Nonlocal pseudopotential list
	// First index: atom
	// Second index: nonlocal pseudopotential
	std::vector<std::vector<NonlocalPP> >    vnlDoubleList_;
	// Fermi energy
	Real                        fermi_;
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
			const Domain                   &dm, 
			const std::vector<Atom>        &atomList, 
			const std::string               pseudoType,
			const xc_func_type             &XCFuncType, 
			const Int                       numExtraState, 
      const Int                       numDensityComponent);


	// *********************************************************************
	// Operations
	// *********************************************************************

	virtual void CalculatePseudoCharge( PeriodTable &ptable ) = 0;

	virtual void CalculateNonlocalPP( PeriodTable &ptable ) = 0;

	virtual void CalculateDensity( const Spinor &psi, const DblNumVec &occrate, Real &val ) = 0;

	virtual void CalculateXC (Real &val) = 0;

	virtual void CalculateHartree( Fourier& fft ) = 0;

	virtual void CalculateVtot( DblNumVec& vtot ) = 0;

	// Matrix vector multiplication
	virtual void MultSpinor(Spinor& psi, NumTns<Scalar>& a3, Fourier& fft) = 0;

	void CalculateOccupationRate( const Real Tbeta );

	// *********************************************************************
	// Access
	// *********************************************************************
  DblNumVec&  Vtot() { return vtot_; }


	// *********************************************************************
	// Inquiry
	// *********************************************************************
	Int NumStateTotal() const { return numExtraState_ + numOccupiedState_; }
	Int NumOccupiedState() const { return numOccupiedState_; }
	Int NumExtraState() const { return numExtraState_; }
	Int NumDensityComponent() const { return density_.n(); }
	
};


// *********************************************************************
// One-component Kohn-Sham class
// *********************************************************************
class KohnSham: public Hamiltonian {
public:

	// *********************************************************************
	// Lifecycle
	// *********************************************************************
  KohnSham() {}
  ~KohnSham() {} 
	KohnSham( 
			const Domain                   &dm, 
			const std::vector<Atom>        &atomList, 
			const std::string               pseudoType,
			const xc_func_type             &XCFuncType, 
			const Int                       numExtraState, 
      const Int                       numDensityComponent );

	// *********************************************************************
	// Operations
	// *********************************************************************

	virtual void CalculatePseudoCharge( PeriodTable &ptable );

	virtual void CalculateNonlocalPP( PeriodTable &ptable );

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