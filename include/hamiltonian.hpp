#ifndef _HAMILTONIAN_HPP_
#define _HAMILTONIAN_HPP_

#include  "environment_impl.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"
#include  "spinor.hpp"
//#include  <xc.h>

namespace dgdft{

// *********************************************************************
// Base Hamiltonian class 
// *********************************************************************
class Hamiltonian{
};

//class Hamiltonian {
//private:
//	Domain domain_;
//	std::vector<Atom>  atomList_;
//	Int numExtraState_;
//	Int numOccupiedState_;
//
//
//
//public:
//  Domain _domain;
//  vector<Atom> _atomvec;
//  int _nexstate;
//  string _PStype; // the default value is HGH
//
//  int _nocstate;
//  DblNumVec _PSden;  // rho0
//  vector<SparseVec> _atomPSden; // pseodupotentials stuff should not be
//  // included in the general Hamiltonian class
//  // but we will work with PS+DFT for real problems
//  DblNumMat _density; // rho
//
//  // scalar potentials
//  DblNumVec _vext;
//  DblNumVec _vhart;
//  DblNumMat _vxc;   // the same dimension as the dimension of _density
//  DblNumVec _vtot;
//  DblNumVec _epsxc; // the exchange-correlation density vector
//  int _ntotalPS;
//  int _npsi;
//
//  // nonlocal potentials
//  vector< vector< pair<SparseVec,double> > > _vnlss;
//
//public:
//  Hamiltonian();
//  virtual ~Hamiltonian();
//  Hamiltonian(Domain &dm, Index2 &val);
//  Hamiltonian(Domain &dm, int n); 
//  Hamiltonian(Domain &dm, vector<Atom> &atvec, int nexstate, string PStype, int n);
//  
//  virtual int get_density(Spinor &psi, DblNumVec &occrate, double &val);
//
//  // density dependent potentials
//  int set_hartree(FFTPrepare &fp);
//  virtual int set_XC(xc_func_type& XCFunc, double &val);
//  virtual int set_atomPSden(PeriodTable &ptable);
//
//  // density independet potentials
//  int set_external(); 
//  virtual int set_nonlocalPS(PeriodTable &ptable, int cnt);
//  virtual int set_total();
//  virtual int set_total(DblNumVec &vtot);
//
//  virtual int set_PS(PeriodTable &ptable);
//    
//  // H*Psi
//  virtual int act_spinor(Spinor &psi0, CpxNumTns &a3, FFTPrepare &fp);
//};
//
//// Two-component Kohn-Sham with spin orbit coupling.
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
//
//// Four-component Dirac-Kohn-Sham
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
//
//int generate_uniformmesh(Domain &dm, vector<DblNumVec> &gridpos);
} // namespace dgdft


#endif

