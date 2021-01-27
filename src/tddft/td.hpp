//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia, Lin Lin

/// @file tddft.hpp
/// @brief time dependent DFT
/// @date 2017-09-05 
#ifndef _TDDFT_HPP_
#define _TDDFT_HPP_

#include  "environment.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"
#include  "periodtable.hpp"
#include  "esdf.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "hamiltonian.hpp"
#include  "sgmres.hpp"
#ifdef GPU
#include  "cuSolver.hpp"
#include  "magma.hpp"
#endif
using namespace std;

#ifdef _COMPLEX_
namespace scales{


  typedef struct{

    /// @brief polarization direction
    std::vector<Real>  pol;

    /// @brief carrier frequency (angular)
    double freq;

    /// @brief carrier-envelop phase
    double phase;

    /// @brief type of envelope
    std::string env;

    /// @brief Amplitude
    double Amp;

    /// @brief shift parameter
    double t0;

    /// @brief scale parameter, positive real
    double tau;

  } eField;

  /// @brief set up default options for the eField
  void setDefaultEfieldOptions( eField * extEfield);
  void setEfieldPolarization( eField* extEfield, std::vector<Real> & pol);
  void setEfieldFrequency( eField* extEfield, Real freq);
  void setEfieldPhase( eField* extEfield, Real phase);
  void setEfieldEnv( eField* extEfield, std::string env);
  void setEfieldAmplitude( eField* extEfield, Real Amp);
  void setEfieldT0( eField* extEfield, Real t0);
  void setEfieldTau( eField* extEfield, Real tau);

 
  typedef struct{
  
    /// @brief auto save interval
    int auto_save; 
  
    /// @brief whether to load from an autosave
    bool load_save;
  
    /// @brief propagation scheme
    std::string method;
  
    /// @brief whether to include nuclear motion
    bool ehrenfest;
  
    /// @brief time step
    Real simulateTime;
  
    /// @brief time step
    Real dt;

    /// @brief number of inner iterations for GMRES
    int gmres_restart;
  
    /// @brief convergence criteria in Krylov methods
    Real krylovTol;
  
    /// @brief maximum dimension of Krylov subspace
    Real krylovMax;
  
    /// @brief maximum dimension of diis iteration
    Real diisMaxIter;
  
    /// @brief maximum dimension of phi iteration
    Real phiMaxIter;

    /// @brief convergence criteria in phi iteration
    Real phiTol;

    /// @brief convergence criteria in phi iteration
    Real diisTol;
  
    /// @brief dimension of the adiabatic subspace
    int adNum ;
  
    /// @brief update interval of the adiabatic projector
    int adUpdate;
  
    /// @brief the external field with time t
    eField eField_;

    /// @brief output the atomic position in the XYZ format.  for
    /// visualization purpose, only valid when ehrenfest dynamics is
    /// used.
    bool   isOutputXYZ;

  } TDDFTOptions;
  
  void setDefaultTDDFTOptions( TDDFTOptions * options);
  void setTDDFTMethod( TDDFTOptions * options, std::string method);
  void setTDDFTEhrenfest( TDDFTOptions * options, bool ehrenfest);
  void setTDDFTDt( TDDFTOptions * options, Real dT);
  void setTDDFTkrylovTol( TDDFTOptions * options, Real krylovTol);
  void setTDDFTkrylovMax( TDDFTOptions *options, int krylovMax);
  void setTDDFTDiisTol( TDDFTOptions *options, Real tol);
  void setTDDFTPhiTol ( TDDFTOptions *options, Real tol);

  class TDDFT{

    private: 

      Hamiltonian*        hamPtr_;
      Fourier*            fftPtr_;
      Spinor*             psiPtr_;
      Int                 k_;
      Int                 maxHist_;

      /// @brief the Xr, Yr, Zr for the dipole
      Int                 calDipole_;
      Int                 calVext_;

      DblNumVec           Xr_;
      DblNumVec           Yr_;
      DblNumVec           Zr_;
      DblNumVec           D_;

      std::vector<Atom>*   atomListPtr_;
      std::vector<Real>    tlist_;

      /// @brief atomListHist_[0] stores the lastest atomic configuration
      ///
      /// FIXME
      /// This should be combined with GeoOptVars information later
      std::vector<std::vector<Atom> >   atomListHist_;

      DblNumVec            atomMass_;

      // Supercell parameters : grab this from esdf
      Real supercell_x_;
      Real supercell_y_;
      Real supercell_z_;

      TDDFTOptions options_;
       
      // for Anderson iteration
      DblNumMat           dfMat_;
      DblNumMat           dvMat_;

      Sgmres sgmres_solver;

      Real Ekin_;
      Real Ehart_;
      Real EVxc_;
      Real Exc_;
      Real Eself_;
      Real EIonSR_;
      Real EVdw_;
      Real Ecor_;
      Real Etot_;
      Real Efree_;
      Real Eext_;
      Real AtomKin_;


      std::ofstream etotOFS;
      std::ofstream vextOFS;
      std::ofstream dipoleOFS;
      std::ifstream velocityOFS;
      
      bool isCalculateGradRho_; 

    public:

      /// @brief Main program to move the ions.
      ///
      /// Will determine both geometry optimization and molecular dynamics
      // ************** LIFECYCLE ********************
      TDDFT (){};
      ~TDDFT(){};

      // ************** OPERATORS ********************
      void Setup(
         Hamiltonian& ham,
         Spinor& psi,
         Fourier& fft,
         std::vector<Atom>& atomList,
         PeriodTable& ptable) ;

      /// @brief Adjust the atomic positions to be within the box
      void AdjustAtomPos(std::vector<Point3> & atomPos);
      void CalculateDipole(Real t);
#ifdef GPU
#ifdef _COMPLEX_
      void advanceRK4_GPU(PeriodTable& ptable) ;
      void advancePTTRAPDIIS_GPU(PeriodTable& ptable) ;
      void advancePTTRAPDIIS_GPU_BookKeeping(PeriodTable& ptable) ;
#endif
#endif
      void advanceRK4(PeriodTable& ptable) ;
      void advancePTTRAP(PeriodTable& ptable) ;
      void advancePTTRAPDIIS(PeriodTable& ptable) ;
      void done();  
      void Propagate(PeriodTable& ptable );
      Real getEfield( Real t);
      /// @brief Calculate electric field potential, energy and force
      /// (which only depends on the atomic position and time)
      void CalculateEfieldExt(PeriodTable& ptable, Real t);
      void Update();
      void CalculateEnergy(PeriodTable& ptable, Real t) ;
      void PrintState( Int step );
#ifdef GPU
      Real InnerSolve_GPU(Int step, Spinor & psiFinal, NumTns<Complex> & tnsTemp, CpxNumMat & HX, CpxNumMat &X, CpxNumMat & HPSI, CpxNumMat &psiF, CpxNumMat & XHX, CpxNumMat & XHXtemp, CpxNumMat & RX, CpxNumMat &Xmid, Real & dT, CpxNumMat & psiRes, CpxNumVec & vin, CpxNumVec & vout, std::vector<CpxNumMat> & dfMat, std::vector<CpxNumMat> & dvMat, DblNumMat & rhoFinal );
      Real InnerSolve_GPU( int iscf, Spinor & psiFinal, NumTns<Complex> & tnsTemp, CpxNumMat & HX, CpxNumMat &X, CpxNumMat &HPSI, CpxNumMat & psiF, CpxNumMat & XHX, CpxNumMat & XHXtemp, CpxNumMat & RX, CpxNumMat & Xmid, Real & dT, CpxNumMat & psiRes, CpxNumVec & vin, CpxNumVec & vout, std::vector<CpxNumMat> & dfMat, std::vector<CpxNumMat> & dvMat, DblNumMat & rhoFinal, cuCpxNumVec & vin_dev, cuCpxNumVec & vout_dev, cuCpxNumMat & cu_psiRes, cuCpxNumMat & cu_RX , cuCpxNumMat & cu_Xmid);
      Real InnerSolve_GPU2( int iscf, Spinor & psiFinal, NumTns<Complex> & tnsTemp, CpxNumMat & HX, CpxNumMat &X, CpxNumMat &HPSI, CpxNumMat & psiF, CpxNumMat & XHX, CpxNumMat & XHXtemp, CpxNumMat & RX, CpxNumMat & Xmid, Real & dT, CpxNumMat & psiRes, CpxNumVec & vin, CpxNumVec & vout, std::vector<CpxNumMat> & dfMat, std::vector<CpxNumMat> & dvMat, DblNumMat & rhoFinal, std::vector<cuCpxNumMat> & dfMat_dev, std::vector<cuCpxNumMat> & dvMat_dev, cuCpxNumVec & vin_dev, cuCpxNumVec & vout_dev, cuCpxNumMat & cu_psiRes, cuCpxNumMat & cu_RX );
#endif
      Real InnerSolve(Int step, Spinor & psiFinal, NumTns<Complex> & tnsTemp, CpxNumMat & HX, CpxNumMat &X, CpxNumMat & HPSI, CpxNumMat &psiF, CpxNumMat & XHX, CpxNumMat & XHXtemp, CpxNumMat & RX, CpxNumMat &Xmid, Real & dT, CpxNumMat & psiRes, CpxNumVec & vin, CpxNumVec & vout, std::vector<CpxNumMat> & dfMat, std::vector<CpxNumMat> & dvMat, DblNumMat & rhoFinal );
      
      // Mixing
      void  AndersonMix( 
          Int iter,
          Real            mixStepLength,
          std::string     mixType,
          DblNumVec&      vMix,
          DblNumVec&      vOld,
          DblNumVec&      vNew,
          DblNumMat&      dfMat,
          DblNumMat&      dvMat );

      void  Store4Restart();

  };

}
#endif

#endif
