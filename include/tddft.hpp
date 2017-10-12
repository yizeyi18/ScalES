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


using namespace std;

namespace dgdft{


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
  
    /// @brief convergence criteria in SCF
    Real scfTol;
  
    /// @brief dimension of the adiabatic subspace
    int adNum ;
  
    /// @brief update interval of the adiabatic projector
    int adUpdate;
  
    /// @brief the external field with time t
    eField eField_;

    /// @brief output options, will include later
   
    
  } TDDFTOptions;
  
  void setDefaultTDDFTOptions( TDDFTOptions * options);
  void setTDDFTMethod( TDDFTOptions * options, std::string method);
  void setTDDFTEhrenfest( TDDFTOptions * options, bool ehrenfest);
  void setTDDFTDt( TDDFTOptions * options, Real dT);
  void setTDDFTkrylovTol( TDDFTOptions * options, Real krylovTol);
  void setTDDFTkrylovMax( TDDFTOptions *options, int krylovMax);
  void setTDDFTScfTol( TDDFTOptions *options, Real scfTol);

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

      /// @brief VelocityVerlet for NVE simulation
      ///
      void VelocityVerlet( Int ionIter );

    public:

      /// @brief Main program to move the ions.
      ///
      /// Will determine both geometry optimization and molecular dynamics
      // ************** LIFECYCLE ********************
      TDDFT (){};
      ~TDDFT(){};

      // ************** OPERATORS ********************
      void SetUp(
         Hamiltonian& ham,
         Spinor& psi,
         Fourier& fft,
         std::vector<Atom>& atomList,
         PeriodTable& ptable) ;

      void calculateDipole();
      void MoveIons( Int ionIter );

      void advanceRK4(PeriodTable& ptable) ;
      void advancePTTRAP(PeriodTable& ptable) ;
      void done();  
      void propagate(PeriodTable& ptable );
      Real getEfield( Real t);
      void calculateVext(Real t);
      void Update();

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


  };


}
#endif
