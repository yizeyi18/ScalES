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
/// @file tddft.cpp
/// @brief time dependent density functional theory with ehrenfest dynamics.
/// @date 2017-09-05 
#ifndef _TDDFT_CPP_
#define _TDDFT_CPP_

#include "tddft.hpp"
#include "utility.hpp"


namespace dgdft{

  using namespace dgdft::esdf;

  void setDefaultEfieldOptions( eField * eF)
  {
    // set the polarization to x 
    eF->pol.resize(3);
    eF->pol[0] = 1.0;
    eF->pol[1] = 0.0;
    eF->pol[2] = 0.0;

    // set frequency to 0.0 
    eF->freq = 18.0/27.211385;

    // set phase to 0.0 
    eF->phase = 0.0;

    // set phase to 0.0 
    eF->env = "gaussian";

    // set Amp to 0
    eF->Amp = 0.0194;

    // set t0 to 0
    eF->t0 = 13.6056925;

    // set tau to 0
    eF->tau = 13.6056925;
  }


  void setEfieldPolarization( eField* eF, std::vector<Real> & pol)
  {
    Real scale = std::sqrt(pol[0]*pol[0] + pol[1]*pol[1] + pol[2] * pol[2]);

    eF->pol.resize(3);
    for(int i =0; i < 3; i++)
      eF->pol[i] = pol[i]/scale;
  }

  void setEfieldFrequency( eField* eF, Real freq)
  {
    eF->freq = freq;
  }

  void setEfieldPhase( eField* eF, Real phase)
  {
    eF->phase = phase;
  }

  void setEfieldEnv( eField* eF, std::string env)
  {
    if(env != "constant" && env != "gaussian" &&
       env != "erf"      && env != "sinsq"    &&
       env != "hann"     && env != "kick")
      env = "gaussian"; // set gaussian as default

    eF->env = env;
  }

  void setEfieldAmplitude( eField* eF, Real Amp)
  {
    eF->Amp = Amp;
  }

  void setEfieldT0( eField* eF, Real t0)
  {
    eF->t0 = t0;
  }

  void setEfieldTau( eField* eF, Real tau)
  {
    if(tau <= 0.0){
      tau = 1.0;
      statusOFS << " Warning: Tau must be positive number; reset to 1.0 " << std::endl;
    }
    eF->tau = tau;
  }

  void setDefaultTDDFTOptions( TDDFTOptions * options)
  {
     options->auto_save     = 0;
     options->load_save     = false;
     options->method        = "RK4";
     //options->ehrenfest     = true;
     options->ehrenfest     = false;
     options->simulateTime  = 40.00;
     options->dt            = 0.005;
     options->gmres_restart = 10; // not sure.
     options->krylovTol     = 1.0E-7;
     options->krylovMax     = 30; 
     options->scfTol        = 1.0E-7; 
     options->adNum         = 20; 
     options->adUpdate      = 1;
     setDefaultEfieldOptions( & options->eField_);
  }
  void setTDDFTMethod( TDDFTOptions * options, std::string method)
  {
     if(method == "RK4" || method == "TPTRAP")
       options->method = method;
     else
       statusOFS << std::endl << " Warning: Method set failed, still use " << options->method << std::endl;
  }
  void setTDDFTEhrenfest( TDDFTOptions * options, bool ehrenfest)
  {
     options->ehrenfest = ehrenfest;
  }
  void setTDDFTDt( TDDFTOptions * options, Real dT)
  {
     options->dt = dT;
  }
  void setTDDFTTime( TDDFTOptions * options, Real time)
  {
     options->simulateTime = time;
  }
  void setTDDFTkrylovTol( TDDFTOptions * options, Real krylovTol)
  {
     options->krylovTol = krylovTol;
  }
  void setTDDFTkrylovMax( TDDFTOptions *options, int krylovMax)
  {
     options->krylovMax = krylovMax;
  }
  void setTDDFTScfTol( TDDFTOptions *options, Real scfTol)
  {
     options->scfTol = scfTol;
  }

  void TDDFT::SetUp(
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft,
    std::vector<Atom>& atomList,
    PeriodTable& ptable) {

      hamPtr_ = &ham;
      psiPtr_ = &psi;
      fftPtr_ = &fft;
      atomListPtr_ = &atomList;

      // Grab the supercell info
      supercell_x_ = esdfParam.domain.length[0];
      supercell_y_ = esdfParam.domain.length[1];
      supercell_z_ = esdfParam.domain.length[2];

      // History of atomic position
      maxHist_ = 4;  // hard coded
      atomListHist_.resize(maxHist_);
      for( Int l = 0; l < maxHist_; l++ ){
        atomListHist_[l] = atomList;
      }

      Int numAtom = atomList.size();
      atomMass_.Resize( numAtom );
      for(Int a=0; a < numAtom; a++) {
        Int atype = atomList[a].type;
        if (ptable.ptemap().find(atype)==ptable.ptemap().end() ){
          ErrorHandling( "Cannot find the atom type." );
        }
        atomMass_[a]=amu2au*ptable.Mass(atype);
      }

      setDefaultTDDFTOptions( & options_);// set the default options.
 
      // init the time list
      int size = options_.simulateTime/ options_.dt + 1;
      tlist_.resize(size);
      for(int i = 0; i < size; i++)
 	     tlist_[i] = i * options_.dt;

      // might need to setup others, will add here.
      k_ = 0;

      // CHECK CHECK: change this to a input parameter.
      calDipole_ = 1;
      calVext_ = 1;

      if( calDipole_) {
        statusOFS << " ************************ WARNING ******************************** " << std::endl;
        statusOFS << " Warning: Please make sure that your atoms are centered at (0,0,0) " << std::endl;
        statusOFS << " ************************ WARNING ******************************** " << std::endl;

        Xr_.Resize( fft.domain.NumGridTotalFine() );
        Yr_.Resize( fft.domain.NumGridTotalFine() );
        Zr_.Resize( fft.domain.NumGridTotalFine() );
        D_.Resize( fft.domain.NumGridTotalFine() );

        Real * xr = Xr_.Data();
        Real * yr = Yr_.Data();
        Real * zr = Zr_.Data();
 
        Int  idx;
        Real Xtmp, Ytmp, Ztmp;       

        for( Int k = 0; k < fft.domain.numGridFine[2]; k++ ){
          for( Int j = 0; j < fft.domain.numGridFine[1]; j++ ){
            for( Int i = 0; i < fft.domain.numGridFine[0]; i++ ){

               idx = i + j * fft.domain.numGridFine[0] + k * fft.domain.numGridFine[0] * fft.domain.numGridFine[1];
               Xtmp = (Real(i) - Real( round(Real(i)/Real(fft.domain.numGridFine[0])) * Real(fft.domain.numGridFine[0]) ) ) / Real(fft.domain.numGridFine[0]);
               Ytmp = (Real(j) - Real( round(Real(j)/Real(fft.domain.numGridFine[1])) * Real(fft.domain.numGridFine[1]) ) ) / Real(fft.domain.numGridFine[1]);
               Ztmp = (Real(k) - Real( round(Real(k)/Real(fft.domain.numGridFine[2])) * Real(fft.domain.numGridFine[2]) ) ) / Real(fft.domain.numGridFine[2]);

               // should be AL(0,0) * X + AL(0,1) * Y + AL(0,2) * Z
               // the other parts are zeros. 
               xr[idx] = Xtmp * supercell_x_ ;
               yr[idx] = Ytmp * supercell_y_ ;
               zr[idx] = Ztmp * supercell_z_ ;

              // get the p.D corresponding to the matlab KSSOLV 
              D_[idx] = xr[idx] * options_.eField_.pol[0] + yr[idx] * options_.eField_.pol[1] + zr[idx] * options_.eField_.pol[2]; 
            }
          }
        }

        for( Int i = 0; i < fft.domain.numGridFine[0]; i++ ){
          statusOFS << " Xr " << i << " " << xr[i] << std::endl;
          //statusOFS << " Yr " << i << " " << yr[i] << std::endl;
          //statusOFS << " Zr " << i << " " << zr[i] << std::endl;
        }

      } 

  } // TDDFT::Setup function

  Real TDDFT::getEfield(Real t)
  {
    Real et = 0.0;
    if (options_.eField_.env == "gaussian" ) {
      Real temp = (t-options_.eField_.t0)/options_.eField_.tau;
      et = options_.eField_.Amp * exp( - temp * temp / 2.0) * cos(options_.eField_.freq * t + options_.eField_.phase);
      return et;
    }
    else{
      statusOFS<< " Wrong Efield input, should be constant/gaussian/erf/sinsq/hann/kick" << std::endl;
      exit(0);
    }
  }

  void TDDFT::calculateVext(Real t)
  {
    Hamiltonian& ham = *hamPtr_;
    Fourier&     fft = *fftPtr_;

    if(calVext_){
      Real et = getEfield(t);
      DblNumVec & vext = ham.Vext();
      // then Vext = Vext0 + et * options_.D
      // Here we suppose the Vext0 are zeros.
      Int idx;
      for( Int k = 0; k < fft.domain.numGridFine[2]; k++ ){
        for( Int j = 0; j < fft.domain.numGridFine[1]; j++ ){
          for( Int i = 0; i < fft.domain.numGridFine[0]; i++ ){
            idx = i + j * fft.domain.numGridFine[0] + k * fft.domain.numGridFine[0] * fft.domain.numGridFine[1];
            vext[idx] = et * D_[idx];
          }
        }
      }
    }

  }


  void TDDFT::calculateDipole()
  {
     Hamiltonian& ham = *hamPtr_;
     Fourier&     fft = *fftPtr_;
     Spinor&      psi = *psiPtr_;
     
     // Di = ∫ Rho(i, j, k) * Xr(i, j, k) *dx
     // Density is not distributed
     Real *density = ham.Density().Data();
     
     Real Dx = 0.0;
     Real Dy = 0.0;
     Real Dz = 0.0;
     Real sumRho = 0.0;
     Real sumDx  = 0.0;
     Real sumDy  = 0.0;
     Real sumDz  = 0.0;

     Real * xr = Xr_.Data();
     Real * yr = Yr_.Data();
     Real * zr = Zr_.Data();
     for( Int k = 0; k < fft.domain.numGridFine[2]; k++ ){
       for( Int j = 0; j < fft.domain.numGridFine[1]; j++ ){
         for( Int i = 0; i < fft.domain.numGridFine[0]; i++ ){

           sumRho += ( * density);
           sumDx  += ( *xr);
           sumDy  += ( *yr);
           sumDz  += ( *zr);

           Dx -=( *density ) * ( *xr++);
           Dy -=( *density ) * ( *yr++);
           Dz -=( *density++ ) * ( *zr++);
         }
       }
     }
     Dx *= Real(supercell_x_ * supercell_y_ * supercell_z_) / Real( fft.domain.numGridFine[0] * fft.domain.numGridFine[1]* fft.domain.numGridFine[2]);
     Dy *= Real(supercell_x_ * supercell_y_ * supercell_z_) / Real( fft.domain.numGridFine[0] * fft.domain.numGridFine[1]* fft.domain.numGridFine[2]);
     Dz *= Real(supercell_x_ * supercell_y_ * supercell_z_) / Real( fft.domain.numGridFine[0] * fft.domain.numGridFine[1]* fft.domain.numGridFine[2]);

#if ( _DEBUGlevel_ >= 0 )
     statusOFS<< "**** Dipole  Calculated *****" << std::endl << std::endl;
     statusOFS << " Dipole x " << Dx << std::endl;
     statusOFS << " Dipole y " << Dy << std::endl;
     statusOFS << " Dipole z " << Dz << std::endl;
     statusOFS<< "**** Dipole  Calculated *****" << std::endl << std::endl;
#if 0
     statusOFS << " sum Dx : " << sumDx << std::endl;
     statusOFS << " sum Dy : " << sumDy << std::endl;
     statusOFS << " sum Dz : " << sumDz << std::endl;
     statusOFS << " sum Rho : " << sumRho << std::endl;
     statusOFS << " super cell: " << supercell_x_ << " " << supercell_y_ << " " << supercell_z_ << std::endl;
     statusOFS << " FFT size  : " << fft.domain.numGridFine[0] << " " << fft.domain.numGridFine[1] << " " << fft.domain.numGridFine[2]  << std::endl;
#endif
#endif
     /*
     */
  }    // -----  end of method TDDFT::calculateDipole ---- 

  void
  TDDFT::VelocityVerlet    ( Int ionIter )
  {
    return ;
  }         // -----  end of method TDDFT::VelocityVerlet  ----- 

  void
  TDDFT::MoveIons    ( Int ionIter )
  {
    Int mpirank, mpisize;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
    MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

    return ;
  }         // -----  end of method TDDFT::MoveIons  ----- 

  void TDDFT::advanceRK4( PeriodTable& ptable ) {

     Int mpirank, mpisize;
     MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
     MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

     Hamiltonian& ham = *hamPtr_;
     Fourier&     fft = *fftPtr_;
     Spinor&      psi = *psiPtr_;

     std::vector<Atom>&   atomList = *atomListPtr_;
     Int numAtom = atomList.size();

     // print the options_ when first step. 
     if(k_ == 0){
       statusOFS<< std::endl;
       statusOFS<< " -------   Print the TDDFT Options  ---------- "     << std::endl;
       statusOFS<< " options.auto_save      " << options_.auto_save      << std::endl;
       statusOFS<< " options.load_save      " << options_.load_save      << std::endl;
       statusOFS<< " options.method         " << options_.method         << std::endl;
       statusOFS<< " options.ehrenfest      " << options_.ehrenfest      << std::endl;
       statusOFS<< " options.simulateTime   " << options_.simulateTime   << std::endl;
       statusOFS<< " options.dt             " << options_.dt             << std::endl;
       statusOFS<< " options.gmres_restart  " << options_.gmres_restart  << std::endl;
       statusOFS<< " options.krylovTol      " << options_.krylovTol      << std::endl;
       statusOFS<< " options.scfTol         " << options_.scfTol         << std::endl;
       statusOFS<< " options.adNum          " << options_.adNum          << std::endl;
       statusOFS<< " options.adUpdate       " << options_.adUpdate       << std::endl;
       statusOFS<< " --------------------------------------------- "     << std::endl;
       statusOFS<< std::endl;
     }
 
     // Update saved atomList. 0 is the latest one
     for( Int l = maxHist_-1; l > 0; l-- ){
       atomListHist_[l] = atomListHist_[l-1];
     }
     atomListHist_[0] = atomList;

     // do the verlocity verlet algorithm to move ion
     Int ionIter = k_;
     //VelocityVerlet( ionIter );

     std::vector<Point3>  atompos(numAtom);
     std::vector<Point3>  atomvel(numAtom);
     std::vector<Point3>  atomvel_temp(numAtom);
     std::vector<Point3>  atomforce(numAtom);
 
     std::vector<Point3>  atompos_mid(numAtom);
     std::vector<Point3>  atompos_fin(numAtom);
     {
       if( mpirank == 0 ){
   
         Real& dt = options_.dt;
         DblNumVec& atomMass = atomMass_;
   
         for( Int a = 0; a < numAtom; a++ ){
           atompos[a]     = atomList[a].pos;
           atompos_mid[a] = atomList[a].pos;
           atompos_fin[a] = atomList[a].pos;
           atomvel[a]     = atomList[a].vel;
           atomforce[a]   = atomList[a].force;
         }

#if ( _DEBUGlevel_ >= 0 )
           statusOFS<< "****************************************************************************" << std::endl << std::endl;
           statusOFS<< "TDDFT RK4 Method, step " << k_ << "  t = " << dt << std::endl;
  
           for( Int a = 0; a < numAtom; a++ ){
             statusOFS << "time: " << k_*24.19*dt << " atom " << a << " position: " << std::setprecision(12) << atompos[a]   << std::endl;
             statusOFS << "time: " << k_*24.19*dt << " atom " << a << " velocity: " << std::setprecision(12) << atomvel[a]   << std::endl;
             statusOFS << "time: " << k_*24.19*dt << " atom " << a << " Force:    " << std::setprecision(12) << atomforce[a] << std::endl;
           }
           statusOFS<< std::endl;
           statusOFS<< "****************************************************************************" << std::endl << std::endl;
#endif

         // Update velocity and position when doing ehrenfest dynamics
         if(options_.ehrenfest){
           for(Int a=0; a<numAtom; a++) {
             atomvel_temp[a] = atomvel[a]/2.0 + atomforce[a]*dt/atomMass[a]/8.0; 
             atompos_mid[a]  = atompos[a] + atomvel_temp[a] * dt;
  
             atomvel_temp[a] = atomvel[a] + atomforce[a]*dt/atomMass[a]/2.0; 
             atompos_fin[a]  = atompos[a] + atomvel_temp[a] * dt;
           }
         }
       }
     }

     // have the atompos_mid and atompos_final
     if(options_.ehrenfest){
       for(Int a = 0; a < numAtom; a++){
         MPI_Bcast( &atompos_mid[a][0], 3, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 
         MPI_Bcast( &atompos_fin[a][0], 3, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 
       }
     }

     // k_ is the current K
     Complex i_Z_One = Complex(0.0, 1.0);
     Int k = k_;
     Real ti = tlist_[k];
     Real tf = tlist_[k+1];
     Real dT = tf - ti;
     Real tmid =  (ti + tf)/2.0;

     // 4-th order Runge-Kutta  Start now 
     DblNumVec &occupationRate = ham.OccupationRate();
     occupationRate.Resize( psi.NumStateTotal() );
     SetValue( occupationRate, 1.0);
     //statusOFS << " Occupation Rate: " << occupationRate << std::endl;
     if(calDipole_)  calculateDipole();
     if(k == 0) {
       Real totalCharge_;
       ham.CalculateDensity(
            psi,
            ham.OccupationRate(),
            totalCharge_, 
            fft );

#if ( _DEBUGlevel_ >= 2 )
       statusOFS << " total Charge init " << setw(16) << totalCharge_ << std::endl;
#endif
       //get the new V(r,t+dt) from the rho(r,t+dt)
       Real Exc_;
       ham.CalculateXC( Exc_, fft ); 
       ham.CalculateHartree( fft );
       calculateVext(ti);

#if ( _DEBUGlevel_ >= 2 )
       Real et0 = getEfield(0);
       Real eti = getEfield(ti);
       Real etmid = getEfield(tmid);
       Real etf = getEfield(tf);
       statusOFS << " DEBUG INFORMATION ON THE EFILED " << std::endl;
       statusOFS << " et0: " << et0 << " eti " << eti << " etmid " << etmid << " etf " << etf << std::endl;
       statusOFS << " DEBUG INFORMATION ON THE EFILED " << std::endl;
#endif
       ham.CalculateVtot( ham.Vtot() );

     }

     // HX1 = (H1 * psi)
     Int ntot  = fft.domain.NumGridTotal();
     Int numStateLocal = psi.NumState();
     CpxNumMat Xtemp(ntot, numStateLocal); // X2, X3, X4
     CpxNumMat HX1(ntot, numStateLocal);
     NumTns<Complex> tnsTemp(ntot, 1, numStateLocal, false, HX1.Data());
     ham.MultSpinor( psi, tnsTemp, fft );

     // test psi * conj(psi) 
#if ( _DEBUGlevel_ >= 2 )
     statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
     statusOFS << " step: " << k_ << " K1 = -i1 * ( H1 * psi ) " << std::endl;
     Int width = numStateLocal;
     Int heightLocal = ntot;
     CpxNumMat  XTXtemp1( width, width );

     blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi.Wavefun().Data(), 
      heightLocal, psi.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );

     for( Int i = 0; i < width; i++)
       statusOFS << " Psi * conjg( Psi) : "  << XTXtemp1(i,i) << std::endl;
       Complex *ptr = psi.Wavefun().Data();
       DblNumVec vtot = ham.Vtot();
       statusOFS << " Psi  0 : " << ptr[0] << " HX1 " << HX1(0,0) << " vtot " << vtot[0] << std::endl;
     statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
#endif
 
     //  1. set up Psi <-- X - i ( HX1 ) * dt/2
     Complex* dataPtr = Xtemp.Data();
     Complex* psiDataPtr = psi.Wavefun().Data();
     Complex* HpsiDataPtr = HX1.Data();
     Int numStateTotal = psi.NumStateTotal();
     Spinor psi2 (fft.domain, 1, numStateTotal, numStateLocal, false, Xtemp.Data() );
     for( Int i = 0; i < numStateLocal; i ++)
       for( Int j = 0; j < ntot; j ++){
	       Int index = i* ntot +j;
	       dataPtr[index] = psiDataPtr[index] -  i_Z_One * HpsiDataPtr[index] * options_.dt/2.0;
       }

     // 2. if ehrenfest dynamics, re-calculate the Vlocal and Vnonlocal
     if(options_.ehrenfest){
      for( Int a = 0; a < numAtom; a++ ){
        atomList[a].pos  = atompos_mid[a];
      }
      ham.UpdateHamiltonian( atomList );
      ham.CalculatePseudoPotential( ptable );
     }
     
     // 3. Update the H matrix. 
     {
       Real totalCharge_;
       ham.CalculateDensity(
            psi2,
            ham.OccupationRate(),
            totalCharge_, 
            fft );

       Real Exc_;
       ham.CalculateXC( Exc_, fft ); 
       ham.CalculateHartree( fft );
       calculateVext(tmid);
       ham.CalculateVtot( ham.Vtot() );
     }
 
     // 4. Calculate the K2 = H2 * X2
     CpxNumMat HX2(ntot, numStateLocal);
     NumTns<Complex> tnsTemp2(ntot, 1, numStateLocal, false, HX2.Data());
     ham.MultSpinor( psi2, tnsTemp2, fft );

     // check the psi * conj(psi)
#if ( _DEBUGlevel_ >= 2 )
     {
     statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
     statusOFS << " step: " << k_ << " K2 = ( H2 * X2 ) " << std::endl;
     Int width = numStateLocal;
     Int heightLocal = ntot;
     CpxNumMat  XTXtemp1( width, width );

     blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi2.Wavefun().Data(), 
      heightLocal, psi2.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );

     for( Int i = 0; i < width; i++)
       statusOFS << " Psi2 * conjg( Psi2) : " << XTXtemp1(i,i) << std::endl;
       Complex *ptr = psi2.Wavefun().Data();
       DblNumVec vtot = ham.Vtot();
       statusOFS << " Psi 2   0 : " << ptr[0]  << " HX2 " << HX2(0,0)<< " vtot " << vtot[0] << std::endl;
     }
     statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
#endif 

     //  1. set up Psi <-- X - i ( HX2) * dt/2
     Spinor psi3 (fft.domain, 1, numStateTotal, numStateLocal, false, Xtemp.Data() );
     dataPtr = Xtemp.Data();
     psiDataPtr = psi.Wavefun().Data();
     HpsiDataPtr = HX2.Data();
     for( Int i = 0; i < numStateLocal; i ++)
       for( Int j = 0; j < ntot; j ++)
       {
	       Int index = i* ntot +j;
	       dataPtr[index] = psiDataPtr[index] -  i_Z_One * HpsiDataPtr[index] * options_.dt/2.0;
       }

     // 2. if ehrenfest dynamics, re-calculate the Vlocal and Vnonlocal
     if(options_.ehrenfest){
       for( Int a = 0; a < numAtom; a++ ){
         atomList[a].pos  = atompos_mid[a];
       }
       ham.UpdateHamiltonian( atomList );
       ham.CalculatePseudoPotential( ptable );
     }

     // 3. Update the H matrix. 
     // CHECK CHECK: The Vext is zero, not updated here. 
     {
       Real totalCharge_;
       ham.CalculateDensity(
            psi3,
            ham.OccupationRate(),
            totalCharge_, 
            fft );
       Real Exc_;
       ham.CalculateXC( Exc_, fft ); 
       ham.CalculateHartree( fft );
       ham.CalculateVtot( ham.Vtot() );
     }
     // 4. Calculate the K3 = H3 * X3
     CpxNumMat HX3(ntot, numStateLocal);
     NumTns<Complex> tnsTemp3(ntot, 1, numStateLocal, false, HX3.Data());
     ham.MultSpinor( psi3, tnsTemp3, fft );

     // check psi3*conj(psi3)
#if ( _DEBUGlevel_ >= 2 )
     {
       statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
       statusOFS << " step: " << k_ << " K3 = -i1 * ( H3 * X3 ) " << std::endl;
       Int width = numStateLocal;
       Int heightLocal = ntot;
       CpxNumMat  XTXtemp1( width, width );
  
       blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi3.Wavefun().Data(), 
        heightLocal, psi3.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
  
       for( Int i = 0; i < width; i++)
         statusOFS << " Psi3 * conjg( Psi3) : " << XTXtemp1(i,i) << std::endl;
       Complex *ptr = psi3.Wavefun().Data();
       DblNumVec vtot = ham.Vtot();
       statusOFS << " Psi 3   0 : " << ptr[0]  << " HX3 " << HX2(0,0) << " vtot " << vtot[0] << std::endl;
       statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
     }
#endif


     //  1. set up Psi <-- X - i ( HX3) * dt
     Spinor psi4 (fft.domain, 1, numStateTotal, numStateLocal, false, Xtemp.Data() );
     dataPtr = Xtemp.Data();
     psiDataPtr = psi.Wavefun().Data();
     HpsiDataPtr = HX3.Data();
     for( Int i = 0; i < numStateLocal; i ++)
       for( Int j = 0; j < ntot; j ++)
       {
	       Int index = i* ntot +j;
	       dataPtr[index] = psiDataPtr[index] -  i_Z_One * HpsiDataPtr[index] * options_.dt;
       }

     // 2. if ehrenfest dynamics, re-calculate the Vlocal and Vnonlocal
     if(options_.ehrenfest){
      for( Int a = 0; a < numAtom; a++ ){
        atomList[a].pos  = atompos_fin[a];
      }
      ham.UpdateHamiltonian( atomList );
      ham.CalculatePseudoPotential( ptable );
     }

     // 3. Update the H matrix. 
     {
       Real totalCharge_;
       ham.CalculateDensity(
            psi4,
            ham.OccupationRate(),
            totalCharge_, 
            fft );

       Real Exc_;
       ham.CalculateXC( Exc_, fft ); 
       ham.CalculateHartree( fft );
       calculateVext(tf);
       ham.CalculateVtot( ham.Vtot() );
     }
 
     // 4. Calculate the K3 = H3 * X3
     // Now Hpsi is H2 * X2
     // K2 = -i1(H2 * psi) 
     CpxNumMat HX4(ntot, numStateLocal);
     NumTns<Complex> tnsTemp4(ntot, 1, numStateLocal, false, HX4.Data());
     ham.MultSpinor( psi4, tnsTemp4, fft );

#if ( _DEBUGlevel_ >= 2 )
     {
     statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
     statusOFS << " step: " << k_ << " K4 = -i1 * ( H4 * X4 ) " << std::endl;
     Int width = numStateLocal;
     Int heightLocal = ntot;
     CpxNumMat  XTXtemp1( width, width );

     blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi4.Wavefun().Data(), 
      heightLocal, psi4.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );

     for( Int i = 0; i < width; i++)
       statusOFS << " Psi4 * conjg( Psi4) : " << XTXtemp1(i,i) << std::endl;
     Complex *ptr = psi4.Wavefun().Data();
     DblNumVec vtot = ham.Vtot();
     statusOFS << " Psi 4   0 : " << ptr[0]  << " HX4 " << HX4(0,0)<< " vtot " << vtot[0] << std::endl;
     statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
     }
#endif 

     // Xf <--  X - i ( K1 + 2K2 + 2K3 + K4) * dt / 6.0
     psiDataPtr = psi.Wavefun().Data();
     Complex *XfPtr = Xtemp.Data();
     Complex *K1    = HX1.Data();
     Complex *K2    = HX2.Data();
     Complex *K3    = HX3.Data();
     Complex *K4    = HX4.Data();
     for( Int i = 0; i < numStateLocal; i ++)
       for( Int j = 0; j < ntot; j ++){
	       Int index = i* ntot +j;
	       XfPtr[index] =  psiDataPtr[index] 
                         - i_Z_One * ( K1[index] + 2.0*K2[index] 
                         + 2.0*K3[index] + K4[index]) *options_.dt /6.0;
       }
     
     Spinor psiFinal (fft.domain, 1, numStateTotal, numStateLocal, false, Xtemp.Data() );

     // 2. if ehrenfest dynamics, re-calculate the Vlocal and Vnonlocal
     if(options_.ehrenfest){
      for( Int a = 0; a < numAtom; a++ ){
        atomList[a].pos  = atompos_fin[a];
      }
      ham.UpdateHamiltonian( atomList );
      ham.CalculatePseudoPotential( ptable );
     }

     //get the new V(r,t+dt) from the rho(r,t+dt)
     {
       Real totalCharge_;
       ham.CalculateDensity(
            psiFinal,
            ham.OccupationRate(),
            totalCharge_, 
            fft );

       Real Exc_;
       ham.CalculateXC( Exc_, fft ); 
       ham.CalculateHartree( fft );
       DblNumVec vtot;
       Int ntotFine  = fft.domain.NumGridTotalFine();
       vtot.Resize(ntotFine);
       SetValue(vtot, 0.0);
       ham.CalculateVtot( vtot);
       Real *vtot0 = ham.Vtot().Data() ;
#if ( _DEBUGlevel_ >= 2 )
         statusOFS << "Xf delta vtot " << vtot(0) - vtot0[0] << std::endl;
#endif
       blas::Copy( ntotFine, vtot.Data(), 1, ham.Vtot().Data(), 1 );
     }

     psiDataPtr = psi.Wavefun().Data();
     Complex* psiDataPtrFinal = psiFinal.Wavefun().Data();
     for( Int i = 0; i < numStateLocal; i ++)
       for( Int j = 0; j < ntot; j ++){
	       Int index = i* ntot +j;
	       psiDataPtr[index] =  psiDataPtrFinal[index];
       }

#if ( _DEBUGlevel_ >= 2 )
     {
       statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
       Int width = numStateLocal;
       Int heightLocal = ntot;
       CpxNumMat  XTXtemp1( width, width );
  
       blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi.Wavefun().Data(), 
        heightLocal, psi.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
  
       for( Int i = 0; i < width; i++)
         statusOFS << " Psi End * conjg( Psi End) : " << XTXtemp1(i,i) << std::endl;
       Complex *ptr = psi.Wavefun().Data();
       DblNumVec vtot = ham.Vtot();
       statusOFS << " Psi End 0 : " << ptr[0] << " vtot " << vtot[0] << std::endl;
       statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
     }
#endif
       
     //update Velocity
     if(options_.ehrenfest){
       ham.CalculateForce( psi, fft);
       Real& dt = options_.dt;
       DblNumVec& atomMass = atomMass_;
       for( Int a = 0; a < numAtom; a++ ){
         atomList[a].vel = atomList[a].vel + (atomforce[a]/atomMass[a] + atomList[a].force/atomMass[a])*dt/2.0;
       } 
     }

     ++k_;
  }

  void TDDFT::propagate( PeriodTable& ptable ) {
    Int totalSteps = tlist_.size();
    for( Int i = 0; i < totalSteps; i++)
        advanceRK4( ptable );
   }
}
#endif
