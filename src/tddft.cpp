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

  void setDefaultTDDFTOptions( TDDFTOptions * options)
  {
     options->auto_save     = 0;
     options->load_save     = false;
     options->method        = "RK4";
     options->ehrenfest     = false;
     options->simulateTime  = 1.00;
     options->dt            = 0.05;
     options->gmres_restart = 10; // not sure.
     options->krylovTol     = 1.0E-7;
     options->krylovMax     = 30; 
     options->scfTol        = 1.0E-7; 
     options->adNum         = 20; 
     options->adUpdate      = 1;
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

  } // TDDFT::Setup function

  void
  TDDFT::VelocityVerlet    ( Int ionIter )
  {
    Int mpirank, mpisize;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
    MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

    std::vector<Atom>&   atomList = *atomListPtr_;

    Int numAtom = atomList.size();

    std::vector<Point3>  atompos(numAtom);
    std::vector<Point3>  atomvel(numAtom);
    std::vector<Point3>  atomforce(numAtom);

    if( mpirank == 0 ){

      // some aliasing to be compatible with implementation before
      Real& dt = options_.dt;
      DblNumVec& atomMass = atomMass_;
      Real  K;

      for( Int a = 0; a < numAtom; a++ ){
        atompos[a]   = atomList[a].pos;
        atomvel[a]   = atomList[a].vel;
        atomforce[a] = atomList[a].force;
      }


      // Propagate velocity. This is the second part of Verlet step

      for( Int a = 0; a < numAtom; a++ ){
        atomvel[a] = atomvel[a] + atomforce[a]*dt*0.5/atomMass[a]; 
      }


      // Propagate the chain. This is due to the remaining update of the
      // chain variables.
      // used in verlet, commented out. Weile
      /*
      K=0.;
      for(Int a=0; a<numAtom; a++){
        for(Int j=0; j<3; j++){
          K += atomMass[a]*atomvel[a][j]*atomvel[a][j]/2.;
        }
      }
      */
      // At this point, the position, velocity and thermostat variables are
      // synced at the same time step

      // CHECK CHECK 
      /*
      Ekinetic_  = K;
      Econserve_ = Ekinetic_ + Epot_;
      if(ionIter == 1)
        EconserveInit_ = Econserve_;
      Edrift_ = (Econserve_-EconserveInit_)/EconserveInit_;
     
      Print(statusOFS, "TDDFT_Ekin    =  ", Ekinetic_);
      Print(statusOFS, "TDDFT_Epot    =  ", Epot_);
      Print(statusOFS, "TDDFT_Econ    =  ", Econserve_);
      Print(statusOFS, "TDDFT_Edrift  =  ", Edrift_);
      */

      // Output the XYZ format for movie
      // Once this is written, all work associated with the current atomic
      // position is DONE.
      /*
      if( isOutputXYZ_ ){
        std::fstream fout;
        fout.open("TDDFT.xyz",std::ios::out | std::ios::app) ;
        if( !fout.good() ){
          ErrorHandling( "Cannot open TDDFT.xyz!" );
        }
        fout << numAtom << std::endl;
        fout << "TDDFT step # "<< ionIter << std::endl;
        for(Int a=0; a<numAtom; a++){
          fout<< std::setw(6)<< atomList[a].type
	      << std::setw(16)<< atompos[a][0]*au2ang
	      << std::setw(16)<< atompos[a][1]*au2ang
	      << std::setw(16)<< atompos[a][2]*au2ang
	      << std::endl;
        }
        fout.close();
      }
      */

      // Update velocity and position
      for(Int a=0; a<numAtom; a++) {
        atomvel[a] = atomvel[a] + atomforce[a]*dt*0.5/atomMass[a]; 
        atompos[a] = atompos[a] + atomvel[a] * dt;
      }

      // Output the position and thermostat variable. 
      // These are the configuration that SCF will work on next. 
      // Hence if the job is stopped in the middle of SCF (which is most
      // likely), the TDDFT job should continue from this configuration

      // CHECK CHECK 
      /*
      if(isOutputVelocity_){
        std::fstream fout_v;
        fout_v.open("lastVel.out",std::ios::out);
        if( !fout_v.good() ){
          ErrorHandling( "File cannot be opened !" );
        }
        for(Int i=0; i<numAtom; i++){
          fout_v << std::setiosflags(std::ios::scientific)
		 << std::setiosflags(std::ios::showpos)
		 << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][0]
		 << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][1]
		 << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][2]
		 << std::resetiosflags(std::ios::scientific)
		 << std::resetiosflags(std::ios::showpos)
		 << std::endl;
        }
        fout_v.close();
      }
      */

    } // if( mpirank == 0 )

    // Sync the atomic position and velocity
    for(Int a = 0; a < numAtom; a++){
      MPI_Bcast( &atompos[a][0], 3, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 
      MPI_Bcast( &atomvel[a][0], 3, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 
    }

    // Update atomic position and velocity to store in atomListPtr_
    // NOTE: Force is NOT consistent with the position yet.
    for(Int a = 0; a < numAtom; a++){
      atomList[a].pos = atompos[a];
      atomList[a].vel = atomvel[a];
    }


    return ;
  }         // -----  end of method TDDFT::VelocityVerlet  ----- 

  void
  TDDFT::MoveIons    ( Int ionIter )
  {
    Int mpirank, mpisize;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
    MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

    std::vector<Atom>&   atomList = *atomListPtr_;
    Int numAtom = atomList.size();

    // Update saved atomList. 0 is the latest one
    for( Int l = maxHist_-1; l > 0; l-- ){
      atomListHist_[l] = atomListHist_[l-1];
    }
    atomListHist_[0] = atomList;

    // *********************************************************************
    // Molecular dynamics methods
    // *********************************************************************
    // if( ionMove_ == "verlet" ){
    VelocityVerlet( ionIter );
    //}
   
    // Output the new coordinates
    /*
    {
      Print(statusOFS, ""); 
      Print(statusOFS, "Atom Type and Coordinates");
      Print(statusOFS, ""); 
      for(Int i=0; i < atomList.size(); i++) {
        statusOFS << std::setiosflags(std::ios::left) 
		  << std::setw(LENGTH_VAR_NAME) << "Type = "
		  << std::setw(6) << atomList[i].type
		  << std::setiosflags(std::ios::scientific)
		  << std::setiosflags(std::ios::showpos)
		  << std::setw(LENGTH_VAR_NAME) << "Pos  = "
		  << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].pos[0]
		  << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].pos[1]
		  << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].pos[2]
		  << std::setw(LENGTH_VAR_NAME) << "Vel  = "
		  << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].vel[0]
		  << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].vel[1]
		  << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].vel[2]
		  << std::resetiosflags(std::ios::scientific)
		  << std::resetiosflags(std::ios::showpos)
		  << std::endl;
      }
    }
    */

    // Output the position. Common to all routines
    /*
    if( mpirank == 0 ){
      if(isOutputPosition_){
        std::fstream fout;
        fout.open("lastPos.out",std::ios::out);
        if( !fout.good() ){
          ErrorHandling( "File cannot be opened !" );
        }
        for(Int i=0; i<numAtom; i++){
          fout << std::setiosflags(std::ios::scientific)
	       << std::setiosflags(std::ios::showpos)
	       << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[i].pos[0]
	       << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[i].pos[1]
	       << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[i].pos[2]
	       << std::resetiosflags(std::ios::scientific)
	       << std::resetiosflags(std::ios::showpos)
	       << std::endl;
        }
        fout.close();
      }
    }
    */
     // Output the XYZ format for movie
     /*
    if( mpirank == 0 ){
      if( isOutputXYZ_ ){
	std::fstream fout;
	fout.open("MD.xyz",std::ios::out | std::ios::app) ;
	if( !fout.good() ){
	  ErrorHandling( "Cannot open MD.xyz!" );
	}
	fout << numAtom << std::endl;
	fout << "MD step # "<< ionIter << std::endl;
	for(Int a=0; a<numAtom; a++){
	  fout<< std::setw(6)<< atomList[a].type
	      << std::setw(16)<< atomList[a].pos[0]*au2ang
	      << std::setw(16)<< atomList[a].pos[1]*au2ang
	      << std::setw(16)<< atomList[a].pos[2]*au2ang
	      << std::endl;
	}
	fout.close();
      }
    } // if( mpirank == 0 )
    */

    return ;
  }         // -----  end of method TDDFT::MoveIons  ----- 



  void TDDFT::advance()
  {
     Int mpirank, mpisize;
     MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
     MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

     Hamiltonian& ham = *hamPtr_;
     Fourier&     fft = *fftPtr_;
     Spinor&      psi = *psiPtr_;

     std::vector<Atom>&   atomList = *atomListPtr_;
     Int numAtom = atomList.size();
 
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
   
         // some aliasing to be compatible with implementation before
         Real& dt = options_.dt;
         DblNumVec& atomMass = atomMass_;
         Real  K;
   
         for( Int a = 0; a < numAtom; a++ ){
           atompos[a]   = atomList[a].pos;
           atompos_mid[a]   = atomList[a].pos;
           atompos_fin[a]   = atomList[a].pos;
           atomvel[a]   = atomList[a].vel;
           atomforce[a] = atomList[a].force;
         }
   
         // Propagate velocity. This is the second part of Verlet step
	 // this is for the Ek, not in tddft, 
	 // CHECK CHECK
	 /*
         for( Int a = 0; a < numAtom; a++ ){
           atomvel[a] = atomvel[a] + atomforce[a]*dt*0.5/atomMass[a]; 
         }
         */
         // Update velocity and position
         for(Int a=0; a<numAtom; a++) {
           atomvel_temp[a] = atomvel[a]/2.0 + atomforce[a]*dt/atomMass[a]/8.0; 
           atompos_mid[a] = atompos[a] + atomvel_temp[a] * dt;

           atomvel_temp[a] = atomvel[a] + atomforce[a]*dt/atomMass[a]/2.0; 
           atompos_fin[a] = atompos[a] + atomvel_temp[a] * dt;
         }
       }
     }

     // have the atompos_mid and atompos_final
     for(Int a = 0; a < numAtom; a++){
       MPI_Bcast( &atompos_mid[a][0], 3, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 
       MPI_Bcast( &atompos_fin[a][0], 3, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 
     }

     // k_ is the current K
     Int k = k_;
     Real ti = tlist_[k];
     Real tf = tlist_[k+1];
     Real dT = tf - ti;
     Real tmid =  (ti + tf)/2.0;
     statusOFS<< " RK4 step " << k << "  t = " << ti << std::endl;



     // 4-th order Runge-Kutta  Start now 


     //set the OccupationRate to 1.0
     //hamPtr_ = &ham;
     //Fourier& fft = *fftPtr_;
     DblNumVec &occupationRate = ham.OccupationRate();
     occupationRate.Resize( psi.NumStateTotal() );
     SetValue( occupationRate, 1.0);

     // K1 = -i1(H1 * psi)
     Int ntot      = fft.domain.NumGridTotal();
     Int numStateLocal = psi.NumState();
     CpxNumMat Hpsi(ntot, numStateLocal);
     NumTns<Complex> tnsTemp(ntot, 1, numStateLocal, false, Hpsi.Data());
     ham.MultSpinor( psi, tnsTemp, fft );
     

     // CHECK CHECK, not multiply the by -i yet. 
     CpxNumMat X2(ntot, numStateLocal);
     Complex* dataPtr = X2.Data();
     Complex* psiDataPtr = psi.Wavefun().Data();
     Complex* HpsiDataPtr = Hpsi.Data();
     for( Int i = 0; i < numStateLocal; i ++)
       for( Int j = 0; i < ntot; j ++)
       {
	       Int index = i* ntot +j;
	       dataPtr[index] = psiDataPtr[index] +  HpsiDataPtr[index] * options_.dt/2.0;
       }

     // Update the H matrix. 
     // CHECK CHECK: The Vext is not updated here. 
     {
       // use the mid atom position
       for( Int a = 0; a < numAtom; a++ ){
           atomList[a].pos   =  atompos_mid[a];
       }
       Real totalCharge_;
       ham.CalculateDensity(
            psi,
            ham.OccupationRate(),
            totalCharge_, 
            fft );

       //get the new V(r,t+dt) from the rho(r,t+dt)
       Real Exc_;
       ham.CalculateXC( Exc_, fft ); 
       ham.CalculateHartree( fft );
       Int ntotFine = fft.domain.NumGridTotalFine();
       DblNumVec vtotNew_;
       vtotNew_.Resize(ntotFine); 
       SetValue(vtotNew_, 0.0);
       ham.CalculateVtot( vtotNew_ );
       blas::Copy( ntotFine, vtotNew_.Data(), 1, ham.Vtot().Data(), 1 );
     }
 
     // Now Hpsi is H2 * X2
     // K2 = -i1(H2 * psi) CHECK CHECK, not multply by -i yet. 
     Int numStateTotal = psi.NumStateTotal();
     Spinor psi2 (fft.domain, 1, numStateTotal, numStateLocal, false, X2.Data() );
     ham.MultSpinor( psi2, tnsTemp, fft );

     // X3 = X + dt/2 * K2
     CpxNumMat X3(ntot, numStateLocal);
     dataPtr = X3.Data();
     psiDataPtr = psi.Wavefun().Data();
     HpsiDataPtr = Hpsi.Data();
     for( Int i = 0; i < numStateLocal; i ++)
       for( Int j = 0; i < ntot; j ++)
       {
	       Int index = i* ntot +j;
	       dataPtr[index] = psiDataPtr[index] +  HpsiDataPtr[index] * options_.dt/2.0;
       }

     // Update the H matrix. 
     // CHECK CHECK: The Vext is not updated here. 
     {
       // use the mid atom position
       for( Int a = 0; a < numAtom; a++ ){
           atomList[a].pos   =  atompos_mid[a];
       }
       Real totalCharge_;
       ham.CalculateDensity(
            psi,
            ham.OccupationRate(),
            totalCharge_, 
            fft );
       //get the new V(r,t+dt) from the rho(r,t+dt)
       Real Exc_;
       ham.CalculateXC( Exc_, fft ); 
       ham.CalculateHartree( fft );
       Int ntotFine = fft.domain.NumGridTotalFine();
       DblNumVec vtotNew_;
       vtotNew_.Resize(ntotFine);
       SetValue(vtotNew_, 0.0);
       ham.CalculateVtot( vtotNew_ );
       blas::Copy( ntotFine, vtotNew_.Data(), 1, ham.Vtot().Data(), 1 );
     }
 
     // Now Hpsi is H2 * X2
     // K2 = -i1(H2 * psi) CHECK CHECK, not multply by -i yet. 
     Spinor psi3 (fft.domain, 1, numStateTotal, numStateLocal, false, X3.Data() );
     ham.MultSpinor( psi3, tnsTemp, fft );
     

     // X4 = X + dt/2 * K2
     CpxNumMat X4(ntot, numStateLocal);
     dataPtr = X4.Data();
     psiDataPtr = psi.Wavefun().Data();
     HpsiDataPtr = Hpsi.Data();
     for( Int i = 0; i < numStateLocal; i ++)
       for( Int j = 0; i < ntot; j ++)
       {
	       Int index = i* ntot +j;
	       dataPtr[index] = psiDataPtr[index] +  HpsiDataPtr[index] * options_.dt;
       }

     // Update the H matrix. 
     // CHECK CHECK: The Vext is not updated here. 
     {
       // use the mid atom position
       for( Int a = 0; a < numAtom; a++ ){
           atomList[a].pos   =  atompos_fin[a];
       }
       Real totalCharge_;
       ham.CalculateDensity(
            psi,
            ham.OccupationRate(),
            totalCharge_, 
            fft );
       //get the new V(r,t+dt) from the rho(r,t+dt)
       Real Exc_;
       ham.CalculateXC( Exc_, fft ); 
       ham.CalculateHartree( fft );
       Int ntotFine = fft.domain.NumGridTotalFine();
       DblNumVec vtotNew_;
       vtotNew_.Resize(ntotFine); SetValue(vtotNew_, 0.0);
       ham.CalculateVtot( vtotNew_ );
       blas::Copy( ntotFine, vtotNew_.Data(), 1, ham.Vtot().Data(), 1 );
     }
 
     // Now Hpsi is H2 * X2
     // K2 = -i1(H2 * psi) CHECK CHECK, not multply by -i yet. 
     Spinor psi4 (fft.domain, 1, numStateTotal, numStateLocal, false, X3.Data() );
     ham.MultSpinor( psi4, tnsTemp, fft );
 



     Real totalCharge_;
     ham.CalculateDensity(
            psi,
            ham.OccupationRate(),
            totalCharge_, 
            fft );

      

     // 4-th order Runge-Kutta  Start now 

     {
       //get the new V(r,t+dt) from the rho(r,t+dt)
       Real Exc_;
       ham.CalculateXC( Exc_, fft ); 
       ham.CalculateHartree( fft );
       Int ntotFine = fft.domain.NumGridTotalFine();
       DblNumVec vtotNew_;
       vtotNew_.Resize(ntotFine); SetValue(vtotNew_, 0.0);
       ham.CalculateVtot( vtotNew_ );
       blas::Copy( ntotFine, vtotNew_.Data(), 1, ham.Vtot().Data(), 1 );
     }
     ++k_;
  }
  
}
#endif
