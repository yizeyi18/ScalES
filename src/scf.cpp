/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

   Author: Lin Lin, Wei Hu and Amartya Banerjee
	 
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
/// @file scf.cpp
/// @brief SCF class for the global domain or extended element.
/// @date 2012-10-25
/// @date 2014-02-01 Dual grid implementation
/// @date 2014-08-07 Parallelization for PWDFT
#include  "scf.hpp"
#include	"blas.hpp"
#include	"lapack.hpp"
#include  "utility.hpp"

namespace  dgdft{

using namespace dgdft::DensityComponent;

SCF::SCF	(  )
{
#ifndef _RELEASE_
	PushCallStack("SCF::SCF");
#endif
	eigSolPtr_ = NULL;
	ptablePtr_ = NULL;

#ifndef _RELEASE_
	PopCallStack();
#endif
} 		// -----  end of method SCF::SCF  ----- 

SCF::~SCF	(  )
{
#ifndef _RELEASE_
	PushCallStack("SCF::~SCF");
#endif

#ifndef _RELEASE_
	PopCallStack();
#endif
} 		// -----  end of method SCF::~SCF  ----- 


void
SCF::Setup	( const esdf::ESDFInputParam& esdfParam, EigenSolver& eigSol, PeriodTable& ptable )
{
#ifndef _RELEASE_
    PushCallStack("SCF::Setup");
#endif

    // esdf parameters
    {
        mixMaxDim_     = esdfParam.mixMaxDim;
        mixType_       = esdfParam.mixType;
        mixStepLength_ = esdfParam.mixStepLength;
        // Note: for PW SCF there is no inner loop. Use the parameter value
        // for the outer SCF loop only.
        eigTolerance_  = esdfParam.eigTolerance;
        eigMinTolerance_  = esdfParam.eigMinTolerance;
        eigMaxIter_    = esdfParam.eigMaxIter;
        scfTolerance_  = esdfParam.scfOuterTolerance;
        scfMaxIter_    = esdfParam.scfOuterMaxIter;
        scfPhiMaxIter_ = esdfParam.scfPhiMaxIter;
        scfPhiTolerance_ = esdfParam.scfPhiTolerance;
        numUnusedState_ = esdfParam.numUnusedState;
        isEigToleranceDynamic_ = esdfParam.isEigToleranceDynamic;
        isRestartDensity_ = esdfParam.isRestartDensity;
        isRestartWfn_     = esdfParam.isRestartWfn;
        isOutputDensity_  = esdfParam.isOutputDensity;
        isCalculateForceEachSCF_       = esdfParam.isCalculateForceEachSCF;
        Tbeta_         = esdfParam.Tbeta;

        numGridWavefunctionElem_ = esdfParam.numGridWavefunctionElem;
        numGridDensityElem_      = esdfParam.numGridDensityElem;  

        XCType_                  = esdfParam.XCType;
        VDWType_                 = esdfParam.VDWType;

        isHybridACEOutside_      = esdfParam.isHybridACEOutside;
	
	// Chebyshev Filtering related parameters
	Diag_SCF_PWDFT_by_Cheby_ = esdfParam.Diag_SCF_PWDFT_by_Cheby;
        First_SCF_PWDFT_ChebyFilterOrder_ = esdfParam.First_SCF_PWDFT_ChebyFilterOrder;
        First_SCF_PWDFT_ChebyCycleNum_ = esdfParam.First_SCF_PWDFT_ChebyCycleNum;
        General_SCF_PWDFT_ChebyFilterOrder_ = esdfParam.General_SCF_PWDFT_ChebyFilterOrder;
	
	
    }

    // other SCF parameters
    {
        eigSolPtr_ = &eigSol;
        ptablePtr_ = &ptable;

        //		Int ntot = eigSolPtr_->Psi().NumGridTotal();
        Int ntot = esdfParam.domain.NumGridTotal();
        Int ntotFine = esdfParam.domain.NumGridTotalFine();

        vtotNew_.Resize(ntotFine); SetValue(vtotNew_, 0.0);
        dfMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
        dvMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dvMat_, 0.0 );

        restartDensityFileName_ = "DEN";
        restartWfnFileName_     = "WFN";
    }

    // Density
    {
        DblNumMat&  density = eigSolPtr_->Ham().Density();

        if( isRestartDensity_ ) {
            std::istringstream rhoStream;      
            SharedRead( restartDensityFileName_, rhoStream);
            // TODO Error checking
            deserialize( density, rhoStream, NO_MASK );    
        } // else using the zero initial guess
        else {
            // make sure the pseudocharge is initialized
            DblNumVec&  pseudoCharge = eigSolPtr_->Ham().PseudoCharge();

            SetValue( density, 0.0 );

            Int ntot = esdfParam.domain.NumGridTotal();
            Int ntotFine = esdfParam.domain.NumGridTotalFine();

            Real sum0 = 0.0, sum1 = 0.0;
            Real EPS = 1e-6;

            // make sure that the electron density is positive
            for (Int i=0; i<ntotFine; i++){
                density(i, RHO) = ( pseudoCharge(i) > EPS ) ? pseudoCharge(i) : EPS;
                sum0 += density(i, RHO);
                sum1 += pseudoCharge(i);
            }

            // Rescale the density
            for (int i=0; i <ntotFine; i++){
                density(i, RHO) *= sum1 / sum0;
            } 
        }
    }

    if( !isRestartWfn_ ) {
        //		UniformRandom( eigSolPtr_->Psi().Wavefun() );
    }
    else {
        std::istringstream iss;
        SharedRead( restartWfnFileName_, iss );
        deserialize( eigSolPtr_->Psi().Wavefun(), iss, NO_MASK );
    }

    // XC functional
    {
        isCalculateGradRho_ = false;
        if( XCType_ == "XC_GGA_XC_PBE" || 
                XCType_ == "XC_HYB_GGA_XC_HSE06" ) {
            isCalculateGradRho_ = true;
        }
    }

#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
} 		// -----  end of method SCF::Setup  ----- 

void
SCF::Update	( )
{
#ifndef _RELEASE_
	PushCallStack("SCF::Update");
#endif
	{
    Int ntotFine  = eigSolPtr_->FFT().domain.NumGridTotalFine();

    vtotNew_.Resize(ntotFine); SetValue(vtotNew_, 0.0);
		dfMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
		dvMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dvMat_, 0.0 );
	}
#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCF::Update  ----- 



void
SCF::Iterate (  )
{
#ifndef _RELEASE_
	PushCallStack("SCF::Iterate");
#endif
	Real timeSta, timeEnd;
  // Only works for KohnSham class
  Hamiltonian& ham = eigSolPtr_->Ham();
  Fourier&     fft = eigSolPtr_->FFT();
  Spinor&      psi = eigSolPtr_->Psi();

  // EXX: Only allow hybrid functional here

  // Compute the exchange-correlation potential and energy
	if( isCalculateGradRho_ ){
		ham.CalculateGradDensity( fft );
	}
	ham.CalculateXC( Exc_, fft ); 

	// Compute the Hartree energy
	ham.CalculateHartree( fft );
	// No external potential

	// Compute the total potential
	ham.CalculateVtot( ham.Vtot() );

  Real timeIterStart(0), timeIterEnd(0);
  Real timePhiIterStart(0), timePhiIterEnd(0);
  
  // EXX: Run SCF::Iterate here
  bool isPhiIterConverged = false;

  // Fock energies
  Real fock0 = 0.0, fock1 = 0.0, fock2 = 0.0;

  if( ham.IsHybrid() == false || isHybridACEOutside_ == true ){
    // Let the hybrid functional be handledo outside the SCF loop
    scfPhiMaxIter_ = 1;
  }

  for( Int phiIter = 1; phiIter <= scfPhiMaxIter_; phiIter++ ){
    bool isSCFConverged = false;

    if( ham.IsHybrid() && isHybridACEOutside_ == false )
    {
      if ( isPhiIterConverged ) break;
      GetTime( timePhiIterStart );
      std::ostringstream msg;
      msg << "Phi iteration # " << phiIter;
      PrintBlock( statusOFS, msg.str() );
    }


    // Regular SCF iter
    for (Int iter=1; iter <= scfMaxIter_; iter++) {
      if ( isSCFConverged ) break;

      // *********************************************************************
      // Performing each iteartion
      // *********************************************************************
      {
        std::ostringstream msg;
        msg << "SCF iteration # " << iter;
        PrintBlock( statusOFS, msg.str() );
      }

      GetTime( timeIterStart );

      // Solve the eigenvalue problem

      Real eigTolNow;
      if( isEigToleranceDynamic_ ){
        // Dynamic strategy to control the tolerance
        if( iter == 1 )
          eigTolNow = 1e-2;
        else
          eigTolNow = std::max( std::min( scfNorm_*1e-2, 1e-2 ) , eigTolerance_);
      }
      else{
        // Static strategy to control the tolerance
        eigTolNow = eigTolerance_;
      }

      Int numEig = (psi.NumStateTotal())-numUnusedState_;

      statusOFS << "The current tolerance used by the eigensolver is " 
        << eigTolNow << std::endl;
      statusOFS << "The target number of converged eigenvectors is " 
        << numEig << std::endl;

      GetTime( timeSta );
      
      if(Diag_SCF_PWDFT_by_Cheby_ == 1)
      {
	// Use CheFSI
	if(iter <= 1){
	//eigSolPtr_->LOBPCGSolveReal2(numEig, eigMaxIter_, eigTolNow );
	eigSolPtr_->FirstChebyStep(numEig, First_SCF_PWDFT_ChebyCycleNum_, First_SCF_PWDFT_ChebyFilterOrder_);
      }
      else{
	//eigSolPtr_->LOBPCGSolveReal2(numEig, eigMaxIter_, eigTolNow );
	eigSolPtr_->GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
	}
      }
      else
      {
	// Use LOBPCG
	eigSolPtr_->LOBPCGSolveReal2(numEig, eigMaxIter_, eigMinTolerance_, eigTolNow );	
      }

      
      GetTime( timeEnd );

      ham.EigVal() = eigSolPtr_->EigVal();

      statusOFS << std::endl << " Time for the eigensolver is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;


      // No need for normalization using LOBPCG

      // Compute the occupation rate
      CalculateOccupationRate( ham.EigVal(), 
          ham.OccupationRate() );

      // Compute the electron density
      ham.CalculateDensity(
          psi,
          ham.OccupationRate(),
          totalCharge_, 
          fft );


      // Compute the exchange-correlation potential and energy
      if( isCalculateGradRho_ ){
        ham.CalculateGradDensity( fft );
      }
      ham.CalculateXC( Exc_, fft ); 

      // Compute the Hartree energy
      ham.CalculateHartree( fft );
      // No external potential

      // Compute the total potential
      ham.CalculateVtot( vtotNew_ );

      Real normVtotDif = 0.0, normVtotOld;
      DblNumVec& vtotOld_ = ham.Vtot();
      Int ntot = vtotOld_.m();
      for( Int i = 0; i < ntot; i++ ){
        normVtotDif += pow( vtotOld_(i) - vtotNew_(i), 2.0 );
        normVtotOld += pow( vtotOld_(i), 2.0 );
      }
      normVtotDif = sqrt( normVtotDif );
      normVtotOld = sqrt( normVtotOld );
      scfNorm_    = normVtotDif / normVtotOld;

      Evdw_ = 0.0;

      CalculateEnergy();

      PrintState( iter );

      if( scfNorm_ < scfTolerance_ ){
        /* converged */
        statusOFS << "SCF is converged in " << iter << " steps !" << std::endl;
        isSCFConverged = true;
      }

      // Potential mixing
      if( mixType_ == "anderson" ){
        AndersonMix(iter);
      }

      if( mixType_ == "kerker" ){
        KerkerMix();  
        AndersonMix(iter);
      }

      GetTime( timeIterEnd );

      statusOFS << "Total wall clock time for this SCF iteration = " << timeIterEnd - timeIterStart
        << " [s]" << std::endl;

    }

    // EXX
    if( ham.IsHybrid() && isHybridACEOutside_ == false ){
      Real dExx;
      if( phiIter == 1 ){
        ham.SetEXXActive(true);
        // Update Phi <- Psi
        GetTime( timeSta );
        ham.SetPhiEXX( psi, fft ); 
        if( ham.IsHybridACE() ){
          ham.CalculateVexxACE ( psi, fft );
        }
        GetTime( timeEnd );
        statusOFS << "Time for updating Phi related variable is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

        GetTime( timeSta );
        fock2 = ham.CalculateEXXEnergy( psi, fft ); 
        GetTime( timeEnd );
        statusOFS << "Time for computing the EXX energy is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

        // Update the energy
        Efock_ = fock2;
        Etot_ = Etot_ - Efock_;
        Efree_ = Efree_ - Efock_;
        Print(statusOFS, "Fock energy       = ",  Efock_, "[au]");
        Print(statusOFS, "Etot(with fock)   = ",  Etot_, "[au]");
        Print(statusOFS, "Efree(with fock)  = ",  Efree_, "[au]");
      }
      else{
        // Calculate first
        fock1 = ham.CalculateEXXEnergy( psi, fft ); 

        // Update Phi <- Psi
        GetTime( timeSta );
        ham.SetPhiEXX( psi, fft ); 
        if( ham.IsHybridACE() ){
          ham.CalculateVexxACE ( psi, fft );
        }
        GetTime( timeEnd );
        statusOFS << "Time for updating Phi related variable is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;


        fock0 = fock2;
        // Calculate again
        GetTime( timeSta );
        fock2 = ham.CalculateEXXEnergy( psi, fft ); 
        GetTime( timeEnd );
        statusOFS << "Time for computing the EXX energy is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
        dExx = fock1 - 0.5 * (fock0 + fock2);

        Efock_ = fock2;
        Etot_ = Etot_ - Efock_;
        Efree_ = Efree_ - Efock_;
        Print(statusOFS, "dExx              = ",  dExx, "[au]");
        Print(statusOFS, "Fock energy       = ",  Efock_, "[au]");
        Print(statusOFS, "Etot(with fock)   = ",  Etot_, "[au]");
        Print(statusOFS, "Efree(with fock)  = ",  Efree_, "[au]");

        if( dExx < scfPhiTolerance_ ){
          statusOFS << "SCF for hybrid functional is converged in " 
            << phiIter << " steps !" << std::endl;
          isPhiIterConverged = true;
        }
      }

      GetTime( timePhiIterEnd );

      statusOFS << "Total wall clock time for this Phi iteration = " << 
        timePhiIterEnd - timePhiIterStart << " [s]" << std::endl;
    } // if (hybrid)
  } // for(phiIter)

  // Calculate the Force
  if(0){
    ham.CalculateForce( psi, fft );
  }
  if(1){
    ham.CalculateForce2( psi, fft );
  }
  
  // Calculate the VDW energy
  if( VDWType_ == "DFT-D2"){
    CalculateVDW ( Evdw_, forceVdw_ );
    // Update energy
    Etot_  += Evdw_;
    Efree_ += Evdw_;
    Ecor_  += Evdw_;

    // Update force
    std::vector<Atom>& atomList = ham.AtomList();
    for( Int a = 0; a < atomList.size(); a++ ){
      atomList[a].force += Point3( forceVdw_(a,0), forceVdw_(a,1), forceVdw_(a,2) );
    }
  } 

  // Output the information after SCF
  {
    // Energy
    Real HOMO, LUMO;
    HOMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState()-1);
    if( eigSolPtr_->Ham().NumExtraState() > 0 )
      LUMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState());

    // Print out the energy
    PrintBlock( statusOFS, "Energy" );
    statusOFS 
      << "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself + Evdw" << std::endl
      << "       Etot  = Ekin + Ecor" << std::endl
      << "       Efree = Etot	+ Entropy" << std::endl << std::endl;
    Print(statusOFS, "! Etot            = ",  Etot_, "[au]");
    Print(statusOFS, "! Efree           = ",  Efree_, "[au]");
    Print(statusOFS, "! Evdw            = ",  Evdw_, "[au]"); 
    Print(statusOFS, "! Fermi           = ",  fermi_, "[au]");
    Print(statusOFS, "! HOMO            = ",  HOMO*au2ev, "[ev]");
    if( ham.NumExtraState() > 0 ){
      Print(statusOFS, "! LUMO            = ",  LUMO*au2ev, "[eV]");
    }
    Print(statusOFS, "! norm(out-in)/norm(in) = ",  scfNorm_ ); 
  }
  
  {
    // Print out the force
    PrintBlock( statusOFS, "Atomic Force" );
    
    Point3 forceCM(0.0, 0.0, 0.0);
    std::vector<Atom>& atomList = ham.AtomList();
    Int numAtom = atomList.size();

    for( Int a = 0; a < numAtom; a++ ){
      Print( statusOFS, "atom", a, "force", atomList[a].force );
      forceCM += atomList[a].force;
    }
    statusOFS << std::endl;
    Print( statusOFS, "force for centroid  : ", forceCM );
    Print( statusOFS, "Max force magnitude : ", MaxForce(atomList) );
    statusOFS << std::endl;
  }


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCF::Iterate  ----- 



void
SCF::CalculateOccupationRate	( DblNumVec& eigVal, DblNumVec& occupationRate )
{
#ifndef _RELEASE_
	PushCallStack("SCF::CalculateOccupationRate");
#endif
	// For a given finite temperature, update the occupation number */
	// FIXME Magic number here
	Real tol = 1e-10; 
	Int maxiter = 100;  

	Real lb, ub, flb, fub, occsum;
	Int ilb, iub, iter;

	Int npsi       = eigSolPtr_->Ham().NumStateTotal();
	Int nOccStates = eigSolPtr_->Ham().NumOccupiedState();

	if( eigVal.m() != npsi ){
		std::ostringstream msg;
		msg 
			<< "The number of eigenstates do not match."  << std::endl
			<< "eigVal         ~ " << eigVal.m() << std::endl
			<< "numStateTotal  ~ " << npsi << std::endl;
		throw std::logic_error( msg.str().c_str() );
	}


	if( occupationRate.m() != npsi ) occupationRate.Resize( npsi );

	if( npsi > nOccStates )  {
		/* use bisection to find efermi such that 
		 * sum_i fermidirac(ev(i)) = nocc
		 */
		ilb = nOccStates-1;
		iub = nOccStates+1;

		lb = eigVal(ilb-1);
		ub = eigVal(iub-1);

		/* Calculate Fermi-Dirac function and make sure that
		 * flb < nocc and fub > nocc
		 */

		flb = 0.0;
		fub = 0.0;
		for(Int j = 0; j < npsi; j++) {
			flb += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-lb)));
			fub += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-ub))); 
		}

		while( (nOccStates-flb)*(fub-nOccStates) < 0 ) {
			if( flb > nOccStates ) {
				if(ilb > 0){
					ilb--;
					lb = eigVal(ilb-1);
					flb = 0.0;
					for(Int j = 0; j < npsi; j++) flb += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-lb)));
				}
				else {
					throw std::logic_error( "Cannot find a lower bound for efermi" );
				}
			}

			if( fub < nOccStates ) {
				if( iub < npsi ) {
					iub++;
					ub = eigVal(iub-1);
					fub = 0.0;
					for(Int j = 0; j < npsi; j++) fub += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-ub)));
				}
				else {
					throw std::logic_error( "Cannot find a lower bound for efermi, try to increase the number of wavefunctions" );
				}
			}
		}  /* end while */

		fermi_ = (lb+ub)*0.5;
		occsum = 0.0;
		for(Int j = 0; j < npsi; j++) {
			occupationRate(j) = 1.0 / (1.0 + exp(Tbeta_*(eigVal(j) - fermi_)));
			occsum += occupationRate(j);
		}

		/* Start bisection iteration */
		iter = 1;
		while( (fabs(occsum - nOccStates) > tol) && (iter < maxiter) ) {
			if( occsum < nOccStates ) {lb = fermi_;}
			else {ub = fermi_;}

			fermi_ = (lb+ub)*0.5;
			occsum = 0.0;
			for(Int j = 0; j < npsi; j++) {
				occupationRate(j) = 1.0 / (1.0 + exp(Tbeta_*(eigVal(j) - fermi_)));
				occsum += occupationRate(j);
			}
			iter++;
		}
	}
	else {
		if (npsi == nOccStates ) {
			for(Int j = 0; j < npsi; j++) 
				occupationRate(j) = 1.0;
			fermi_ = eigVal(npsi-1);
		}
		else {
			throw std::logic_error( "The number of eigenvalues in ev should be larger than nocc" );
		}
	}

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCF::CalculateOccupationRate  ----- 


void
SCF::CalculateEnergy	(  )
{
#ifndef _RELEASE_
	PushCallStack("SCF::CalculateEnergy");
#endif
	Ekin_ = 0.0;
	DblNumVec&  eigVal         = eigSolPtr_->Ham().EigVal();
	DblNumVec&  occupationRate = eigSolPtr_->Ham().OccupationRate();

	// Kinetic energy
	Int numSpin = eigSolPtr_->Ham().NumSpin();
	for (Int i=0; i < eigVal.m(); i++) {
		Ekin_  += numSpin * eigVal(i) * occupationRate(i);
	}

	// Hartree and xc part
	Int  ntot = eigSolPtr_->FFT().domain.NumGridTotalFine();
	Real vol  = eigSolPtr_->FFT().domain.Volume();
	DblNumMat&  density      = eigSolPtr_->Ham().Density();
  DblNumMat&  vxc          = eigSolPtr_->Ham().Vxc();
  DblNumVec&  pseudoCharge = eigSolPtr_->Ham().PseudoCharge();
	DblNumVec&  vhart        = eigSolPtr_->Ham().Vhart();
	Ehart_ = 0.0;
	EVxc_  = 0.0;
	for (Int i=0; i<ntot; i++) {
		EVxc_  += vxc(i,RHO) * density(i,RHO);
		Ehart_ += 0.5 * vhart(i) * ( density(i,RHO) + pseudoCharge(i) );
	}
	Ehart_ *= vol/Real(ntot);
	EVxc_  *= vol/Real(ntot);

	// Self energy part
	Eself_ = 0;
	std::vector<Atom>&  atomList = eigSolPtr_->Ham().AtomList();
	for(Int a=0; a< atomList.size() ; a++) {
		Int type = atomList[a].type;
		Eself_ +=  ptablePtr_->ptemap()[type].params(PTParam::ESELF);
	}

  // Correction energy
  Ecor_ = (Exc_ - EVxc_) - Ehart_ - Eself_;

  // Total energy
	Etot_ = Ekin_ + Ecor_;

	// Helmholtz fre energy
	if( eigSolPtr_->Ham().NumOccupiedState() == 
			eigSolPtr_->Ham().NumStateTotal() ){
		// Zero temperature
		Efree_ = Etot_;
	}
	else{
		// Finite temperature
		Efree_ = 0.0;
		Real fermi = fermi_;
		Real Tbeta = Tbeta_;
		for(Int l=0; l< eigVal.m(); l++) {
			Real eig = eigVal(l);
			if( eig - fermi >= 0){
				Efree_ += -numSpin / Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
			}
			else{
				Efree_ += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
			}
		}
		Efree_ += Ecor_ + fermi * eigSolPtr_->Ham().NumOccupiedState() * numSpin; 
	}


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCF::CalculateEnergy  ----- 


void
SCF::CalculateVDW	( Real& VDWEnergy, DblNumMat& VDWForce )
{
#ifndef _RELEASE_
  PushCallStack("SCF::CalculateVDW");
#endif

  //Real& VDWEnergy = Evdw_;
  //DblNumMat& VDWForce = forceVdw_;

  std::vector<Atom>&  atomList = eigSolPtr_->Ham().AtomList();
  Evdw_ = 0.0;
  forceVdw_.Resize( atomList.size(), DIM );
  SetValue( forceVdw_, 0.0 );

  Int numAtom = atomList.size();

  Domain& dm = eigSolPtr_->FFT().domain;

  // std::vector<Point3>  atompos(numAtom);
  // for( Int i = 0; i < numAtom; i++ ){
  //   atompos[i]   = atomList[i].pos;
  // }

  if( VDWType_ == "DFT-D2"){

    const Int vdw_nspecies = 55;
    Int ia,is1,is2,is3,itypat,ja,jtypat,npairs,nshell;
    bool need_gradient,newshell;
    const Real vdw_d = 20.0;
    const Real vdw_tol_default = 1e-10;
    const Real vdw_s_pbe = 0.75;
    Real c6,c6r6,ex,fr,fred1,fred2,fred3,gr,grad,r0,r1,r2,r3,rcart1,rcart2,rcart3;
    //real(dp) :: rcut,rcut2,rsq,rr,sfact,ucvol,vdw_s
    //character(len=500) :: msg
    //type(atomdata_t) :: atom
    //integer,allocatable :: ivdw(:)
    //real(dp) :: gmet(3,3),gprimd(3,3),rmet(3,3)
    //real(dp),allocatable :: vdw_c6(:,:),vdw_r0(:,:),xred01(:,:)
    //DblNumVec vdw_c6_dftd2(vdw_nspecies);

    double vdw_c6_dftd2[vdw_nspecies] = 
    { 0.14, 0.08, 1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63,
      5.71, 5.71,10.79, 9.23, 7.84, 5.57, 5.07, 4.61,10.80,10.80,
      10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,
      16.99,17.10,16.37,12.64,12.47,12.01,24.67,24.67,24.67,24.67,
      24.67,24.67,24.67,24.67,24.67,24.67,24.67,24.67,37.32,38.71,
      38.44,31.74,31.50,29.99, 0.00 };

    // DblNumVec vdw_r0_dftd2(vdw_nspecies);
    double vdw_r0_dftd2[vdw_nspecies] =
    { 1.001,1.012,0.825,1.408,1.485,1.452,1.397,1.342,1.287,1.243,
      1.144,1.364,1.639,1.716,1.705,1.683,1.639,1.595,1.485,1.474,
      1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,
      1.650,1.727,1.760,1.771,1.749,1.727,1.628,1.606,1.639,1.639,
      1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.672,1.804,
      1.881,1.892,1.892,1.881,1.000 };

    for(Int i=0; i<vdw_nspecies; i++) {
      vdw_c6_dftd2[i] = vdw_c6_dftd2[i] / 2625499.62 * pow(10/0.52917706, 6);
      vdw_r0_dftd2[i] = vdw_r0_dftd2[i] / 0.52917706;
    }

    DblNumMat vdw_c6(vdw_nspecies, vdw_nspecies);
    DblNumMat vdw_r0(vdw_nspecies, vdw_nspecies);
    SetValue( vdw_c6, 0.0 );
    SetValue( vdw_r0, 0.0 );

    for(Int i=0; i<vdw_nspecies; i++) {
      for(Int j=0; j<vdw_nspecies; j++) {
        vdw_c6(i, j) = std::sqrt( vdw_c6_dftd2[i] * vdw_c6_dftd2[j] );
        vdw_r0(i, j) = vdw_r0_dftd2[i] + vdw_r0_dftd2[j];
      }
    }

    Real vdw_s;

    if (XCType_ == "XC_GGA_XC_PBE") {
      vdw_s = vdw_s_pbe;
    }
    else {
      throw std::logic_error( "Van der Waals DFT-D2 correction in only compatible with GGA-PBE!" );
    }

    // Calculate the number of atom types.
//    Real numAtomType = 0;   
//    for(Int a=0; a< atomList.size() ; a++) {
//      Int type1 = atomList[a].type;
//      Int a1 = 0;
//      Int a2 = 0;
//      for(Int b=0; b<a ; b++) {
//        a1 = a1 + 1;
//        Int type2 = atomList[b].type;
//        if ( type1 != type2 ) {
//          a2 = a2 + 1;
//        }
//      }
//
//      if ( a1 == a2 ) {
//        numAtomType = numAtomType + 1;
//      }
//
//    }


    //    IntNumVec  atomType ( numAtomType );
    //    SetValue( atomType, 0 );

    //    Real numAtomType1 = 0;
    //    atomType(0) = atomList[0].type;


    //    for(Int a=0; a< atomList.size() ; a++) {
    //      Int type1 = atomList[a].type;
    //      Int a1 = 0;
    //      Int a2 = 0;
    //      for(Int b=0; b<a ; b++) {
    //        a1 = a1 + 1;
    //        Int type2 = atomList[b].type;
    //        if ( type1 != type2 ) {
    //          a2 = a2 + 1;
    //        }
    //      }
    //      if ( a1 == a2 ) {
    //        numAtomType1 = numAtomType1 + 1;
    //        atomType(numAtomType1-1) = atomList[a].type;
    //      }
    //    }


    //    DblNumMat  vdw_c6 ( numAtomType, numAtomType );
    //    DblNumMat  vdw_r0 ( numAtomType, numAtomType );
    //    SetValue( vdw_c6, 0.0 );
    //    SetValue( vdw_r0, 0.0 );
    //
    //    for(Int i=0; i< numAtomType; i++) {
    //      for(Int j=0; j< numAtomType; j++) {
    //        vdw_c6(i,j)=std::sqrt(vdw_c6_dftd2[atomType(i)-1]*vdw_c6_dftd2[atomType(j)-1]);
    //        //vdw_r0(i,j)=(vdw_r0_dftd2(atomType(i))+vdw_r0_dftd2(atomType(j)))/Bohr_Ang;
    //        vdw_r0(i,j)=(vdw_r0_dftd2[atomType(i)-1]+vdw_r0_dftd2[atomType(j)-1]);
    //      }
    //    }

    //    statusOFS << "vdw_c6 = " << vdw_c6 << std::endl;
    //    statusOFS << "vdw_r0 = " << vdw_r0 << std::endl;

    for(Int ii=-1; ii<2; ii++) {
      for(Int jj=-1; jj<2; jj++) {
        for(Int kk=-1; kk<2; kk++) {

          for(Int i=0; i<atomList.size(); i++) {
            Int iType = atomList[i].type;
            for(Int j=0; j<(i+1); j++) {
              Int jType = atomList[j].type;

              Real rx = atomList[i].pos[0] - atomList[j].pos[0] + ii * dm.length[0];
              Real ry = atomList[i].pos[1] - atomList[j].pos[1] + jj * dm.length[1];
              Real rz = atomList[i].pos[2] - atomList[j].pos[2] + kk * dm.length[2];
              Real rr = std::sqrt( rx * rx + ry * ry + rz * rz );

              if ( ( rr > 0.0001 ) && ( rr < 75.0 ) ) {

                Real sfact = vdw_s;
                if ( i == j ) sfact = sfact * 0.5;

                Real c6 = vdw_c6(iType-1, jType-1);
                Real r0 = vdw_r0(iType-1, jType-1);

                Real ex = exp( -vdw_d * ( rr / r0 - 1 ));
                Real fr = 1.0 / ( 1.0 + ex );
                Real c6r6 = c6 / pow(rr, 6.0);

                // Contribution to energy
                Evdw_ = Evdw_ - sfact * fr * c6r6;

                // Contribution to force
                if( i != j ) {

                  Real gr = ( vdw_d / r0 ) * ( fr * fr ) * ex;
                  Real grad = sfact * ( gr - 6.0 * fr / rr ) * c6r6 / rr; 

                  Real fx = grad * rx;
                  Real fy = grad * ry;
                  Real fz = grad * rz;

                  forceVdw_( i, 0 ) = forceVdw_( i, 0 ) + fx; 
                  forceVdw_( i, 1 ) = forceVdw_( i, 1 ) + fy; 
                  forceVdw_( i, 2 ) = forceVdw_( i, 2 ) + fz; 
                  forceVdw_( j, 0 ) = forceVdw_( j, 0 ) - fx; 
                  forceVdw_( j, 1 ) = forceVdw_( j, 1 ) - fy; 
                  forceVdw_( j, 2 ) = forceVdw_( j, 2 ) - fz; 

                } // end for i != j

              } // end if


            } // end for j
          } // end for i

        } // end for ii
      } // end for jj
    } // end for kk


    //#endif 

  } // If DFT-D2

  VDWEnergy = Evdw_;
  VDWForce = forceVdw_;


#ifndef _RELEASE_
  PopCallStack();
#endif

  return ;
} 		// -----  end of method SCF::CalculateVDW  ----- 


void
SCF::AndersonMix	( const Int iter )
{
#ifndef _RELEASE_
  PushCallStack("SCF::AndersonMix");
#endif
  DblNumVec vin, vout, vinsave, voutsave;

  Int ntot  = eigSolPtr_->FFT().domain.NumGridTotalFine();

  vin.Resize(ntot);
  vout.Resize(ntot);
  vinsave.Resize(ntot);
  voutsave.Resize(ntot);

  DblNumVec& vtot = eigSolPtr_->Ham().Vtot();

  for (Int i=0; i<ntot; i++) {
    vin(i) = vtot(i);
    vout(i) = vtotNew_(i) - vtot(i);
  }

  for(Int i = 0; i < ntot; i++){
    vinsave(i) = vin(i);
    voutsave(i) = vout(i);
  }

  Int iterused = std::min( iter-1, mixMaxDim_ ); // iter should start from 1
  Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;

  // TODO Set verbose level 
  Print( statusOFS, "Anderson mixing" );
  Print( statusOFS, "  iterused = ", iterused );
  Print( statusOFS, "  ipos     = ", ipos );


  if( iter > 1 ){
    for(Int i = 0; i < ntot; i++){
      dfMat_(i, ipos-1) -= vout(i);
      dvMat_(i, ipos-1) -= vin(i);
    }

    // Calculating pseudoinverse

    DblNumVec gammas, S;
    DblNumMat dftemp;
    Int rank;
    // FIXME Magic number
    Real rcond = 1e-6;

    S.Resize(iterused);

    gammas = vout;
    dftemp = dfMat_;

    lapack::SVDLeastSquare( ntot, iterused, 1, 
        dftemp.Data(), ntot, gammas.Data(), ntot,
        S.Data(), rcond, &rank );

    Print( statusOFS, "  Rank of dfmat = ", rank );

    // Update vin, vout

    blas::Gemv('N', ntot, iterused, -1.0, dvMat_.Data(),
        ntot, gammas.Data(), 1, 1.0, vin.Data(), 1 );

    blas::Gemv('N', ntot, iterused, -1.0, dfMat_.Data(),
        ntot, gammas.Data(), 1, 1.0, vout.Data(), 1 );
  }

  Int inext = iter - ((iter-1)/ mixMaxDim_) * mixMaxDim_;
  for (Int i=0; i<ntot; i++) {
    dfMat_(i, inext-1) = voutsave(i);
    dvMat_(i, inext-1) = vinsave(i);
  }

  for (Int i=0; i<ntot; i++) {
    vtot(i) = vin(i) + mixStepLength_ * vout(i);
  }

#ifndef _RELEASE_
  PopCallStack();
#endif

  return ;

} 		// -----  end of method SCF::AndersonMix  ----- 


void
SCF::KerkerMix	(  )
{
#ifndef _RELEASE_
  PushCallStack("SCF::KerkerMix");
#endif
  // FIXME Magic number here
  Real mixStepLengthKerker = 0.8; 
  Int ntot  = eigSolPtr_->FFT().domain.NumGridTotalFine();
  DblNumVec& vtot = eigSolPtr_->Ham().Vtot();

  for (Int i=0; i<ntot; i++) {
    eigSolPtr_->FFT().inputComplexVec(i) = 
      Complex(vtotNew_(i) - vtot(i), 0.0);
    // Why?
    vtot(i) += mixStepLengthKerker * (vtotNew_(i) - vtot(i));
  }
  fftw_execute( eigSolPtr_->FFT().forwardPlan );

  DblNumVec&  gkk = eigSolPtr_->FFT().gkk;

  for(Int i=0; i<ntot; i++) {
    if( gkk(i) == 0 ){
      eigSolPtr_->FFT().outputComplexVec(i) = Z_ZERO;
    }
    else{
      // FIXME Magic number
      eigSolPtr_->FFT().outputComplexVec(i) *= 
        mixStepLengthKerker * ( gkk(i) / (gkk(i)+0.5) - 1.0 );
    }
  }
  fftw_execute( eigSolPtr_->FFT().backwardPlan );

  // Update vtot
  for (Int i=0; i<ntot; i++)
	  vtot(i) += eigSolPtr_->FFT().inputComplexVec(i).real() / ntot;	

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCF::KerkerMix  ----- 

void
SCF::PrintState	( const Int iter  )
{
#ifndef _RELEASE_
	PushCallStack("SCF::PrintState");
#endif
  Real HOMO, LUMO;
  HOMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState()-1);
  if( eigSolPtr_->Ham().NumExtraState() > 0 )
    LUMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState());
	for(Int i = 0; i < eigSolPtr_->EigVal().m(); i++){
    Print(statusOFS, 
				"band#    = ", i, 
	      "eigval   = ", eigSolPtr_->EigVal()(i),
	      "resval   = ", eigSolPtr_->ResVal()(i),
	      "occrate  = ", eigSolPtr_->Ham().OccupationRate()(i));
	}
	statusOFS << std::endl;
	statusOFS 
		<< "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself + Evdw" << std::endl
	  << "       Etot  = Ekin + Ecor" << std::endl
	  << "       Efree = Etot	+ Entropy" << std::endl << std::endl;
	Print(statusOFS, "Etot              = ",  Etot_, "[au]");
	Print(statusOFS, "Efree             = ",  Efree_, "[au]");
	Print(statusOFS, "Ekin              = ",  Ekin_, "[au]");
	Print(statusOFS, "Ehart             = ",  Ehart_, "[au]");
	Print(statusOFS, "EVxc              = ",  EVxc_, "[au]");
	Print(statusOFS, "Exc               = ",  Exc_, "[au]"); 
	Print(statusOFS, "Evdw              = ",  Evdw_, "[au]"); 
	Print(statusOFS, "Eself             = ",  Eself_, "[au]");
	Print(statusOFS, "Ecor              = ",  Ecor_, "[au]");
	Print(statusOFS, "Fermi             = ",  fermi_, "[au]");
	Print(statusOFS, "Total charge      = ",  totalCharge_, "[au]");
	Print(statusOFS, "HOMO              = ",  HOMO*au2ev, "[eV]");
  if( eigSolPtr_->Ham().NumExtraState() > 0 ){
    Print(statusOFS, "LUMO              = ",  LUMO*au2ev, "[eV]");
  }
	Print(statusOFS, "norm(vout-vin)/norm(vin) = ", scfNorm_ );

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCF::PrintState  ----- 


void SCF::OutputState	(  )
{
#ifndef _RELEASE_
	PushCallStack("SCF::OutputState");
#endif
  if( isOutputDensity_ ){
		std::ofstream ofs(restartDensityFileName_.c_str());
		if( !ofs.good() ){
			throw std::logic_error( "Density file cannot be opened." );
		}
		serialize( eigSolPtr_->Ham().Density(), ofs, NO_MASK );
		ofs.close();
	}	


//  if( isOutputWfn_ ){
//		std::ofstream ofs(restartWfnFileName_.c_str());
//		if( !ofs.good() ){
//			throw std::logic_error( "Wavefunction file cannot be opened." );
//		}
//		serialize( eigSolPtr_->Psi().Wavefun(), ofs, NO_MASK );
//		ofs.close();
//	}	
#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCF::OutputState  ----- 


} // namespace dgdft
