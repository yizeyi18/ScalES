
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
    eigMaxIter_    = esdfParam.eigMaxIter;
		scfTolerance_  = esdfParam.scfOuterTolerance;
		scfMaxIter_    = esdfParam.scfOuterMaxIter;
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
SCF::Iterate	(  )
{
#ifndef _RELEASE_
	PushCallStack("SCF::Iterate");
#endif
	Real timeSta, timeEnd;

#ifndef _RELEASE_
	PushCallStack("SCF::Iterate::Initialize");
#endif

  // Compute the exchange-correlation potential and energy
	if( XCType_ == "XC_GGA_XC_PBE"){
		eigSolPtr_->Ham().CalculateGradDensity( eigSolPtr_->FFT() );
	}
	eigSolPtr_->Ham().CalculateXC( Exc_, eigSolPtr_->FFT() ); 

	// Compute the Hartree energy
	eigSolPtr_->Ham().CalculateHartree( eigSolPtr_->FFT() );
	// No external potential

	// Compute the total potential
	eigSolPtr_->Ham().CalculateVtot( eigSolPtr_->Ham().Vtot() );

  Real timeIterStart(0), timeIterEnd(0);
  
	bool isSCFConverged = false;

#ifndef _RELEASE_
	PopCallStack();
#endif

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



    Int ntotCoarse  = eigSolPtr_->FFT().domain.NumGridTotal();
    Int ntotFine  = eigSolPtr_->FFT().domain.NumGridTotalFine();

    DblNumVec& vtotFine = eigSolPtr_->Ham().Vtot();
    DblNumVec& vtotCoarse = eigSolPtr_->Ham().VtotCoarse();

    Fourier& fft =  eigSolPtr_->FFT();

    for( Int i = 0; i < ntotFine; i++ ){
      fft.inputComplexVecFine(i) = Complex( vtotFine(i), 0.0 ); 
    }

    fftw_execute( fft.forwardPlanFine );


    // vtotCoarse is no longer needed. To be removed in the next version.

    // fft Fine to Coarse 

//    Int PtrC = 0;
//    Int PtrF = 0;
//
//    Int iF = 0;
//    Int jF = 0;
//    Int kF = 0;
//
//    SetValue( fft.outputComplexVec, Z_ZERO );
//
//    for( Int kk = 0; kk < fft.domain.numGrid[2]; kk++ ){
//      for( Int jj = 0; jj <  fft.domain.numGrid[1]; jj++ ){
//        for( Int ii = 0; ii <  fft.domain.numGrid[0]; ii++ ){
//
//          PtrC = ii + jj * fft.domain.numGrid[0] + kk * fft.domain.numGrid[0] * fft.domain.numGrid[1];
//
//          if ( (0 <= ii) && (ii <=  fft.domain.numGrid[0] / 2) ) { iF = ii; } 
//          else {iF =  fft.domain.numGridFine[0] - fft.domain.numGrid[0] + ii; } 
//
//          if ( (0 <= jj) && (jj <=  fft.domain.numGrid[1] / 2) ) { jF = jj; }
//          else { jF =  fft.domain.numGridFine[1] - fft.domain.numGrid[1] + jj; }
//
//          if ( (0 <= kk) && (kk <=  fft.domain.numGrid[2] / 2) ) { kF = kk; }
//          else { kF =  fft.domain.numGridFine[2] - fft.domain.numGrid[2] + kk; }
//
//          PtrF = iF + jF *  fft.domain.numGridFine[0] + kF *  fft.domain.numGridFine[0] *  fft.domain.numGridFine[1];
//
//          fft.outputComplexVec(PtrC) = fft.outputComplexVecFine(PtrF);
//
//        }
//      }
//    }
//
////    for( Int i = 0; i < ntotCoarse; i++ ){
////      if( fft.gkk(i) == 0 ){
////        fft.outputComplexVec(i) = Z_ZERO; 
////      }
////      else{
////       fft.outputComplexVec(i) *= 2.0 * PI / fft.gkk(i);
////       }
////    }
//    fftw_execute( fft.backwardPlan );
//
//    for( Int i = 0; i < ntotCoarse; i++ ){
//      vtotCoarse(i) = fft.inputComplexVec(i).real() / ntotFine;
//    }

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

    Int numEig = (eigSolPtr_->Psi().NumStateTotal())-numUnusedState_;

    statusOFS << "The current tolerance used by the eigensolver is " 
      << eigTolNow << std::endl;
    statusOFS << "The target number of converged eigenvectors is " 
      << numEig << std::endl;
   
    
    GetTime( timeSta );
    if(0){
      eigSolPtr_->Solve();
    }
  
    if(0){
      eigSolPtr_->LOBPCGSolveReal(numEig, eigMaxIter_, eigTolNow );
    }

    if(1){
      eigSolPtr_->LOBPCGSolveReal2(numEig, eigMaxIter_, eigTolNow );
    }
    GetTime( timeEnd );

    eigSolPtr_->Ham().EigVal() = eigSolPtr_->EigVal();

    statusOFS << "Time for the eigensolver is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;


		// No need for normalization using LOBPCG

		// Compute the occupation rate
		CalculateOccupationRate( eigSolPtr_->Ham().EigVal(), 
				eigSolPtr_->Ham().OccupationRate() );

		// Compute the electron density
		eigSolPtr_->Ham().CalculateDensity(
				eigSolPtr_->Psi(),
				eigSolPtr_->Ham().OccupationRate(),
        totalCharge_, 
        eigSolPtr_->FFT() );

		// Compute the exchange-correlation potential and energy
    if( XCType_ == "XC_GGA_XC_PBE"){
      eigSolPtr_->Ham().CalculateGradDensity( eigSolPtr_->FFT() );
    }
    eigSolPtr_->Ham().CalculateXC( Exc_, eigSolPtr_->FFT() ); 
		
		// Compute the Hartree energy
		eigSolPtr_->Ham().CalculateHartree( eigSolPtr_->FFT() );
		// No external potential
		
		// Compute the total potential
		eigSolPtr_->Ham().CalculateVtot( vtotNew_ );

		Real normVtotDif = 0.0, normVtotOld;
		DblNumVec& vtotOld_ = eigSolPtr_->Ham().Vtot();
		Int ntot = vtotOld_.m();
		for( Int i = 0; i < ntot; i++ ){
			normVtotDif += pow( vtotOld_(i) - vtotNew_(i), 2.0 );
			normVtotOld += pow( vtotOld_(i), 2.0 );
		}
		normVtotDif = sqrt( normVtotDif );
		normVtotOld = sqrt( normVtotOld );
		scfNorm_    = normVtotDif / normVtotOld;
		
    CalculateEnergy();

    PrintState( iter );

    if( isCalculateForceEachSCF_ ){
      GetTime( timeSta );
      eigSolPtr_->Ham().CalculateForce2( eigSolPtr_->Psi(), eigSolPtr_->FFT() );
      GetTime( timeEnd );
      statusOFS << "Time for computing the force is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

      // Print out the force
      PrintBlock( statusOFS, "Atomic Force" );
      {
        Point3 forceCM(0.0, 0.0, 0.0);
        std::vector<Atom>& atomList = eigSolPtr_->Ham().AtomList();
        Int numAtom = atomList.size();
        for( Int a = 0; a < numAtom; a++ ){
          Print( statusOFS, "atom", a, "force", atomList[a].force );
          forceCM += atomList[a].force;
        }
        statusOFS << std::endl;
        Print( statusOFS, "force for centroid: ", forceCM );
        statusOFS << std::endl;
      }
    }

    if( scfNorm_ < scfTolerance_ ){
      /* converged */
      Print( statusOFS, "SCF is converged!\n" );
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
	Ecor_   = (Exc_ - EVxc_) - Ehart_ - Eself_;

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
	for(Int i = 0; i < eigSolPtr_->EigVal().m(); i++){
    Print(statusOFS, 
				"band#    = ", i, 
	      "eigval   = ", eigSolPtr_->EigVal()(i),
	      "resval   = ", eigSolPtr_->ResVal()(i),
	      "occrate  = ", eigSolPtr_->Ham().OccupationRate()(i));
	}
	statusOFS << std::endl;
	statusOFS 
		<< "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself" << std::endl
	  << "       Etot  = Ekin + Ecor" << std::endl
	  << "       Efree = Etot	+ Entropy" << std::endl << std::endl;
	Print(statusOFS, "Etot              = ",  Etot_, "[au]");
	Print(statusOFS, "Efree             = ",  Efree_, "[au]");
	Print(statusOFS, "Ekin              = ",  Ekin_, "[au]");
	Print(statusOFS, "Ehart             = ",  Ehart_, "[au]");
	Print(statusOFS, "EVxc              = ",  EVxc_, "[au]");
	Print(statusOFS, "Exc               = ",  Exc_, "[au]"); 
	Print(statusOFS, "Eself             = ",  Eself_, "[au]");
	Print(statusOFS, "Ecor              = ",  Ecor_, "[au]");
	Print(statusOFS, "Fermi             = ",  fermi_, "[au]");
	Print(statusOFS, "Total charge      = ",  totalCharge_, "[au]");
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
