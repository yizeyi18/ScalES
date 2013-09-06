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
/// @file hamiltonian.cpp
/// @brief Hamiltonian class for planewave basis diagonalization method.
/// @date 2012-09-16
#include  "hamiltonian.hpp"
#include  "blas.hpp"

namespace dgdft{

using namespace dgdft::PseudoComponent;
using namespace dgdft::DensityComponent;

// *********************************************************************
// Hamiltonian class (base class)
// *********************************************************************

Hamiltonian::Hamiltonian	( 
			const esdf::ESDFInputParam& esdfParam,
      const Int                   numDensityComponent )
{
#ifndef _RELEASE_
	PushCallStack("Hamiltonian::Hamiltonian");
#endif
	this->Setup( 
			esdfParam.domain,
			esdfParam.atomList,
			esdfParam.pseudoType,
			esdfParam.XCId,
			esdfParam.numExtraState,
			numDensityComponent );
#ifndef _RELEASE_
	PopCallStack();
#endif
	return ;
} 		// -----  end of method Hamiltonian::Hamiltonian  ----- 

void
Hamiltonian::Setup ( 
		const Domain&              dm,
		const std::vector<Atom>&   atomList,
		std::string                pseudoType,
		Int                        XCId,
		Int                        numExtraState,
    Int                        numDensityComponent )
{
#ifndef _RELEASE_
	PushCallStack("Hamiltonian::Setup");
#endif
	domain_        = dm;
	atomList_      = atomList;
	pseudoType_    = pseudoType;
	XCId_          = XCId;
	numExtraState_ = numExtraState;

	// NOTE: NumSpin variable will be determined in derivative classes.

  Int ntot = domain_.NumGridTotal();

	density_.Resize( ntot, numDensityComponent );   
	SetValue( density_, 0.0 );

	pseudoCharge_.Resize( ntot );
	SetValue( pseudoCharge_, 0.0 );
	
	vext_.Resize( ntot );
	SetValue( vext_, 0.0 );

	vhart_.Resize( ntot );
	SetValue( vhart_, 0.0 );

	vtot_.Resize( ntot );
	SetValue( vtot_, 0.0 );

	epsxc_.Resize( ntot );
	SetValue( epsxc_, 0.0 );

	vxc_.Resize( ntot, numDensityComponent );
	SetValue( vxc_, 0.0 );

#ifndef _RELEASE_
	PopCallStack();
#endif
	return ;
} 		// -----  end of method Hamiltonian::Setup  ----- 


// *********************************************************************
// KohnSham class
// *********************************************************************

KohnSham::KohnSham() {
	XCInitialized_ = false;
}

KohnSham::~KohnSham() {
	if( XCInitialized_ )
		xc_func_end(&XCFuncType_);
}

KohnSham::
	KohnSham( 
			const esdf::ESDFInputParam& esdfParam,
      const Int                   numDensityComponent ) : 
		Hamiltonian( esdfParam , numDensityComponent ) 
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::KohnSham");
#endif
	// Initialize the XC functional.  
	// Spin-unpolarized functional is used here
	if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
    throw std::runtime_error( "XC functional initialization error." );
	} 

	XCInitialized_ = true;

	if( numDensityComponent != 1 ){
		throw std::runtime_error( "KohnSham currently only supports numDensityComponent == 1." );
	}

	// Since the number of density components is always 1 here, set numSpin = 2.
	numSpin_ = 2;
#ifndef _RELEASE_
	PopCallStack();
#endif
}


void
KohnSham::Setup	(
		const Domain&              dm,
		const std::vector<Atom>&   atomList,
		std::string                pseudoType,
		Int                        XCId,
		Int                        numExtraState,
    Int                        numDensityComponent )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::Setup");
#endif
	Hamiltonian::Setup(
		dm,
		atomList,
		pseudoType,
		XCId,
		numExtraState,
    numDensityComponent);

	// Initialize the XC functional.  
	// Spin-unpolarized functional is used here
	if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
    throw std::runtime_error( "XC functional initialization error." );
	} 

	if( numDensityComponent != 1 ){
		throw std::runtime_error( "KohnSham currently only supports numDensityComponent == 1." );
	}

	// Since the number of density components is always 1 here, set numSpin = 2.
	numSpin_ = 2;
  	

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::Setup  ----- 

void
KohnSham::CalculatePseudoPotential	( PeriodTable &ptable ){
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculatePseudoPotential");
#endif
	Int ntot = domain_.NumGridTotal();
	Int numAtom = atomList_.size();
	Real vol = domain_.Volume();

	pseudo_.resize( numAtom );

	std::vector<DblNumVec> gridpos;
  UniformMesh ( domain_, gridpos );

  // calculate the number of occupied states
  Int nelec = 0;
  for (Int a=0; a<numAtom; a++) {
    Int atype  = atomList_[a].type;
		if( ptable.ptemap().find(atype) == ptable.ptemap().end() ){
			throw std::logic_error( "Cannot find the atom type." );
		}
    nelec = nelec + ptable.ptemap()[atype].params(PTParam::ZION);
  }
	// FIXME Deal with the case when this is a buffer calculation and the
	// number of electrons is not a even number.
	//
//	if( nelec % 2 != 0 ){
//		throw std::runtime_error( "This is spin-restricted calculation. nelec should be even." );
//	}
	numOccupiedState_ = nelec / numSpin_;

	// Compute pseudocharge
	SetValue( pseudoCharge_, 0.0 );
  for (Int a=0; a<numAtom; a++) {
    ptable.CalculatePseudoCharge( atomList_[a], domain_, 
				gridpos, pseudo_[a].pseudoCharge );
    //accumulate to the global vector
    IntNumVec &idx = pseudo_[a].pseudoCharge.first;
    DblNumMat &val = pseudo_[a].pseudoCharge.second;
    for (Int k=0; k<idx.m(); k++) 
			pseudoCharge_[idx(k)] += val(k, VAL);
  }

  Real sumrho = 0.0;
  for (Int i=0; i<ntot; i++) 
		sumrho += pseudoCharge_[i]; 
  sumrho *= vol / Real(ntot);

	Print( statusOFS, "Sum of Pseudocharge                          = ", 
			sumrho );
	Print( statusOFS, "Number of Occupied States                    = ", 
			numOccupiedState_ );
  
  Real diff = ( numSpin_ * numOccupiedState_ - sumrho ) / vol;
  for (Int i=0; i<ntot; i++) 
		pseudoCharge_(i) += diff; 

	Print( statusOFS, "After adjustment, Sum of Pseudocharge        = ", 
			numSpin_ * numOccupiedState_ );


	// Nonlocal projectors
	

  Int cnt = 0; // the total number of PS used
  for ( Int a=0; a < atomList_.size(); a++ ) {
		ptable.CalculateNonlocalPP( atomList_[a], domain_, gridpos,
				pseudo_[a].vnlList ); 
		cnt = cnt + pseudo_[a].vnlList.size();
  }

	Print( statusOFS, "Total number of nonlocal pseudopotential = ",  cnt );

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculatePseudoPotential ----- 



void
KohnSham::CalculateDensity ( const Spinor &psi, const DblNumVec &occrate, Real &val )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculateDensity");
#endif
	Int ntot  = psi.NumGridTotal();
	Int ncom  = psi.NumComponent();
	Int nocc  = psi.NumState();
	Real vol  = domain_.Volume();

	SetValue( density_, 0.0 );
	for (Int k=0; k<nocc; k++) {
		for (Int j=0; j<ncom; j++) {
			for (Int i=0; i<ntot; i++) {
				density_(i,RHO) += numSpin_ * occrate(k) * 
					pow( std::abs(psi.Wavefun(i,j,k)), 2.0 );
			}
		}
	}

	// Scale the density
	blas::Scal( ntot, ntot / vol, density_.VecData(RHO), 1 );
  val = 0.0; // sum of density
  for (Int i=0; i<ntot; i++) {
    val  += density_(i, RHO) * vol / ntot;
  }
	
#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculateDensity  ----- 


void
KohnSham::CalculateXC	( Real &val )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculateXC");
#endif
	Int ntot = domain_.NumGridTotal();
	Real vol = domain_.Volume();

  switch( XCFuncType_.info->family ){
    case XC_FAMILY_LDA:
       xc_lda_exc_vxc( &XCFuncType_, ntot, density_.VecData(RHO), 
		      epsxc_.Data(), vxc_.Data() );
      break;
    default:
			throw std::logic_error( "Unsupported XC family!" );
      break;
  }

  // Compute the total exchange-correlation energy
  val = 0.0;
  for(Int i = 0; i < ntot; i++){
    val += density_(i, RHO) * epsxc_(i) * vol / (Real) ntot;
  }


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculateXC  ----- 


void KohnSham::CalculateHartree( Fourier& fft ) {
#ifndef _RELEASE_ 
	PushCallStack("KohnSham::CalculateHartree");
#endif
	if( !fft.isInitialized ){
		throw std::runtime_error("Fourier is not prepared.");
	}
 	
  Int ntot = domain_.NumGridTotal();
	if( fft.domain.NumGridTotal() != ntot ){
		throw std::logic_error( "Grid size does not match!" );
	}

	// The contribution of the pseudoCharge is subtracted. So the Poisson
	// equation is well defined for neutral system.
	for( Int i = 0; i < ntot; i++ ){
		fft.inputComplexVec(i) = Complex( 
				density_(i,RHO) - pseudoCharge_(i), 0.0 );
	}
	fftw_execute( fft.forwardPlan );

	for( Int i = 0; i < ntot; i++ ){
		if( fft.gkk(i) == 0 ){
			fft.outputComplexVec(i) = Z_ZERO;
		}
		else{
			// NOTE: gkk already contains the factor 1/2.
			fft.outputComplexVec(i) *= 2.0 * PI / fft.gkk(i);
		}
	}
	fftw_execute( fft.backwardPlan );

	for( Int i = 0; i < ntot; i++ ){
		vhart_(i) = fft.inputComplexVec(i).real() / ntot;
	}

#ifndef _RELEASE_
	PopCallStack();
#endif
	return; 
}  // -----  end of method KohnSham::CalculateHartree ----- 


void
KohnSham::CalculateVtot	( DblNumVec& vtot  )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculateVtot");
#endif
	Int ntot = domain_.NumGridTotal();
  for (int i=0; i<ntot; i++) {
		vtot(i) = vext_(i) + vhart_(i) + vxc_(i, RHO);
  }

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculateVtot  ----- 


void
KohnSham::MultSpinor	( Spinor& psi, NumTns<Scalar>& a3, Fourier& fft )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::MultSpinor");
#endif
	SetValue( a3, SCALAR_ZERO );
	psi.AddScalarDiag( vtot_, a3 );
	psi.AddLaplacian( a3, &fft );
  psi.AddNonlocalPP( pseudo_, a3 );
#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::MultSpinor  ----- 



} // namespace dgdft
