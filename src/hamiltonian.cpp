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
			esdfParam.XCType,
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
		std::string                XCType,
		Int                        numExtraState,
    Int                        numDensityComponent )
{
#ifndef _RELEASE_
	PushCallStack("Hamiltonian::Setup");
#endif
	domain_        = dm;
	atomList_      = atomList;
	pseudoType_    = pseudoType;
	numExtraState_ = numExtraState;

	// Obtain the exchange-correlation id
  {
    if( XCType == "XC_LDA_XC_TETER93" )
    { XCId_ = XC_LDA_XC_TETER93;
      // Teter 93
      // S Goedecker, M Teter, J Hutter, Phys. Rev B 54, 1703 (1996) 
    }    
    else if( XCType == "XC_GGA_XC_PBE" )
    {
      XId_ = XC_GGA_X_PBE;
      CId_ = XC_GGA_C_PBE;
      // Perdew, Burke & Ernzerhof correlation
      // JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
      // JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 78, 1396(E) (1997)
    }
    else
      throw std::logic_error("Unrecognized exchange-correlation type");
  }

	// NOTE: NumSpin variable will be determined in derivative classes.

  Int ntotCoarse = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();

  density_.Resize( ntotFine, numDensityComponent );   
  SetValue( density_, 0.0 );

  gradDensity_.resize( DIM );
  for( Int d = 0; d < DIM; d++ ){
    gradDensity_[d].Resize( ntotFine, numDensityComponent );
    SetValue (gradDensity_[d], 0.0);
  }

  pseudoCharge_.Resize( ntotFine );
  SetValue( pseudoCharge_, 0.0 );

  vext_.Resize( ntotFine );
  SetValue( vext_, 0.0 );

  vhart_.Resize( ntotFine );
  SetValue( vhart_, 0.0 );

	vtot_.Resize( ntotFine );
	SetValue( vtot_, 0.0 );

  vtotCoarse_.Resize( ntotCoarse );
  SetValue( vtot_, 0.0 );

	epsxc_.Resize( ntotFine );
	SetValue( epsxc_, 0.0 );

	vxc_.Resize( ntotFine, numDensityComponent );
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
  if( XCInitialized_ ){
    if( XCId_ == 20 )
    {
      xc_func_end(&XCFuncType_);
    }    
    else if( ( XId_ == 101 ) && ( CId_ == 130 )  )
    {
      xc_func_end(&XFuncType_);
      xc_func_end(&CFuncType_);
    }
    else
      throw std::logic_error("Unrecognized exchange-correlation type");
  }
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
 
  if( XCId_ == 20 )
  {
    if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
      throw std::runtime_error( "XC functional initialization error." );
    } 
  }    
  else if( ( XId_ == 101 ) && ( CId_ == 130 )  )
  {
    if( ( xc_func_init(&XFuncType_, XId_, XC_UNPOLARIZED) != 0 )
        && ( xc_func_init(&CFuncType_, CId_, XC_UNPOLARIZED) != 0 ) ){
      throw std::runtime_error( "XC functional initialization error." );
    }
  }
  else
    throw std::logic_error("Unrecognized exchange-correlation type");

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
		std::string                XCType,
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
		XCType,
		numExtraState,
    numDensityComponent);

  // Initialize the XC functional.  
  // Spin-unpolarized functional is used here
 
  xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED);
  xc_func_init(&XFuncType_, XId_, XC_UNPOLARIZED);
  xc_func_init(&CFuncType_, CId_, XC_UNPOLARIZED);
  
  if( XCType == "XC_LDA_XC_TETER93" )
  {
    if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
      throw std::runtime_error( "XC functional initialization error." );
    } 
  }    
  else if( XCType == "XC_GGA_XC_PBE" )
  {
    if( ( xc_func_init(&XFuncType_, XId_, XC_UNPOLARIZED) != 0 )
        && ( xc_func_init(&CFuncType_, CId_, XC_UNPOLARIZED) != 0 ) ){
      throw std::runtime_error( "XC functional initialization error." );
    }
  }
  else
    throw std::logic_error("Unrecognized exchange-correlation type");

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
KohnSham::Update	( std::vector<Atom>&   atomList )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::Update");
#endif	
  atomList_ = atomList;
  
#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::Update  ----- 


void
KohnSham::CalculatePseudoPotential	( PeriodTable &ptable ){
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculatePseudoPotential");
#endif
	Int ntotFine = domain_.NumGridTotalFine();
	Int numAtom = atomList_.size();
	Real vol = domain_.Volume();

  pseudo_.clear();
  pseudo_.resize( numAtom );

	std::vector<DblNumVec> gridpos;
  UniformMeshFine ( domain_, gridpos );

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
  for (Int i=0; i<ntotFine; i++) 
		sumrho += pseudoCharge_[i]; 
  sumrho *= vol / Real(ntotFine);

	Print( statusOFS, "Sum of Pseudocharge                          = ", 
			sumrho );
	Print( statusOFS, "Number of Occupied States                    = ", 
			numOccupiedState_ );
  
  Real diff = ( numSpin_ * numOccupiedState_ - sumrho ) / vol;
  for (Int i=0; i<ntotFine; i++) 
		pseudoCharge_(i) += diff; 

	Print( statusOFS, "After adjustment, Sum of Pseudocharge        = ", 
			numSpin_ * numOccupiedState_ );


	// Nonlocal projectors
	
	std::vector<DblNumVec> gridposCoarse;
  UniformMesh ( domain_, gridposCoarse );

  Int cnt = 0; // the total number of PS used
  for ( Int a=0; a < atomList_.size(); a++ ) {
		ptable.CalculateNonlocalPP( atomList_[a], domain_, gridposCoarse,
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
KohnSham::CalculateDensity ( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier &fft)
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculateDensity");
#endif
	Int ntot  = psi.NumGridTotal();
	Int ncom  = psi.NumComponent();
	Int nocc  = psi.NumState();
	Real vol  = domain_.Volume();

  Int ntotFine  = fft.domain.NumGridTotalFine();

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

//  IntNumVec& wavefunIdx = psi.WavefunIdx();

  DblNumMat   densityLocal;
	densityLocal.Resize( ntotFine, ncom );   
	SetValue( densityLocal, 0.0 );

	SetValue( density_, 0.0 );
  for (Int k=0; k<nocc; k++) {
		for (Int j=0; j<ncom; j++) {

      for( Int i = 0; i < ntot; i++ ){
        fft.inputComplexVec(i) = Complex( psi.Wavefun(i,j,k), 0.0 ); 
      }
      fftw_execute( fft.forwardPlan );
 
      // fft Coarse to Fine 

      Int PtrC = 0;
      Int PtrF = 0;

      Int iF = 0;
      Int jF = 0;
      Int kF = 0;

      SetValue( fft.outputComplexVecFine, Z_ZERO );

      for( Int kk = 0; kk < fft.domain.numGrid[2]; kk++ ){
        for( Int jj = 0; jj < fft.domain.numGrid[1]; jj++ ){
          for( Int ii = 0; ii < fft.domain.numGrid[0]; ii++ ){

            PtrC = ii + jj * fft.domain.numGrid[0] + kk * fft.domain.numGrid[0] * fft.domain.numGrid[1];

            if ( (0 <= ii) && (ii <= fft.domain.numGrid[0] / 2) ) { iF = ii; } 
            else { iF = fft.domain.numGridFine[0] - fft.domain.numGrid[0] + ii; } 

            if ( (0 <= jj) && (jj <= fft.domain.numGrid[1] / 2) ) { jF = jj; } 
            else { jF = fft.domain.numGridFine[1] - fft.domain.numGrid[1] + jj; } 

            if ( (0 <= kk) && (kk <= fft.domain.numGrid[2] / 2) ) { kF = kk; } 
            else { kF = fft.domain.numGridFine[2] - fft.domain.numGrid[2] + kk; } 

            PtrF = iF + jF * fft.domain.numGridFine[0] + kF * fft.domain.numGridFine[0] * fft.domain.numGridFine[1];

            fft.outputComplexVecFine(PtrF) = fft.outputComplexVec(PtrC);

          } 
        }
      }

      // end fft Coarse to Fine
    
//    	for( Int i = 0; i < ntotFine; i++ ){
//		    if( fft.gkkFine(i) == 0 ){
//			    fft.outputComplexVecFine(i) = Z_ZERO; }
//	  	  else{
//			    fft.outputComplexVecFine(i) *= 2.0 * PI / fft.gkkFine(i);
//	   	  }
//     	}
      fftw_execute( fft.backwardPlanFine );

      for( Int i = 0; i < ntotFine; i++ ){
				densityLocal(i,RHO) += numSpin_ * occrate(psi.WavefunIdx(k)) * pow( std::abs(fft.inputComplexVecFine(i).real() / ntot), 2.0 );
      }
		}
	}

//	SetValue( density_, 0.0 );
//	for (Int k=0; k<nocc; k++) {
//		for (Int j=0; j<ncom; j++) {
//			for (Int i=0; i<ntot; i++) {
//				density_(i,RHO) += numSpin_ * occrate(k) * 
//					pow( std::abs(psi.Wavefun(i,j,k)), 2.0 );
//			}
//		}
//	}

	// Scale the density
//	blas::Scal( ntotFine, ntotFine / vol, density_.VecData(RHO), 1 );
//  val = 0.0; // sum of density
//  for (Int i=0; i<ntotFine; i++) {
//    val  += density_(i, RHO) * vol / ntotFine;
//  }

	mpi::Allreduce( densityLocal.Data(), density_.Data(), ntotFine, MPI_SUM, domain_.comm );

  val = 0.0; // sum of density
  for (Int i=0; i<ntotFine; i++) {
    val  += density_(i, RHO);
  }

  // Scale the density
  blas::Scal( ntotFine, (numSpin_ * numOccupiedState_ * ntotFine) / ( vol * val ), density_.VecData(RHO), 1 );

  // Double check (can be neglected)
  val = 0.0; // sum of density
  for (Int i=0; i<ntotFine; i++) {
    val  += density_(i, RHO) * vol / ntotFine;
  }
  

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculateDensity  ----- 


void
KohnSham::CalculateGradDensity ( Fourier& fft )
{
#ifndef _RELEASE_
  PushCallStack("KohnSham::CalculateGradDensity");
#endif
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Real vol  = domain_.Volume();

  for( Int i = 0; i < ntotFine; i++ ){
    fft.inputComplexVecFine(i) = Complex( density_(i,RHO), 0.0 ); 
  }
  fftw_execute( fft.forwardPlanFine );

  CpxNumVec  cpxVec( ntotFine );
  blas::Copy( ntotFine, fft.outputComplexVecFine.Data(), 1,
      cpxVec.Data(), 1 );

  // Compute the derivative of the Density via Fourier
  for( Int d = 0; d < DIM; d++ ){
    CpxNumVec& ik = fft.ikFine[d];

    for( Int i = 0; i < ntotFine; i++ ){
      if( fft.gkkFine(i) == 0 ){
        fft.outputComplexVecFine(i) = Z_ZERO;
      }
      else{
        fft.outputComplexVecFine(i) = cpxVec(i) * ik(i); 
      }
    }

    fftw_execute( fft.backwardPlanFine );

    DblNumMat& gradDensity = gradDensity_[d];
    for( Int i = 0; i < ntotFine; i++ ){
      gradDensity(i, RHO) = fft.inputComplexVecFine(i).real() / ntotFine;
    }
  } // for d

#ifndef _RELEASE_
  PopCallStack();
#endif

  return ;
} 		// -----  end of method KohnSham::CalculateGradDensity  ----- 


void
KohnSham::CalculateXC	( Real &val, Fourier& fft )
{
#ifndef _RELEASE_
  PushCallStack("KohnSham::CalculateXC");
#endif
  Int ntot = domain_.NumGridTotalFine();
  Int numDensityComponent = vxc_.n();
  Real vol = domain_.Volume();

  if( XCId_ == 20 ) //XC_FAMILY_LDA
  {
    xc_lda_exc_vxc( &XCFuncType_, ntot, density_.VecData(RHO), 
        epsxc_.Data(), vxc_.Data() );
  }
  else if( ( XId_ == 101 ) && ( CId_ == 130 ) ) //XC_FAMILY_GGA
  {
    DblNumMat     vxc1;             
    DblNumMat     vxc2;             
    vxc1.Resize( ntot, numDensityComponent );
    vxc2.Resize( ntot, numDensityComponent );

    DblNumMat     vxc1temp;             
    DblNumMat     vxc2temp;             
    vxc1temp.Resize( ntot, numDensityComponent );
    vxc2temp.Resize( ntot, numDensityComponent );

    DblNumVec     epsx; 
    DblNumVec     epsc; 
    epsx.Resize( ntot );
    epsc.Resize( ntot );

    DblNumMat gradDensity;
    gradDensity.Resize( ntot, numDensityComponent );
    SetValue( gradDensity, 0.0 );
    DblNumMat& gradDensity0 = gradDensity_[0];
    DblNumMat& gradDensity1 = gradDensity_[1];
    DblNumMat& gradDensity2 = gradDensity_[2];

    for(Int i = 0; i < ntot; i++){
      gradDensity(i, RHO) = gradDensity0(i, RHO) * gradDensity0(i, RHO)
        + gradDensity1(i, RHO) * gradDensity1(i, RHO)
        + gradDensity2(i, RHO) * gradDensity2(i, RHO);
    }

    SetValue( epsx, 0.0 );
    SetValue( vxc1, 0.0 );
    SetValue( vxc2, 0.0 );
    xc_gga_exc_vxc( &XFuncType_, ntot, density_.VecData(RHO), 
        gradDensity.VecData(RHO), epsx.Data(), vxc1.Data(), vxc2.Data() );

    SetValue( epsc, 0.0 );
    SetValue( vxc1temp, 0.0 );
    SetValue( vxc2temp, 0.0 );
    xc_gga_exc_vxc( &CFuncType_, ntot, density_.VecData(RHO), 
        gradDensity.VecData(RHO), epsc.Data(), vxc1temp.Data(), vxc2temp.Data() );

    for( Int i = 0; i < ntot; i++ ){
      epsxc_(i) = epsx(i) + epsc(i) ;
      vxc1( i, RHO ) += vxc1temp( i, RHO );
      vxc2( i, RHO ) += vxc2temp( i, RHO );
      vxc_( i, RHO ) = vxc1( i, RHO );
    }

    for( Int d = 0; d < DIM; d++ ){

      DblNumMat& gradDensityd = gradDensity_[d];

      for(Int i = 0; i < ntot; i++){
        fft.inputComplexVecFine(i) = Complex( gradDensityd( i, RHO ) * 2.0 * vxc2( i, RHO ), 0.0 ); 
      }

      fftw_execute( fft.forwardPlanFine );

      CpxNumVec& ik = fft.ikFine[d];

      for( Int i = 0; i < ntot; i++ ){
        if( fft.gkkFine(i) == 0 ){
          fft.outputComplexVecFine(i) = Z_ZERO;
        }
        else{
          fft.outputComplexVecFine(i) *= ik(i);
        }
      }

      fftw_execute( fft.backwardPlanFine );

      for( Int i = 0; i < ntot; i++ ){
        vxc_( i, RHO ) -= fft.inputComplexVecFine(i).real() / ntot;
      }

    } // for d

  } // XC_FAMILY_GGA
  else
    throw std::logic_error( "Unsupported XC family!" );

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
 	
  Int ntot = domain_.NumGridTotalFine();
	if( fft.domain.NumGridTotalFine() != ntot ){
		throw std::logic_error( "Grid size does not match!" );
	}

	// The contribution of the pseudoCharge is subtracted. So the Poisson
	// equation is well defined for neutral system.
	for( Int i = 0; i < ntot; i++ ){
		fft.inputComplexVecFine(i) = Complex( 
				density_(i,RHO) - pseudoCharge_(i), 0.0 );
	}
	fftw_execute( fft.forwardPlanFine );

	for( Int i = 0; i < ntot; i++ ){
		if( fft.gkkFine(i) == 0 ){
			fft.outputComplexVecFine(i) = Z_ZERO;
		}
		else{
			// NOTE: gkk already contains the factor 1/2.
			fft.outputComplexVecFine(i) *= 2.0 * PI / fft.gkkFine(i);
		}
	}
	fftw_execute( fft.backwardPlanFine );

	for( Int i = 0; i < ntot; i++ ){
		vhart_(i) = fft.inputComplexVecFine(i).real() / ntot;
	}

#ifndef _RELEASE_
	PopCallStack();
#endif
	return; 
}  // -----  end of method KohnSham::CalculateHartree ----- 


void
KohnSham::CalculateVtot	( DblNumVec& vtot )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculateVtot");
#endif
	Int ntot = domain_.NumGridTotalFine();
  for (int i=0; i<ntot; i++) {
		vtot(i) = vext_(i) + vhart_(i) + vxc_(i, RHO);
  }

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculateVtot  ----- 


void
KohnSham::CalculateForce	( Spinor& psi, Fourier& fft  )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculateForce");
#endif

//  Int ntot      = fft.numGridTotal;
  Int ntot      = fft.domain.NumGridTotalFine();
  Int numAtom   = atomList_.size();

	DblNumMat  force( numAtom, DIM );
	SetValue( force, 0.0 );
	DblNumMat  forceLocal( numAtom, DIM );
	SetValue( forceLocal, 0.0 );

	// *********************************************************************
	// Compute the derivative of the Hartree potential for computing the 
	// local pseudopotential contribution to the Hellmann-Feynman force
	// *********************************************************************
	DblNumVec               vhart;
	std::vector<DblNumVec>  vhartDrv(DIM);

	DblNumVec  tempVec(ntot);
	SetValue( tempVec, 0.0 );

	// tempVec = density_ - pseudoCharge_
	// FIXME No density
	blas::Copy( ntot, density_.VecData(0), 1, tempVec.Data(), 1 );
	blas::Axpy( ntot, -1.0, pseudoCharge_.Data(),1,
			tempVec.Data(), 1 );
	
	// cpxVec saves the Fourier transform of 
	// density_ - pseudoCharge_ 
	CpxNumVec  cpxVec( tempVec.Size() );

	for( Int i = 0; i < ntot; i++ ){
		fft.inputComplexVecFine(i) = Complex( 
				tempVec(i), 0.0 );
	}

	fftw_execute( fft.forwardPlanFine );

	blas::Copy( ntot, fft.outputComplexVecFine.Data(), 1,
			cpxVec.Data(), 1 );

	// Compute the derivative of the Hartree potential via Fourier
	// transform 
	{
		for( Int i = 0; i < ntot; i++ ){
			if( fft.gkkFine(i) == 0 ){
				fft.outputComplexVecFine(i) = Z_ZERO;
			}
			else{
				// NOTE: gkk already contains the factor 1/2.
				fft.outputComplexVecFine(i) = cpxVec(i) *
					2.0 * PI / fft.gkkFine(i);
			}
		}

		fftw_execute( fft.backwardPlanFine );

		vhart.Resize( ntot );

		for( Int i = 0; i < ntot; i++ ){
			vhart(i) = fft.inputComplexVecFine(i).real() / ntot;
		}
	}
	
	for( Int d = 0; d < DIM; d++ ){
		CpxNumVec& ik = fft.ikFine[d];
		for( Int i = 0; i < ntot; i++ ){
			if( fft.gkkFine(i) == 0 ){
				fft.outputComplexVecFine(i) = Z_ZERO;
			}
			else{
				// NOTE: gkk already contains the factor 1/2.
				fft.outputComplexVecFine(i) = cpxVec(i) *
					2.0 * PI / fft.gkkFine(i) * ik(i);
			}
		}

		fftw_execute( fft.backwardPlanFine );

		// vhartDrv saves the derivative of the Hartree potential
		vhartDrv[d].Resize( ntot );

		for( Int i = 0; i < ntot; i++ ){
			vhartDrv[d](i) = fft.inputComplexVecFine(i).real() / ntot;
		}

	} // for (d)


	// *********************************************************************
	// Compute the force from local pseudopotential
	// *********************************************************************
	// Method 1: Using the derivative of the pseudopotential
	if(1){
		for (Int a=0; a<numAtom; a++) {
			PseudoPot& pp = pseudo_[a];
			SparseVec& sp = pp.pseudoCharge;
			IntNumVec& idx = sp.first;
			DblNumMat& val = sp.second;

			Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
			Real resX = 0.0;
			Real resY = 0.0;
			Real resZ = 0.0;
			for( Int l = 0; l < idx.m(); l++ ){
				resX -= val(l, DX) * vhart[idx(l)] * wgt;
				resY -= val(l, DY) * vhart[idx(l)] * wgt;
				resZ -= val(l, DZ) * vhart[idx(l)] * wgt;
			}
			force( a, 0 ) += resX;
			force( a, 1 ) += resY;
			force( a, 2 ) += resZ;

		} // for (a)
	}


	// *********************************************************************
	// Compute the force from nonlocal pseudopotential
	// *********************************************************************
	// Method 1: Using the derivative of the pseudopotential
	if(1)
	{
		// Loop over atoms and pseudopotentials
		Int numEig = occupationRate_.m();
    Int numStateTotal = psi.NumStateTotal();
    Int numStateLocal = psi.NumState();

    MPI_Barrier(domain_.comm);
    int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
    int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

    if( numEig != numStateTotal ){
      throw std::runtime_error( "numEig != numStateTotal in CalculateForce" );
    }

		for( Int a = 0; a < numAtom; a++ ){
			std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlList;
			for( Int l = 0; l < vnlList.size(); l++ ){
				SparseVec& bl = vnlList[l].first;
				Real  gamma   = vnlList[l].second;
        // FIXME Change to caorse
				Real wgt = domain_.Volume() / domain_.NumGridTotal();
				IntNumVec& idx = bl.first;
				DblNumMat& val = bl.second;

				for( Int g = 0; g < numStateLocal; g++ ){
					DblNumVec res(4);
					SetValue( res, 0.0 );
					Real* psiPtr = psi.Wavefun().VecData(0, g);
					for( Int i = 0; i < idx.Size(); i++ ){
						res(VAL) += val(i, VAL ) * psiPtr[ idx(i) ] * sqrt(wgt);
						res(DX) += val(i, DX ) * psiPtr[ idx(i) ] * sqrt(wgt);
						res(DY) += val(i, DY ) * psiPtr[ idx(i) ] * sqrt(wgt);
						res(DZ) += val(i, DZ ) * psiPtr[ idx(i) ] * sqrt(wgt);
					}

					// forceLocal( a, 0 ) += 4.0 * occupationRate_( g + mpirank * psi.Blocksize() ) * gamma * res[VAL] * res[DX];
					// forceLocal( a, 1 ) += 4.0 * occupationRate_( g + mpirank * psi.Blocksize() ) * gamma * res[VAL] * res[DY];
					// forceLocal( a, 2 ) += 4.0 * occupationRate_( g + mpirank * psi.Blocksize() ) * gamma * res[VAL] * res[DZ];
				
					forceLocal( a, 0 ) += 4.0 * occupationRate_( psi.WavefunIdx(g) ) * gamma * res[VAL] * res[DX];
					forceLocal( a, 1 ) += 4.0 * occupationRate_( psi.WavefunIdx(g) ) * gamma * res[VAL] * res[DY];
					forceLocal( a, 2 ) += 4.0 * occupationRate_( psi.WavefunIdx(g) ) * gamma * res[VAL] * res[DZ];
       
        } // for (g)
			} // for (l)
		} // for (a)
	}

  DblNumMat  forceTmp( numAtom, DIM );
	SetValue( forceTmp, 0.0 );
      
  mpi::Allreduce( forceLocal.Data(), forceTmp.Data(), numAtom * DIM, MPI_SUM, domain_.comm );

  for( Int a = 0; a < numAtom; a++ ){
    force( a, 0 ) = force( a, 0 ) + forceTmp( a, 0 );
    force( a, 1 ) = force( a, 1 ) + forceTmp( a, 1 );
    force( a, 2 ) = force( a, 2 ) + forceTmp( a, 2 );
  }


	// Method 2: Using integration by parts, and throw the derivative to the wavefunctions
  // FIXME: Assumign real arithmetic is used here.
	if(0)
	{
    // Compute the derivative of the wavefunctions

    Fourier* fftPtr = &(fft);

    Int ntothalf = fftPtr->numGridTotalR2C;
    Int ntot  = psi.NumGridTotal();
    Int ncom  = psi.NumComponent();
    Int nocc  = psi.NumState();

    DblNumVec realInVec(ntot);
		CpxNumVec cpxSaveVec(ntothalf);
		CpxNumVec cpxOutVec(ntothalf);

    std::vector<DblNumTns>   psiDrv(DIM);
    for( Int d = 0; d < DIM; d++ ){
      psiDrv[d].Resize( ntot, ncom, nocc );
      SetValue( psiDrv[d], 0.0 );
    }

    for (Int k=0; k<nocc; k++) {
      for (Int j=0; j<ncom; j++) {
        // For c2r and r2c transforms, the default is to DESTROY the
        // input, therefore a copy of the original matrix is necessary. 
        blas::Copy( ntot, psi.Wavefun().VecData(j, k), 1, 
            realInVec.Data(), 1 );
        fftw_execute_dft_r2c(
            fftPtr->forwardPlanR2C, 
            realInVec.Data(),
            reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ));

        cpxSaveVec = cpxOutVec;

        for( Int d = 0; d < DIM; d++ ){
          Complex* ptr1   = fftPtr->ikR2C[d].Data();
          Complex* ptr2   = cpxSaveVec.Data();
          Complex* ptr3   = cpxOutVec.Data();
          for (Int i=0; i<ntothalf; i++) {
            *(ptr3++) = (*(ptr1++)) * (*(ptr2++));
          }

          fftw_execute_dft_c2r(
              fftPtr->backwardPlanR2C,
              reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ),
              realInVec.Data() );

          blas::Axpy( ntot, 1.0 / Real(ntot), realInVec.Data(), 1, 
              psiDrv[d].VecData(j, k), 1 );
        }
      }
    }

    // Loop over atoms and pseudopotentials
    Int numEig = occupationRate_.m();

    for( Int a = 0; a < numAtom; a++ ){
      std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlList;
      for( Int l = 0; l < vnlList.size(); l++ ){
        SparseVec& bl = vnlList[l].first;
        Real  gamma   = vnlList[l].second;
        Real wgt = domain_.Volume() / domain_.NumGridTotal();
        IntNumVec& idx = bl.first;
        DblNumMat& val = bl.second;

        for( Int g = 0; g < numEig; g++ ){
          DblNumVec res(4);
          SetValue( res, 0.0 );
          Real* psiPtr = psi.Wavefun().VecData(0, g);
          Real* DpsiXPtr = psiDrv[0].VecData(0, g);
          Real* DpsiYPtr = psiDrv[1].VecData(0, g);
          Real* DpsiZPtr = psiDrv[2].VecData(0, g);
          for( Int i = 0; i < idx.Size(); i++ ){
            res(VAL) += val(i, VAL ) * psiPtr[ idx(i) ] * sqrt(wgt);
            res(DX)  += val(i, VAL ) * DpsiXPtr[ idx(i) ] * sqrt(wgt);
            res(DY)  += val(i, VAL ) * DpsiYPtr[ idx(i) ] * sqrt(wgt);
            res(DZ)  += val(i, VAL ) * DpsiZPtr[ idx(i) ] * sqrt(wgt);
          }

          force( a, 0 ) += -4.0 * occupationRate_(g) * gamma * res[VAL] * res[DX];
          force( a, 1 ) += -4.0 * occupationRate_(g) * gamma * res[VAL] * res[DY];
          force( a, 2 ) += -4.0 * occupationRate_(g) * gamma * res[VAL] * res[DZ];
        } // for (g)
      } // for (l)
    } // for (a)
	}

	// *********************************************************************
	// Compute the total force and give the value to atomList
	// *********************************************************************

	for( Int a = 0; a < numAtom; a++ ){
		atomList_[a].force = Point3( force(a,0), force(a,1), force(a,2) );
	} 


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculateForce  ----- 

void
KohnSham::MultSpinor	( Spinor& psi, NumTns<Scalar>& a3, Fourier& fft )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::MultSpinor");
#endif
  SetValue( a3, SCALAR_ZERO );

#ifdef _USE_OPENMP_
#pragma omp parallel
  {
#endif
    // FIXME
    psi.AddScalarDiag( vtotCoarse_, a3 );
    psi.AddLaplacian( &fft, a3 );
    psi.AddNonlocalPP( pseudo_, a3 );
#ifdef _USE_OPENMP_
  }
#endif

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::MultSpinor  ----- 

void
KohnSham::MultSpinor	( Int iocc, Spinor& psi, NumMat<Scalar>& y, Fourier& fft )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::MultSpinor");
#endif
  // Make sure that the address corresponding to the pointer y has been
  // allocated.
  SetValue( y, SCALAR_ZERO );

	psi.AddScalarDiag( iocc, vtotCoarse_, y );
	psi.AddLaplacian( iocc, &fft, y );
  psi.AddNonlocalPP( iocc, pseudo_, y );

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::MultSpinor  ----- 



} // namespace dgdft
