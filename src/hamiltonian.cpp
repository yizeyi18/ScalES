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
#include  "lapack.hpp"

namespace dgdft{

using namespace dgdft::PseudoComponent;
using namespace dgdft::DensityComponent;


// *********************************************************************
// KohnSham class
// *********************************************************************

KohnSham::KohnSham() {
	XCInitialized_ = false;
}

KohnSham::~KohnSham() {
  if( XCInitialized_ ){
    if( XCId_ == XC_LDA_XC_TETER93 )
    {
      xc_func_end(&XCFuncType_);
    }    
    else if( ( XId_ == XC_GGA_X_PBE ) && ( CId_ == XC_GGA_C_PBE ) )
    {
      xc_func_end(&XFuncType_);
      xc_func_end(&CFuncType_);
    }
    else if( XCId_ == XC_HYB_GGA_XC_HSE06 ){
      xc_func_end(&XCFuncType_);
    }
    else
      throw std::logic_error("Unrecognized exchange-correlation type");
  }
}



void
KohnSham::Setup	(
    const esdf::ESDFInputParam& esdfParam,
    const Domain&              dm,
    const std::vector<Atom>&   atomList )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::Setup");
#endif
	domain_              = dm;
	atomList_            = atomList;
	pseudoType_          = esdfParam.pseudoType;
	numExtraState_       = esdfParam.numExtraState;
  XCType_              = esdfParam.XCType;
  isHybridVexxProj_    = esdfParam.isHybridVexxProj;

  // FIXME Hard coded
  numDensityComponent_ = 1;

  // Since the number of density components is always 1 here, set numSpin = 2.
	numSpin_ = 2;

	// NOTE: NumSpin variable will be determined in derivative classes.

  Int ntotCoarse = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();

  density_.Resize( ntotFine, numDensityComponent_ );   
  SetValue( density_, 0.0 );

  gradDensity_.resize( DIM );
  for( Int d = 0; d < DIM; d++ ){
    gradDensity_[d].Resize( ntotFine, numDensityComponent_ );
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
  SetValue( vtotCoarse_, 0.0 );

	epsxc_.Resize( ntotFine );
	SetValue( epsxc_, 0.0 );

	vxc_.Resize( ntotFine, numDensityComponent_ );
	SetValue( vxc_, 0.0 );



  // Initialize the XC functionals, only spin-unpolarized case
	// Obtain the exchange-correlation id
  {
    isHybrid_ = false;

    if( XCType_ == "XC_LDA_XC_TETER93" )
    { 
      XCId_ = XC_LDA_XC_TETER93;
      if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
        throw std::runtime_error( "XC functional initialization error." );
      } 
      // Teter 93
      // S Goedecker, M Teter, J Hutter, Phys. Rev B 54, 1703 (1996) 
    }    
    else if( XCType_ == "XC_GGA_XC_PBE" )
    {
      XId_ = XC_GGA_X_PBE;
      CId_ = XC_GGA_C_PBE;
      // Perdew, Burke & Ernzerhof correlation
      // JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
      // JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 78, 1396(E) (1997)
      if( xc_func_init(&XFuncType_, XId_, XC_UNPOLARIZED) != 0 ){
        throw std::runtime_error( "X functional initialization error." );
      }
      if( xc_func_init(&CFuncType_, CId_, XC_UNPOLARIZED) != 0 ){
        throw std::runtime_error( "C functional initialization error." );
      }
    }
    else if( XCType_ == "XC_HYB_GGA_XC_HSE06" )
    {
      XCId_ = XC_HYB_GGA_XC_HSE06;
      if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
        throw std::runtime_error( "XC functional initialization error." );
      } 

      isHybrid_ = true;
      // FIXME Not considering restarting yet
      isEXXActive_ = false;

      // J. Heyd, G. E. Scuseria, and M. Ernzerhof, J. Chem. Phys. 118, 8207 (2003) (doi: 10.1063/1.1564060)
      // J. Heyd, G. E. Scuseria, and M. Ernzerhof, J. Chem. Phys. 124, 219906 (2006) (doi: 10.1063/1.2204597)
      // A. V. Krukau, O. A. Vydrov, A. F. Izmaylov, and G. E. Scuseria, J. Chem. Phys. 125, 224106 (2006) (doi: 10.1063/1.2404663)
      //
      // This is the same as the "hse" functional in QE 5.1
    }
    else {
      throw std::logic_error("Unrecognized exchange-correlation type");
    }
  }
  	

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

  Print( statusOFS, "Computing the local pseudopotential" );
	SetValue( pseudoCharge_, 0.0 );
  for (Int a=0; a<numAtom; a++) {
    ptable.CalculatePseudoCharge( atomList_[a], domain_, 
				gridpos, pseudo_[a].pseudoCharge );
    //accumulate to the global vector
    IntNumVec &idx = pseudo_[a].pseudoCharge.first;
    DblNumMat &val = pseudo_[a].pseudoCharge.second;
    for (Int k=0; k<idx.m(); k++) 
			pseudoCharge_[idx(k)] += val(k, VAL);
    // For debug purpose, check the summation of the derivative
    Real sumVDX = 0.0, sumVDY = 0.0, sumVDZ = 0.0;
    for (Int k=0; k<idx.m(); k++) {
      sumVDX += val(k, DX);
      sumVDY += val(k, DY);
      sumVDZ += val(k, DZ);
    }
    sumVDX *= vol / Real(ntotFine);
    sumVDY *= vol / Real(ntotFine);
    sumVDZ *= vol / Real(ntotFine);
    Print( statusOFS, "For Atom ", a );
    Print( statusOFS, "Sum dV_a / dx = ", sumVDX );
    Print( statusOFS, "Sum dV_a / dy = ", sumVDY );
    Print( statusOFS, "Sum dV_a / dz = ", sumVDZ );
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
  
  Print( statusOFS, "Computing the non-local pseudopotential" );
  Print( statusOFS, "Print out the summation of derivative of pseudopotential on the fine grid " );

  Int cnt = 0; // the total number of PS used
  for ( Int a=0; a < atomList_.size(); a++ ) {
		ptable.CalculateNonlocalPP( atomList_[a], domain_, gridposCoarse,
				pseudo_[a].vnlList ); 
    // Introduce the nonlocal pseudopotential on the fine grid.
		ptable.CalculateNonlocalPP( atomList_[a], domain_, gridpos,
				pseudo_[a].vnlListFine ); 
		cnt = cnt + pseudo_[a].vnlList.size();

    // For debug purpose, check the summation of the derivative
    std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlListFine;
    for( Int l = 0; l < vnlList.size(); l++ ){
      SparseVec& bl = vnlList[l].first;
      IntNumVec& idx = bl.first;
      DblNumMat& val = bl.second;
      Real sumVDX = 0.0, sumVDY = 0.0, sumVDZ = 0.0;
      for (Int k=0; k<idx.m(); k++) {
        sumVDX += val(k, DX);
        sumVDY += val(k, DY);
        sumVDZ += val(k, DZ);
      }
      sumVDX *= vol / Real(ntotFine);
      sumVDY *= vol / Real(ntotFine);
      sumVDZ *= vol / Real(ntotFine);
      statusOFS << "For atom " << a << ", projector " << l << std::endl;
      Print( statusOFS, "Sum dV_a / dx = ", sumVDX );
      Print( statusOFS, "Sum dV_a / dy = ", sumVDY );
      Print( statusOFS, "Sum dV_a / dz = ", sumVDZ );
    }

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

      SetValue( fft.outputComplexVecFine, Z_ZERO );
      for( Int i = 0; i < ntot; i++ ){
        fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i);
      }

      fftw_execute( fft.backwardPlanFine );

      // FIXME Factor to be simplified
      Real fac = numSpin_ * occrate(psi.WavefunIdx(k)) / (double(ntot) * double(ntotFine));
      for( Int i = 0; i < ntotFine; i++ ){
				densityLocal(i,RHO) +=  pow( std::abs(fft.inputComplexVecFine(i).real()), 2.0 ) * fac;
      }
		}
	}
	
  mpi::Allreduce( densityLocal.Data(), density_.Data(), ntotFine, MPI_SUM, domain_.comm );

  val = 0.0; // sum of density
  for (Int i=0; i<ntotFine; i++) {
    val  += density_(i, RHO);
  }

	Print( statusOFS, "Raw data, sum of density = ",  val );
	Print( statusOFS, "Expected sum of density  = ",  (numSpin_ * numOccupiedState_ ) );

  // Scale the density
  blas::Scal( ntotFine, (numSpin_ * Real(numOccupiedState_) * Real(ntotFine)) / ( vol * val ), 
      density_.VecData(RHO), 1 );

  // Double check (can be neglected)
  val = 0.0; // sum of density
  for (Int i=0; i<ntotFine; i++) {
    val  += density_(i, RHO) * vol / ntotFine;
  }
	Print( statusOFS, "Raw data, sum of adjusted density = ",  val );
  

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
  Real vol = domain_.Volume();

  if( XCId_ == XC_LDA_XC_TETER93 ) 
  {
    xc_lda_exc_vxc( &XCFuncType_, ntot, density_.VecData(RHO), 
        epsxc_.Data(), vxc_.Data() );
  }//XC_FAMILY_LDA
  else if( ( XId_ == XC_GGA_X_PBE ) && ( CId_ == XC_GGA_C_PBE ) ) {
    DblNumMat     vxc1;             
    DblNumMat     vxc2;             
    vxc1.Resize( ntot, numDensityComponent_ );
    vxc2.Resize( ntot, numDensityComponent_ );

    DblNumMat     vxc1temp;             
    DblNumMat     vxc2temp;             
    vxc1temp.Resize( ntot, numDensityComponent_ );
    vxc2temp.Resize( ntot, numDensityComponent_ );

    DblNumVec     epsx; 
    DblNumVec     epsc; 
    epsx.Resize( ntot );
    epsc.Resize( ntot );

    DblNumMat gradDensity;
    gradDensity.Resize( ntot, numDensityComponent_ );
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
    statusOFS << "1" << std::endl;
    xc_gga_exc_vxc( &XFuncType_, ntot, density_.VecData(RHO), 
        gradDensity.VecData(RHO), epsx.Data(), vxc1.Data(), vxc2.Data() );

    SetValue( epsc, 0.0 );
    SetValue( vxc1temp, 0.0 );
    SetValue( vxc2temp, 0.0 );
    statusOFS << "2" << std::endl;
    xc_gga_exc_vxc( &CFuncType_, ntot, density_.VecData(RHO), 
        gradDensity.VecData(RHO), epsc.Data(), vxc1temp.Data(), vxc2temp.Data() );
    statusOFS << "3" << std::endl;

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
  else if( XCId_ == XC_HYB_GGA_XC_HSE06 ){
    // FIXME Condensify with the previous
    DblNumMat     vxc1;             
    DblNumMat     vxc2;             
    vxc1.Resize( ntot, numDensityComponent_ );
    vxc2.Resize( ntot, numDensityComponent_ );


    DblNumMat gradDensity;
    gradDensity.Resize( ntot, numDensityComponent_ );
    SetValue( gradDensity, 0.0 );
    DblNumMat& gradDensity0 = gradDensity_[0];
    DblNumMat& gradDensity1 = gradDensity_[1];
    DblNumMat& gradDensity2 = gradDensity_[2];

    for(Int i = 0; i < ntot; i++){
      gradDensity(i, RHO) = gradDensity0(i, RHO) * gradDensity0(i, RHO)
        + gradDensity1(i, RHO) * gradDensity1(i, RHO)
        + gradDensity2(i, RHO) * gradDensity2(i, RHO);
    }

    SetValue( epsxc_, 0.0 );
    SetValue( vxc1, 0.0 );
    SetValue( vxc2, 0.0 );
    xc_gga_exc_vxc( &XCFuncType_, ntot, density_.VecData(RHO), 
        gradDensity.VecData(RHO), epsxc_.Data(), vxc1.Data(), vxc2.Data() );


    for( Int i = 0; i < ntot; i++ ){
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
  } // XC_FAMILY Hybrid
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

  // Method 2: Using integration by parts
	if(0)
	{
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
				resX += val(l, VAL) * vhartDrv[0][idx(l)] * wgt;
				resY += val(l, VAL) * vhartDrv[1][idx(l)] * wgt;
				resZ += val(l, VAL) * vhartDrv[2][idx(l)] * wgt;
			}
			force( a, 0 ) += resX;
			force( a, 1 ) += resY;
			force( a, 2 ) += resZ;

		} // for (a)
	}

  // Method 3: Evaluating the derivative by expliciting computing the
  // derivative of the local pseudopotential
	if(0){
    Int ntotFine = domain_.NumGridTotalFine();

    DblNumVec totalCharge(ntotFine);
    blas::Copy( ntot, density_.VecData(0), 1, totalCharge.Data(), 1 );
    blas::Axpy( ntot, -1.0, pseudoCharge_.Data(),1,
        totalCharge.Data(), 1 );

    std::vector<DblNumVec> vlocDrv(DIM);
    for( Int d = 0; d < DIM; d++ ){
      vlocDrv[d].Resize(ntotFine);
    }
    CpxNumVec cpxTempVec(ntotFine);

		for (Int a=0; a<numAtom; a++) {
			PseudoPot& pp = pseudo_[a];
			SparseVec& sp = pp.pseudoCharge;
			IntNumVec& idx = sp.first;
			DblNumMat& val = sp.second;

      // Solve the Poisson equation for the pseudo-charge of atom a
      SetValue( fft.inputComplexVecFine, Z_ZERO );

      for( Int k = 0; k < idx.m(); k++ ){
        fft.inputComplexVecFine(idx(k)) = Complex( val(k,VAL), 0.0 );
      }

      fftw_execute( fft.forwardPlanFine );

      // Save the vector for multiple differentiation
      blas::Copy( ntot, fft.outputComplexVecFine.Data(), 1,
          cpxTempVec.Data(), 1 );

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

        for( Int i = 0; i < ntotFine; i++ ){
          vlocDrv[d](i) = fft.inputComplexVecFine(i).real() / ntotFine;
        }
      } // for (d)


			Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
			Real resX = 0.0;
			Real resY = 0.0;
			Real resZ = 0.0;
      for( Int i = 0; i < ntotFine; i++ ){
        resX -= vlocDrv[0](i) * totalCharge(i) * wgt;
        resY -= vlocDrv[1](i) * totalCharge(i) * wgt;
        resZ -= vlocDrv[2](i) * totalCharge(i) * wgt;
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
	if(0)
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
        // FIXME Change to coarse
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



	// Method 2: Using integration by parts, and throw the derivative to the wavefunctions
  // FIXME: Assumign real arithmetic is used here.
	if(0)
	{
    // Compute the derivative of the wavefunctions

    Fourier* fftPtr = &(fft);

    Int ntothalf = fftPtr->numGridTotalR2C;
    Int ntot  = psi.NumGridTotal();
    Int ncom  = psi.NumComponent();
    Int nocc  = psi.NumState(); // Local number of states

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

        for( Int g = 0; g < nocc; g++ ){
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

          forceLocal( a, 0 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * res[VAL] * res[DX];
          forceLocal( a, 1 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * res[VAL] * res[DY];
          forceLocal( a, 2 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * res[VAL] * res[DZ];
        } // for (g)
      } // for (l)
    } // for (a)
	}


	// Method 3: Using the derivative of the pseudopotential, but evaluated on a fine grid
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

    DblNumVec wfnFine(domain_.NumGridTotalFine());

		for( Int a = 0; a < numAtom; a++ ){
      // Use nonlocal pseudopotential on the fine grid 
			std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlListFine;
			for( Int l = 0; l < vnlList.size(); l++ ){
				SparseVec& bl = vnlList[l].first;
				Real  gamma   = vnlList[l].second;
				Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
				IntNumVec& idx = bl.first;
				DblNumMat& val = bl.second;

				for( Int g = 0; g < numStateLocal; g++ ){
					DblNumVec res(4);
					SetValue( res, 0.0 );
					Real* psiPtr = psi.Wavefun().VecData(0, g);

          // Interpolate the wavefunction from coarse to fine grid

          for( Int i = 0; i < domain_.NumGridTotal(); i++ ){
            fft.inputComplexVec(i) = Complex( psiPtr[i], 0.0 ); 
          }
          fftw_execute( fft.forwardPlan );

          // fft Coarse to Fine 

          SetValue( fft.outputComplexVecFine, Z_ZERO );
          for( Int i = 0; i < domain_.NumGridTotal(); i++ ){
            fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i);
          }

          fftw_execute( fft.backwardPlanFine );

          Real fac = 1.0 / sqrt( double(domain_.NumGridTotal())  *
             double(domain_.NumGridTotalFine()) ); 
          for( Int i = 0; i < domain_.NumGridTotalFine(); i++ ){
            wfnFine(i) = fft.inputComplexVecFine(i).real() * fac;
          }

					for( Int i = 0; i < idx.Size(); i++ ){
						res(VAL) += val(i, VAL ) * wfnFine[ idx(i) ] * sqrt(wgt);
						res(DX) += val(i, DX ) * wfnFine[ idx(i) ] * sqrt(wgt);
						res(DY) += val(i, DY ) * wfnFine[ idx(i) ] * sqrt(wgt);
						res(DZ) += val(i, DZ ) * wfnFine[ idx(i) ] * sqrt(wgt);
					}

					forceLocal( a, 0 ) += 4.0 * occupationRate_( psi.WavefunIdx(g) ) * gamma * res[VAL] * res[DX];
					forceLocal( a, 1 ) += 4.0 * occupationRate_( psi.WavefunIdx(g) ) * gamma * res[VAL] * res[DY];
					forceLocal( a, 2 ) += 4.0 * occupationRate_( psi.WavefunIdx(g) ) * gamma * res[VAL] * res[DZ];
       
        } // for (g)
			} // for (l)
		} // for (a)
	}

	// *********************************************************************
	// Compute the total force and give the value to atomList
	// *********************************************************************

  // Sum over the force
  DblNumMat  forceTmp( numAtom, DIM );
	SetValue( forceTmp, 0.0 );
      
  mpi::Allreduce( forceLocal.Data(), forceTmp.Data(), numAtom * DIM, MPI_SUM, domain_.comm );

  for( Int a = 0; a < numAtom; a++ ){
    force( a, 0 ) = force( a, 0 ) + forceTmp( a, 0 );
    force( a, 1 ) = force( a, 1 ) + forceTmp( a, 1 );
    force( a, 2 ) = force( a, 2 ) + forceTmp( a, 2 );
  }




	for( Int a = 0; a < numAtom; a++ ){
		atomList_[a].force = Point3( force(a,0), force(a,1), force(a,2) );
	} 


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculateForce  ----- 


void
KohnSham::CalculateForce2	( Spinor& psi, Fourier& fft  )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculateForce2");
#endif
  
  Real timeSta, timeEnd;

  // DEBUG purpose: special time on FFT
  Real timeFFTSta, timeFFTEnd, timeFFTTotal = 0.0;

  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int ntot  = psi.NumGridTotal();
  Int ncom  = psi.NumComponent();
  Int numStateLocal = psi.NumState(); // Local number of states

  Int numAtom   = atomList_.size();

	DblNumMat  force( numAtom, DIM );
	SetValue( force, 0.0 );
	DblNumMat  forceLocal( numAtom, DIM );
	SetValue( forceLocal, 0.0 );

	// *********************************************************************
	// Compute the derivative of the Hartree potential for computing the 
	// local pseudopotential contribution to the Hellmann-Feynman force
	// *********************************************************************
  GetTime( timeSta );

	std::vector<DblNumVec>  vhartDrv(DIM);

	DblNumVec  totalCharge(ntotFine);
	SetValue( totalCharge, 0.0 );

	// totalCharge = density_ - pseudoCharge_
	blas::Copy( ntotFine, density_.VecData(0), 1, totalCharge.Data(), 1 );
	blas::Axpy( ntotFine, -1.0, pseudoCharge_.Data(),1,
			totalCharge.Data(), 1 );
	
  // Total charge in the Fourier space
	CpxNumVec  totalChargeFourier( ntotFine );

	for( Int i = 0; i < ntotFine; i++ ){
		fft.inputComplexVecFine(i) = Complex( totalCharge(i), 0.0 );
	}

  GetTime( timeFFTSta );
	fftw_execute( fft.forwardPlanFine );
  GetTime( timeFFTEnd );
  timeFFTTotal += timeFFTEnd - timeFFTSta;


	blas::Copy( ntotFine, fft.outputComplexVecFine.Data(), 1,
			totalChargeFourier.Data(), 1 );

	// Compute the derivative of the Hartree potential via Fourier
	// transform 
	for( Int d = 0; d < DIM; d++ ){
		CpxNumVec& ikFine = fft.ikFine[d];
		for( Int i = 0; i < ntotFine; i++ ){
			if( fft.gkkFine(i) == 0 ){
				fft.outputComplexVecFine(i) = Z_ZERO;
			}
			else{
				// NOTE: gkk already contains the factor 1/2.
				fft.outputComplexVecFine(i) = totalChargeFourier(i) *
					2.0 * PI / fft.gkkFine(i) * ikFine(i);
			}
		}

    GetTime( timeFFTSta );
		fftw_execute( fft.backwardPlanFine );
    GetTime( timeFFTEnd );
    timeFFTTotal += timeFFTEnd - timeFFTSta;

		// vhartDrv saves the derivative of the Hartree potential
		vhartDrv[d].Resize( ntotFine );

		for( Int i = 0; i < ntotFine; i++ ){
			vhartDrv[d](i) = fft.inputComplexVecFine(i).real() / ntotFine;
		}

	} // for (d)

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the derivative of Hartree potential is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

	// *********************************************************************
	// Compute the force from local pseudopotential
	// *********************************************************************
  // Method 2: Using integration by parts for local pseudopotential.
  // No need to evaluate the derivative of the local pseudopotential.
  // This could potentially save some coding effort, and perhaps better for other 
  // pseudopotential such as Troullier-Martins
  GetTime( timeSta );
	if(1)
	{
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
				resX += val(l, VAL) * vhartDrv[0][idx(l)] * wgt;
				resY += val(l, VAL) * vhartDrv[1][idx(l)] * wgt;
				resZ += val(l, VAL) * vhartDrv[2][idx(l)] * wgt;
			}
			force( a, 0 ) += resX;
			force( a, 1 ) += resY;
			force( a, 2 ) += resZ;

		} // for (a)
	}

  GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the local potential contribution of the force is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	// *********************************************************************
	// Compute the force from nonlocal pseudopotential
	// *********************************************************************
  GetTime( timeSta );
	// Method 4: Using integration by parts, and throw the derivative to the wavefunctions
  // No need to evaluate the derivative of the non-local pseudopotential.
  // This could potentially save some coding effort, and perhaps better for other 
  // pseudopotential such as Troullier-Martins
	if(1)
  {
    Int ntot  = psi.NumGridTotal(); 
    Int ncom  = psi.NumComponent();
    Int numStateLocal = psi.NumState(); // Local number of states

    DblNumVec                psiFine( ntotFine );
    std::vector<DblNumVec>   psiDrvFine(DIM);
    for( Int d = 0; d < DIM; d++ ){
      psiDrvFine[d].Resize( ntotFine );
    }

    CpxNumVec psiFourier(ntotFine);

    // Loop over atoms and pseudopotentials
    Int numEig = occupationRate_.m();
    for( Int g = 0; g < numStateLocal; g++ ){
      // Compute the derivative of the wavefunctions on a fine grid
      Real* psiPtr = psi.Wavefun().VecData(0, g);
      for( Int i = 0; i < domain_.NumGridTotal(); i++ ){
        fft.inputComplexVec(i) = Complex( psiPtr[i], 0.0 ); 
      }

      GetTime( timeFFTSta );
      fftw_execute( fft.forwardPlan );
      GetTime( timeFFTEnd );
      timeFFTTotal += timeFFTEnd - timeFFTSta;
      // fft Coarse to Fine 

      SetValue( psiFourier, Z_ZERO );
      for( Int i = 0; i < ntot; i++ ){
        psiFourier(fft.idxFineGrid(i)) = fft.outputComplexVec(i);
      }

      // psi on a fine grid
      for( Int i = 0; i < ntotFine; i++ ){
        fft.outputComplexVecFine(i) = psiFourier(i);
      }

      GetTime( timeFFTSta );
      fftw_execute( fft.backwardPlanFine );
      GetTime( timeFFTEnd );
      timeFFTTotal += timeFFTEnd - timeFFTSta;
      
      Real fac = 1.0 / sqrt( double(domain_.NumGridTotal())  *
          double(domain_.NumGridTotalFine()) ); 
//      for( Int i = 0; i < domain_.NumGridTotalFine(); i++ ){
//        psiFine(i) = fft.inputComplexVecFine(i).real() * fac;
//      }
      blas::Copy( ntotFine, reinterpret_cast<Real*>(fft.inputComplexVecFine.Data()),
          2, psiFine.Data(), 1 );
      blas::Scal( ntotFine, fac, psiFine.Data(), 1 );

      // derivative of psi on a fine grid
      for( Int d = 0; d < DIM; d++ ){
        Complex* ikFinePtr = fft.ikFine[d].Data();
        Complex* psiFourierPtr    = psiFourier.Data();
        Complex* fftOutFinePtr = fft.outputComplexVecFine.Data();
        for( Int i = 0; i < ntotFine; i++ ){
//          fft.outputComplexVecFine(i) = psiFourier(i) * ikFine(i);
          *(fftOutFinePtr++) = *(psiFourierPtr++) * *(ikFinePtr++);
        }

        GetTime( timeFFTSta );
        fftw_execute( fft.backwardPlanFine );
        GetTime( timeFFTEnd );
        timeFFTTotal += timeFFTEnd - timeFFTSta;

//        for( Int i = 0; i < domain_.NumGridTotalFine(); i++ ){
//          psiDrvFine[d](i) = fft.inputComplexVecFine(i).real() * fac;
//        }
        blas::Copy( ntotFine, reinterpret_cast<Real*>(fft.inputComplexVecFine.Data()),
            2, psiDrvFine[d].Data(), 1 );
        blas::Scal( ntotFine, fac, psiDrvFine[d].Data(), 1 );

      } // for (d)

      // Evaluate the contribution to the atomic force
      for( Int a = 0; a < numAtom; a++ ){
        std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlListFine;
        for( Int l = 0; l < vnlList.size(); l++ ){
          SparseVec& bl = vnlList[l].first;
          Real  gamma   = vnlList[l].second;
          Real  wgt = domain_.Volume() / domain_.NumGridTotalFine();
          IntNumVec& idx = bl.first;
          DblNumMat& val = bl.second;

          DblNumVec res(4);
          SetValue( res, 0.0 );
          Real* psiPtr = psiFine.Data();
          Real* DpsiXPtr = psiDrvFine[0].Data();
          Real* DpsiYPtr = psiDrvFine[1].Data();
          Real* DpsiZPtr = psiDrvFine[2].Data();
          Real* valPtr   = val.VecData(VAL);
          Int*  idxPtr = idx.Data();
          for( Int i = 0; i < idx.Size(); i++ ){
            res(VAL) += *valPtr * psiPtr[ *idxPtr ] * sqrt(wgt);
            res(DX)  += *valPtr * DpsiXPtr[ *idxPtr ] * sqrt(wgt);
            res(DY)  += *valPtr * DpsiYPtr[ *idxPtr ] * sqrt(wgt);
            res(DZ)  += *valPtr * DpsiZPtr[ *idxPtr ] * sqrt(wgt);
            valPtr++;
            idxPtr++;
          }

          forceLocal( a, 0 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * res[VAL] * res[DX];
          forceLocal( a, 1 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * res[VAL] * res[DY];
          forceLocal( a, 2 ) += -4.0 * occupationRate_(psi.WavefunIdx(g)) * gamma * res[VAL] * res[DZ];
        } // for (l)
      } // for (a)

    } // for (g)
  }
	
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the nonlocal potential contribution of the force is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Total time for FFT in the computation of the force is " <<
    timeFFTTotal << " [s]" << std::endl << std::endl;
#endif
	// *********************************************************************
	// Compute the total force and give the value to atomList
	// *********************************************************************

  // Sum over the force
  DblNumMat  forceTmp( numAtom, DIM );
	SetValue( forceTmp, 0.0 );
      
  mpi::Allreduce( forceLocal.Data(), forceTmp.Data(), numAtom * DIM, MPI_SUM, domain_.comm );

  for( Int a = 0; a < numAtom; a++ ){
    force( a, 0 ) = force( a, 0 ) + forceTmp( a, 0 );
    force( a, 1 ) = force( a, 1 ) + forceTmp( a, 1 );
    force( a, 2 ) = force( a, 2 ) + forceTmp( a, 2 );
  }

	for( Int a = 0; a < numAtom; a++ ){
		atomList_[a].force = Point3( force(a,0), force(a,1), force(a,2) );
	} 


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculateForce2  ----- 

void
KohnSham::MultSpinor	( Spinor& psi, NumTns<Scalar>& a3, Fourier& fft )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::MultSpinor");
#endif
  SetValue( a3, SCALAR_ZERO );

  // DO not use OpenMP for now.
#ifdef _USE_OPENMP_
//#pragma omp parallel
  {
#endif
    psi.AddMultSpinorFineR2C( fft, vtot_, pseudo_, a3 );
#ifdef _USE_OPENMP_
  }
#endif

  if( isHybrid_ && isEXXActive_ ){
    if( this->IsHybridVexxProj() ){
      // temporarily just implement here
      // Directly use projector
      Int numProj = vexxProj_.n();
      Int numStateTotal = this->NumStateTotal();
      Int ntot = psi.NumGridTotal();

      //        statusOFS << "numProj = " << numProj << std::endl;
      //        statusOFS << "numSTate= " << numStateTotal << std::endl;
      DblNumMat M(numProj, numStateTotal);

      // 
      blas::Gemm( 'T', 'N', numProj, numStateTotal, ntot, 1.0,
          vexxProj_.Data(), ntot, psi.Wavefun().Data(), ntot, 
          0.0, M.Data(), M.m() );
      // Minus sign comes from that all eigenvalues are negative
      blas::Gemm( 'N', 'N', ntot, numStateTotal, numProj, -1.0,
          vexxProj_.Data(), ntot, M.Data(), numProj,
          1.0, a3.Data(), ntot );
    }
    else{
      psi.AddMultSpinorEXX( fft, phiEXX_, exxFraction_,  numSpin_, occupationRate_, a3 );
    }
  }

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


void
KohnSham::SetPhiEXX	(const Spinor& psi, Fourier& fft)
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::SetPhiEXX");
#endif
  // FIXME collect Psi into a globally shared array in the MPI context.
  const NumTns<Scalar>& wavefun = psi.Wavefun();
  Int ntot = wavefun.m();
  Int ncom = wavefun.n();
  Int numStateLocal = wavefun.p();
  Int numStateTotal = this->NumStateTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Real vol = fft.domain.Volume();

  phiEXX_.Resize( ntotFine, ncom, numStateTotal );
  SetValue( phiEXX_, SCALAR_ZERO );

  // Temporary buffer for collecting contribution from different MPI procs.
//  NumTns<Scalar> phiEXXTmp = phiEXX_;
//  SetValue(phiEXXTmp, SCALAR_ZERO);

  // Buffer
  DblNumVec psiFine(ntotFine);

  // From coarse to fine grid
  // FIXME Put in a more proper place
  for (Int k=0; k<numStateLocal; k++) {
    for (Int j=0; j<ncom; j++) {

      SetValue( psiFine, 0.0 );

      SetValue( fft.inputVecR2C, 0.0 ); 
      SetValue( fft.inputVecR2CFine, 0.0 ); 
      SetValue( fft.outputVecR2C, Z_ZERO ); 
      SetValue( fft.outputVecR2CFine, Z_ZERO ); 

      // For c2r and r2c transforms, the default is to DESTROY the
      // input, therefore a copy of the original matrix is necessary. 
      blas::Copy( ntot, wavefun.VecData(j,k), 1, 
          fft.inputVecR2C.Data(), 1 );

      fftw_execute_dft_r2c(
          fft.forwardPlanR2C, 
          fft.inputVecR2C.Data(),
          reinterpret_cast<fftw_complex*>(fft.outputVecR2C.Data() ));

      // Interpolate wavefunction from coarse to fine grid
      {
        Int *idxPtr = fft.idxFineGridR2C.Data();
        Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
        Complex *fftOutPtr = fft.outputVecR2C.Data();
        for( Int ig = 0; ig < fft.numGridTotalR2C; ig++ ){
          fftOutFinePtr[*(idxPtr++)] = *(fftOutPtr++);
        }
      }

      fftw_execute_dft_c2r(
          fft.backwardPlanR2CFine, 
          reinterpret_cast<fftw_complex*>(fft.outputVecR2CFine.Data() ),
          fft.inputVecR2CFine.Data() );


      // Factor normalize so that integration in the real space is 1
      Real fac = 1.0 / std::sqrt( double(ntot) * double(ntotFine) );
      fac *= std::sqrt( double(ntotFine) / vol );
      blas::Copy( ntotFine, fft.inputVecR2CFine.Data(), 1, psiFine.Data(), 1 );
      blas::Scal( ntotFine, fac, psiFine.Data(), 1 );
      statusOFS << "int (psiFine^2) dx = " << Energy(psiFine)*vol / double(ntotFine) << std::endl;
      
      blas::Copy( ntotFine, psiFine.Data(), 1, phiEXX_.VecData(j,psi.WavefunIdx(k)), 1);

    } // for (j)
  } // for (k)

//  mpi::Allreduce( phiEXXTmp.Data(), phiEXX_.Data(), ntotFine * ncom * numStateTotal, MPI_SUM, 
//     domain_.comm );
  
#ifndef _RELEASE_
  PopCallStack();
#endif

  return ;
} 		// -----  end of method KohnSham::SetPhiEXX  ----- 


void
KohnSham::CalculateVexxPsi ( Spinor& psi, Fourier& fft )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculateVexxPsi");
#endif
  // FIXME
  Real SVDTolerance = 1e-4;
  // This assumes SetPhiEXX has been called so that phiEXX and psi
  // contain the same information. 
  
  // Since this is a projector, it should be done on the COARSE grid,
  // i.e. to the wavefunction directly

  // Only works for single processor
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = phiEXX_.p();
  NumTns<Scalar>  vexxPsi( ntot, 1, numStateTotal );

  // VexxPsi = V_{exx}*Phi.
  SetValue( vexxPsi, SCALAR_ZERO );
  psi.AddMultSpinorEXX( fft, phiEXX_, exxFraction_,  numSpin_, 
      occupationRate_, vexxPsi );

  // Compute M = Phi'*vexxPsi
  DblNumMat  M(numStateTotal, numStateTotal);
  blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, 
      1.0, psi.Wavefun().Data(), ntot, vexxPsi.Data(), ntot,
      0.0, M.Data(), numStateTotal );

  DblNumMat  U( numStateTotal, numStateTotal );
  DblNumMat VT( numStateTotal, numStateTotal );
  DblNumVec  S( numStateTotal );
  SetValue( S, 0.0 );
  
  lapack::QRSVD( numStateTotal, numStateTotal, M.Data(), numStateTotal,
      S.Data(), U.Data(), U.m(), VT.Data(), VT.m() );


  for( Int g = 0; g < numStateTotal; g++ ){
    S[g] = std::sqrt( S[g] );
  }

  Int rankM = 0;
  for( Int g = 0; g < numStateTotal; g++ ){
    if( S[g] / S[0] > SVDTolerance ){
      rankM++;
    }
  }
  statusOFS << "rank of Phi'*VPhi matrix = " << rankM << std::endl;
  for( Int g = 0; g < rankM; g++ ){
    blas::Scal( numStateTotal, 1.0 / S[g], U.VecData(g), 1 );
  }

  vexxProj_.Resize( ntot, rankM );
  blas::Gemm( 'N', 'N', ntot, rankM, numStateTotal, 1.0, 
      vexxPsi.Data(), ntot, U.Data(), numStateTotal, 0.0,
      vexxProj_.Data(), ntot );

  // Sanity check. For debugging only
  if(0){
  // Make sure U and VT are the same. Should be an identity matrix
    blas::Gemm( 'N', 'N', numStateTotal, numStateTotal, numStateTotal, 1.0, 
        VT.Data(), numStateTotal, U.Data(), numStateTotal, 0.0,
        M.Data(), numStateTotal );
    statusOFS << "M = " << M << std::endl;

    NumTns<Scalar> vpsit = psi.Wavefun();
    Int numProj = rankM;
    DblNumMat Mt(numProj, numStateTotal);
    
    blas::Gemm( 'T', 'N', numProj, numStateTotal, ntot, 1.0,
        vexxProj_.Data(), ntot, psi.Wavefun().Data(), ntot, 
        0.0, Mt.Data(), Mt.m() );
    // Minus sign comes from that all eigenvalues are negative
    blas::Gemm( 'N', 'N', ntot, numStateTotal, numProj, -1.0,
        vexxProj_.Data(), ntot, Mt.Data(), numProj,
        0.0, vpsit.Data(), ntot );

    for( Int k = 0; k < numStateTotal; k++ ){
      Real norm = 0.0;
      for( Int ir = 0; ir < ntot; ir++ ){
        norm = norm + std::pow(vexxPsi(ir,0,k) - vpsit(ir,0,k), 2.0);
      }
      statusOFS << "Diff of vexxPsi " << std::sqrt(norm) << std::endl;
    }
  }

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculateVexxPsi  ----- 


// This comes from exxenergy2() function in exx.f90 in QE.
Real
KohnSham::CalculateEXXEnergy	( Spinor& psi, Fourier& fft )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculateEXXEnergy");
#endif
  Real fockEnergy = 0.0;
  
  // Repeat the calculation of Vexx
  // FIXME Will be replaced by the stored VPhi matrix in the new
  // algorithm to reduce the cost, but this should be a new function
  
  // FIXME Should be combined better with the addition of exchange part in spinor
  NumTns<Scalar>& wavefun = psi.Wavefun();

  if( !fft.isInitialized ){
    throw std::runtime_error("Fourier is not prepared.");
  }
  Index3& numGrid = fft.domain.numGrid;
  Index3& numGridFine = fft.domain.numGridFine;
  Int ntot = wavefun.m();
  Int ncom = wavefun.n();
  Int numStateLocal = wavefun.p();
  Int ntotFine = fft.domain.NumGridTotalFine();
  Real vol = fft.domain.Volume();
  NumTns<Scalar>& phi = phiEXX_;
  Int ncomPhi = phi.n();
  if( ncomPhi != 1 || ncom != 1 ){
    throw std::logic_error("Spin polarized case not implemented.");
  }
  Int numStateTotalPhi = phi.p();

  if( fft.domain.NumGridTotal() != ntot ){
    throw std::logic_error("Domain size does not match.");
  }

  // Directly use the phiEXX_ and vexxProj_ to calculate the exchange energy
  if( isHybridVexxProj_ ){
    // temporarily just implement here
    // Directly use projector
    Int numProj = vexxProj_.n();
    Int numStateTotal = this->NumStateTotal();
    Int ntot = psi.NumGridTotal();

    DblNumMat M(numProj, numStateTotal);

    // 
    NumTns<Scalar>  vexxPsi( ntot, 1, numStateTotalPhi );
    SetValue( vexxPsi, SCALAR_ZERO );

    blas::Gemm( 'T', 'N', numProj, numStateTotal, ntot, 1.0,
        vexxProj_.Data(), ntot, psi.Wavefun().Data(), ntot, 
        0.0, M.Data(), M.m() );
    // Minus sign comes from that all eigenvalues are negative
    blas::Gemm( 'N', 'N', ntot, numStateTotal, numProj, -1.0,
        vexxProj_.Data(), ntot, M.Data(), numProj,
        0.0, vexxPsi.Data(), ntot );

    for( Int k = 0; k < numStateTotalPhi; k++ ){
      for( Int j = 0; j < ncom; j++ ){
        for( Int ir = 0; ir < ntot; ir++ ){
          fockEnergy += vexxPsi(ir,j,k) * wavefun(ir,j,k) * occupationRate_[k];
        }
      }
    }
  }
  else{
    NumTns<Scalar>  vexxPsi( ntot, 1, numStateTotalPhi );
    SetValue( vexxPsi, SCALAR_ZERO );
    psi.AddMultSpinorEXX( fft, phiEXX_, exxFraction_,  numSpin_, occupationRate_, 
       vexxPsi );
    // Compute the exchange energy:
    // Note: no additional normalization factor due to the
    // normalization rule of psi, NOT phi!!
    for( Int k = 0; k < numStateTotalPhi; k++ ){
      for( Int j = 0; j < ncom; j++ ){
        for( Int ir = 0; ir < ntot; ir++ ){
          fockEnergy += vexxPsi(ir,j,k) * wavefun(ir,j,k) * occupationRate_[k];
        }
      }
    }
  }
  
#ifndef _RELEASE_
	PopCallStack();
#endif

	return fockEnergy;
} 		// -----  end of method KohnSham::CalculateEXXEnergy  ----- 


} // namespace dgdft
