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
using namespace dgdft::esdf;


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
      ErrorHandling("Unrecognized exchange-correlation type");
  }
}



void
KohnSham::Setup    (
    const Domain&               dm,
    const std::vector<Atom>&    atomList )
{
  domain_              = dm;
  atomList_            = atomList;
  numExtraState_       = esdfParam.numExtraState;
  XCType_              = esdfParam.XCType;
  numMuHybridDF_                   = esdfParam.numMuHybridDF;
  numGaussianRandomHybridDF_       = esdfParam.numGaussianRandomHybridDF;
  numProcScaLAPACKHybridDF_        = esdfParam.numProcScaLAPACKHybridDF;
  BlockSizeScaLAPACK_      = esdfParam.BlockSizeScaLAPACK;
  exxDivergenceType_   = esdfParam.exxDivergenceType;

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
        ErrorHandling( "XC functional initialization error." );
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
        ErrorHandling( "X functional initialization error." );
      }
      if( xc_func_init(&CFuncType_, CId_, XC_UNPOLARIZED) != 0 ){
        ErrorHandling( "C functional initialization error." );
      }
    }
    else if( XCType_ == "XC_HYB_GGA_XC_HSE06" )
    {
      XCId_ = XC_HYB_GGA_XC_HSE06;
      if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
        ErrorHandling( "XC functional initialization error." );
      } 

      isHybrid_ = true;

      // J. Heyd, G. E. Scuseria, and M. Ernzerhof, J. Chem. Phys. 118, 8207 (2003) (doi: 10.1063/1.1564060)
      // J. Heyd, G. E. Scuseria, and M. Ernzerhof, J. Chem. Phys. 124, 219906 (2006) (doi: 10.1063/1.2204597)
      // A. V. Krukau, O. A. Vydrov, A. F. Izmaylov, and G. E. Scuseria, J. Chem. Phys. 125, 224106 (2006) (doi: 10.1063/1.2404663)
      //
      // This is the same as the "hse" functional in QE 5.1
    }
    else {
      ErrorHandling("Unrecognized exchange-correlation type");
    }
  }

  // ~~~ * ~~~
  // Set up wavefunction filter options: useful for CheFSI in PWDFT, for example
  // Affects the MATVEC operations in MultSpinor
  if(esdfParam.PWSolver == "CheFSI")
    set_wfn_filter(esdfParam.PWDFT_Cheby_apply_wfn_ecut_filt, 1, esdfParam.ecutWavefunction);
  else
    set_wfn_filter(0, 0, esdfParam.ecutWavefunction);



  return ;
}         // -----  end of method KohnSham::Setup  ----- 


void
KohnSham::CalculatePseudoPotential    ( PeriodTable &ptable ){
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
      ErrorHandling( "Cannot find the atom type." );
    }
    nelec = nelec + ptable.Zion(atype);
  }
  // FIXME Deal with the case when this is a buffer calculation and the
  // number of electrons is not a even number.
  //
  //    if( nelec % 2 != 0 ){
  //        ErrorHandling( "This is spin-restricted calculation. nelec should be even." );
  //    }
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
    if(0){
      Real sumVDX = 0.0, sumVDY = 0.0, sumVDZ = 0.0;
      for (Int k=0; k<idx.m(); k++) {
        sumVDX += val(k, DX);
        sumVDY += val(k, DY);
        sumVDZ += val(k, DZ);
      }
      sumVDX *= vol / Real(ntotFine);
      sumVDY *= vol / Real(ntotFine);
      sumVDZ *= vol / Real(ntotFine);
      if( std::sqrt(sumVDX * sumVDX + sumVDY * sumVDY + sumVDZ * sumVDZ) 
          > 1e-8 ){
        Print( statusOFS, "Local pseudopotential may not be constructed correctly" );
        Print( statusOFS, "For Atom ", a );
        Print( statusOFS, "Sum dV_a / dx = ", sumVDX );
        Print( statusOFS, "Sum dV_a / dy = ", sumVDY );
        Print( statusOFS, "Sum dV_a / dz = ", sumVDZ );
      }
    }
  }

  Real sumrho = 0.0;
  for (Int i=0; i<ntotFine; i++) 
    sumrho += pseudoCharge_[i]; 
  sumrho *= vol / Real(ntotFine);

  Print( statusOFS, "Sum of Pseudocharge                          = ", 
      sumrho );
  Print( statusOFS, "Number of Occupied States                    = ", 
      numOccupiedState_ );

  // adjustment should be multiplicative
  Real fac = nelec / sumrho;
  for (Int i=0; i<ntotFine; i++) 
    pseudoCharge_(i) *= fac; 

  Print( statusOFS, "After adjustment, Sum of Pseudocharge        = ", 
      (Real) nelec );

  // Nonlocal projectors
  std::vector<DblNumVec> gridposCoarse;
  UniformMesh ( domain_, gridposCoarse );

  Print( statusOFS, "Computing the non-local pseudopotential" );

  Int cnt = 0; // the total number of PS used
  for ( Int a=0; a < atomList_.size(); a++ ) {
    ptable.CalculateNonlocalPP( atomList_[a], domain_, gridposCoarse,
        pseudo_[a].vnlList ); 
    // Introduce the nonlocal pseudopotential on the fine grid.
    ptable.CalculateNonlocalPP( atomList_[a], domain_, gridpos,
        pseudo_[a].vnlListFine ); 
    cnt = cnt + pseudo_[a].vnlList.size();

    // For debug purpose, check the summation of the derivative
    if(0){
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
        if( std::sqrt(sumVDX * sumVDX + sumVDY * sumVDY + sumVDZ * sumVDZ) 
            > 1e-8 ){
          Print( statusOFS, "Local pseudopotential may not be constructed correctly" );
          statusOFS << "For atom " << a << ", projector " << l << std::endl;
          Print( statusOFS, "Sum dV_a / dx = ", sumVDX );
          Print( statusOFS, "Sum dV_a / dy = ", sumVDY );
          Print( statusOFS, "Sum dV_a / dz = ", sumVDZ );
        }
      }
    }

  }

  Print( statusOFS, "Total number of nonlocal pseudopotential = ",  cnt );


  return ;
}         // -----  end of method KohnSham::CalculatePseudoPotential ----- 

void KohnSham::CalculateAtomDensity ( PeriodTable &ptable, Fourier &fft ){
  if( esdfParam.pseudoType == "HGH" ){
    ErrorHandling("HGH pseudopotential does not yet support the computation of atomic density!");
  }

  Int ntotFine = domain_.NumGridTotalFine();
  Int numAtom = atomList_.size();
  Real vol = domain_.Volume();
  std::vector<DblNumVec> gridpos;
  UniformMeshFine ( domain_, gridpos );

  // The number of electrons for normalization purpose. 
  Int nelec = 0;
  for (Int a=0; a<numAtom; a++) {
    Int atype  = atomList_[a].type;
    if( ptable.ptemap().find(atype) == ptable.ptemap().end() ){
      ErrorHandling( "Cannot find the atom type." );
    }
    nelec = nelec + ptable.Zion(atype);
  }
  if( nelec % 2 != 0 ){
    ErrorHandling( "This is spin-restricted calculation. nelec should be even." );
  }


  // Search for the number of atom types and build a list of atom types
  std::set<Int> atomTypeSet;
  for( Int a = 0; a < numAtom; a++ ){
    atomTypeSet.insert( atomList_[a].type );
  } // for (a)

  // For each atom type, construct the atomic pseudocharge within the
  // cutoff radius starting from the origin in the real space, and
  // construct the structure factor

  // Origin-centered atomDensity in the real space and Fourier space
  DblNumVec atomDensityR( ntotFine );
  CpxNumVec atomDensityG( ntotFine );
  atomDensity_.Resize( ntotFine );
  SetValue( atomDensity_, 0.0 );
  SetValue( atomDensityR, 0.0 );
  SetValue( atomDensityG, Z_ZERO );

  for( std::set<Int>::iterator itype = atomTypeSet.begin(); 
    itype != atomTypeSet.end(); itype++ ){
    Int atype = *itype;
    Atom fakeAtom;
    fakeAtom.type = atype;
    fakeAtom.pos = domain_.posStart;

    ptable.CalculateAtomDensity( fakeAtom, domain_, gridpos, atomDensityR );

    // Compute the structure factor
    CpxNumVec ccvec(ntotFine);
    SetValue( ccvec, Z_ZERO );

    Complex* ccvecPtr = ccvec.Data();
    Complex* ikxPtr = fft.ikFine[0].Data();
    Complex* ikyPtr = fft.ikFine[1].Data();
    Complex* ikzPtr = fft.ikFine[2].Data();
    Real xx, yy, zz;
    Complex phase;

    for (Int a=0; a<numAtom; a++) {
      if( atomList_[a].type == atype ){
        xx = atomList_[a].pos[0];
        yy = atomList_[a].pos[1];
        zz = atomList_[a].pos[2];
        for( Int i = 0; i < ntotFine; i++ ){
          phase = -(ikxPtr[i] * xx + ikyPtr[i] * yy + ikzPtr[i] * zz);
          ccvecPtr[i] += std::exp( phase );
        }
      }
    }

    // Transfer the atomic charge from real space to Fourier space, and
    // multiply with the structure factor
    for(Int i = 0; i < ntotFine; i++){
      fft.inputComplexVecFine[i] = Complex( atomDensityR[i], 0.0 ); 
    }

    FFTWExecute ( fft, fft.forwardPlanFine );

    for( Int i = 0; i < ntotFine; i++ ){
      // Make it smoother: AGGREESIVELY truncate components beyond EcutWavefunction
      if( fft.gkkFine[i] < esdfParam.ecutWavefunction ){
        atomDensityG[i] += fft.outputComplexVecFine[i] * ccvec[i];
      }
    }
  }

  // Transfer back to the real space and add to atomDensity_ 
  {
    for(Int i = 0; i < ntotFine; i++){
      fft.outputComplexVecFine[i] = atomDensityG[i];
    }
  
    FFTWExecute ( fft, fft.backwardPlanFine );

    for( Int i = 0; i < ntotFine; i++ ){
      atomDensity_[i] = fft.inputComplexVecFine[i].real();
    }
  }


  Real sumrho = 0.0;
  for (Int i=0; i<ntotFine; i++) 
    sumrho += atomDensity_[i]; 
  sumrho *= vol / Real(ntotFine);

  Print( statusOFS, "Sum of atomic density                        = ", 
      sumrho );

  // adjustment should be multiplicative
  Real fac = nelec / sumrho;
  for (Int i=0; i<ntotFine; i++) 
    atomDensity_[i] *= fac; 

  Print( statusOFS, "After adjustment, Sum of atomic density = ", (Real) nelec );

  return ;
}         // -----  end of method KohnSham::CalculateAtomDensity  ----- 


void
KohnSham::CalculateDensity ( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier &fft)
{
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

  Real fac;

  SetValue( density_, 0.0 );
  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {

      for( Int i = 0; i < ntot; i++ ){
        fft.inputComplexVec(i) = Complex( psi.Wavefun(i,j,k), 0.0 ); 
      }

      FFTWExecute ( fft, fft.forwardPlan );

      // fft Coarse to Fine 

      SetValue( fft.outputComplexVecFine, Z_ZERO );
      for( Int i = 0; i < ntot; i++ ){
        fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i) * 
          sqrt( double(ntot) / double(ntotFine) );
      } 

      FFTWExecute ( fft, fft.backwardPlanFine );

      // FIXME Factor to be simplified
      fac = numSpin_ * occrate(psi.WavefunIdx(k));
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

  Real val1 = val;

  // Scale the density
  blas::Scal( ntotFine, (numSpin_ * Real(numOccupiedState_) * Real(ntotFine)) / ( vol * val ), 
      density_.VecData(RHO), 1 );

  // Double check (can be neglected)
  val = 0.0; // sum of density
  for (Int i=0; i<ntotFine; i++) {
    val  += density_(i, RHO) * vol / ntotFine;
  }

  Real val2 = val;

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Raw data, sum of density          = " << val1 << std::endl;
  statusOFS << "Expected sum of density           = " << numSpin_ * numOccupiedState_ << std::endl;
  statusOFS << "Raw data, sum of adjusted density = " << val2 << std::endl;
#endif


  return ;
}         // -----  end of method KohnSham::CalculateDensity  ----- 


void
KohnSham::CalculateGradDensity ( Fourier& fft )
{
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Real vol  = domain_.Volume();

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  MPI_Comm rowComm = MPI_COMM_NULL;
  MPI_Comm colComm = MPI_COMM_NULL;
  
  Int mpirankRow, mpisizeRow, mpirankCol, mpisizeCol;
  Int dmCol = DIM;
  Int dmRow = mpisize / dmCol;

  if(mpisize >= DIM){

    IntNumVec mpiRowMap(mpisize);
    IntNumVec mpiColMap(mpisize);

    for( Int i = 0; i < mpisize; i++ ){
      mpiRowMap(i) = i / dmCol;
      mpiColMap(i) = i % dmCol;
    } 

    if( mpisize > dmRow * dmCol ){
      for( Int k = dmRow * dmCol; k < mpisize; k++ ){
        mpiRowMap(k) = dmRow - 1;
      }
    } 

    MPI_Comm_split( domain_.comm, mpiRowMap(mpirank), mpirank, &rowComm );
    MPI_Comm_split( domain_.comm, mpiColMap(mpirank), mpirank, &colComm );

    MPI_Comm_rank(rowComm, &mpirankRow);
    MPI_Comm_size(rowComm, &mpisizeRow);

    MPI_Comm_rank(colComm, &mpirankCol);
    MPI_Comm_size(colComm, &mpisizeCol);

  }

  for( Int i = 0; i < ntotFine; i++ ){
    fft.inputComplexVecFine(i) = Complex( density_(i,RHO), 0.0 ); 
  }

  FFTWExecute ( fft, fft.forwardPlanFine );

  CpxNumVec  cpxVec( ntotFine );
  blas::Copy( ntotFine, fft.outputComplexVecFine.Data(), 1,
      cpxVec.Data(), 1 );

  // Compute the derivative of the Density via Fourier

  if(0){

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

      FFTWExecute ( fft, fft.backwardPlanFine );

      DblNumMat& gradDensity = gradDensity_[d];
      for( Int i = 0; i < ntotFine; i++ ){
        gradDensity(i, RHO) = fft.inputComplexVecFine(i).real();
      }
    } // for d

  } //if(0)

  if(1){
    
    Int d;
    if( mpisize < DIM ){ // mpisize < 3
      for( d = 0; d < DIM; d++ ){
        DblNumMat& gradDensity = gradDensity_[d];
        CpxNumVec& ik = fft.ikFine[d];
        for( Int i = 0; i < ntotFine; i++ ){
          if( fft.gkkFine(i) == 0 ){
            fft.outputComplexVecFine(i) = Z_ZERO;
          }
          else{
            fft.outputComplexVecFine(i) = cpxVec(i) * ik(i); 
          }
        }

        FFTWExecute ( fft, fft.backwardPlanFine );

        for( Int i = 0; i < ntotFine; i++ ){
          gradDensity(i, RHO) = fft.inputComplexVecFine(i).real();
        }
      } // for d

    } // mpisize < 3
    else { // mpisize > 3
  
      for( d = 0; d < DIM; d++ ){
        DblNumMat& gradDensity = gradDensity_[d];
        if ( d == mpirank % dmCol ){ 
          CpxNumVec& ik = fft.ikFine[d];
          for( Int i = 0; i < ntotFine; i++ ){
            if( fft.gkkFine(i) == 0 ){
              fft.outputComplexVecFine(i) = Z_ZERO;
            }
            else{
              fft.outputComplexVecFine(i) = cpxVec(i) * ik(i); 
            }
          }

          FFTWExecute ( fft, fft.backwardPlanFine );

          for( Int i = 0; i < ntotFine; i++ ){
            gradDensity(i, RHO) = fft.inputComplexVecFine(i).real();
          }
        } // d == mpirank
      } // for d

      for( d = 0; d < DIM; d++ ){
        DblNumMat& gradDensity = gradDensity_[d];
        MPI_Bcast( gradDensity.Data(), ntotFine, MPI_DOUBLE, d, rowComm );
      } // for d

    } // mpisize > 3

  } //if(1)

  if( rowComm != MPI_COMM_NULL ) MPI_Comm_free( & rowComm );
  if( colComm != MPI_COMM_NULL ) MPI_Comm_free( & colComm );

  return ;
}         // -----  end of method KohnSham::CalculateGradDensity  ----- 


void
KohnSham::CalculateXC    ( Real &val, Fourier& fft )
{
  Int ntot = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
  
  MPI_Comm rowComm = MPI_COMM_NULL;
  MPI_Comm colComm = MPI_COMM_NULL;
  
  Int mpirankRow, mpisizeRow, mpirankCol, mpisizeCol;
  Int dmCol = DIM;
  Int dmRow = mpisize / dmCol;

  if(mpisize >= DIM){

    IntNumVec mpiRowMap(mpisize);
    IntNumVec mpiColMap(mpisize);

    for( Int i = 0; i < mpisize; i++ ){
      mpiRowMap(i) = i / dmCol;
      mpiColMap(i) = i % dmCol;
    } 

    if( mpisize > dmRow * dmCol ){
      for( Int k = dmRow * dmCol; k < mpisize; k++ ){
        mpiRowMap(k) = dmRow - 1;
      }
    } 

    MPI_Comm_split( domain_.comm, mpiRowMap(mpirank), mpirank, &rowComm );
    MPI_Comm_split( domain_.comm, mpiColMap(mpirank), mpirank, &colComm );

    MPI_Comm_rank(rowComm, &mpirankRow);
    MPI_Comm_size(rowComm, &mpisizeRow);

    MPI_Comm_rank(colComm, &mpirankCol);
    MPI_Comm_size(colComm, &mpisizeCol);

  }

  Int ntotBlocksize = ntot / mpisize;
  Int ntotLocal = ntotBlocksize;
  if(mpirank < (ntot % mpisize)){
    ntotLocal = ntotBlocksize + 1;
  } 
  IntNumVec localSize(mpisize);
  IntNumVec localSizeDispls(mpisize);
  SetValue( localSize, 0 );
  SetValue( localSizeDispls, 0 );
  MPI_Allgather( &ntotLocal, 1, MPI_INT, localSize.Data(), 1, MPI_INT, domain_.comm );

  for (Int i = 1; i < mpisize; i++ ){
    localSizeDispls[i] = localSizeDispls[i-1] + localSize[i-1];
  }

  Real fac;
  // Cutoff 
  Real epsRho = 1e-8, epsGRho = 1e-10;

  Real timeSta, timeEnd;

  Real timeFFT = 0.00;
  Real timeOther = 0.00;

  if( XCId_ == XC_LDA_XC_TETER93 ) 
  {
    xc_lda_exc_vxc( &XCFuncType_, ntot, density_.VecData(RHO), 
        epsxc_.Data(), vxc_.Data() );

    // Modify "bad points"
    if(1){
      for( Int i = 0; i < ntot; i++ ){
        if( density_(i,RHO) < epsRho ){
          epsxc_(i) = 0.0;
          vxc_( i, RHO ) = 0.0;
        }
      }
    }


  }//XC_FAMILY_LDA
  else if( ( XId_ == XC_GGA_X_PBE ) && ( CId_ == XC_GGA_C_PBE ) ) {

    DblNumMat gradDensity( ntotLocal, numDensityComponent_ );
    DblNumMat& gradDensity0 = gradDensity_[0];
    DblNumMat& gradDensity1 = gradDensity_[1];
    DblNumMat& gradDensity2 = gradDensity_[2];

    GetTime( timeSta );
    for(Int i = 0; i < ntotLocal; i++){
      Int ii = i + localSizeDispls(mpirank);
      gradDensity(i, RHO) = gradDensity0(ii, RHO) * gradDensity0(ii, RHO)
        + gradDensity1(ii, RHO) * gradDensity1(ii, RHO)
        + gradDensity2(ii, RHO) * gradDensity2(ii, RHO);
    }
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing gradDensity in XC GGA-PBE is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    DblNumMat densityTemp;
    densityTemp.Resize( ntotLocal, numDensityComponent_ );

    for( Int i = 0; i < ntotLocal; i++ ){
      densityTemp(i, RHO) = density_(i + localSizeDispls(mpirank), RHO);
    }

    DblNumVec vxc1(ntotLocal);             
    DblNumVec vxc2(ntotLocal);             
    DblNumVec vxc1Temp(ntotLocal);             
    DblNumVec vxc2Temp(ntotLocal);             
    DblNumVec epsx(ntotLocal); 
    DblNumVec epsc(ntotLocal); 

    SetValue( vxc1, 0.0 );
    SetValue( vxc2, 0.0 );
    SetValue( vxc1Temp, 0.0 );
    SetValue( vxc2Temp, 0.0 );
    SetValue( epsx, 0.0 );
    SetValue( epsc, 0.0 );

    GetTime( timeSta );

    xc_gga_exc_vxc( &XFuncType_, ntotLocal, densityTemp.VecData(RHO), 
        gradDensity.VecData(RHO), epsx.Data(), vxc1.Data(), vxc2.Data() );

    xc_gga_exc_vxc( &CFuncType_, ntotLocal, densityTemp.VecData(RHO), 
        gradDensity.VecData(RHO), epsc.Data(), vxc1Temp.Data(), vxc2Temp.Data() );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for calling the XC kernel in XC GGA-PBE is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    DblNumVec     epsxcTemp( ntotLocal );
    DblNumVec     vxcTemp( ntot );
    DblNumVec     vxc2Temp2( ntot );
    for( Int i = 0; i < ntotLocal; i++ ){
      epsxcTemp(i) = epsx(i) + epsc(i) ;
      vxc1(i) += vxc1Temp( i );
      vxc2(i) += vxc2Temp( i );
    }


    // Modify "bad points"
    if(1){
      for( Int i = 0; i < ntotLocal; i++ ){
//        if( densityTemp(i,RHO) < epsRho ){
        if( densityTemp(i,RHO) < epsRho || gradDensity(i,RHO) < epsGRho ){
          epsxcTemp(i) = 0.0;
          vxc1(i) = 0.0;
          vxc2(i) = 0.0;
        }
      }
    }


    SetValue( epsxc_, 0.0 );
    SetValue( vxcTemp, 0.0 );
    SetValue( vxc2Temp2, 0.0 );

    GetTime( timeSta );

    MPI_Allgatherv( epsxcTemp.Data(), ntotLocal, MPI_DOUBLE, epsxc_.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );
    MPI_Allgatherv( vxc1.Data(), ntotLocal, MPI_DOUBLE, vxcTemp.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );
    MPI_Allgatherv( vxc2.Data(), ntotLocal, MPI_DOUBLE, vxc2Temp2.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for MPI_Allgatherv in XC GGA-PBE is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


    for( Int i = 0; i < ntot; i++ ){
      vxc_( i, RHO ) = vxcTemp(i);
    }
    Int d;
    if( mpisize < DIM ){ // mpisize < 3

      for( Int d = 0; d < DIM; d++ ){
        DblNumMat& gradDensityd = gradDensity_[d];
        for(Int i = 0; i < ntot; i++){
          fft.inputComplexVecFine(i) = Complex( gradDensityd( i, RHO ) * 2.0 * vxc2Temp2(i), 0.0 ); 
        }

        FFTWExecute ( fft, fft.forwardPlanFine );

        CpxNumVec& ik = fft.ikFine[d];

        for( Int i = 0; i < ntot; i++ ){
          if( fft.gkkFine(i) == 0 ){
            fft.outputComplexVecFine(i) = Z_ZERO;
          }
          else{
            fft.outputComplexVecFine(i) *= ik(i);
          }
        }

        FFTWExecute ( fft, fft.backwardPlanFine );

        for( Int i = 0; i < ntot; i++ ){
          vxc_( i, RHO ) -= fft.inputComplexVecFine(i).real();
        }

      } // for d

    } // mpisize < 3
    else { // mpisize > 3

      std::vector<DblNumVec>      vxcTemp3d;
      vxcTemp3d.resize( DIM );
      for( Int d = 0; d < DIM; d++ ){
        vxcTemp3d[d].Resize(ntot);
        SetValue (vxcTemp3d[d], 0.0);
      }

      for( d = 0; d < DIM; d++ ){
        DblNumMat& gradDensityd = gradDensity_[d];
        DblNumVec& vxcTemp3 = vxcTemp3d[d]; 
        if ( d == mpirank % dmCol ){ 
          for(Int i = 0; i < ntot; i++){
            fft.inputComplexVecFine(i) = Complex( gradDensityd( i, RHO ) * 2.0 * vxc2Temp2(i), 0.0 ); 
          }

          FFTWExecute ( fft, fft.forwardPlanFine );

          CpxNumVec& ik = fft.ikFine[d];

          for( Int i = 0; i < ntot; i++ ){
            if( fft.gkkFine(i) == 0 ){
              fft.outputComplexVecFine(i) = Z_ZERO;
            }
            else{
              fft.outputComplexVecFine(i) *= ik(i);
            }
          }

          FFTWExecute ( fft, fft.backwardPlanFine );

          for( Int i = 0; i < ntot; i++ ){
            vxcTemp3(i) = fft.inputComplexVecFine(i).real();
          }
        } // d == mpirank
      } // for d

      for( d = 0; d < DIM; d++ ){
        DblNumVec& vxcTemp3 = vxcTemp3d[d]; 
        MPI_Bcast( vxcTemp3.Data(), ntot, MPI_DOUBLE, d, rowComm );
        for( Int i = 0; i < ntot; i++ ){
          vxc_( i, RHO ) -= vxcTemp3(i);
        }
      } // for d

    } // mpisize > 3



  } // XC_FAMILY_GGA
  else if( XCId_ == XC_HYB_GGA_XC_HSE06 ){
    // FIXME Condensify with the previous

    DblNumMat gradDensity;
    gradDensity.Resize( ntotLocal, numDensityComponent_ );
    SetValue( gradDensity, 0.0 );
    DblNumMat& gradDensity0 = gradDensity_[0];
    DblNumMat& gradDensity1 = gradDensity_[1];
    DblNumMat& gradDensity2 = gradDensity_[2];

    GetTime( timeSta );
    for(Int i = 0; i < ntotLocal; i++){
      Int ii = i + localSizeDispls(mpirank);
      gradDensity(i, RHO) = gradDensity0(ii, RHO) * gradDensity0(ii, RHO)
        + gradDensity1(ii, RHO) * gradDensity1(ii, RHO)
        + gradDensity2(ii, RHO) * gradDensity2(ii, RHO);
    }
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << " " << std::endl;
    statusOFS << "Time for computing gradDensity in XC HSE06 is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    DblNumMat densityTemp;
    densityTemp.Resize( ntotLocal, numDensityComponent_ );

    for( Int i = 0; i < ntotLocal; i++ ){
      densityTemp(i, RHO) = density_(i + localSizeDispls(mpirank), RHO);
    }

    DblNumVec vxc1Temp(ntotLocal);             
    DblNumVec vxc2Temp(ntotLocal);             
    DblNumVec epsxcTemp(ntotLocal); 

    SetValue( vxc1Temp, 0.0 );
    SetValue( vxc2Temp, 0.0 );
    SetValue( epsxcTemp, 0.0 );

    GetTime( timeSta );
    xc_gga_exc_vxc( &XCFuncType_, ntotLocal, densityTemp.VecData(RHO), 
        gradDensity.VecData(RHO), epsxcTemp.Data(), vxc1Temp.Data(), vxc2Temp.Data() );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing xc_gga_exc_vxc in XC HSE06 is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    DblNumVec vxc1(ntot);             
    DblNumVec vxc2(ntot);             

    SetValue( vxc1, 0.0 );
    SetValue( vxc2, 0.0 );
    SetValue( epsxc_, 0.0 );

    // Modify "bad points"
    if(1){
      for( Int i = 0; i < ntotLocal; i++ ){
        if( densityTemp(i,RHO) < epsRho || gradDensity(i,RHO) < epsGRho ){
          epsxcTemp(i) = 0.0;
          vxc1Temp(i) = 0.0;
          vxc2Temp(i) = 0.0;
        }
      }
    }

    GetTime( timeSta );

    MPI_Allgatherv( epsxcTemp.Data(), ntotLocal, MPI_DOUBLE, epsxc_.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );
    MPI_Allgatherv( vxc1Temp.Data(), ntotLocal, MPI_DOUBLE, vxc1.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );
    MPI_Allgatherv( vxc2Temp.Data(), ntotLocal, MPI_DOUBLE, vxc2.Data(), 
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for MPI_Allgatherv in XC HSE06 is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    for( Int i = 0; i < ntot; i++ ){
      vxc_( i, RHO ) = vxc1(i);
    }

    Int d;
    if( mpisize < DIM ){ // mpisize < 3

      for( d = 0; d < DIM; d++ ){
        DblNumMat& gradDensityd = gradDensity_[d];
        for(Int i = 0; i < ntot; i++){
          fft.inputComplexVecFine(i) = Complex( gradDensityd( i, RHO ) * 2.0 * vxc2(i), 0.0 ); 
        }

        FFTWExecute ( fft, fft.forwardPlanFine );

        CpxNumVec& ik = fft.ikFine[d];

        for( Int i = 0; i < ntot; i++ ){
          if( fft.gkkFine(i) == 0 ){
            fft.outputComplexVecFine(i) = Z_ZERO;
          }
          else{
            fft.outputComplexVecFine(i) *= ik(i);
          }
        }

        FFTWExecute ( fft, fft.backwardPlanFine );

        GetTime( timeSta );
        for( Int i = 0; i < ntot; i++ ){
          vxc_( i, RHO ) -= fft.inputComplexVecFine(i).real();
        }
        GetTime( timeEnd );
        timeOther = timeOther + ( timeEnd - timeSta );

      } // for d
    
    } // mpisize < 3
    else { // mpisize > 3
      
      std::vector<DblNumVec>      vxcTemp3d;
      vxcTemp3d.resize( DIM );
      for( Int d = 0; d < DIM; d++ ){
        vxcTemp3d[d].Resize(ntot);
        SetValue (vxcTemp3d[d], 0.0);
      }

      for( d = 0; d < DIM; d++ ){
        DblNumMat& gradDensityd = gradDensity_[d];
        DblNumVec& vxcTemp3 = vxcTemp3d[d]; 
        if ( d == mpirank % dmCol ){ 
          for(Int i = 0; i < ntot; i++){
            fft.inputComplexVecFine(i) = Complex( gradDensityd( i, RHO ) * 2.0 * vxc2(i), 0.0 ); 
          }

          FFTWExecute ( fft, fft.forwardPlanFine );

          CpxNumVec& ik = fft.ikFine[d];

          for( Int i = 0; i < ntot; i++ ){
            if( fft.gkkFine(i) == 0 ){
              fft.outputComplexVecFine(i) = Z_ZERO;
            }
            else{
              fft.outputComplexVecFine(i) *= ik(i);
            }
          }

          FFTWExecute ( fft, fft.backwardPlanFine );

          for( Int i = 0; i < ntot; i++ ){
            vxcTemp3(i) = fft.inputComplexVecFine(i).real();
          }

        } // d == mpirank
      } // for d

      for( d = 0; d < DIM; d++ ){
        DblNumVec& vxcTemp3 = vxcTemp3d[d]; 
        MPI_Bcast( vxcTemp3.Data(), ntot, MPI_DOUBLE, d, rowComm );
        for( Int i = 0; i < ntot; i++ ){
          vxc_( i, RHO ) -= vxcTemp3(i);
        }
      } // for d

    } // mpisize > 3

  } // XC_FAMILY Hybrid
  else
    ErrorHandling( "Unsupported XC family!" );

  if( rowComm != MPI_COMM_NULL ) MPI_Comm_free( & rowComm );
  if( colComm != MPI_COMM_NULL ) MPI_Comm_free( & colComm );

  // Compute the total exchange-correlation energy
  val = 0.0;
  GetTime( timeSta );
  for(Int i = 0; i < ntot; i++){
    val += density_(i, RHO) * epsxc_(i) * vol / (Real) ntot;
  }
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << " " << std::endl;
  statusOFS << "Time for computing total xc energy in XC is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  return ;
}         // -----  end of method KohnSham::CalculateXC  ----- 


void KohnSham::CalculateHartree( Fourier& fft ) {
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  Int ntot = domain_.NumGridTotalFine();
  if( fft.domain.NumGridTotalFine() != ntot ){
    ErrorHandling( "Grid size does not match!" );
  }

  // The contribution of the pseudoCharge is subtracted. So the Poisson
  // equation is well defined for neutral system.
  for( Int i = 0; i < ntot; i++ ){
    fft.inputComplexVecFine(i) = Complex( 
        density_(i,RHO) - pseudoCharge_(i), 0.0 );
  }

  FFTWExecute ( fft, fft.forwardPlanFine );

  for( Int i = 0; i < ntot; i++ ){
    if( fft.gkkFine(i) == 0 ){
      fft.outputComplexVecFine(i) = Z_ZERO;
    }
    else{
      // NOTE: gkk already contains the factor 1/2.
      fft.outputComplexVecFine(i) *= 2.0 * PI / fft.gkkFine(i);
    }
  }

  FFTWExecute ( fft, fft.backwardPlanFine );

  for( Int i = 0; i < ntot; i++ ){
    vhart_(i) = fft.inputComplexVecFine(i).real();
  }

  return; 
}  // -----  end of method KohnSham::CalculateHartree ----- 


void
KohnSham::CalculateVtot    ( DblNumVec& vtot )
{
  Int ntot = domain_.NumGridTotalFine();
  for (int i=0; i<ntot; i++) {
    vtot(i) = vext_(i) + vhart_(i) + vxc_(i, RHO);
  }


  return ;
}         // -----  end of method KohnSham::CalculateVtot  ----- 


void
KohnSham::CalculateForce    ( Spinor& psi, Fourier& fft  )
{

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

  FFTWExecute ( fft, fft.forwardPlanFine );

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

    FFTWExecute ( fft, fft.backwardPlanFine );

    vhart.Resize( ntot );

    for( Int i = 0; i < ntot; i++ ){
      vhart(i) = fft.inputComplexVecFine(i).real();
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

    FFTWExecute ( fft, fft.backwardPlanFine );

    // vhartDrv saves the derivative of the Hartree potential
    vhartDrv[d].Resize( ntot );

    for( Int i = 0; i < ntot; i++ ){
      vhartDrv[d](i) = fft.inputComplexVecFine(i).real();
    }

  } // for (d)


  // *********************************************************************
  // Compute the force from local pseudopotential
  // *********************************************************************
  // Method 1: Using the derivative of the pseudopotential
  if(0){
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
  // This formulation must be used when ONCV pseudopotential is used.
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

      FFTWExecute ( fft, fft.forwardPlanFine );

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

        FFTWExecute ( fft, fft.backwardPlanFine );

        for( Int i = 0; i < ntotFine; i++ ){
          vlocDrv[d](i) = fft.inputComplexVecFine(i).real();
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
      ErrorHandling( "numEig != numStateTotal in CalculateForce" );
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
  // FIXME: Assuming real arithmetic is used here.
  // This formulation must be used when ONCV pseudopotential is used.
  if(1)
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
      ErrorHandling( "numEig != numStateTotal in CalculateForce" );
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

          FFTWExecute ( fft, fft.forwardPlan );

          // fft Coarse to Fine 

          SetValue( fft.outputComplexVecFine, Z_ZERO );
          for( Int i = 0; i < domain_.NumGridTotal(); i++ ){
            fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i) *
              sqrt( double(domain_.NumGridTotal()) / double(domain_.NumGridTotalFine()) );
          }

          FFTWExecute ( fft, fft.backwardPlanFine );

          for( Int i = 0; i < domain_.NumGridTotalFine(); i++ ){
            wfnFine(i) = fft.inputComplexVecFine(i).real();
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



  return ;
}         // -----  end of method KohnSham::CalculateForce  ----- 


void
KohnSham::CalculateForce2    ( Spinor& psi, Fourier& fft  )
{

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
  // Compute the force from local pseudopotential
  // *********************************************************************
  // Method 2: Using integration by parts for local pseudopotential.
  // No need to evaluate the derivative of the local pseudopotential.
  // This could potentially save some coding effort, and perhaps better for other 
  // pseudopotential such as Troullier-Martins
  GetTime( timeSta );
  if(1)
  {
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
    FFTWExecute ( fft, fft.forwardPlanFine );
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
      FFTWExecute ( fft, fft.backwardPlanFine );
      GetTime( timeFFTEnd );
      timeFFTTotal += timeFFTEnd - timeFFTSta;

      // vhartDrv saves the derivative of the Hartree potential
      vhartDrv[d].Resize( ntotFine );

      for( Int i = 0; i < ntotFine; i++ ){
        vhartDrv[d](i) = fft.inputComplexVecFine(i).real();
      }

    } // for (d)


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



  // Method 4 (2016/11/20): Remove the local contribution that involving
  // overlapping pseudocharge contribution on the same atom.
  // 
  // THIS HAS NO EFFECT AT ALL ON THE FINAL RESULT!!
  if(0)
  {
    for (Int a=0; a<numAtom; a++) {
      PseudoPot& pp = pseudo_[a];
      SparseVec& sp = pp.pseudoCharge;
      IntNumVec& idx = sp.first;
      DblNumMat& val = sp.second;


      std::vector<DblNumVec>  vhartDrv(DIM);

      DblNumVec  totalCharge(ntotFine);
      SetValue( totalCharge, 0.0 );

      // totalCharge = density_ - pseudoCharge_
      blas::Copy( ntotFine, density_.VecData(0), 1, totalCharge.Data(), 1 );
      blas::Axpy( ntotFine, -1.0, pseudoCharge_.Data(),1,
          totalCharge.Data(), 1 );

      // Add back the contribution from local pseudocharge
      for (Int k=0; k<idx.m(); k++) 
        totalCharge[idx(k)] += val(k, VAL);

      // Total charge in the Fourier space
      CpxNumVec  totalChargeFourier( ntotFine );

      for( Int i = 0; i < ntotFine; i++ ){
        fft.inputComplexVecFine(i) = Complex( totalCharge(i), 0.0 );
      }

      GetTime( timeFFTSta );
      FFTWExecute ( fft, fft.forwardPlanFine );
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
        FFTWExecute ( fft, fft.backwardPlanFine );
        GetTime( timeFFTEnd );
        timeFFTTotal += timeFFTEnd - timeFFTSta;

        // vhartDrv saves the derivative of the Hartree potential
        vhartDrv[d].Resize( ntotFine );

        for( Int i = 0; i < ntotFine; i++ ){
          vhartDrv[d](i) = fft.inputComplexVecFine(i).real();
        }

      } // for (d)



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

  if(0){
    // Output the local component of the force for debugging purpose
    for( Int a = 0; a < numAtom; a++ ){
      Point3 ft(force(a,0),force(a,1),force(a,2));
      Print( statusOFS, "atom", a, "localforce ", ft );
    }
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
      FFTWExecute ( fft, fft.forwardPlan );
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
      FFTWExecute ( fft, fft.backwardPlanFine );
      GetTime( timeFFTEnd );
      timeFFTTotal += timeFFTEnd - timeFFTSta;

      Real fac = sqrt(double(domain_.NumGridTotal())) / 
        sqrt( double(domain_.NumGridTotalFine()) ); 
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
        FFTWExecute ( fft, fft.backwardPlanFine );
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



  return ;
}         // -----  end of method KohnSham::CalculateForce2  ----- 
#ifdef GPU
void
KohnSham::MultSpinor    ( Spinor& psi, cuNumTns<Real>& a3, Fourier& fft )
{

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;
  
  //SetValue( a3, 0.0 );

  GetTime( timeSta );
  psi.AddMultSpinorFineR2C( fft, vtot_, pseudo_, a3 );
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for psi.AddMultSpinorFineR2C is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


  return ;
}         // -----  end of method KohnSham::MultSpinor  ----- 


#endif
void
KohnSham::MultSpinor    ( Spinor& psi, NumTns<Real>& a3, Fourier& fft )
{

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Real>& wavefun = psi.Wavefun();
  Int ncom = wavefun.n();

  Int ntotR2C = fft.numGridTotalR2C;

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  Real timeGemm = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeAllreduce = 0.0;

  SetValue( a3, 0.0 );

  // Apply an initial filter on the wavefunctions, if required
  if((apply_filter_ == 1 && apply_first_ == 1))
  {

    //statusOFS << std::endl << " In here in 1st filter : " << wfn_cutoff_ << std::endl; 
    apply_first_ = 0;

    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {

        SetValue( fft.inputVecR2C, 0.0 );
        SetValue( fft.outputVecR2C, Z_ZERO );

        blas::Copy( ntot, wavefun.VecData(j,k), 1,
            fft.inputVecR2C.Data(), 1 );
        FFTWExecute ( fft, fft.forwardPlanR2C ); // So outputVecR2C contains the FFT result now


        for (Int i=0; i<ntotR2C; i++)
        {
          if(fft.gkkR2C(i) > wfn_cutoff_)
            fft.outputVecR2C(i) = Z_ZERO;
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );
        blas::Copy( ntot,  fft.inputVecR2C.Data(), 1,
            wavefun.VecData(j,k), 1 );

      }
    }
  }


  GetTime( timeSta );
  psi.AddMultSpinorFineR2C( fft, vtot_, pseudo_, a3 );
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for psi.AddMultSpinorFineR2C is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  if( isHybrid_ && isEXXActive_ ){

    GetTime( timeSta );

    if( esdfParam.isHybridACE ){

      //      if(0)
      //      {
      //        DblNumMat M(numStateTotal, numStateTotal);
      //        blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, 1.0,
      //            vexxProj_.Data(), ntot, psi.Wavefun().Data(), ntot, 
      //            0.0, M.Data(), M.m() );
      //        // Minus sign comes from that all eigenvalues are negative
      //        blas::Gemm( 'N', 'N', ntot, numStateTotal, numStateTotal, -1.0,
      //            vexxProj_.Data(), ntot, M.Data(), numStateTotal,
      //            1.0, a3.Data(), ntot );
      //      }

      if(1){ // for MPI
        // Convert the column partition to row partition

        Int numStateBlocksize = numStateTotal / mpisize;
        Int ntotBlocksize = ntot / mpisize;

        Int numStateLocal = numStateBlocksize;
        Int ntotLocal = ntotBlocksize;

        if(mpirank < (numStateTotal % mpisize)){
          numStateLocal = numStateBlocksize + 1;
        }

        if(mpirank < (ntot % mpisize)){
          ntotLocal = ntotBlocksize + 1;
        }

        DblNumMat psiCol( ntot, numStateLocal );
        SetValue( psiCol, 0.0 );

        DblNumMat vexxProjCol( ntot, numStateLocal );
        SetValue( vexxProjCol, 0.0 );

        DblNumMat psiRow( ntotLocal, numStateTotal );
        SetValue( psiRow, 0.0 );

        DblNumMat vexxProjRow( ntotLocal, numStateTotal );
        SetValue( vexxProjRow, 0.0 );

        lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
        lapack::Lacpy( 'A', ntot, numStateLocal, vexxProj_.Data(), ntot, vexxProjCol.Data(), ntot );

        GetTime( timeSta1 );
        AlltoallForward (psiCol, psiRow, domain_.comm);
        AlltoallForward (vexxProjCol, vexxProjRow, domain_.comm);
        GetTime( timeEnd1 );
        timeAlltoallv = timeAlltoallv + ( timeEnd1 - timeSta1 );

        DblNumMat MTemp( numStateTotal, numStateTotal );
        SetValue( MTemp, 0.0 );

        GetTime( timeSta1 );
        blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal,
            1.0, vexxProjRow.Data(), ntotLocal, 
            psiRow.Data(), ntotLocal, 0.0,
            MTemp.Data(), numStateTotal );
        GetTime( timeEnd1 );
        timeGemm = timeGemm + ( timeEnd1 - timeSta1 );

        DblNumMat M(numStateTotal, numStateTotal);
        SetValue( M, 0.0 );
        GetTime( timeSta1 );
        MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );
        GetTime( timeEnd1 );
        timeAllreduce = timeAllreduce + ( timeEnd1 - timeSta1 );

        DblNumMat a3Col( ntot, numStateLocal );
        SetValue( a3Col, 0.0 );

        DblNumMat a3Row( ntotLocal, numStateTotal );
        SetValue( a3Row, 0.0 );

        GetTime( timeSta1 );
        blas::Gemm( 'N', 'N', ntotLocal, numStateTotal, numStateTotal, 
            -1.0, vexxProjRow.Data(), ntotLocal, 
            M.Data(), numStateTotal, 0.0, 
            a3Row.Data(), ntotLocal );
        GetTime( timeEnd1 );
        timeGemm = timeGemm + ( timeEnd1 - timeSta1 );

        GetTime( timeSta1 );
        AlltoallBackward (a3Row, a3Col, domain_.comm);
        GetTime( timeEnd1 );
        timeAlltoallv = timeAlltoallv + ( timeEnd1 - timeSta1 );

        GetTime( timeSta1 );
        for (Int k=0; k<numStateLocal; k++) {
          for (Int j=0; j<ncom; j++) {
            Real *p1 = a3Col.VecData(k);
            Real *p2 = a3.VecData(j, k);
            for (Int i=0; i<ntot; i++) { 
              *(p2++) += *(p1++); 
            }
          }
        }
        GetTime( timeEnd1 );
        timeGemm = timeGemm + ( timeEnd1 - timeSta1 );

      } //if(1)

    }
    else{
      psi.AddMultSpinorEXX( fft, phiEXX_, exxgkkR2C_,
          exxFraction_,  numSpin_, occupationRate_, a3 );
    }

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for updating hybrid Spinor is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
    statusOFS << "Time for Gemm is " <<
      timeGemm << " [s]" << std::endl << std::endl;
    statusOFS << "Time for Alltoallv is " <<
      timeAlltoallv << " [s]" << std::endl << std::endl;
    statusOFS << "Time for Allreduce is " <<
      timeAllreduce << " [s]" << std::endl << std::endl;
#endif


  }

  // Apply filter on the wavefunctions before exit, if required
  if((apply_filter_ == 1))
  {
    //statusOFS << std::endl << " In here in 2nd filter : "  << wfn_cutoff_<< std::endl; 
    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {

        SetValue( fft.inputVecR2C, 0.0 );
        SetValue( fft.outputVecR2C, Z_ZERO );

        blas::Copy( ntot, a3.VecData(j,k), 1,
            fft.inputVecR2C.Data(), 1 );
        FFTWExecute ( fft, fft.forwardPlanR2C ); // So outputVecR2C contains the FFT result now


        for (Int i=0; i<ntotR2C; i++)
        {
          if(fft.gkkR2C(i) > wfn_cutoff_)
            fft.outputVecR2C(i) = Z_ZERO;
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );
        blas::Copy( ntot,  fft.inputVecR2C.Data(), 1,
            a3.VecData(j,k), 1 );

      }
    }
  }



  return ;
}         // -----  end of method KohnSham::MultSpinor  ----- 




//void
//KohnSham::MultSpinor    ( Int iocc, Spinor& psi, NumMat<Real>& y, Fourier& fft )
//{
//  // Make sure that the address corresponding to the pointer y has been
//  // allocated.
//  SetValue( y, 0.0 );
//
//    psi.AddRealDiag( iocc, vtotCoarse_, y );
//    psi.AddLaplacian( iocc, &fft, y );
//  psi.AddNonlocalPP( iocc, pseudo_, y );
//
//
//    return ;
//}         // -----  end of method KohnSham::MultSpinor  ----- 


void KohnSham::InitializeEXX ( Real ecutWavefunction, Fourier& fft )
{
  const Real epsDiv = 1e-8;

  isEXXActive_ = false;

  Int numGridTotalR2C = fft.numGridTotalR2C;
  exxgkkR2C_.Resize(numGridTotalR2C);
  SetValue( exxgkkR2C_, 0.0 );


  // extra 2.0 factor for ecutWavefunction compared to QE due to unit difference
  // tpiba2 in QE is just a unit for G^2. Do not include it here
  Real exxAlpha = 10.0 / (ecutWavefunction * 2.0);

  // Gygi-Baldereschi regularization. Currently set to zero and compare
  // with QE without the regularization 
  // Set exxdiv_treatment to "none"
  // NOTE: I do not quite understand the detailed derivation
  // Compute the divergent term for G=0
  Real gkk2;
  if(exxDivergenceType_ == 0){
    exxDiv_ = 0.0;
  }
  else if (exxDivergenceType_ == 1){
    exxDiv_ = 0.0;
    // no q-point
    // NOTE: Compared to the QE implementation, it is easier to do below.
    // Do the integration over the entire G-space rather than just the
    // R2C grid. This is because it is an integration in the G-space.
    // This implementation fully agrees with the QE result.
    for( Int ig = 0; ig < fft.numGridTotal; ig++ ){
      gkk2 = fft.gkk(ig) * 2.0;
      if( gkk2 > epsDiv ){
        if( screenMu_ > 0.0 ){
          exxDiv_ += std::exp(-exxAlpha * gkk2) / gkk2 * 
            (1.0 - std::exp(-gkk2 / (4.0*screenMu_*screenMu_)));
        }
        else{
          exxDiv_ += std::exp(-exxAlpha * gkk2) / gkk2;
        }
      }
    } // for (ig)

    if( screenMu_ > 0.0 ){
      exxDiv_ += 1.0 / (4.0*screenMu_*screenMu_);
    }
    else{
      exxDiv_ -= exxAlpha;
    }
    exxDiv_ *= 4.0 * PI;


    Int nqq = 100000;
    Real dq = 5.0 / std::sqrt(exxAlpha) / nqq;
    Real aa = 0.0;
    Real qt, qt2;
    for( Int iq = 0; iq < nqq; iq++ ){
      qt = dq * (iq+0.5);
      qt2 = qt*qt;
      if( screenMu_ > 0.0 ){
        aa -= std::exp(-exxAlpha *qt2) * 
          std::exp(-qt2 / (4.0*screenMu_*screenMu_)) * dq;
      }
    }
    aa = aa * 2.0 / PI + 1.0 / std::sqrt(exxAlpha*PI);
    exxDiv_ -= domain_.Volume()*aa;
  }

  if(1){
    statusOFS << "computed exxDiv_ = " << exxDiv_ << std::endl;
  }


  for( Int ig = 0; ig < numGridTotalR2C; ig++ ){
    gkk2 = fft.gkkR2C(ig) * 2.0;
    if( gkk2 > epsDiv ){
      if( screenMu_ > 0 ){
        // 2.0*pi instead 4.0*pi due to gkk includes a factor of 2
        exxgkkR2C_[ig] = 4.0 * PI / gkk2 * (1.0 - 
            std::exp( -gkk2 / (4.0*screenMu_*screenMu_) ));
      }
      else{
        exxgkkR2C_[ig] = 4.0 * PI / gkk2;
      }
    }
    else{
      exxgkkR2C_[ig] = -exxDiv_;
      if( screenMu_ > 0 ){
        exxgkkR2C_[ig] += PI / (screenMu_*screenMu_);
      }
    }
  } // for (ig)


  if(1){
    statusOFS << "Hybrid mixing parameter  = " << exxFraction_ << std::endl; 
    statusOFS << "Hybrid screening length = " << screenMu_ << std::endl;
  }


  return ;
}        // -----  end of function KohnSham::InitializeEXX  ----- 

void
KohnSham::SetPhiEXX    (const Spinor& psi, Fourier& fft)
{
  // FIXME collect Psi into a globally shared array in the MPI context.
  const NumTns<Real>& wavefun = psi.Wavefun();
  Int ntot = wavefun.m();
  Int ncom = wavefun.n();
  Int numStateLocal = wavefun.p();
  Int numStateTotal = this->NumStateTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Real vol = fft.domain.Volume();

  phiEXX_.Resize( ntot, ncom, numStateLocal );
  SetValue( phiEXX_, 0.0 );

  // FIXME Put in a more proper place
  for (Int k=0; k<numStateLocal; k++) {
    for (Int j=0; j<ncom; j++) {

      Real fac = std::sqrt( double(ntot) / vol );
      blas::Copy( ntot, wavefun.VecData(j,k), 1, phiEXX_.VecData(j,k), 1 );
      blas::Scal( ntot, fac, phiEXX_.VecData(j,k), 1 );

      if(0){
        DblNumVec psiTemp(ntot);
        blas::Copy( ntot, phiEXX_.VecData(j,k), 1, psiTemp.Data(), 1 );
        statusOFS << "int (phi^2) dx = " << Energy(psiTemp)*vol / double(ntot) << std::endl;
      }

    } // for (j)
  } // for (k)


  return ;
}         // -----  end of method KohnSham::SetPhiEXX  ----- 


void
KohnSham::CalculateVexxACE ( Spinor& psi, Fourier& fft )
{
  // This assumes SetPhiEXX has been called so that phiEXX and psi
  // contain the same information. 

  // Since this is a projector, it should be done on the COARSE grid,
  // i.e. to the wavefunction directly

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  // Only works for single processor
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Real>  vexxPsi( ntot, 1, numStateLocal );

  // VexxPsi = V_{exx}*Phi.
  SetValue( vexxPsi, 0.0 );
  psi.AddMultSpinorEXX( fft, phiEXX_, exxgkkR2C_,
      exxFraction_,  numSpin_, occupationRate_, vexxPsi );

  // Implementation based on SVD
  DblNumMat  M(numStateTotal, numStateTotal);

  if(0){
    // FIXME
    Real SVDTolerance = 1e-4;
    // M = Phi'*vexxPsi
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
  }

  // Implementation based on Cholesky
  if(0){
    // M = -Phi'*vexxPsi. The minus sign comes from vexx is a negative
    // semi-definite matrix.
    blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, 
        -1.0, psi.Wavefun().Data(), ntot, vexxPsi.Data(), ntot,
        0.0, M.Data(), numStateTotal );

    lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);

    blas::Trsm( 'R', 'L', 'T', 'N', ntot, numStateTotal, 1.0, 
        M.Data(), numStateTotal, vexxPsi.Data(), ntot );

    vexxProj_.Resize( ntot, numStateTotal );
    blas::Copy( ntot * numStateTotal, vexxPsi.Data(), 1, vexxProj_.Data(), 1 );
  }

  if(1){ //For MPI

    // Convert the column partition to row partition
    Int numStateBlocksize = numStateTotal / mpisize;
    Int ntotBlocksize = ntot / mpisize;

    Int numStateLocal = numStateBlocksize;
    Int ntotLocal = ntotBlocksize;

    if(mpirank < (numStateTotal % mpisize)){
      numStateLocal = numStateBlocksize + 1;
    }

    if(mpirank < (ntot % mpisize)){
      ntotLocal = ntotBlocksize + 1;
    }

    DblNumMat localPsiCol( ntot, numStateLocal );
    SetValue( localPsiCol, 0.0 );

    DblNumMat localVexxPsiCol( ntot, numStateLocal );
    SetValue( localVexxPsiCol, 0.0 );

    DblNumMat localPsiRow( ntotLocal, numStateTotal );
    SetValue( localPsiRow, 0.0 );

    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    SetValue( localVexxPsiRow, 0.0 );

    // Initialize
    lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, localPsiCol.Data(), ntot );
    lapack::Lacpy( 'A', ntot, numStateLocal, vexxPsi.Data(), ntot, localVexxPsiCol.Data(), ntot );

    AlltoallForward (localPsiCol, localPsiRow, domain_.comm);
    AlltoallForward (localVexxPsiCol, localVexxPsiRow, domain_.comm);

    DblNumMat MTemp( numStateTotal, numStateTotal );
    SetValue( MTemp, 0.0 );

    blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal,
        -1.0, localPsiRow.Data(), ntotLocal, 
        localVexxPsiRow.Data(), ntotLocal, 0.0,
        MTemp.Data(), numStateTotal );

    SetValue( M, 0.0 );
    MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

    if ( mpirank == 0) {
      lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);
    }

    MPI_Bcast(M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, 0, domain_.comm);

    blas::Trsm( 'R', 'L', 'T', 'N', ntotLocal, numStateTotal, 1.0, 
        M.Data(), numStateTotal, localVexxPsiRow.Data(), ntotLocal );

    vexxProj_.Resize( ntot, numStateLocal );

    AlltoallBackward (localVexxPsiRow, vexxProj_, domain_.comm);
  } //if(1)

  // Sanity check. For debugging only
  //  if(0){
  //  // Make sure U and VT are the same. Should be an identity matrix
  //    blas::Gemm( 'N', 'N', numStateTotal, numStateTotal, numStateTotal, 1.0, 
  //        VT.Data(), numStateTotal, U.Data(), numStateTotal, 0.0,
  //        M.Data(), numStateTotal );
  //    statusOFS << "M = " << M << std::endl;
  //
  //    NumTns<Real> vpsit = psi.Wavefun();
  //    Int numProj = rankM;
  //    DblNumMat Mt(numProj, numStateTotal);
  //    
  //    blas::Gemm( 'T', 'N', numProj, numStateTotal, ntot, 1.0,
  //        vexxProj_.Data(), ntot, psi.Wavefun().Data(), ntot, 
  //        0.0, Mt.Data(), Mt.m() );
  //    // Minus sign comes from that all eigenvalues are negative
  //    blas::Gemm( 'N', 'N', ntot, numStateTotal, numProj, -1.0,
  //        vexxProj_.Data(), ntot, Mt.Data(), numProj,
  //        0.0, vpsit.Data(), ntot );
  //
  //    for( Int k = 0; k < numStateTotal; k++ ){
  //      Real norm = 0.0;
  //      for( Int ir = 0; ir < ntot; ir++ ){
  //        norm = norm + std::pow(vexxPsi(ir,0,k) - vpsit(ir,0,k), 2.0);
  //      }
  //      statusOFS << "Diff of vexxPsi " << std::sqrt(norm) << std::endl;
  //    }
  //  }


  return ;
}         // -----  end of method KohnSham::CalculateVexxACE  ----- 


void
KohnSham::CalculateVexxACEDF ( Spinor& psi, Fourier& fft, bool isFixColumnDF )
{
  // This assumes SetPhiEXX has been called so that phiEXX and psi
  // contain the same information. 

  // Since this is a projector, it should be done on the COARSE grid,
  // i.e. to the wavefunction directly

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  // Only works for single processor
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Real>  vexxPsi( ntot, 1, numStateLocal );

  // VexxPsi = V_{exx}*Phi.
  DblNumMat  M(numStateTotal, numStateTotal);
  SetValue( vexxPsi, 0.0 );
  SetValue( M, 0.0 );
  // M = -Phi'*vexxPsi. The minus sign comes from vexx is a negative
  // semi-definite matrix.
  psi.AddMultSpinorEXXDF4( fft, phiEXX_, exxgkkR2C_, exxFraction_,  numSpin_, 
      occupationRate_, numMuHybridDF_, numGaussianRandomHybridDF_,
      numProcScaLAPACKHybridDF_, BlockSizeScaLAPACK_,
      vexxPsi, M, isFixColumnDF );

  // Implementation based on Cholesky
  if(0){
    lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);

    blas::Trsm( 'R', 'L', 'T', 'N', ntot, numStateTotal, 1.0, 
        M.Data(), numStateTotal, vexxPsi.Data(), ntot );

    vexxProj_.Resize( ntot, numStateTotal );
    blas::Copy( ntot * numStateTotal, vexxPsi.Data(), 1, vexxProj_.Data(), 1 );
  }

  if(1){ //For MPI

    // Convert the column partition to row partition
    Int numStateBlocksize = numStateTotal / mpisize;
    Int ntotBlocksize = ntot / mpisize;

    Int numStateLocal = numStateBlocksize;
    Int ntotLocal = ntotBlocksize;

    if(mpirank < (numStateTotal % mpisize)){
      numStateLocal = numStateBlocksize + 1;
    }

    if(mpirank < (ntot % mpisize)){
      ntotLocal = ntotBlocksize + 1;
    }

    DblNumMat localVexxPsiCol( ntot, numStateLocal );
    SetValue( localVexxPsiCol, 0.0 );

    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    SetValue( localVexxPsiRow, 0.0 );

    // Initialize
    lapack::Lacpy( 'A', ntot, numStateLocal, vexxPsi.Data(), ntot, localVexxPsiCol.Data(), ntot );

    AlltoallForward (localVexxPsiCol, localVexxPsiRow, domain_.comm);

    if ( mpirank == 0) {
      lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);
    }

    MPI_Bcast(M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, 0, domain_.comm);

    blas::Trsm( 'R', 'L', 'T', 'N', ntotLocal, numStateTotal, 1.0, 
        M.Data(), numStateTotal, localVexxPsiRow.Data(), ntotLocal );

    vexxProj_.Resize( ntot, numStateLocal );

    AlltoallBackward (localVexxPsiRow, vexxProj_, domain_.comm);
  } //if(1)
  return ;
}         // -----  end of method KohnSham::CalculateVexxACEDF  ----- 


// This comes from exxenergy2() function in exx.f90 in QE.
Real
KohnSham::CalculateEXXEnergy    ( Spinor& psi, Fourier& fft )
{

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Real fockEnergy = 0.0;
  Real fockEnergyLocal = 0.0;

  // Repeat the calculation of Vexx
  // FIXME Will be replaced by the stored VPhi matrix in the new
  // algorithm to reduce the cost, but this should be a new function

  // FIXME Should be combined better with the addition of exchange part in spinor
  NumTns<Real>& wavefun = psi.Wavefun();

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Index3& numGrid = fft.domain.numGrid;
  Index3& numGridFine = fft.domain.numGridFine;
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  Real vol = fft.domain.Volume();
  Int ncom = wavefun.n();
  NumTns<Real>& phi = phiEXX_;
  Int ncomPhi = phi.n();
  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin polarized case not implemented.");
  }
  Int numStateLocalPhi = phi.p();

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  // Directly use the phiEXX_ and vexxProj_ to calculate the exchange energy
  if( esdfParam.isHybridACE ){
    // temporarily just implement here
    // Directly use projector
    Int numProj = vexxProj_.n();
    Int numStateTotal = this->NumStateTotal();
    Int ntot = psi.NumGridTotal();

    if(0)
    {
      DblNumMat M(numProj, numStateTotal);

      NumTns<Real>  vexxPsi( ntot, 1, numStateLocalPhi );
      SetValue( vexxPsi, 0.0 );

      blas::Gemm( 'T', 'N', numProj, numStateTotal, ntot, 1.0,
          vexxProj_.Data(), ntot, psi.Wavefun().Data(), ntot, 
          0.0, M.Data(), M.m() );
      // Minus sign comes from that all eigenvalues are negative
      blas::Gemm( 'N', 'N', ntot, numStateTotal, numProj, -1.0,
          vexxProj_.Data(), ntot, M.Data(), numProj,
          0.0, vexxPsi.Data(), ntot );

      for( Int k = 0; k < numStateLocalPhi; k++ ){
        for( Int j = 0; j < ncom; j++ ){
          for( Int ir = 0; ir < ntot; ir++ ){
            fockEnergy += vexxPsi(ir,j,k) * wavefun(ir,j,k) * occupationRate_[psi.WavefunIdx(k)];
          }
        }
      }

    }

    if(1) // For MPI
    {
      Int numStateBlocksize = numStateTotal / mpisize;
      Int ntotBlocksize = ntot / mpisize;

      Int numStateLocal = numStateBlocksize;
      Int ntotLocal = ntotBlocksize;

      if(mpirank < (numStateTotal % mpisize)){
        numStateLocal = numStateBlocksize + 1;
      }

      if(mpirank < (ntot % mpisize)){
        ntotLocal = ntotBlocksize + 1;
      }

      DblNumMat psiCol( ntot, numStateLocal );
      SetValue( psiCol, 0.0 );

      DblNumMat psiRow( ntotLocal, numStateTotal );
      SetValue( psiRow, 0.0 );

      DblNumMat vexxProjCol( ntot, numStateLocal );
      SetValue( vexxProjCol, 0.0 );

      DblNumMat vexxProjRow( ntotLocal, numStateTotal );
      SetValue( vexxProjRow, 0.0 );

      DblNumMat vexxPsiCol( ntot, numStateLocal );
      SetValue( vexxPsiCol, 0.0 );

      DblNumMat vexxPsiRow( ntotLocal, numStateTotal );
      SetValue( vexxPsiRow, 0.0 );

      lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
      lapack::Lacpy( 'A', ntot, numStateLocal, vexxProj_.Data(), ntot, vexxProjCol.Data(), ntot );

      AlltoallForward (psiCol, psiRow, domain_.comm);
      AlltoallForward (vexxProjCol, vexxProjRow, domain_.comm);

      DblNumMat MTemp( numStateTotal, numStateTotal );
      SetValue( MTemp, 0.0 );

      blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal,
          1.0, vexxProjRow.Data(), ntotLocal, 
          psiRow.Data(), ntotLocal, 0.0,
          MTemp.Data(), numStateTotal );

      DblNumMat M(numStateTotal, numStateTotal);
      SetValue( M, 0.0 );

      MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

      blas::Gemm( 'N', 'N', ntotLocal, numStateTotal, numStateTotal, -1.0,
          vexxProjRow.Data(), ntotLocal, M.Data(), numStateTotal,
          0.0, vexxPsiRow.Data(), ntotLocal );

      AlltoallBackward (vexxPsiRow, vexxPsiCol, domain_.comm);

      fockEnergy = 0.0;
      fockEnergyLocal = 0.0;

      for( Int k = 0; k < numStateLocal; k++ ){
        for( Int j = 0; j < ncom; j++ ){
          for( Int ir = 0; ir < ntot; ir++ ){
            fockEnergyLocal += vexxPsiCol(ir,k) * wavefun(ir,j,k) * occupationRate_[psi.WavefunIdx(k)];
          }
        }
      }
      mpi::Allreduce( &fockEnergyLocal, &fockEnergy, 1, MPI_SUM, domain_.comm );
    } //if(1) 
  }
  else{
    NumTns<Real>  vexxPsi( ntot, 1, numStateLocalPhi );
    SetValue( vexxPsi, 0.0 );
    psi.AddMultSpinorEXX( fft, phiEXX_, exxgkkR2C_, 
        exxFraction_,  numSpin_, occupationRate_, 
        vexxPsi );
    // Compute the exchange energy:
    // Note: no additional normalization factor due to the
    // normalization rule of psi, NOT phi!!
    fockEnergy = 0.0;
    fockEnergyLocal = 0.0;
    for( Int k = 0; k < numStateLocalPhi; k++ ){
      for( Int j = 0; j < ncom; j++ ){
        for( Int ir = 0; ir < ntot; ir++ ){
          fockEnergyLocal += vexxPsi(ir,j,k) * wavefun(ir,j,k) * occupationRate_[psi.WavefunIdx(k)];
        }
      }
    }
    mpi::Allreduce( &fockEnergyLocal, &fockEnergy, 1, MPI_SUM, domain_.comm );
  }


  return fockEnergy;
}         // -----  end of method KohnSham::CalculateEXXEnergy  ----- 



//void
//KohnSham::UpdateHybrid ( Int phiIter, const Spinor& psi, Fourier& fft, Real Efock )
//{
//
//
//    return ;
//}        // -----  end of function KohnSham::UpdateHybrid  ----- 
//
} // namespace dgdft
