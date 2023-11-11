//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin, Wei Hu

/// @file spinor.cpp
/// @brief Spinor (wavefunction) for the global domain or extended
/// element.
/// @date 2012-10-06
#include  "spinor.h"
#include  "utility.h"
#include  <blas.hh>
#include  <lapack.hh>
#include "linalg_extensions.h"
#include  "scalapack.h"
#include  "mpi_interf.h"

#include "block_distributor_decl.h"

namespace scales{

using namespace scales::scalapack;

using namespace scales::PseudoComponent;

Spinor::Spinor () { }         
Spinor::~Spinor    () {}

Spinor::Spinor ( 
    const Domain &dm, 
    const Int     numComponent,
    const Int     numStateTotal,
    Int     numStateLocal,
    const Real  val ) {
  this->Setup( dm, numComponent, numStateTotal, numStateLocal, val );

}         // -----  end of method Spinor::Spinor  ----- 

Spinor::Spinor ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    Int numStateLocal,
    const bool owndata, 
    Real* data )
{
  this->Setup( dm, numComponent, numStateTotal, numStateLocal, owndata, data );

}         // -----  end of method Spinor::Spinor  ----- 

void Spinor::Setup ( 
    const Domain &dm, 
    const Int     numComponent,
    const Int     numStateTotal,
    Int     numStateLocal,
    const Real  val ) {

  domain_       = dm;
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int blocksize;

  if ( numStateTotal <=  mpisize ) {
    blocksize = 1;
  }
  else {  // numStateTotal >  mpisize
    if ( numStateTotal % mpisize == 0 ){
      blocksize = numStateTotal / mpisize;
    }
    else {
      blocksize = ((numStateTotal - 1) / mpisize) + 1;
    }    
  }

  numStateTotal_ = numStateTotal;
  blocksize_ = blocksize;

  wavefunIdx_.Resize( numStateLocal );
  SetValue( wavefunIdx_, 0 );
  for (Int i = 0; i < numStateLocal; i++){
    wavefunIdx_[i] = i * mpisize + mpirank ;
  }

  wavefun_.Resize( dm.NumGridTotal(), numComponent, numStateLocal );
  SetValue( wavefun_, val );

}         // -----  end of method Spinor::Setup  ----- 

void Spinor::Setup ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    Int numStateLocal,
    const bool owndata, 
    Real* data )
{

  domain_       = dm;
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  wavefun_      = NumTns<Real>( dm.NumGridTotal(), numComponent, numStateLocal,
      owndata, data );

  Int blocksize;

  if ( numStateTotal <=  mpisize ) {
    blocksize = 1;
  }
  else {  // numStateTotal >  mpisize
    if ( numStateTotal % mpisize == 0 ){
      blocksize = numStateTotal / mpisize;
    }
    else {
      blocksize = ((numStateTotal - 1) / mpisize) + 1;
    }    
  }

  numStateTotal_ = numStateTotal;
  blocksize_ = blocksize;

  wavefunIdx_.Resize( numStateLocal );
  SetValue( wavefunIdx_, 0 );
  for (Int i = 0; i < numStateLocal; i++){
    wavefunIdx_[i] = i * mpisize + mpirank ;
  }

}         // -----  end of method Spinor::Setup  ----- 

void
Spinor::Normalize    ( )
{
  Int size = wavefun_.m() * wavefun_.n();
  Int nocc = wavefun_.p();

  for (Int k=0; k<nocc; k++) {
    Real *ptr = wavefun_.MatData(k);
    Real   sum = 0.0;
    for (Int i=0; i<size; i++) {
      sum += pow(abs(*ptr++), 2.0);
    }
    sum = sqrt(sum);
    if (sum != 0.0) {
      ptr = wavefun_.MatData(k);
      for (Int i=0; i<size; i++) *(ptr++) /= sum;
    }
  }
  return ;
}         // -----  end of method Spinor::Normalize  ----- 


void
Spinor::AddTeterPrecond (Fourier* fftPtr, NumTns<Real>& Hpsi)
{
  Fourier& fft = *fftPtr;
  if( !fftPtr->isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int nocc = wavefun_.p();

  if( fftPtr->domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  //#ifdef _USE_OPENMP_
  //#pragma omp parallel
  //  {
  //#endif
  Int ntothalf = fftPtr->numGridTotalR2C;
  // These two are private variables in the OpenMP context

  //#ifdef _USE_OPENMP_
  //#pragma omp for schedule (dynamic,1) nowait
  //#endif
  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
      // For c2r and r2c transforms, the default is to DESTROY the
      // input, therefore a copy of the original matrix is necessary. 
      blas::copy( ntot, wavefun_.VecData(j,k), 1, 
          reinterpret_cast<Real*>(fft.inputVecR2C.Data()), 1 );

      FFTWExecute ( fft, fft.forwardPlanR2C );

      Real*    ptr1d   = fftPtr->TeterPrecondR2C.Data();
      Complex* ptr2    = fft.outputVecR2C.Data();
      for (Int i=0; i<ntothalf; i++) 
        *(ptr2++) *= *(ptr1d++);

      FFTWExecute ( fft, fft.backwardPlanR2C);

      blas::axpy( ntot, 1.0, fft.inputVecR2C.Data(), 1, Hpsi.VecData(j,k), 1 );
    }
  }
  //#ifdef _USE_OPENMP_
  //  }
  //#endif


  return ;
}         // -----  end of method Spinor::AddTeterPrecond ----- 

void
Spinor::AddMultSpinor ( Fourier& fft, const DblNumVec& vtot, 
    const std::vector<PseudoPot>& pseudo, NumTns<Real>& Hpsi )
{

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  auto ntot = wavefun_.m();
  auto ncom = wavefun_.n();
  auto numStateLocal = wavefun_.p();
  auto ntotFine = domain_.NumGridTotalFine();
  auto vol = domain_.Volume();

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  // Temporary variable for saving wavefunction on the fine grid
  DblNumVec psiFine(ntotFine);
  DblNumVec psiUpdateFine(ntotFine);

  for (auto k=0; k<numStateLocal; k++) {
    for (auto j=0; j<ncom; j++) {

      SetValue( psiFine, 0.0 );
      SetValue( psiUpdateFine, 0.0 );

      SetValue( fft.inputComplexVec, Z_ZERO );
      blas::copy( ntot, wavefun_.VecData(j,k), 1,
          reinterpret_cast<Real*>(fft.inputComplexVec.Data()), 2 );

      // Fourier transform of wavefunction saved in fft.outputComplexVec
      FFTWExecute( fft, fft.forwardPlan );

      // Interpolate wavefunction from coarse to fine grid
      {
        auto fac = sqrt( double(ntot) / double(ntotFine) );
        SetValue( fft.outputComplexVecFine, Z_ZERO ); 
        auto idxPtr = fft.idxFineGrid.Data();
        auto fftOutFinePtr = fft.outputComplexVecFine.Data();
        auto fftOutPtr = fft.outputComplexVec.Data();
        for( auto i = 0; i < ntot; i++ ){
          fftOutFinePtr[idxPtr[i]] = fftOutPtr[i] * fac;
        }
        FFTWExecute( fft, fft.backwardPlanFine );

        blas::copy( ntotFine, reinterpret_cast<Real*>(fft.inputComplexVecFine.Data()),
            2, psiFine.Data(), 1 );
      }

      // Apply the local potential
      {
        auto psiUpdateFinePtr = psiUpdateFine.Data();
        auto psiFinePtr = psiFine.Data();
        auto vtotPtr = vtot.Data();
        for( auto i = 0; i < ntotFine; i++ ){
          psiUpdateFinePtr[i] += psiFinePtr[i] * vtotPtr[i];
        }
      }

      // Add the contribution from nonlocal pseudopotential
      {
        auto natm = pseudo.size();
        for (auto iatm=0; iatm<natm; iatm++) {
          auto nobt = pseudo[iatm].vnlList.size();
          for (auto iobt=0; iobt<nobt; iobt++) {
            auto& vnlwgt = pseudo[iatm].vnlList[iobt].second;
            auto& vnlvec = pseudo[iatm].vnlList[iobt].first;
            auto& iv = vnlvec.first;
            auto& dv = vnlvec.second;

            auto weight = 0.0; 
            auto ivptr = iv.Data();
            auto dvptr = dv.VecData(VAL);
            for (auto i=0; i<iv.m(); i++) {
              weight += dvptr[i] * psiFine[ivptr[i]];
            }
            weight *= vol/Real(ntotFine)*vnlwgt;

            ivptr = iv.Data();
            dvptr = dv.VecData(VAL);
            for (auto i=0; i<iv.m(); i++) {
              psiUpdateFine[ivptr[i]] += dvptr[i] * weight;
            }
          } // for (iobt)
        } // for (iatm)
      }

      // Laplacian operator. Perform inverse Fourier transform in the end
      {
        for (auto i=0; i<ntot; i++) 
          fft.outputComplexVec[i] *= fft.gkk[i];
      }

      // Restrict psiUpdateFine from fine grid in the real space to
      // coarse grid in the Fourier space. Combine with the Laplacian contribution

      // Fine to coarse grid
      // Note the update is important since the Laplacian contribution is already taken into account.
      // The computation order is also important
      {
        SetValue( fft.inputComplexVecFine, Z_ZERO ); // Do not forget this
        blas::copy( ntotFine, psiUpdateFine.Data(), 1,
            reinterpret_cast<Real*>(fft.inputComplexVecFine.Data()), 2 );
        FFTWExecute( fft, fft.forwardPlanFine );

        auto fac = sqrt( double(ntotFine) / double(ntot) );
        auto idxPtr = fft.idxFineGrid.Data();
        auto fftOutFinePtr = fft.outputComplexVecFine.Data();
        auto fftOutPtr = fft.outputComplexVec.Data();

        for( auto i = 0; i < ntot; i++ ){
          fftOutPtr[i] += fftOutFinePtr[idxPtr[i]] * fac;
        }
      }

      // Inverse Fourier transform to save back to the output vector
      FFTWExecute( fft, fft.backwardPlan );

      blas::axpy( ntot, 1.0, reinterpret_cast<Real*>(fft.inputComplexVec.Data()), 2,
          Hpsi.VecData(j,k), 1 );
    }
  }

  return ;
}        // -----  end of method Spinor::AddMultSpinor  ----- 


// LL: 1/3/2021
// This function requires the coarse grid to be an odd number along each
// direction, and therefore should be deprecated in the future.
/*
void
Spinor::AddMultSpinorR2C ( Fourier& fft, const DblNumVec& vtot, 
    const std::vector<PseudoPot>& pseudo, NumTns<Real>& Hpsi )
{

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int ntotFine = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  // Temporary variable for saving wavefunction on a fine grid
  DblNumVec psiFine(ntotFine);
  DblNumVec psiUpdateFine(ntotFine);

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;
  
  Real timeFFTCoarse = 0.0;
  Real timeFFTFine = 0.0;
  Real timeNonlocal = 0.0;
  Real timeOther = 0.0;
  Int  iterFFTCoarse = 0;
  Int  iterFFTFine = 0;
  Int  iterNonlocal = 0;
  Int  iterOther = 0;

  GetTime( timeSta1 );
 


  //#ifdef _USE_OPENMP_
  //#pragma omp parallel
  //    {
  //#endif
  for (Int k=0; k<numStateLocal; k++) {
    for (Int j=0; j<ncom; j++) {

      SetValue( psiFine, 0.0 );
      SetValue( psiUpdateFine, 0.0 );

      // R2C version
      if(1)
      {
        SetValue( fft.inputVecR2C, 0.0 ); 
        SetValue( fft.inputVecR2CFine, 0.0 ); 
        SetValue( fft.outputVecR2C, Z_ZERO ); 
        SetValue( fft.outputVecR2CFine, Z_ZERO ); 


        // For c2r and r2c transforms, the default is to DESTROY the
        // input, therefore a copy of the original vector is necessary. 
        blas::copy( ntot, wavefun_.VecData(j,k), 1, 
            fft.inputVecR2C.Data(), 1 );

        GetTime( timeSta );
        FFTWExecute ( fft, fft.forwardPlanR2C );
        GetTime( timeEnd );
        iterFFTCoarse = iterFFTCoarse + 1;
        timeFFTCoarse = timeFFTCoarse + ( timeEnd - timeSta );

        // statusOFS << std::endl << " Input vec = " << fft.inputVecR2C << std::endl;
        // statusOFS << std::endl << " Output vec = " << fft.outputVecR2C << std::endl;


        // Interpolate wavefunction from coarse to fine grid
        {
          Real fac = sqrt( double(ntot) / double(ntotFine) );
          Int *idxPtr = fft.idxFineGridR2C.Data();
          Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
          Complex *fftOutPtr = fft.outputVecR2C.Data();
          for( Int i = 0; i < ntotR2C; i++ ){
            fftOutFinePtr[*(idxPtr++)] = *(fftOutPtr++) * fac;
          }
        }

        GetTime( timeSta );
        FFTWExecute ( fft, fft.backwardPlanR2CFine );
        GetTime( timeEnd );
        iterFFTFine = iterFFTFine + 1;
        timeFFTFine = timeFFTFine + ( timeEnd - timeSta );

        blas::copy( ntotFine, fft.inputVecR2CFine.Data(), 1, psiFine.Data(), 1 );

      }  // if (1)

      // Add the contribution from local pseudopotential
      //      for( Int i = 0; i < ntotFine; i++ ){
      //        psiUpdateFine(i) += psiFine(i) * vtot(i);
      //      }
      {
        Real *psiUpdateFinePtr = psiUpdateFine.Data();
        Real *psiFinePtr = psiFine.Data();
        Real *vtotPtr = vtot.Data();
        for( Int i = 0; i < ntotFine; i++ ){
          *(psiUpdateFinePtr++) += *(psiFinePtr++) * *(vtotPtr++);
        }
      }

      // Add the contribution from nonlocal pseudopotential
      GetTime( timeSta );
      if(1){
        Int natm = pseudo.size();
        for (Int iatm=0; iatm<natm; iatm++) {
          Int nobt = pseudo[iatm].vnlList.size();
          for (Int iobt=0; iobt<nobt; iobt++) {
            const Real       vnlwgt = pseudo[iatm].vnlList[iobt].second;
            const SparseVec &vnlvec = pseudo[iatm].vnlList[iobt].first;
            const IntNumVec &iv = vnlvec.first;
            const DblNumMat &dv = vnlvec.second;

            Real    weight = 0.0; 
            const Int    *ivptr = iv.Data();
            const Real   *dvptr = dv.VecData(VAL);
            for (Int i=0; i<iv.m(); i++) {
              weight += (*(dvptr++)) * psiFine[*(ivptr++)];
            }
            weight *= vol/Real(ntotFine)*vnlwgt;

            ivptr = iv.Data();
            dvptr = dv.VecData(VAL);
            for (Int i=0; i<iv.m(); i++) {
              psiUpdateFine[*(ivptr++)] += (*(dvptr++)) * weight;
            }
          } // for (iobt)
        } // for (iatm)
      }
      GetTime( timeEnd );
      iterNonlocal = iterNonlocal + 1;
      timeNonlocal = timeNonlocal + ( timeEnd - timeSta );


      // Laplacian operator. Perform inverse Fourier transform in the end
      {
        for (Int i=0; i<ntotR2C; i++) 
          fft.outputVecR2C(i) *= fft.gkkR2C(i);
      }

      // Restrict psiUpdateFine from fine grid in the real space to
      // coarse grid in the Fourier space. Combine with the Laplacian contribution
      //      for( Int i = 0; i < ntotFine; i++ ){
      //        fft.inputComplexVecFine(i) = Complex( psiUpdateFine(i), 0.0 ); 
      //      }
      //SetValue( fft.inputComplexVecFine, Z_ZERO );
      SetValue( fft.inputVecR2CFine, 0.0 );
      blas::copy( ntotFine, psiUpdateFine.Data(), 1, fft.inputVecR2CFine.Data(), 1 );

      // Fine to coarse grid
      // Note the update is important since the Laplacian contribution is already taken into account.
      // The computation order is also important
      // fftw_execute( fft.forwardPlanFine );
      GetTime( timeSta );
      FFTWExecute ( fft, fft.forwardPlanR2CFine );
      GetTime( timeEnd );
      iterFFTFine = iterFFTFine + 1;
      timeFFTFine = timeFFTFine + ( timeEnd - timeSta );
      //      {
      //        Real fac = std::sqrt(Real(ntot) / (Real(ntotFine)));
      //        Int* idxPtr = fft.idxFineGrid.Data();
      //        Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
      //        Complex *fftOutPtr = fft.outputComplexVec.Data();
      //
      //        for( Int i = 0; i < ntot; i++ ){
      //          //          fft.outputComplexVec(i) += fft.outputComplexVecFine(fft.idxFineGrid(i)) * fac;
      //          *(fftOutPtr++) += fftOutFinePtr[*(idxPtr++)] * fac;
      //        }
      //      }


      {
        Real fac = sqrt( double(ntotFine) / double(ntot) );
        Int *idxPtr = fft.idxFineGridR2C.Data();
        Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
        Complex *fftOutPtr = fft.outputVecR2C.Data();
        for( Int i = 0; i < ntotR2C; i++ ){
          *(fftOutPtr++) += fftOutFinePtr[*(idxPtr++)] * fac;
        }
      }

      GetTime( timeSta );
      FFTWExecute ( fft, fft.backwardPlanR2C );
      GetTime( timeEnd );
      iterFFTCoarse = iterFFTCoarse + 1;
      timeFFTCoarse = timeFFTCoarse + ( timeEnd - timeSta );

      // Inverse Fourier transform to save back to the output vector
      //fftw_execute( fft.backwardPlan );

      blas::axpy( ntot, 1.0, fft.inputVecR2C.Data(), 1, Hpsi.VecData(j,k), 1 );

    } // j++
  } // k++
  //#ifdef _USE_OPENMP_
  //    }
  //#endif


  GetTime( timeEnd1 );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd1 - timeSta1 ) - timeFFTCoarse - timeFFTFine - timeNonlocal;

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for iterFFTCoarse    = " << iterFFTCoarse       << "  timeFFTCoarse    = " << timeFFTCoarse << std::endl;
    statusOFS << "Time for iterFFTFine      = " << iterFFTFine         << "  timeFFTFine      = " << timeFFTFine << std::endl;
    statusOFS << "Time for iterNonlocal     = " << iterNonlocal        << "  timeNonlocal     = " << timeNonlocal << std::endl;
    statusOFS << "Time for iterOther        = " << iterOther           << "  timeOther        = " << timeOther << std::endl;
#endif

  return ;
}        // -----  end of method Spinor::AddMultSpinorR2C  ----- 
*/

void Spinor::AddMultSpinorEXX ( Fourier& fft, 
    const NumTns<Real>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Real  numSpin,
    const DblNumVec& occupationRate,
    NumTns<Real>& Hpsi )
{
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;

  Int ntot     = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();
  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int numStateTotal = numStateTotal_;

  Int ncomPhi = phi.n();

  Real vol = domain_.Volume();

  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin polarized case not implemented.");
  }

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  // Temporary variable for saving wavefunction on a fine grid
  DblNumVec phiTemp(ntot);

  Int numStateLocalTemp;

  MPI_Barrier(domain_.comm);

  for( Int iproc = 0; iproc < mpisize; iproc++ ){

    if( iproc == mpirank )
      numStateLocalTemp = numStateLocal;

    MPI_Bcast( &numStateLocalTemp, 1, MPI_INT, iproc, domain_.comm );

    IntNumVec wavefunIdxTemp(numStateLocalTemp);
    if( iproc == mpirank ){
      wavefunIdxTemp = wavefunIdx_;
    }

    MPI_Bcast( wavefunIdxTemp.Data(), numStateLocalTemp, MPI_INT, iproc, domain_.comm );

    // FIXME OpenMP does not work since all variables are shared
    for( Int kphi = 0; kphi < numStateLocalTemp; kphi++ ){
      for( Int jphi = 0; jphi < ncomPhi; jphi++ ){

        SetValue( phiTemp, 0.0 );

        if( iproc == mpirank )
        { 
          Real* phiPtr = phi.VecData(jphi, kphi);
          for( Int ir = 0; ir < ntot; ir++ ){
            phiTemp(ir) = phiPtr[ir];
          }
        }

        MPI_Bcast( phiTemp.Data(), ntot, MPI_DOUBLE, iproc, domain_.comm );

        for (Int k=0; k<numStateLocal; k++) {
          for (Int j=0; j<ncom; j++) {

            Real* psiPtr = wavefun_.VecData(j,k);
            for( Int ir = 0; ir < ntot; ir++ ){
              fft.inputVecR2C(ir) = psiPtr[ir] * phiTemp(ir);
            }

            FFTWExecute ( fft, fft.forwardPlanR2C );

            // Solve the Poisson-like problem for exchange
            for( Int ig = 0; ig < ntotR2C; ig++ ){
              fft.outputVecR2C(ig) *= exxgkkR2C(ig);
            }

            FFTWExecute ( fft, fft.backwardPlanR2C );

            Real* HpsiPtr = Hpsi.VecData(j,k);
            Real fac = -exxFraction * occupationRate[wavefunIdxTemp(kphi)];  
            for( Int ir = 0; ir < ntot; ir++ ){
              HpsiPtr[ir] += fft.inputVecR2C(ir) * phiTemp(ir) * fac;
            }

          } // for (j)
        } // for (k)

        MPI_Barrier(domain_.comm);


      } // for (jphi)
    } // for (kphi)

  } //iproc

  MPI_Barrier(domain_.comm);


  return ;
}        // -----  end of method Spinor::AddMultSpinorEXX  ----- 


// This is the new density matrix based algorithm for compressing the Coulomb integrals
void Spinor::AddMultSpinorEXXDF ( Fourier& fft, 
    const NumTns<Real>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Real  numSpin,
    const DblNumVec& occupationRate,
    const Real numMuFac,
    const Real numGaussianRandomFac,
    const Int numProcScaLAPACKPotrf,  
    const Int scaPotrfBlockSize,  
    NumTns<Real>& Hpsi, 
    NumMat<Real>& VxMat,
    bool isFixColumnDF )
{
  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;

  Int ntot     = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();
  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int numStateTotal = numStateTotal_;

  Int ncomPhi = phi.n();

  Real vol = domain_.Volume();

  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin polarized case not implemented.");
  }

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  // Pre-processing. Perform SCDM to align the orbitals into localized orbitals
  // This assumes that the phi and psi orbitals are the same
  IntNumVec permPhi(ntot);
  DblNumMat Q(numStateTotal, numStateTotal);
  DblNumMat phiSave(ntot, numStateTotal);
  DblNumMat psiSave(ntot, numStateTotal);
  lapack::lacpy( lapack::MatrixType::General, ntot, numStateTotal, phi.Data(), ntot, phiSave.Data(), ntot );
  lapack::lacpy( lapack::MatrixType::General, ntot, numStateTotal, wavefun_.Data(), ntot, psiSave.Data(), ntot );

  if(1){
    if( mpisize > 1 )
      ErrorHandling("Only works for mpisize == 1.");

    DblNumMat phiT;
    Transpose( DblNumMat(ntot, numStateTotal, false, phi.Data()), phiT );

    // SCDM using sequential QRCP
    DblNumMat R(numStateTotal, ntot);
    QRCP( numStateTotal, ntot, phiT.Data(), Q.Data(), R.Data(), numStateTotal, 
        permPhi.Data() );

    // Make a copy before GEMM

    // Rotate phi
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, ntot, numStateTotal, numStateTotal, 1.0, 
        phiSave.Data(), ntot, Q.Data(), numStateTotal, 0.0,
        phi.Data(), ntot );
    // Rotate psi
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, ntot, numStateTotal, numStateTotal, 1.0, 
        psiSave.Data(), ntot, Q.Data(), numStateTotal, 0.0,
        wavefun_.Data(), ntot );
  }



  if(1){ //For MPI

    // *********************************************************************
    // Perform interpolative separable density fitting
    // *********************************************************************

    //numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
    numMu_ = IRound(numStateTotal*numMuFac);

    /// @todo The factor 2.0 is hard coded.  The PhiG etc should in
    /// principle be a tensor, but only treated as matrix.
    //Int numPre = std::min(IRound(std::sqrt(numMu_*2.0)), numStateTotal);
    //Int numPre = std::min(IRound(std::sqrt(numMu_))+5, numStateTotal);
    //Int numPre = IRound(std::sqrt(numMu_*numGaussianRandomFac));
    if( IRound(std::sqrt(numMu_*numGaussianRandomFac)) > numStateTotal ){
      ErrorHandling("numMu is too large for interpolative separable density fitting!");
    }

    Int numPre = numMuFac*numGaussianRandomFac;

    statusOFS << "numMu  = " << numMu_ << std::endl;
    statusOFS << "numPre*numStateTotal = " << numPre*numStateTotal << std::endl;

    // Convert the column partition to row partition
    Int numStateBlocksize = numStateTotal / mpisize;
    Int ntotBlocksize = ntot / mpisize;

    Int numMuBlocksize = numMu_ / mpisize;

    Int numStateLocal = numStateBlocksize;
    Int ntotLocal = ntotBlocksize;

    Int numMuLocal = numMuBlocksize;

    if(mpirank < (numStateTotal % mpisize)){
      numStateLocal = numStateBlocksize + 1;
    }

    if(mpirank < (numMu_ % mpisize)){
      numMuLocal = numMuBlocksize + 1;
    }

    if(mpirank < (ntot % mpisize)){
      ntotLocal = ntotBlocksize + 1;
    }

    DblNumMat localVexxPsiCol( ntot, numStateLocal );
    SetValue( localVexxPsiCol, 0.0 );

    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    SetValue( localVexxPsiRow, 0.0 );

    //DblNumMat localphiGRow( ntotLocal, numPre );
    //SetValue( localphiGRow, 0.0 );

    DblNumMat localpsiGRow( ntotLocal, numPre );
    SetValue( localpsiGRow, 0.0 );

    DblNumMat G( numStateTotal, numPre );
    SetValue( G, 0.0 );

    DblNumMat phiCol( ntot, numStateLocal );
    SetValue( phiCol, 0.0 );
    DblNumMat phiRow( ntotLocal, numStateTotal );
    SetValue( phiRow, 0.0 );

    DblNumMat psiCol( ntot, numStateLocal );
    SetValue( psiCol, 0.0 );
    DblNumMat psiRow( ntotLocal, numStateTotal );
    SetValue( psiRow, 0.0 );

    lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, phi.Data(), ntot, phiCol.Data(), ntot );
    lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, wavefun_.Data(), ntot, psiCol.Data(), ntot );

    auto bdist = 
      make_block_distributor<double>( BlockDistAlg::HostGeneric, domain_.comm,
                                      ntot, numStateTotal );
    bdist.redistribute_col_to_row( phiCol, phiRow );
    bdist.redistribute_col_to_row( psiCol, psiRow );
    

    // Computing the indices is optional

    Int ntotLocalMG, ntotMG;

    if( (ntot % mpisize) == 0 ){
      ntotLocalMG = ntotBlocksize;
    }
    else{
      ntotLocalMG = ntotBlocksize + 1;
    }


    if( isFixColumnDF == false ){
      GetTime( timeSta );

      // Step 1: Pre-compression of the wavefunctions. This uses
      // multiplication with orthonormalized random Gaussian matrices
      if ( mpirank == 0) {
        GaussianRandom(G);
        Orth( numStateTotal, numPre, G.Data(), numStateTotal );
      }

      MPI_Bcast(G.Data(), numStateTotal * numPre, MPI_DOUBLE, 0, domain_.comm);

      //blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numPre, numStateTotal, 1.0, 
      //    phiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
      //    localphiGRow.Data(), ntotLocal );

      blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numPre, numStateTotal, 1.0, 
          psiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
          localpsiGRow.Data(), ntotLocal );

      // Step 2: Pivoted QR decomposition  for the Hadamard product of
      // the compressed matrix. Transpose format for QRCP

      // NOTE: All processors should have the same ntotLocalMG
      ntotMG = ntotLocalMG * mpisize;



      DblNumMat MG( numStateTotal*numPre, ntotLocalMG );
      SetValue( MG, 0.0 );

      if(1){

        for( Int j = 0; j < numPre; j++ ){
          for( Int i = 0; i < numStateTotal; i++ ){
            for( Int ir = 0; ir < ntotLocal; ir++ ){
              MG(i+j*numStateTotal,ir) = phiRow(ir,i) * ( localpsiGRow(ir,j) - psiRow(ir,i) * G(i, j) );
            }
          }
        }

      } // if(1)

      DblNumVec tau(ntotMG);
      pivQR_.Resize(ntotMG);
      SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
      // Q factor does not need to be used

      Real timeQRCPSta, timeQRCPEnd;
      GetTime( timeQRCPSta );

      if(1){ // ScaLAPACL QRCP
        Int contxt;
        Int nprow, npcol, myrow, mycol, info;
        Cblacs_get(0, 0, &contxt);
        nprow = 1;
        npcol = mpisize;

        Cblacs_gridinit(&contxt, "C", nprow, npcol);
        Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);
        Int desc_MG[9];
        Int desc_QR[9];

        Int irsrc = 0;
        Int icsrc = 0;

        Int mb_MG = numStateTotal*numPre;
        Int nb_MG = ntotLocalMG;

        // FIXME The current routine does not actually allow ntotLocal to be different on different processors.
        // This must be fixed.
        SCALAPACK(descinit)(&desc_MG[0], &mb_MG, &ntotMG, &mb_MG, &nb_MG, &irsrc, 
            &icsrc, &contxt, &mb_MG, &info);

        IntNumVec pivQRTmp(ntotMG), pivQRLocal(ntotMG);
        if( mb_MG > ntot ){
          std::ostringstream msg;
          msg << "numStateTotal*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
          ErrorHandling( msg.str().c_str() );
        }
        // DiagR is only for debugging purpose
        //        DblNumVec diagRLocal( mb_MG );
        //        DblNumVec diagR( mb_MG );

        SetValue( pivQRTmp, 0 );
        SetValue( pivQRLocal, 0 );
        SetValue( pivQR_, 0 );

        //        SetValue( diagRLocal, 0.0 );
        //        SetValue( diagR, 0.0 );

        scalapack::QRCPF( mb_MG, ntotMG, MG.Data(), &desc_MG[0], 
            pivQRTmp.Data(), tau.Data() );

        //        scalapack::QRCPR( mb_MG, ntotMG, numMu_, MG.Data(), &desc_MG[0], 
        //            pivQRTmp.Data(), tau.Data(), 80, 40 );

        // Combine the local pivQRTmp to global pivQR_
        for( Int j = 0; j < ntotLocalMG; j++ ){
          pivQRLocal[j + mpirank * ntotLocalMG] = pivQRTmp[j];
        }

        //        std::cout << "diag of MG = " << std::endl;
        //        if(mpirank == 0){
        //          std::cout << pivQRLocal << std::endl;
        //          for( Int j = 0; j < mb_MG; j++ ){
        //            std::cout << MG(j,j) << std::endl;
        //          }
        //        }
        MPI_Allreduce( pivQRLocal.Data(), pivQR_.Data(), 
            ntotMG, MPI_INT, MPI_SUM, domain_.comm );

        if(contxt >= 0) {
          Cblacs_gridexit( contxt );
        }
      } //

      GetTime( timeQRCPEnd );

      //statusOFS << std::endl<< "All processors exit with abort in spinor.cpp." << std::endl;

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for QRCP alone is " <<
        timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;
#endif


      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for density fitting with QRCP is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    }


    // *********************************************************************
    // Compute the interpolation matrix via the density matrix formulation
    // *********************************************************************

    GetTime( timeSta );

    DblNumMat XiRow(ntotLocal, numMu_);
    DblNumMat psiMu(numStateTotal, numMu_);
    // PhiMu is scaled by the occupation number to reflect the "true" density matrix
    DblNumMat PcolPhiMu(ntotLocal, numMu_);
    IntNumVec pivMu(numMu_);


    for( Int mu = 0; mu < numMu_; mu++ ){
      pivMu(mu) = pivQR_(mu);
    }

    // These three matrices are used only once. 
    // Used before reduce
    DblNumMat psiMuRow(numStateTotal, numMu_);
    DblNumMat phiMuRow(numStateTotal, numMu_);
    //DblNumMat PcolMuNuRow(numMu_, numMu_);
    DblNumMat PcolPsiMuRow(ntotLocal, numMu_);

    // Collecting the matrices obtained from row partition
    DblNumMat phiMu(numStateTotal, numMu_);
    DblNumMat PcolMuNu(numMu_, numMu_);
    DblNumMat PcolPsiMu(ntotLocal, numMu_);

    SetValue( psiMuRow, 0.0 );
    SetValue( phiMuRow, 0.0 );
    //SetValue( PcolMuNuRow, 0.0 );
    SetValue( PcolPsiMuRow, 0.0 );

    SetValue( phiMu, 0.0 );
    SetValue( PcolMuNu, 0.0 );
    SetValue( PcolPsiMu, 0.0 );

    GetTime( timeSta1 );

    for( Int mu = 0; mu < numMu_; mu++ ){
      Int muInd = pivMu(mu);
      // NOTE Hard coded here with the row partition strategy
      if( muInd <  mpirank * ntotLocalMG ||
          muInd >= (mpirank + 1) * ntotLocalMG )
        continue;

      Int muIndRow = muInd - mpirank * ntotLocalMG;

      for (Int k=0; k<numStateTotal; k++) {
        psiMuRow(k, mu) = psiRow(muIndRow,k);
        phiMuRow(k, mu) = phiRow(muIndRow,k) * occupationRate[k];
      }
    }

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing the MuRow is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    MPI_Allreduce( psiMuRow.Data(), psiMu.Data(), 
        numStateTotal * numMu_, MPI_DOUBLE, MPI_SUM, domain_.comm );
    MPI_Allreduce( phiMuRow.Data(), phiMu.Data(), 
        numStateTotal * numMu_, MPI_DOUBLE, MPI_SUM, domain_.comm );

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for MPI_Allreduce is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numMu_, numStateTotal, 1.0, 
        psiRow.Data(), ntotLocal, psiMu.Data(), numStateTotal, 0.0,
        PcolPsiMu.Data(), ntotLocal );
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numMu_, numStateTotal, 1.0, 
        phiRow.Data(), ntotLocal, phiMu.Data(), numStateTotal, 0.0,
        PcolPhiMu.Data(), ntotLocal );

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for GEMM is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    DblNumMat psiphiRow(ntotLocal, numStateTotal);
    SetValue( psiphiRow, 0.0 );

    for( Int ir = 0; ir < ntotLocal; ir++ ){
      for (Int i=0; i<numStateTotal; i++) {
        psiphiRow(ir,i) = psiRow(ir,i) * phiRow(ir, i); 
      }
    }       

    DblNumMat psiphiMu(numStateTotal, numMu_);
    SetValue( psiphiMu, 0.0 );

    for( Int i = 0; i < numStateTotal; i++ ){
      for (Int j = 0; j < numMu_; j++) {
        psiphiMu(i,j) = psiMu(i,j) * phiMu(i,j); 
      }
    }       

    DblNumMat PcolPsiPhiMu(ntotLocal, numMu_);
    SetValue( PcolPsiPhiMu, 0.0 );

    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numMu_, numStateTotal, 1.0, 
        psiphiRow.Data(), ntotLocal, psiphiMu.Data(), numStateTotal, 0.0,
        PcolPsiPhiMu.Data(), ntotLocal );

    Real* xiPtr = XiRow.Data();
    Real* PcolPsiMuPtr = PcolPsiMu.Data();
    Real* PcolPhiMuPtr = PcolPhiMu.Data();
    Real* PcolPsiPhiMuPtr = PcolPsiPhiMu.Data();

    GetTime( timeSta1 );

    for( Int g = 0; g < ntotLocal * numMu_; g++ ){
      xiPtr[g] = PcolPsiMuPtr[g] * PcolPhiMuPtr[g] - PcolPsiPhiMuPtr[g];
    }

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for xiPtr is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif
    {
      GetTime( timeSta1 );

      DblNumMat PcolMuNuRow(numMu_, numMu_);
      SetValue( PcolMuNuRow, 0.0 );

      for( Int mu = 0; mu < numMu_; mu++ ){

        Int muInd = pivMu(mu);
        // NOTE Hard coded here with the row partition strategy
        if( muInd <  mpirank * ntotLocalMG ||
            muInd >= (mpirank + 1) * ntotLocalMG )
          continue;
        Int muIndRow = muInd - mpirank * ntotLocalMG;

        for (Int nu=0; nu < numMu_; nu++) {
          PcolMuNuRow( mu, nu ) = XiRow( muIndRow, nu );
        }

      }//for mu

      GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for PcolMuNuRow is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeSta1 );

      MPI_Allreduce( PcolMuNuRow.Data(), PcolMuNu.Data(), 
          numMu_* numMu_, MPI_DOUBLE, MPI_SUM, domain_.comm );

      GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for MPI_Allreduce is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    }

    GetTime( timeSta1 );

    if(1){ // Parallel Portf

      Int contxt;
      Int nprow, npcol, myrow, mycol, info;
      Cblacs_get(0, 0, &contxt);

      for( Int i = IRound(sqrt(double(numProcScaLAPACKPotrf))); 
          i <= numProcScaLAPACKPotrf; i++){
        nprow = i; npcol = numProcScaLAPACKPotrf / nprow;
        if( nprow * npcol == numProcScaLAPACKPotrf ) break;
      }

      IntNumVec pmap(numProcScaLAPACKPotrf);
      // Take the first numProcScaLAPACK processors for diagonalization
      for ( Int i = 0; i < numProcScaLAPACKPotrf; i++ ){
        pmap[i] = i;
      }

      Cblacs_gridmap(&contxt, &pmap[0], nprow, nprow, npcol);

      if( contxt >= 0 ){

        Int numKeep = numMu_; 
        Int lda = numMu_;

        scalapack::ScaLAPACKMatrix<Real> square_mat_scala;

        scalapack::Descriptor descReduceSeq, descReducePar;

        // Leading dimension provided
        descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt, lda );

        // Automatically comptued Leading Dimension
        descReducePar.Init( numKeep, numKeep, scaPotrfBlockSize, scaPotrfBlockSize, I_ZERO, I_ZERO, contxt );

        square_mat_scala.SetDescriptor( descReducePar );

        DblNumMat&  square_mat = PcolMuNu;
        // Redistribute the input matrix over the process grid
        SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
            &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt );

        // Make the ScaLAPACK call
        char LL = 'L';
        //SCALAPACK(pdpotrf)(&LL, &numMu_, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &info);
        scalapack::Potrf(LL, square_mat_scala);

        // Redistribute back eigenvectors
        SetValue(square_mat, 0.0 );
        SCALAPACK(pdgemr2d)( &numKeep, &numKeep, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
            square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt );

      } // if(contxt >= 0)

      if(contxt >= 0) {
        Cblacs_gridexit( contxt );
      }

    } // if(1) for Parallel Portf

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for Potrf is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    { 

      GetTime( timeSta1 );

      MPI_Bcast(PcolMuNu.Data(), numMu_ * numMu_, MPI_DOUBLE, 0, domain_.comm);

      GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for MPI_Bcast is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeSta1 );

    }

    blas::trsm( blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Lower, blas::Op::Trans, blas::Diag::NonUnit, ntotLocal, numMu_, 1.0, 
        PcolMuNu.Data(), numMu_, XiRow.Data(), ntotLocal );

    blas::trsm( blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Lower, blas::Op::NoTrans, blas::Diag::NonUnit, ntotLocal, numMu_, 1.0, 
        PcolMuNu.Data(), numMu_, XiRow.Data(), ntotLocal );

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for Trsm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif


    GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing the interpolation vectors is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // *********************************************************************
    // Solve the Poisson equations
    // Rewrite Xi by the potential of Xi
    // *********************************************************************

    DblNumMat XiCol(ntot, numMuLocal);

    bdist.redistribute_row_to_col( XiRow, XiCol );

    {
      GetTime( timeSta );
      for( Int mu = 0; mu < numMuLocal; mu++ ){
        blas::copy( ntot,  XiCol.VecData(mu), 1, fft.inputVecR2C.Data(), 1 );

        FFTWExecute ( fft, fft.forwardPlanR2C );

        for( Int ig = 0; ig < ntotR2C; ig++ ){
          fft.outputVecR2C(ig) *= -exxFraction * exxgkkR2C(ig);
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );

        blas::copy( ntot, fft.inputVecR2C.Data(), 1, XiCol.VecData(mu), 1 );

      } // for (mu)

      bdist.redistribute_col_to_row( XiCol, XiRow );

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for solving Poisson-like equations is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

    // *********************************************************************
    // Compute the exchange potential and the symmetrized inner product
    // *********************************************************************

    GetTime( timeSta );
    // Rewrite Xi by Xi.*PcolPhi
    DblNumMat HpsiRow( ntotLocal, numStateTotal );
    SetValue( HpsiRow, 0.0 );


    if(1){

      for (Int i = 0; i < numStateTotal; i++) {

        DblNumMat PcolPhiMui = PcolPhiMu;

        // Remove the self-contribution
        for (Int mu = 0; mu < numMu_; mu++) {
          blas::axpy( ntotLocal, -phiMu(i,mu), phiRow.VecData(i),
              1, PcolPhiMui.VecData(mu), 1 );
        }

        DblNumMat XiRowTemp(ntotLocal, numMu_);
        SetValue( XiRowTemp, 0.0 );
        lapack::lacpy( lapack::MatrixType::General, ntotLocal, numMu_, XiRow.Data(), ntotLocal, XiRowTemp.Data(), ntotLocal );

        Real* xiPtr = XiRowTemp.Data();
        Real* PcolPhiMuiPtr = PcolPhiMui.Data();
        for( Int g = 0; g < ntotLocal * numMu_; g++ ){
          xiPtr[g] *= PcolPhiMuiPtr[g];
        }


        for ( Int mu = 0; mu < numMu_; mu++ ){
          blas::axpy( ntotLocal, psiMu(i,mu), XiRowTemp.VecData(mu),
              1, HpsiRow.VecData(i), 1 );
        }


      } //end for i

    } //if(1)

    DblNumMat HpsiCol( ntot, numStateLocal );
    SetValue( HpsiCol, 0.0 );

    bdist.redistribute_row_to_col( HpsiRow, HpsiCol );

    if(1){

      for (Int i=0; i<numStateLocal; i++) {

        for( Int ir = 0; ir < ntot; ir++ ){
          fft.inputVecR2C(ir) = psiCol(ir, i) * phiCol(ir, i);
        }

        FFTWExecute ( fft, fft.forwardPlanR2C );

        // Solve the Poisson-like problem for exchange
        for( Int ig = 0; ig < ntotR2C; ig++ ){
          fft.outputVecR2C(ig) *= exxgkkR2C(ig);
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );

        Real fac = -exxFraction * occupationRate[wavefunIdx_(i)];  
        for( Int ir = 0; ir < ntot; ir++ ){
          HpsiCol(ir,i) += fft.inputVecR2C(ir) * phiCol(ir,i) * fac;
        }

      } // for i

    } //if(1)

    lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, HpsiCol.Data(), ntot, Hpsi.Data(), ntot );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing the exchange potential is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // Unrotate if SCDM is used
    if(1){
      lapack::lacpy( lapack::MatrixType::General, ntot, numStateTotal, phiSave.Data(), ntot, 
          phi.Data(), ntot );
      lapack::lacpy( lapack::MatrixType::General, ntot, numStateTotal, psiSave.Data(), ntot, 
          wavefun_.Data(), ntot );
    }





    // Compute the matrix VxMat = -Psi'* vexxPsi and symmetrize
    // vexxPsi (Hpsi) must be zero before entering this routine
    VxMat.Resize( numStateTotal, numStateTotal );
    {
      // Minus sign so that VxMat is positive semidefinite
      // NOTE: No measure factor vol / ntot due to the normalization
      // factor of psi
      DblNumMat VxMatTemp( numStateTotal, numStateTotal );
      SetValue( VxMatTemp, 0.0 );
      GetTime( timeSta );

      SetValue( HpsiRow, 0.0 );
      bdist.redistribute_col_to_row( HpsiCol, HpsiRow );

      blas::gemm( blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, numStateTotal, numStateTotal, ntotLocal, -1.0,
          psiRow.Data(), ntotLocal, HpsiRow.Data(), ntotLocal, 0.0, 
          VxMatTemp.Data(), numStateTotal );

      SetValue( VxMat, 0.0 );
      MPI_Allreduce( VxMatTemp.Data(), VxMat.Data(), numStateTotal * numStateTotal, 
          MPI_DOUBLE, MPI_SUM, domain_.comm );

      Symmetrize( VxMat );

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing VxMat in the sym format is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

  } //if(1) for For MPI

  MPI_Barrier(domain_.comm);

  return ;
}        // -----  end of method Spinor::AddMultSpinorEXXDF  ----- 



// 2D MPI communication for matrix
// Update: 6/26/2017
void Spinor::AddMultSpinorEXXDF6 ( Fourier& fft, 
    const NumTns<Real>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Real  numSpin,
    const DblNumVec& occupationRate,
    const Real numMuFac,
    const Real numGaussianRandomFac,
    const Int numProcScaLAPACK,  
    const Real hybridDFTolerance,
    const Int BlockSizeScaLAPACK,  
    NumTns<Real>& Hpsi, 
    NumMat<Real>& VxMat,
    bool isFixColumnDF )
{
  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;

  Int ntot     = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();
  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int numStateTotal = numStateTotal_;

  Int ncomPhi = phi.n();

  Real vol = domain_.Volume();

  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin polarized case not implemented.");
  }

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }


  // *********************************************************************
  // Perform interpolative separable density fitting
  // *********************************************************************

  //numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
  numMu_ = IRound(numStateTotal*numMuFac);

  Int numPre = IRound(std::sqrt(numMu_*numGaussianRandomFac));
  if( numPre > numStateTotal ){
    ErrorHandling("numMu is too large for interpolative separable density fitting!");
  }

  statusOFS << "ntot          = " << ntot << std::endl;
  statusOFS << "numMu         = " << numMu_ << std::endl;
  statusOFS << "numPre        = " << numPre << std::endl;
  statusOFS << "numPre*numPre = " << numPre * numPre << std::endl;

  // Convert the column partition to row partition
  Int numStateBlocksize = numStateTotal / mpisize;
  //Int ntotBlocksize = ntot / mpisize;

  Int numMuBlocksize = numMu_ / mpisize;

  Int numStateLocal1 = numStateBlocksize;
  //Int ntotLocal = ntotBlocksize;

  Int numMuLocal = numMuBlocksize;

  if(mpirank < (numStateTotal % mpisize)){
    numStateLocal1 = numStateBlocksize + 1;
  }

  if(numStateLocal !=  numStateLocal1){
    statusOFS << "numStateLocal = " << numStateLocal << " numStateLocal1 = " << numStateLocal1 << std::endl;
    ErrorHandling("The size is not right in interpolative separable density fitting!");
  }

  if(mpirank < (numMu_ % mpisize)){
    numMuLocal = numMuBlocksize + 1;
  }

  //if(mpirank < (ntot % mpisize)){
  //  ntotLocal = ntotBlocksize + 1;
  //}

  //huwei 
  //2D MPI commucation for all the matrix

  Int I_ONE = 1, I_ZERO = 0;
  double D_ONE = 1.0;
  double D_ZERO = 0.0;
  double D_MinusONE = -1.0;

  Int contxt0, contxt1, contxt11, contxt2;
  Int nprow0, npcol0, myrow0, mycol0, info0;
  Int nprow1, npcol1, myrow1, mycol1, info1;
  Int nprow11, npcol11, myrow11, mycol11, info11;
  Int nprow2, npcol2, myrow2, mycol2, info2;

  Int ncolsNgNe1DCol, nrowsNgNe1DCol, lldNgNe1DCol; 
  Int ncolsNgNe1DRow, nrowsNgNe1DRow, lldNgNe1DRow; 
  Int ncolsNgNe2D, nrowsNgNe2D, lldNgNe2D; 
  Int ncolsNgNu1D, nrowsNgNu1D, lldNgNu1D; 
  Int ncolsNgNu2D, nrowsNgNu2D, lldNgNu2D; 
  Int ncolsNuNg2D, nrowsNuNg2D, lldNuNg2D; 
  Int ncolsNeNe0D, nrowsNeNe0D, lldNeNe0D; 
  Int ncolsNeNe2D, nrowsNeNe2D, lldNeNe2D; 
  Int ncolsNuNu1D, nrowsNuNu1D, lldNuNu1D; 
  Int ncolsNuNu2D, nrowsNuNu2D, lldNuNu2D; 
  Int ncolsNeNu1D, nrowsNeNu1D, lldNeNu1D; 
  Int ncolsNeNu2D, nrowsNeNu2D, lldNeNu2D; 
  Int ncolsNuNe2D, nrowsNuNe2D, lldNuNe2D; 

  Int desc_NgNe1DCol[9];
  Int desc_NgNe1DRow[9];
  Int desc_NgNe2D[9];
  Int desc_NgNu1D[9];
  Int desc_NgNu2D[9];
  Int desc_NuNg2D[9];
  Int desc_NeNe0D[9];
  Int desc_NeNe2D[9];
  Int desc_NuNu1D[9];
  Int desc_NuNu2D[9];
  Int desc_NeNu1D[9];
  Int desc_NeNu2D[9];
  Int desc_NuNe2D[9];

  Int Ng = ntot;
  Int Ne = numStateTotal_; 
  Int Nu = numMu_; 

  // 0D MPI
  nprow0 = 1;
  npcol0 = mpisize;

  Cblacs_get(0, 0, &contxt0);
  Cblacs_gridinit(&contxt0, "C", nprow0, npcol0);
  Cblacs_gridinfo(contxt0, &nprow0, &npcol0, &myrow0, &mycol0);

  SCALAPACK(descinit)(desc_NeNe0D, &Ne, &Ne, &Ne, &Ne, &I_ZERO, &I_ZERO, &contxt0, &Ne, &info0);

  // 1D MPI
  nprow1 = 1;
  npcol1 = mpisize;

  Cblacs_get(0, 0, &contxt1);
  Cblacs_gridinit(&contxt1, "C", nprow1, npcol1);
  Cblacs_gridinfo(contxt1, &nprow1, &npcol1, &myrow1, &mycol1);

  //desc_NgNe1DCol
  if(contxt1 >= 0){
    nrowsNgNe1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
    ncolsNgNe1DCol = SCALAPACK(numroc)(&Ne, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNgNe1DCol = std::max( nrowsNgNe1DCol, 1 );
  }    

  SCALAPACK(descinit)(desc_NgNe1DCol, &Ng, &Ne, &Ng, &I_ONE, &I_ZERO, 
      &I_ZERO, &contxt1, &lldNgNe1DCol, &info1);

  nprow11 = mpisize;
  npcol11 = 1;

  Cblacs_get(0, 0, &contxt11);
  Cblacs_gridinit(&contxt11, "C", nprow11, npcol11);
  Cblacs_gridinfo(contxt11, &nprow11, &npcol11, &myrow11, &mycol11);

  Int BlockSizeScaLAPACKTemp = BlockSizeScaLAPACK; 
  //desc_NgNe1DRow
  if(contxt11 >= 0){
    nrowsNgNe1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACKTemp, &myrow11, &I_ZERO, &nprow11);
    ncolsNgNe1DRow = SCALAPACK(numroc)(&Ne, &Ne, &mycol11, &I_ZERO, &npcol11);
    lldNgNe1DRow = std::max( nrowsNgNe1DRow, 1 );
  }    

  SCALAPACK(descinit)(desc_NgNe1DRow, &Ng, &Ne, &BlockSizeScaLAPACKTemp, &Ne, &I_ZERO, 
      &I_ZERO, &contxt11, &lldNgNe1DRow, &info11);

  //desc_NeNu1D
  if(contxt11 >= 0){
    nrowsNeNu1D = SCALAPACK(numroc)(&Ne, &I_ONE, &myrow11, &I_ZERO, &nprow11);
    ncolsNeNu1D = SCALAPACK(numroc)(&Nu, &Nu, &mycol11, &I_ZERO, &npcol11);
    lldNeNu1D = std::max( nrowsNeNu1D, 1 );
  }    

  SCALAPACK(descinit)(desc_NeNu1D, &Ne, &Nu, &I_ONE, &Nu, &I_ZERO, 
      &I_ZERO, &contxt11, &lldNeNu1D, &info11);

  //desc_NgNu1D
  if(contxt1 >= 0){
    nrowsNgNu1D = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
    ncolsNgNu1D = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNgNu1D = std::max( nrowsNgNu1D, 1 );
  }    

  SCALAPACK(descinit)(desc_NgNu1D, &Ng, &Nu, &Ng, &I_ONE, &I_ZERO, 
      &I_ZERO, &contxt1, &lldNgNu1D, &info1);

  //desc_NuNu1D
  if(contxt1 >= 0){
    nrowsNuNu1D = SCALAPACK(numroc)(&Nu, &Nu, &myrow1, &I_ZERO, &nprow1);
    ncolsNuNu1D = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNuNu1D = std::max( nrowsNuNu1D, 1 );
  }    

  SCALAPACK(descinit)(desc_NuNu1D, &Nu, &Nu, &Nu, &I_ONE, &I_ZERO, 
      &I_ZERO, &contxt1, &lldNuNu1D, &info1);


  // 2D MPI
  for( Int i = IRound(sqrt(double(mpisize))); i <= mpisize; i++){
    nprow2 = i; npcol2 = mpisize / nprow2;
    if( (nprow2 >= npcol2) && (nprow2 * npcol2 == mpisize) ) break;
  }

  Cblacs_get(0, 0, &contxt2);
  //Cblacs_gridinit(&contxt2, "C", nprow2, npcol2);

  IntNumVec pmap2(mpisize);
  for ( Int i = 0; i < mpisize; i++ ){
    pmap2[i] = i;
  }
  Cblacs_gridmap(&contxt2, &pmap2[0], nprow2, nprow2, npcol2);

  Int mb2 = BlockSizeScaLAPACK;
  Int nb2 = BlockSizeScaLAPACK;

  //desc_NgNe2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNgNe2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNgNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNgNe2D = std::max( nrowsNgNe2D, 1 );
  }

  SCALAPACK(descinit)(desc_NgNe2D, &Ng, &Ne, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNgNe2D, &info2);

  //desc_NgNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNgNu2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNgNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNgNu2D = std::max( nrowsNgNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NgNu2D, &Ng, &Nu, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNgNu2D, &info2);

  //desc_NuNg2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNuNg2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNuNg2D = SCALAPACK(numroc)(&Ng, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNuNg2D = std::max( nrowsNuNg2D, 1 );
  }

  SCALAPACK(descinit)(desc_NuNg2D, &Nu, &Ng, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNuNg2D, &info2);

  //desc_NeNe2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNeNe2D = SCALAPACK(numroc)(&Ne, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNeNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNeNe2D = std::max( nrowsNeNe2D, 1 );
  }

  SCALAPACK(descinit)(desc_NeNe2D, &Ne, &Ne, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNeNe2D, &info2);

  //desc_NuNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNuNu2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNuNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNuNu2D = std::max( nrowsNuNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NuNu2D, &Nu, &Nu, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNuNu2D, &info2);

  //desc_NeNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNeNu2D = SCALAPACK(numroc)(&Ne, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNeNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNeNu2D = std::max( nrowsNeNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NeNu2D, &Ne, &Nu, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNeNu2D, &info2);

  //desc_NuNe2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNuNe2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNuNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNuNe2D = std::max( nrowsNuNe2D, 1 );
  }

  SCALAPACK(descinit)(desc_NuNe2D, &Nu, &Ne, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &contxt2, &lldNuNe2D, &info2);

  DblNumMat phiCol( ntot, numStateLocal );
  SetValue( phiCol, 0.0 );

  DblNumMat psiCol( ntot, numStateLocal );
  SetValue( psiCol, 0.0 );

  lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, phi.Data(), ntot, phiCol.Data(), ntot );
  lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, wavefun_.Data(), ntot, psiCol.Data(), ntot );

  // Computing the indices is optional


  Int ntotLocal = nrowsNgNe1DRow; 
  //Int ntotLocalMG = nrowsNgNe1DRow;
  //Int ntotMG = ntot;
  Int ntotLocalMG, ntotMG;

  //if( (ntot % mpisize) == 0 ){
  //  ntotLocalMG = ntotBlocksize;
  //}
  //else{
  //  ntotLocalMG = ntotBlocksize + 1;
  // }

  if( isFixColumnDF == false ){

    GetTime( timeSta );

    DblNumMat localphiGRow( ntotLocal, numPre );
    SetValue( localphiGRow, 0.0 );

    DblNumMat localpsiGRow( ntotLocal, numPre );
    SetValue( localpsiGRow, 0.0 );

    DblNumMat G(numStateTotal, numPre);
    SetValue( G, 0.0 );

    DblNumMat phiRow( ntotLocal, numStateTotal );
    SetValue( phiRow, 0.0 );

    DblNumMat psiRow( ntotLocal, numStateTotal );
    SetValue( psiRow, 0.0 );

    SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
        psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1 );

    SCALAPACK(pdgemr2d)(&Ng, &Ne, phiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
        phiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1 );

    // Step 1: Pre-compression of the wavefunctions. This uses
    // multiplication with orthonormalized random Gaussian matrices
    if ( mpirank == 0) {
      GaussianRandom(G);
      Orth( numStateTotal, numPre, G.Data(), numStateTotal );
    }

    GetTime( timeSta1 );

    MPI_Bcast(G.Data(), numStateTotal * numPre, MPI_DOUBLE, 0, domain_.comm);

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for Gaussian MPI_Bcast is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numPre, numStateTotal, 1.0, 
        phiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
        localphiGRow.Data(), ntotLocal );

    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numPre, numStateTotal, 1.0, 
        psiRow.Data(), ntotLocal, G.Data(), numStateTotal, 0.0,
        localpsiGRow.Data(), ntotLocal );

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for localphiGRow and localpsiGRow Gemm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    // Step 2: Pivoted QR decomposition  for the Hadamard product of
    // the compressed matrix. Transpose format for QRCP

    // NOTE: All processors should have the same ntotLocalMG


    int m_MGTemp = numPre*numPre;
    int n_MGTemp = ntot;

    DblNumMat MGCol( m_MGTemp, ntotLocal );

    DblNumVec MGNorm(ntot);
    for( Int k = 0; k < ntot; k++ ){
      MGNorm(k) = 0;
    }
    DblNumVec MGNormLocal(ntotLocal);
    for( Int k = 0; k < ntotLocal; k++ ){
      MGNormLocal(k) = 0;
    }

    GetTime( timeSta1 );

    for( Int j = 0; j < numPre; j++ ){
      for( Int i = 0; i < numPre; i++ ){
        for( Int ir = 0; ir < ntotLocal; ir++ ){
          MGCol(i+j*numPre,ir) = localphiGRow(ir,i) * localpsiGRow(ir,j);
          MGNormLocal(ir) += MGCol(i+j*numPre,ir) * MGCol(i+j*numPre,ir);   
        }
      }
    }

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing MG from localphiGRow and localpsiGRow is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    //int ntotMGTemp, ntotLocalMGTemp;

    IntNumVec MGIdx(ntot);

    {
      Int ncols0D, nrows0D, lld0D; 
      Int ncols1D, nrows1D, lld1D; 

      Int desc_0D[9];
      Int desc_1D[9];

      Cblacs_get(0, 0, &contxt0);
      Cblacs_gridinit(&contxt0, "C", nprow0, npcol0);
      Cblacs_gridinfo(contxt0, &nprow0, &npcol0, &myrow0, &mycol0);

      SCALAPACK(descinit)(desc_0D, &Ng, &I_ONE, &Ng, &I_ONE, &I_ZERO, &I_ZERO, &contxt0, &Ng, &info0);

      if(contxt11 >= 0){
        nrows1D = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACKTemp, &myrow11, &I_ZERO, &nprow11);
        ncols1D = SCALAPACK(numroc)(&I_ONE, &I_ONE, &mycol11, &I_ZERO, &npcol11);
        lld1D = std::max( nrows1D, 1 );
      }    

      SCALAPACK(descinit)(desc_1D, &Ng, &I_ONE, &BlockSizeScaLAPACKTemp, &I_ONE, &I_ZERO, 
          &I_ZERO, &contxt11, &lld1D, &info11);


      SCALAPACK(pdgemr2d)(&Ng, &I_ONE, MGNormLocal.Data(), &I_ONE, &I_ONE, desc_1D, 
          MGNorm.Data(), &I_ONE, &I_ONE, desc_0D, &contxt11 );

      MPI_Bcast( MGNorm.Data(), ntot, MPI_DOUBLE, 0, domain_.comm );


      double MGNormMax = *(std::max_element( MGNorm.Data(), MGNorm.Data() + ntot ) );
      double MGNormMin = *(std::min_element( MGNorm.Data(), MGNorm.Data() + ntot ) );

      ntotMG = 0;
      SetValue( MGIdx, 0 );

      for( Int k = 0; k < ntot; k++ ){
        if(MGNorm(k) > hybridDFTolerance){
          MGIdx(ntotMG) = k; 
          ntotMG = ntotMG + 1;
        }
      }

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "The col size for MG: " << " ntotMG = " << ntotMG << " ntotMG/ntot = " << Real(ntotMG)/Real(ntot) << std::endl << std::endl;
      statusOFS << "The norm range for MG: " <<  " MGNormMax = " << MGNormMax << " MGNormMin = " << MGNormMin << std::endl << std::endl;
#endif

    }

    Int ncols1DCol, nrows1DCol, lld1DCol; 
    Int ncols1DRow, nrows1DRow, lld1DRow; 

    Int desc_1DCol[9];
    Int desc_1DRow[9];

    if(contxt1 >= 0){
      nrows1DCol = SCALAPACK(numroc)(&m_MGTemp, &m_MGTemp, &myrow1, &I_ZERO, &nprow1);
      ncols1DCol = SCALAPACK(numroc)(&n_MGTemp, &BlockSizeScaLAPACKTemp, &mycol1, &I_ZERO, &npcol1);
      lld1DCol = std::max( nrows1DCol, 1 );
    }    

    SCALAPACK(descinit)(desc_1DCol, &m_MGTemp, &n_MGTemp, &m_MGTemp, &BlockSizeScaLAPACKTemp, &I_ZERO, 
        &I_ZERO, &contxt1, &lld1DCol, &info1);

    if(contxt11 >= 0){
      nrows1DRow = SCALAPACK(numroc)(&m_MGTemp, &I_ONE, &myrow11, &I_ZERO, &nprow11);
      ncols1DRow = SCALAPACK(numroc)(&n_MGTemp, &n_MGTemp, &mycol11, &I_ZERO, &npcol11);
      lld1DRow = std::max( nrows1DRow, 1 );
    }    

    SCALAPACK(descinit)(desc_1DRow, &m_MGTemp, &n_MGTemp, &I_ONE, &n_MGTemp, &I_ZERO, 
        &I_ZERO, &contxt11, &lld1DRow, &info11);

    DblNumMat MGRow( nrows1DRow, ntot );
    SetValue( MGRow, 0.0 );


    GetTime( timeSta1 );

    SCALAPACK(pdgemr2d)(&m_MGTemp, &n_MGTemp, MGCol.Data(), &I_ONE, &I_ONE, desc_1DCol, 
        MGRow.Data(), &I_ONE, &I_ONE, desc_1DRow, &contxt1 );

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing MG from Col and Row is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    DblNumMat MG( nrows1DRow, ntotMG );

    for( Int i = 0; i < nrows1DRow; i++ ){
      for( Int j = 0; j < ntotMG; j++ ){
        MG(i,j) = MGRow(i,MGIdx(j));
      }
    }

    DblNumVec tau(ntotMG);
    pivQR_.Resize(ntot);
    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
    // Q factor does not need to be used

    for( Int k = 0; k < ntotMG; k++ ){
      tau[k] = 0.0;
    }

    Real timeQRCPSta, timeQRCPEnd;
    GetTime( timeQRCPSta );


    if(1){ //ScaLAPACL QRCP 2D

      Int contxt1D, contxt2D;
      Int nprow1D, npcol1D, myrow1D, mycol1D, info1D;
      Int nprow2D, npcol2D, myrow2D, mycol2D, info2D;

      Int ncols1D, nrows1D, lld1D; 
      Int ncols2D, nrows2D, lld2D; 

      Int desc_MG1D[9];
      Int desc_MG2D[9];

      Int m_MG = numPre*numPre;
      Int n_MG = ntotMG;

      Int mb_MG1D = 1;
      Int nb_MG1D = ntotMG;

      nprow1D = mpisize;
      npcol1D = 1;

      Cblacs_get(0, 0, &contxt1D);
      Cblacs_gridinit(&contxt1D, "C", nprow1D, npcol1D);
      Cblacs_gridinfo(contxt1D, &nprow1D, &npcol1D, &myrow1D, &mycol1D);

      nrows1D = SCALAPACK(numroc)(&m_MG, &mb_MG1D, &myrow1D, &I_ZERO, &nprow1D);
      ncols1D = SCALAPACK(numroc)(&n_MG, &nb_MG1D, &mycol1D, &I_ZERO, &npcol1D);

      lld1D = std::max( nrows1D, 1 );

      SCALAPACK(descinit)(desc_MG1D, &m_MG, &n_MG, &mb_MG1D, &nb_MG1D, &I_ZERO, 
          &I_ZERO, &contxt1D, &lld1D, &info1D);

      for( Int i = std::min(mpisize, IRound(sqrt(double(mpisize*(n_MG/m_MG))))); 
          i <= mpisize; i++){
        npcol2D = i; nprow2D = mpisize / npcol2D;
        if( (npcol2D >= nprow2D) && (nprow2D * npcol2D == mpisize) ) break;
      }

      Cblacs_get(0, 0, &contxt2D);
      //Cblacs_gridinit(&contxt2D, "C", nprow2D, npcol2D);

      IntNumVec pmap(mpisize);
      for ( Int i = 0; i < mpisize; i++ ){
        pmap[i] = i;
      }
      Cblacs_gridmap(&contxt2D, &pmap[0], nprow2D, nprow2D, npcol2D);

      Int m_MG2DBlocksize = BlockSizeScaLAPACK;
      Int n_MG2DBlocksize = BlockSizeScaLAPACK;

      //Int m_MG2Local = m_MG/(m_MG2DBlocksize*nprow2D)*m_MG2DBlocksize;
      //Int n_MG2Local = n_MG/(n_MG2DBlocksize*npcol2D)*n_MG2DBlocksize;
      Int m_MG2Local, n_MG2Local;

      //MPI_Comm rowComm = MPI_COMM_NULL;
      MPI_Comm colComm = MPI_COMM_NULL;

      Int mpirankRow, mpisizeRow, mpirankCol, mpisizeCol;

      //MPI_Comm_split( domain_.comm, mpirank / nprow2D, mpirank, &rowComm );
      MPI_Comm_split( domain_.comm, mpirank % nprow2D, mpirank, &colComm );

      //MPI_Comm_rank(rowComm, &mpirankRow);
      //MPI_Comm_size(rowComm, &mpisizeRow);

      MPI_Comm_rank(colComm, &mpirankCol);
      MPI_Comm_size(colComm, &mpisizeCol);


      if(contxt2D >= 0){
        Cblacs_gridinfo(contxt2D, &nprow2D, &npcol2D, &myrow2D, &mycol2D);
        nrows2D = SCALAPACK(numroc)(&m_MG, &m_MG2DBlocksize, &myrow2D, &I_ZERO, &nprow2D);
        ncols2D = SCALAPACK(numroc)(&n_MG, &n_MG2DBlocksize, &mycol2D, &I_ZERO, &npcol2D);
        lld2D = std::max( nrows2D, 1 );
      }

      SCALAPACK(descinit)(desc_MG2D, &m_MG, &n_MG, &m_MG2DBlocksize, &n_MG2DBlocksize, &I_ZERO, 
          &I_ZERO, &contxt2D, &lld2D, &info2D);

      m_MG2Local = nrows2D;
      n_MG2Local = ncols2D;

      IntNumVec pivQRTmp1(ntotMG), pivQRTmp2(ntotMG), pivQRLocal(ntotMG);
      if( m_MG > ntot ){
        std::ostringstream msg;
        msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      // DiagR is only for debugging purpose
      //        DblNumVec diagRLocal( mb_MG );
      //        DblNumVec diagR( mb_MG );

      DblNumMat&  MG1D = MG;
      DblNumMat  MG2D (m_MG2Local, n_MG2Local);

      SCALAPACK(pdgemr2d)(&m_MG, &n_MG, MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, 
          MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D, &contxt1D );

      if(contxt2D >= 0){

        Real timeQRCP1, timeQRCP2;
        GetTime( timeQRCP1 );

        SetValue( pivQRTmp1, 0 );
        scalapack::QRCPF( m_MG, n_MG, MG2D.Data(), desc_MG2D, pivQRTmp1.Data(), tau.Data() );

        //scalapack::QRCPR( m_MG, n_MG, numMu_, MG2D.Data(), desc_MG2D, pivQRTmp1.Data(), tau.Data(), BlockSizeScaLAPACK, 32);

        GetTime( timeQRCP2 );

#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time only for QRCP is " << timeQRCP2 - timeQRCP1 << " [s]" << std::endl << std::endl;
#endif

      }

      // Redistribute back eigenvectors
      //SetValue(MG1D, 0.0 );

      //SCALAPACK(pdgemr2d)( &m_MG, &n_MG, MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D,
      //    MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, &contxt1D );

      // Combine the local pivQRTmp to global pivQR_
      SetValue( pivQRLocal, 0 );
      for( Int j = 0; j < n_MG2Local; j++ ){
        pivQRLocal[ (j / n_MG2DBlocksize) * n_MG2DBlocksize * npcol2D + mpirankCol * n_MG2DBlocksize + j % n_MG2DBlocksize] = pivQRTmp1[j];
      }

      SetValue( pivQRTmp2, 0 );
      MPI_Allreduce( pivQRLocal.Data(), pivQRTmp2.Data(), 
          ntotMG, MPI_INT, MPI_SUM, colComm );

      SetValue( pivQR_, 0 );
      for( Int j = 0; j < ntotMG; j++ ){
        pivQR_(j) = MGIdx(pivQRTmp2(j));
      }

      //if( rowComm != MPI_COMM_NULL ) MPI_Comm_free( & rowComm );
      if( colComm != MPI_COMM_NULL ) MPI_Comm_free( & colComm );

      if(contxt2D >= 0) {
        Cblacs_gridexit( contxt2D );
      }

      if(contxt1D >= 0) {
        Cblacs_gridexit( contxt1D );
      }

    } // if(1) ScaLAPACL QRCP

    GetTime( timeQRCPEnd );

    //statusOFS << std::endl<< "All processors exit with abort in spinor.cpp." << std::endl;

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for QRCP alone is " <<
      timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;
#endif


    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for density fitting with QRCP is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  }

  // *********************************************************************
  // Compute the interpolation matrix via the density matrix formulation
  // *********************************************************************

  GetTime( timeSta );

  // PhiMu is scaled by the occupation number to reflect the "true" density matrix
  //IntNumVec pivMu(numMu_);
  IntNumVec pivMu1(numMu_);

  for( Int mu = 0; mu < numMu_; mu++ ){
    pivMu1(mu) = pivQR_(mu);
  }


  //if(ntot % mpisize != 0){
  //  for( Int mu = 0; mu < numMu_; mu++ ){
  //    Int k1 = (pivMu1(mu) / ntotLocalMG) - (ntot % mpisize);
  //    if(k1 > 0){
  //      pivMu1(mu) = pivQR_(mu) - k1;
  //    }
  //  }
  //}

  GetTime( timeSta1 );

  DblNumMat psiMuCol(numStateLocal, numMu_);
  DblNumMat phiMuCol(numStateLocal, numMu_);
  SetValue( psiMuCol, 0.0 );
  SetValue( phiMuCol, 0.0 );

  for (Int k=0; k<numStateLocal; k++) {
    for (Int mu=0; mu<numMu_; mu++) {
      psiMuCol(k, mu) = psiCol(pivMu1(mu),k);
      phiMuCol(k, mu) = phiCol(pivMu1(mu),k) * occupationRate[(k * mpisize + mpirank)];
    }
  }

  DblNumMat psiMu2D(nrowsNeNu2D, ncolsNeNu2D);
  DblNumMat phiMu2D(nrowsNeNu2D, ncolsNeNu2D);
  SetValue( psiMu2D, 0.0 );
  SetValue( phiMu2D, 0.0 );

  SCALAPACK(pdgemr2d)(&Ne, &Nu, psiMuCol.Data(), &I_ONE, &I_ONE, desc_NeNu1D, 
      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, &contxt11 );

  SCALAPACK(pdgemr2d)(&Ne, &Nu, phiMuCol.Data(), &I_ONE, &I_ONE, desc_NeNu1D, 
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, &contxt11 );

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing psiMuRow and phiMuRow is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  GetTime( timeSta1 );

  DblNumMat psi2D(nrowsNgNe2D, ncolsNgNe2D);
  DblNumMat phi2D(nrowsNgNe2D, ncolsNgNe2D);
  SetValue( psi2D, 0.0 );
  SetValue( phi2D, 0.0 );

  SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
      psi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, &contxt1 );
  SCALAPACK(pdgemr2d)(&Ng, &Ne, phiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
      phi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, &contxt1 );

  DblNumMat PpsiMu2D(nrowsNgNu2D, ncolsNgNu2D);
  DblNumMat PphiMu2D(nrowsNgNu2D, ncolsNgNu2D);
  SetValue( PpsiMu2D, 0.0 );
  SetValue( PphiMu2D, 0.0 );

  SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Ne, 
      &D_ONE,
      psi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D,
      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &D_ZERO,
      PpsiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

  SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Ne, 
      &D_ONE,
      phi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D,
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &D_ZERO,
      PphiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for PpsiMu and PphiMu GEMM is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  GetTime( timeSta1 );

  DblNumMat Xi2D(nrowsNgNu2D, ncolsNgNu2D);
  SetValue( Xi2D, 0.0 );

  Real* Xi2DPtr = Xi2D.Data();
  Real* PpsiMu2DPtr = PpsiMu2D.Data();
  Real* PphiMu2DPtr = PphiMu2D.Data();

  for( Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++ ){
    Xi2DPtr[g] = PpsiMu2DPtr[g] * PphiMu2DPtr[g];
  }

  DblNumMat Xi1D(nrowsNgNu1D, ncolsNuNu1D);
  SetValue( Xi1D, 0.0 );

  SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
      Xi1D.Data(), &I_ONE, &I_ONE, desc_NgNu1D, &contxt2 );

  DblNumMat PMuNu1D(nrowsNuNu1D, ncolsNuNu1D);
  SetValue( PMuNu1D, 0.0 );

  for (Int mu=0; mu<nrowsNuNu1D; mu++) {
    for (Int nu=0; nu<ncolsNuNu1D; nu++) {
      PMuNu1D(mu, nu) = Xi1D(pivMu1(mu),nu);
    }
  }

  DblNumMat PMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
  SetValue( PMuNu2D, 0.0 );

  SCALAPACK(pdgemr2d)(&Nu, &Nu, PMuNu1D.Data(), &I_ONE, &I_ONE, desc_NuNu1D, 
      PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &contxt1 );

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing PMuNu is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif


  //Method 2
  if(1){

    GetTime( timeSta1 );

    SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
    SCALAPACK(pdpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu Potrf is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    DblNumMat PMuNu2DTemp(nrowsNuNu2D, ncolsNuNu2D);
    SetValue( PMuNu2DTemp, 0.0 );

    lapack::lacpy( lapack::MatrixType::General, nrowsNuNu2D, ncolsNuNu2D, PMuNu2D.Data(), nrowsNuNu2D, PMuNu2DTemp.Data(), nrowsNuNu2D );

    SCALAPACK(pdtradd)("U", "T", &Nu, &Nu, 
        &D_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, 
        &D_ZERO,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

    DblNumMat Xi2DTemp(nrowsNgNu2D, ncolsNgNu2D);
    SetValue( Xi2DTemp, 0.0 );

    SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Nu, 
        &D_ONE,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D, 
        &D_ZERO,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    SetValue( Xi2D, 0.0 );
    lapack::lacpy( lapack::MatrixType::General, nrowsNgNu2D, ncolsNgNu2D, Xi2DTemp.Data(), nrowsNgNu2D, Xi2D.Data(), nrowsNgNu2D );

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu and Xi pdgemm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  }


  GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the interpolation vectors is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif



  // *********************************************************************
  // Solve the Poisson equations.
  // Store VXi separately. This is not the most memory efficient
  // implementation
  // *********************************************************************

  DblNumMat VXi2D(nrowsNgNu2D, ncolsNgNu2D);

  {
    GetTime( timeSta );
    // XiCol used for both input and output
    DblNumMat XiCol(ntot, numMuLocal);
    SetValue(XiCol, 0.0 );

    SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
        XiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1D, &contxt2 );

    GetTime( timeSta );
    for( Int mu = 0; mu < numMuLocal; mu++ ){
      blas::copy( ntot,  XiCol.VecData(mu), 1, fft.inputVecR2C.Data(), 1 );

      FFTWExecute ( fft, fft.forwardPlanR2C );

      for( Int ig = 0; ig < ntotR2C; ig++ ){
        fft.outputVecR2C(ig) *= -exxFraction * exxgkkR2C(ig);
      }

      FFTWExecute ( fft, fft.backwardPlanR2C );

      blas::copy( ntot, fft.inputVecR2C.Data(), 1, XiCol.VecData(mu), 1 );

    } // for (mu)

    SetValue(VXi2D, 0.0 );
    SCALAPACK(pdgemr2d)(&Ng, &Nu, XiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1D, 
        VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, &contxt1 );

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for solving Poisson-like equations is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  // Prepare for the computation of the M matrix
  GetTime( timeSta );

  DblNumMat MMatMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
  SetValue(MMatMuNu2D, 0.0 );

  SCALAPACK(pdgemm)("T", "N", &Nu, &Nu, &Ng, 
      &D_MinusONE,
      Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
      VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, 
      &D_ZERO,
      MMatMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

  DblNumMat phiMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
  SCALAPACK(pdgemm)("T", "N", &Nu, &Nu, &Ne, 
      &D_ONE,
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D,
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &D_ZERO,
      phiMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

  Real* MMat2DPtr = MMatMuNu2D.Data();
  Real* phi2DPtr  = phiMuNu2D.Data();

  for( Int g = 0; g < nrowsNuNu2D * ncolsNuNu2D; g++ ){
    MMat2DPtr[g] *= phi2DPtr[g];
  }

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for preparing the M matrix is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // *********************************************************************
  // Compute the exchange potential and the symmetrized inner product
  // *********************************************************************

  GetTime( timeSta );

  // Rewrite VXi by VXi.*PcolPhi
  Real* VXi2DPtr = VXi2D.Data();
  Real* PphiMu2DPtr1 = PphiMu2D.Data();
  for( Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++ ){
    VXi2DPtr[g] *= PphiMu2DPtr1[g];
  }

  // NOTE: Hpsi must be zero in order to compute the M matrix later
  DblNumMat Hpsi2D( nrowsNgNe2D, ncolsNgNe2D );
  SetValue( Hpsi2D, 0.0 );

  SCALAPACK(pdgemm)("N", "T", &Ng, &Ne, &Nu, 
      &D_ONE,
      VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
      &D_ZERO,
      Hpsi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D);

  DblNumMat HpsiCol( ntot, numStateLocal );
  SetValue(HpsiCol, 0.0 );

  SCALAPACK(pdgemr2d)(&Ng, &Ne, Hpsi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, 
      HpsiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, &contxt2 );

  lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, HpsiCol.Data(), ntot, Hpsi.Data(), ntot );

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the exchange potential is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // Compute the matrix VxMat = -Psi'* vexxPsi and symmetrize
  // vexxPsi (Hpsi) must be zero before entering this routine
  VxMat.Resize( numStateTotal, numStateTotal );

  GetTime( timeSta );

  if(1){

    DblNumMat VxMat2D( nrowsNeNe2D, ncolsNeNe2D );
    DblNumMat VxMatTemp2D( nrowsNuNe2D, ncolsNuNe2D );

    SCALAPACK(pdgemm)("N", "T", &Nu, &Ne, &Nu, 
        &D_ONE,
        MMatMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
        &D_ZERO,
        VxMatTemp2D.Data(), &I_ONE, &I_ONE, desc_NuNe2D);

    SCALAPACK(pdgemm)("N", "N", &Ne, &Ne, &Nu, 
        &D_ONE,
        psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, 
        VxMatTemp2D.Data(), &I_ONE, &I_ONE, desc_NuNe2D, 
        &D_ZERO,
        VxMat2D.Data(), &I_ONE, &I_ONE, desc_NeNe2D);

    SCALAPACK(pdgemr2d)(&Ne, &Ne, VxMat2D.Data(), &I_ONE, &I_ONE, desc_NeNe2D, 
        VxMat.Data(), &I_ONE, &I_ONE, desc_NeNe0D, &contxt2 );

    //if(mpirank == 0){
    //  MPI_Bcast( VxMat.Data(), Ne * Ne, MPI_DOUBLE, 0, domain_.comm );
    //}

  }
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing VxMat in the sym format is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  if(contxt0 >= 0) {
    Cblacs_gridexit( contxt0 );
  }

  if(contxt1 >= 0) {
    Cblacs_gridexit( contxt1 );
  }

  if(contxt11 >= 0) {
    Cblacs_gridexit( contxt11 );
  }

  if(contxt2 >= 0) {
    Cblacs_gridexit( contxt2 );
  }

  MPI_Barrier(domain_.comm);

  return ;
}        // -----  end of method Spinor::AddMultSpinorEXXDF6  ----- 


// Kmeans for ISDF
// Update: 8/20/2017
void Spinor::AddMultSpinorEXXDF7 ( Fourier& fft, 
    const NumTns<Real>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Real  numSpin,
    const DblNumVec& occupationRate,
    std::string hybridDFType,
    Real  hybridDFKmeansTolerance, 
    Int   hybridDFKmeansMaxIter, 
    const Real numMuFac,
    const Real numGaussianRandomFac,
    const Int numProcScaLAPACK,  
    const Real hybridDFTolerance,
    const Int BlockSizeScaLAPACK,  
    NumTns<Real>& Hpsi, 
    NumMat<Real>& VxMat,
    bool isFixColumnDF )
{
  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;

  Int ntot     = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();
  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int numStateTotal = numStateTotal_;

  Int ncomPhi = phi.n();

  Real vol = domain_.Volume();

  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin polarized case not implemented.");
  }

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }


  // *********************************************************************
  // Perform interpolative separable density fitting
  // *********************************************************************

  //numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
  numMu_ = IRound(numStateTotal*numMuFac);

  Int numPre = IRound(std::sqrt(numMu_*numGaussianRandomFac));
  if( numPre > numStateTotal ){
    ErrorHandling("numMu is too large for interpolative separable density fitting!");
  }

  statusOFS << "ntot          = " << ntot << std::endl;
  statusOFS << "numMu         = " << numMu_ << std::endl;
  statusOFS << "numPre        = " << numPre << std::endl;
  statusOFS << "numPre*numPre = " << numPre * numPre << std::endl;

  // Convert the column partition to row partition
  Int numStateBlocksize = numStateTotal / mpisize;
  //Int ntotBlocksize = ntot / mpisize;

  Int numMuBlocksize = numMu_ / mpisize;

  Int numStateLocal1 = numStateBlocksize;
  //Int ntotLocal = ntotBlocksize;

  Int numMuLocal = numMuBlocksize;

  if(mpirank < (numStateTotal % mpisize)){
    numStateLocal1 = numStateBlocksize + 1;
  }

  if(numStateLocal !=  numStateLocal1){
    statusOFS << "numStateLocal = " << numStateLocal << " numStateLocal1 = " << numStateLocal1 << std::endl;
    ErrorHandling("The size is not right in interpolative separable density fitting!");
  }

  if(mpirank < (numMu_ % mpisize)){
    numMuLocal = numMuBlocksize + 1;
  }

  //if(mpirank < (ntot % mpisize)){
  //  ntotLocal = ntotBlocksize + 1;
  //}

  //huwei 
  //2D MPI commucation for all the matrix

  Int I_ONE = 1, I_ZERO = 0;
  double D_ONE = 1.0;
  double D_ZERO = 0.0;
  double D_MinusONE = -1.0;

  struct CommMatrix CommMat;

  Int Ng = ntot;
  Int Ne = numStateTotal_; 
  Int Nu = numMu_; 

  // 0D MPI
  CommMat.nprow0 = 1;
  CommMat.npcol0 = mpisize;

  Cblacs_get(0, 0, &CommMat.contxt0);
  Cblacs_gridinit(&CommMat.contxt0, "C", CommMat.nprow0, CommMat.npcol0);
  //Cblacs_gridinfo(CommMat.contxt0, &CommMat.nprow0, &CommMat.npcol0, &CommMat.myrow0, &CommMat.mycol0);

  //SCALAPACK(descinit)(CommMat.desc_NeNe0D, &Ne, &Ne, &Ne, &Ne, &I_ZERO, &I_ZERO, &CommMat.contxt0, &Ne, &CommMat.info0);
  //guofeng
  scalapack::Descriptor descNeNe0D;
  descNeNe0D.Init(Ne, Ne, Ne, Ne, I_ZERO, I_ZERO, CommMat.contxt0);

  // 1D MPI
  CommMat.nprow1 = 1;
  CommMat.npcol1 = mpisize;

  Cblacs_get(0, 0, &CommMat.contxt1);
  Cblacs_gridinit(&CommMat.contxt1, "C", CommMat.nprow1, CommMat.npcol1);
  Cblacs_gridinfo(CommMat.contxt1, &CommMat.nprow1, &CommMat.npcol1, &CommMat.myrow1, &CommMat.mycol1);

  //CommMat.desc_NgNe1DCol
  //if(CommMat.contxt1 >= 0){
    //CommMat.nrowsNgNe1DCol = SCALAPACK(numroc)(&Ng, &Ng, &CommMat.myrow1, &I_ZERO, &CommMat.nprow1);
    //CommMat.ncolsNgNe1DCol = SCALAPACK(numroc)(&Ne, &I_ONE, &CommMat.mycol1, &I_ZERO, &CommMat.npcol1);
  //  CommMat.lldNgNe1DCol = std::max( CommMat.nrowsNgNe1DCol, 1 );
  //}    

  //SCALAPACK(descinit)(CommMat.desc_NgNe1DCol, &Ng, &Ne, &Ng, &I_ONE, &I_ZERO, 
  //    &I_ZERO, &CommMat.contxt1, &CommMat.lldNgNe1DCol, &CommMat.info1);
  scalapack::Descriptor descNgNe1DCol;
  descNgNe1DCol.Init(Ng , Ne, Ng, I_ONE, I_ZERO, I_ZERO, CommMat.contxt1);

  CommMat.nprow11 = mpisize;
  CommMat.npcol11 = 1;

  Cblacs_get(0, 0, &CommMat.contxt11);
  Cblacs_gridinit(&CommMat.contxt11, "C", CommMat.nprow11, CommMat.npcol11);
  Cblacs_gridinfo(CommMat.contxt11, &CommMat.nprow11, &CommMat.npcol11, &CommMat.myrow11, &CommMat.mycol11);

  Int BlockSizeScaLAPACKTemp = BlockSizeScaLAPACK; 
  //CommMat.desc_NgNe1DRow
  if(CommMat.contxt11 >= 0){
    CommMat.nrowsNgNe1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACKTemp, &CommMat.myrow11, &I_ZERO, &CommMat.nprow11);
    CommMat.ncolsNgNe1DRow = SCALAPACK(numroc)(&Ne, &Ne, &CommMat.mycol11, &I_ZERO, &CommMat.npcol11);
    CommMat.lldNgNe1DRow = std::max( CommMat.nrowsNgNe1DRow, 1 );
  }    

  SCALAPACK(descinit)(CommMat.desc_NgNe1DRow, &Ng, &Ne, &BlockSizeScaLAPACKTemp, &Ne, &I_ZERO, 
      &I_ZERO, &CommMat.contxt11, &CommMat.lldNgNe1DRow, &CommMat.info11);

  //CommMat.desc_NeNu1D
  #if 0 
  if(CommMat.contxt11 >= 0){
    CommMat.nrowsNeNu1D = SCALAPACK(numroc)(&Ne, &I_ONE, &CommMat.myrow11, &I_ZERO, &CommMat.nprow11);
    CommMat.ncolsNeNu1D = SCALAPACK(numroc)(&Nu, &Nu, &CommMat.mycol11, &I_ZERO, &CommMat.npcol11);
    CommMat.lldNeNu1D = std::max( CommMat.nrowsNeNu1D, 1 );
  }    

  SCALAPACK(descinit)(CommMat.desc_NeNu1D, &Ne, &Nu, &I_ONE, &Nu, &I_ZERO, 
      &I_ZERO, &CommMat.contxt11, &CommMat.lldNeNu1D, &CommMat.info11);
  #else
  scalapack::Descriptor descNeNu1D;
  descNeNu1D.Init(Ne, Nu, I_ONE, Nu, I_ZERO, I_ZERO, CommMat.contxt11);
  #endif

  //CommMat.desc_NgNu1D
  if(CommMat.contxt1 >= 0){
    CommMat.nrowsNgNu1D = SCALAPACK(numroc)(&Ng, &Ng, &CommMat.myrow1, &I_ZERO, &CommMat.nprow1);
    CommMat.ncolsNgNu1D = SCALAPACK(numroc)(&Nu, &I_ONE, &CommMat.mycol1, &I_ZERO, &CommMat.npcol1);
    CommMat.lldNgNu1D = std::max( CommMat.nrowsNgNu1D, 1 );
  }    

  SCALAPACK(descinit)(CommMat.desc_NgNu1D, &Ng, &Nu, &Ng, &I_ONE, &I_ZERO, 
      &I_ZERO, &CommMat.contxt1, &CommMat.lldNgNu1D, &CommMat.info1);

  //CommMat.desc_NuNu1D
  if(CommMat.contxt1 >= 0){
    CommMat.nrowsNuNu1D = SCALAPACK(numroc)(&Nu, &Nu, &CommMat.myrow1, &I_ZERO, &CommMat.nprow1);
    CommMat.ncolsNuNu1D = SCALAPACK(numroc)(&Nu, &I_ONE, &CommMat.mycol1, &I_ZERO, &CommMat.npcol1);
    CommMat.lldNuNu1D = std::max( CommMat.nrowsNuNu1D, 1 );
  }    

  SCALAPACK(descinit)(CommMat.desc_NuNu1D, &Nu, &Nu, &Nu, &I_ONE, &I_ZERO, 
      &I_ZERO, &CommMat.contxt1, &CommMat.lldNuNu1D, &CommMat.info1);


  // 2D MPI
  for( Int i = IRound(sqrt(double(mpisize))); i <= mpisize; i++){
    CommMat.nprow2 = i; CommMat.npcol2 = mpisize / CommMat.nprow2;
    if( (CommMat.nprow2 >= CommMat.npcol2) && (CommMat.nprow2 * CommMat.npcol2 == mpisize) ) break;
  }

  Cblacs_get(0, 0, &CommMat.contxt2);
  //Cblacs_gridinit(&CommMat.contxt2, "C", CommMat.nprow2, CommMat.npcol2);

  IntNumVec pmap2(mpisize);
  for ( Int i = 0; i < mpisize; i++ ){
    pmap2[i] = i;
  }
  Cblacs_gridmap(&CommMat.contxt2, &pmap2[0], CommMat.nprow2, CommMat.nprow2, CommMat.npcol2);

  Int mb2 = BlockSizeScaLAPACK;
  Int nb2 = BlockSizeScaLAPACK;

  //CommMat.desc_NgNe2D
  //if(CommMat.contxt2 >= 0){
  //  Cblacs_gridinfo(CommMat.contxt2, &CommMat.nprow2, &CommMat.npcol2, &CommMat.myrow2, &CommMat.mycol2);
    //CommMat.nrowsNgNe2D = SCALAPACK(numroc)(&Ng, &mb2, &CommMat.myrow2, &I_ZERO, &CommMat.nprow2);
    //CommMat.ncolsNgNe2D = SCALAPACK(numroc)(&Ne, &nb2, &CommMat.mycol2, &I_ZERO, &CommMat.npcol2);
    //CommMat.lldNgNe2D = std::max( CommMat.nrowsNgNe2D, 1 );
  //}

  //SCALAPACK(descinit)(CommMat.desc_NgNe2D, &Ng, &Ne, &mb2, &nb2, &I_ZERO, 
  //    &I_ZERO, &CommMat.contxt2, &CommMat.lldNgNe2D, &CommMat.info2);

  scalapack::Descriptor descNgNe2D;
  descNgNe2D.Init(Ng , Ne, BlockSizeScaLAPACK, BlockSizeScaLAPACK, I_ZERO, I_ZERO, CommMat.contxt2);
  
  //printf("Huwe: values_[0] is %d, m is %d,  n is %d,  mb2 is %d, nb2 is %d LLD is %d\n",CommMat.desc_NgNe2D[0],Ng,Ne,mb2,nb2,CommMat.lldNgNe2D);
  //printf("descNgNe2D_1 = %d\n",CommMat.desc_NgNe2D);
  //CommMat.desc_NgNu2D
  if(CommMat.contxt2 >= 0){
    Cblacs_gridinfo(CommMat.contxt2, &CommMat.nprow2, &CommMat.npcol2, &CommMat.myrow2, &CommMat.mycol2);
    CommMat.nrowsNgNu2D = SCALAPACK(numroc)(&Ng, &mb2, &CommMat.myrow2, &I_ZERO, &CommMat.nprow2);
    CommMat.ncolsNgNu2D = SCALAPACK(numroc)(&Nu, &nb2, &CommMat.mycol2, &I_ZERO, &CommMat.npcol2);
    CommMat.lldNgNu2D = std::max( CommMat.nrowsNgNu2D, 1 );
  }
  //guofeng
  SCALAPACK(descinit)(CommMat.desc_NgNu2D, &Ng, &Nu, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &CommMat.contxt2, &CommMat.lldNgNu2D, &CommMat.info2);
  scalapack::Descriptor descNgNu2D;
  descNgNu2D.Init(Ng, Nu, BlockSizeScaLAPACK, BlockSizeScaLAPACK, I_ZERO, I_ZERO, CommMat.contxt2);



  //CommMat.desc_NuNg2D
  //if(CommMat.contxt2 >= 0){
  //  Cblacs_gridinfo(CommMat.contxt2, &CommMat.nprow2, &CommMat.npcol2, &CommMat.myrow2, &CommMat.mycol2);
  //  CommMat.nrowsNuNg2D = SCALAPACK(numroc)(&Nu, &mb2, &CommMat.myrow2, &I_ZERO, &CommMat.nprow2);
  //  CommMat.ncolsNuNg2D = SCALAPACK(numroc)(&Ng, &nb2, &CommMat.mycol2, &I_ZERO, &CommMat.npcol2);
  //  CommMat.lldNuNg2D = std::max( CommMat.nrowsNuNg2D, 1 );
  //}

  //SCALAPACK(descinit)(CommMat.desc_NuNg2D, &Nu, &Ng, &mb2, &nb2, &I_ZERO, 
  //    &I_ZERO, &CommMat.contxt2, &CommMat.lldNuNg2D, &CommMat.info2);
  // guofeng
  //scalapack::Descriptor descNuNg2D;
  //descNuNg2D.Init(Nu, Ng, BlockSizeScaLAPACK, BlockSizeScaLAPACK, I_ZERO, I_ZERO, CommMat.contxt2);

  //CommMat.desc_NeNe2D
  //if(CommMat.contxt2 >= 0){
  //  Cblacs_gridinfo(CommMat.contxt2, &CommMat.nprow2, &CommMat.npcol2, &CommMat.myrow2, &CommMat.mycol2);
  //  CommMat.nrowsNeNe2D = SCALAPACK(numroc)(&Ne, &mb2, &CommMat.myrow2, &I_ZERO, &CommMat.nprow2);
  //  CommMat.ncolsNeNe2D = SCALAPACK(numroc)(&Ne, &nb2, &CommMat.mycol2, &I_ZERO, &CommMat.npcol2);
  //  CommMat.lldNeNe2D = std::max( CommMat.nrowsNeNe2D, 1 );
  //}

  //SCALAPACK(descinit)(CommMat.desc_NeNe2D, &Ne, &Ne, &mb2, &nb2, &I_ZERO, 
  //    &I_ZERO, &CommMat.contxt2, &CommMat.lldNeNe2D, &CommMat.info2);

  //guofeng
  scalapack::Descriptor descNeNe2D;
  descNeNe2D.Init(Ne, Ne, BlockSizeScaLAPACK, BlockSizeScaLAPACK, I_ZERO, I_ZERO, CommMat.contxt2);

  //CommMat.desc_NuNu2D
  if(CommMat.contxt2 >= 0){
    Cblacs_gridinfo(CommMat.contxt2, &CommMat.nprow2, &CommMat.npcol2, &CommMat.myrow2, &CommMat.mycol2);
    CommMat.nrowsNuNu2D = SCALAPACK(numroc)(&Nu, &mb2, &CommMat.myrow2, &I_ZERO, &CommMat.nprow2);
    CommMat.ncolsNuNu2D = SCALAPACK(numroc)(&Nu, &nb2, &CommMat.mycol2, &I_ZERO, &CommMat.npcol2);
    CommMat.lldNuNu2D = std::max( CommMat.nrowsNuNu2D, 1 );
  }

  SCALAPACK(descinit)(CommMat.desc_NuNu2D, &Nu, &Nu, &mb2, &nb2, &I_ZERO, 
      &I_ZERO, &CommMat.contxt2, &CommMat.lldNuNu2D, &CommMat.info2);

  //CommMat.desc_NeNu2D
  //if(CommMat.contxt2 >= 0){
  //  Cblacs_gridinfo(CommMat.contxt2, &CommMat.nprow2, &CommMat.npcol2, &CommMat.myrow2, &CommMat.mycol2);
  //  CommMat.nrowsNeNu2D = SCALAPACK(numroc)(&Ne, &mb2, &CommMat.myrow2, &I_ZERO, &CommMat.nprow2);
  //  CommMat.ncolsNeNu2D = SCALAPACK(numroc)(&Nu, &nb2, &CommMat.mycol2, &I_ZERO, &CommMat.npcol2);
  //  CommMat.lldNeNu2D = std::max( CommMat.nrowsNeNu2D, 1 );
  //}

  //SCALAPACK(descinit)(CommMat.desc_NeNu2D, &Ne, &Nu, &mb2, &nb2, &I_ZERO, 
  //   &I_ZERO, &CommMat.contxt2, &CommMat.lldNeNu2D, &CommMat.info2);
  // guofeng
  scalapack::Descriptor descNeNu2D;
  descNeNu2D.Init(Ne, Nu, BlockSizeScaLAPACK, BlockSizeScaLAPACK, I_ZERO, I_ZERO, CommMat.contxt2);

  //CommMat.desc_NuNe2D
  //if(CommMat.contxt2 >= 0){
  //  Cblacs_gridinfo(CommMat.contxt2, &CommMat.nprow2, &CommMat.npcol2, &CommMat.myrow2, &CommMat.mycol2);
  //  CommMat.nrowsNuNe2D = SCALAPACK(numroc)(&Nu, &mb2, &CommMat.myrow2, &I_ZERO, &CommMat.nprow2);
  //  CommMat.ncolsNuNe2D = SCALAPACK(numroc)(&Ne, &nb2, &CommMat.mycol2, &I_ZERO, &CommMat.npcol2);
  //  CommMat.lldNuNe2D = std::max( CommMat.nrowsNuNe2D, 1 );
  //}

  //SCALAPACK(descinit)(CommMat.desc_NuNe2D, &Nu, &Ne, &mb2, &nb2, &I_ZERO, 
  //    &I_ZERO, &CommMat.contxt2, &CommMat.lldNuNe2D, &CommMat.info2);
  //guofeng
  scalapack::Descriptor descNuNe2D;
  descNuNe2D.Init(Nu, Ne, BlockSizeScaLAPACK, BlockSizeScaLAPACK, I_ZERO, I_ZERO, CommMat.contxt2);


  DblNumMat phiCol( ntot, numStateLocal );
  SetValue( phiCol, 0.0 );

  DblNumMat psiCol( ntot, numStateLocal );
  SetValue( psiCol, 0.0 );

  lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, phi.Data(), ntot, phiCol.Data(), ntot );
  lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, wavefun_.Data(), ntot, psiCol.Data(), ntot );
  
  //guofeng
  //scalapack::ScaLAPACKMatrix<Real> sca_psiCol;
  //sca_psiCol.SetDescriptor(descNgNe1DCol);
  //DblNumMat& square_mat = psiCol;
  // Redistribute the input matrix over the process grid
  //SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, descNgNe1DCol.Values(),
  //    &sca_psiCol.LocalMatrix()[0], &I_ONE, &I_ONE, descNgNe1DCol.Values(), &CommMat.contxt1);

  // Computing the indices is optional


  Int ntotLocal = CommMat.nrowsNgNe1DRow; 
  //Int ntotLocalMG = CommMat.nrowsNgNe1DRow;
  //Int ntotMG = ntot;
  Int ntotLocalMG, ntotMG;

  //if( (ntot % mpisize) == 0 ){
  //  ntotLocalMG = ntotBlocksize;
  //}
  //else{
  //  ntotLocalMG = ntotBlocksize + 1;
  // }
    
  if ((pivQR_.m_ != ntot) || (hybridDFType == "QRCP")){
    pivQR_.Resize(ntot);
    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
    // Q factor does not need to be used
  }


  if( (isFixColumnDF == false) && ((hybridDFType == "QRCP") || (hybridDFType == "Kmeans+QRCP"))){

    GetTime( timeSta );

    DblNumMat localphiGRow( ntotLocal, numPre );
    SetValue( localphiGRow, 0.0 );

    DblNumMat localpsiGRow( ntotLocal, numPre );
    SetValue( localpsiGRow, 0.0 );

    DblNumMat G(numStateTotal, numPre);
    SetValue( G, 0.0 );

    DblNumMat phiRow( ntotLocal, numStateTotal );
    SetValue( phiRow, 0.0 );

    DblNumMat psiRow( ntotLocal, numStateTotal );
    SetValue( psiRow, 0.0 );

    SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, descNgNe1DCol.Values(), 
        psiRow.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNe1DRow, &CommMat.contxt1 );

    SCALAPACK(pdgemr2d)(&Ng, &Ne, phiCol.Data(), &I_ONE, &I_ONE, descNgNe1DCol.Values(), 
        phiRow.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNe1DRow, &CommMat.contxt1 );

    // Step 1: Pre-compression of the wavefunctions. This uses
    // multiplication with orthonormalized random Gaussian matrices

    GetTime( timeSta1 );

    if (G_.m_!=numStateTotal){
      DblNumMat G(numStateTotal, numPre);
      if ( mpirank == 0 ) {
        GaussianRandom(G);
        Orth( numStateTotal, numPre, G.Data(), numStateTotal );
        statusOFS << "Random projection initialzied." << std::endl << std::endl;
      }
      MPI_Bcast(G.Data(), numStateTotal * numPre, MPI_DOUBLE, 0, domain_.comm);
      G_ = G;
    } else {
      statusOFS << "Random projection reused." << std::endl;
    }
    statusOFS << "G(0,0) = " << G_(0,0) << std::endl << std::endl;

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for Gaussian MPI_Bcast is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numPre, numStateTotal, 1.0, 
        phiRow.Data(), ntotLocal, G_.Data(), numStateTotal, 0.0,
        localphiGRow.Data(), ntotLocal );

    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numPre, numStateTotal, 1.0, 
        psiRow.Data(), ntotLocal, G_.Data(), numStateTotal, 0.0,
        localpsiGRow.Data(), ntotLocal );

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for localphiGRow and localpsiGRow Gemm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    // Step 2: Pivoted QR decomposition  for the Hadamard product of
    // the compressed matrix. Transpose format for QRCP

    // NOTE: All processors should have the same ntotLocalMG


    int m_MGTemp = numPre*numPre;
    int n_MGTemp = ntot;

    DblNumMat MGCol( m_MGTemp, ntotLocal );

    DblNumVec MGNorm(ntot);
    for( Int k = 0; k < ntot; k++ ){
      MGNorm(k) = 0;
    }
    DblNumVec MGNormLocal(ntotLocal);
    for( Int k = 0; k < ntotLocal; k++ ){
      MGNormLocal(k) = 0;
    }

    GetTime( timeSta1 );

    for( Int j = 0; j < numPre; j++ ){
      for( Int i = 0; i < numPre; i++ ){
        for( Int ir = 0; ir < ntotLocal; ir++ ){
          MGCol(i+j*numPre,ir) = localphiGRow(ir,i) * localpsiGRow(ir,j);
          MGNormLocal(ir) += MGCol(i+j*numPre,ir) * MGCol(i+j*numPre,ir);   
        }
      }
    }

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing MG from localphiGRow and localpsiGRow is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    //int ntotMGTemp, ntotLocalMGTemp;

    IntNumVec MGIdx(ntot);

    {
      Int ncols0D, nrows0D, lld0D; 
      Int ncols1D, nrows1D, lld1D; 

      Int desc_0D[9];
      Int desc_1D[9];

      Cblacs_get(0, 0, &CommMat.contxt0);
      Cblacs_gridinit(&CommMat.contxt0, "C", CommMat.nprow0, CommMat.npcol0);
      Cblacs_gridinfo(CommMat.contxt0, &CommMat.nprow0, &CommMat.npcol0, &CommMat.myrow0, &CommMat.mycol0);

      SCALAPACK(descinit)(desc_0D, &Ng, &I_ONE, &Ng, &I_ONE, &I_ZERO, &I_ZERO, &CommMat.contxt0, &Ng, &CommMat.info0);

      if(CommMat.contxt11 >= 0){
        nrows1D = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACKTemp, &CommMat.myrow11, &I_ZERO, &CommMat.nprow11);
        ncols1D = SCALAPACK(numroc)(&I_ONE, &I_ONE, &CommMat.mycol11, &I_ZERO, &CommMat.npcol11);
        lld1D = std::max( nrows1D, 1 );
      }    

      SCALAPACK(descinit)(desc_1D, &Ng, &I_ONE, &BlockSizeScaLAPACKTemp, &I_ONE, &I_ZERO, 
          &I_ZERO, &CommMat.contxt11, &lld1D, &CommMat.info11);


      SCALAPACK(pdgemr2d)(&Ng, &I_ONE, MGNormLocal.Data(), &I_ONE, &I_ONE, desc_1D, 
          MGNorm.Data(), &I_ONE, &I_ONE, desc_0D, &CommMat.contxt11 );

      MPI_Bcast( MGNorm.Data(), ntot, MPI_DOUBLE, 0, domain_.comm );


      double MGNormMax = *(std::max_element( MGNorm.Data(), MGNorm.Data() + ntot ) );
      double MGNormMin = *(std::min_element( MGNorm.Data(), MGNorm.Data() + ntot ) );

      ntotMG = 0;
      SetValue( MGIdx, 0 );

      for( Int k = 0; k < ntot; k++ ){
        if(MGNorm(k) > hybridDFTolerance){
          MGIdx(ntotMG) = k; 
          ntotMG = ntotMG + 1;
        }
      }

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "The col size for MG: " << " ntotMG = " << ntotMG << " ntotMG/ntot = " << Real(ntotMG)/Real(ntot) << std::endl << std::endl;
      statusOFS << "The norm range for MG: " <<  " MGNormMax = " << MGNormMax << " MGNormMin = " << MGNormMin << std::endl << std::endl;
#endif

    }

    Int ncols1DCol, nrows1DCol, lld1DCol; 
    Int ncols1DRow, nrows1DRow, lld1DRow; 

    Int desc_1DCol[9];
    Int desc_1DRow[9];

    if(CommMat.contxt1 >= 0){
      nrows1DCol = SCALAPACK(numroc)(&m_MGTemp, &m_MGTemp, &CommMat.myrow1, &I_ZERO, &CommMat.nprow1);
      ncols1DCol = SCALAPACK(numroc)(&n_MGTemp, &BlockSizeScaLAPACKTemp, &CommMat.mycol1, &I_ZERO, &CommMat.npcol1);
      lld1DCol = std::max( nrows1DCol, 1 );
    }    

    SCALAPACK(descinit)(desc_1DCol, &m_MGTemp, &n_MGTemp, &m_MGTemp, &BlockSizeScaLAPACKTemp, &I_ZERO, 
        &I_ZERO, &CommMat.contxt1, &lld1DCol, &CommMat.info1);

    if(CommMat.contxt11 >= 0){
      nrows1DRow = SCALAPACK(numroc)(&m_MGTemp, &I_ONE, &CommMat.myrow11, &I_ZERO, &CommMat.nprow11);
      ncols1DRow = SCALAPACK(numroc)(&n_MGTemp, &n_MGTemp, &CommMat.mycol11, &I_ZERO, &CommMat.npcol11);
      lld1DRow = std::max( nrows1DRow, 1 );
    }    

    SCALAPACK(descinit)(desc_1DRow, &m_MGTemp, &n_MGTemp, &I_ONE, &n_MGTemp, &I_ZERO, 
        &I_ZERO, &CommMat.contxt11, &lld1DRow, &CommMat.info11);

    DblNumMat MGRow( nrows1DRow, ntot );
    SetValue( MGRow, 0.0 );


    GetTime( timeSta1 );

    SCALAPACK(pdgemr2d)(&m_MGTemp, &n_MGTemp, MGCol.Data(), &I_ONE, &I_ONE, desc_1DCol, 
        MGRow.Data(), &I_ONE, &I_ONE, desc_1DRow, &CommMat.contxt1 );

    GetTime( timeEnd1 );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing MG from Col and Row is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    DblNumMat MG( nrows1DRow, ntotMG );

    for( Int i = 0; i < nrows1DRow; i++ ){
      for( Int j = 0; j < ntotMG; j++ ){
        MG(i,j) = MGRow(i,MGIdx(j));
      }
    }

    DblNumVec tau(ntotMG);

    for( Int k = 0; k < ntotMG; k++ ){
      tau[k] = 0.0;
    }

    Real timeQRCPSta, timeQRCPEnd;
    GetTime( timeQRCPSta );


    if(1){ //ScaLAPACL QRCP 2D

      #if 0
      //GUOFENG
      Int ncols1D, nrows1D, lld1D; 
      Int ncols2D, nrows2D, lld2D; 

      Int desc_MG1D[9];
      Int desc_MG2D[9];
      #endif

      Int m_MG = numPre*numPre;
      Int n_MG = ntotMG;

      Int mb_MG1D = 1;
      Int nb_MG1D = ntotMG;

      CommMat.nprow1D = mpisize;
      CommMat.npcol1D = 1;

      Cblacs_get(0, 0, &CommMat.contxt1D);
      Cblacs_gridinit(&CommMat.contxt1D, "C", CommMat.nprow1D, CommMat.npcol1D);
      #if 0
      Cblacs_gridinfo(CommMat.contxt1D, &CommMat.nprow1D, &CommMat.npcol1D, &CommMat.myrow1D, &CommMat.mycol1D);

      nrows1D = SCALAPACK(numroc)(&m_MG, &mb_MG1D, &CommMat.myrow1D, &I_ZERO, &CommMat.nprow1D);
      ncols1D = SCALAPACK(numroc)(&n_MG, &nb_MG1D, &CommMat.mycol1D, &I_ZERO, &CommMat.npcol1D);

      lld1D = std::max( nrows1D, 1 );

      SCALAPACK(descinit)(desc_MG1D, &m_MG, &n_MG, &mb_MG1D, &nb_MG1D, &I_ZERO, 
          &I_ZERO, &CommMat.contxt1D, &lld1D, &CommMat.info1D);
      #else
      //guofeng
      scalapack::Descriptor descMG1D;
      descMG1D.Init(m_MG, n_MG, mb_MG1D, nb_MG1D, I_ZERO, I_ZERO, CommMat.contxt1D);
      #endif

      for( Int i = std::min(mpisize, IRound(sqrt(double(mpisize*(n_MG/m_MG))))); 
          i <= mpisize; i++){
        CommMat.npcol2D = i; CommMat.nprow2D = mpisize / CommMat.npcol2D;
        if( (CommMat.npcol2D >= CommMat.nprow2D) && (CommMat.nprow2D * CommMat.npcol2D == mpisize) ) break;
      }

      Cblacs_get(0, 0, &CommMat.contxt2D);
      //Cblacs_gridinit(&CommMat.contxt2D, "C", CommMat.nprow2D, CommMat.npcol2D);

      IntNumVec pmap(mpisize);
      for ( Int i = 0; i < mpisize; i++ ){
        pmap[i] = i;
      }
      Cblacs_gridmap(&CommMat.contxt2D, &pmap[0], CommMat.nprow2D, CommMat.nprow2D, CommMat.npcol2D);

      Int m_MG2DBlocksize = BlockSizeScaLAPACK;
      Int n_MG2DBlocksize = BlockSizeScaLAPACK;

      //Int m_MG2Local = m_MG/(m_MG2DBlocksize*CommMat.nprow2D)*m_MG2DBlocksize;
      //Int n_MG2Local = n_MG/(n_MG2DBlocksize*CommMat.npcol2D)*n_MG2DBlocksize;
      Int m_MG2Local, n_MG2Local; 

      //MPI_Comm rowComm = MPI_COMM_NULL;
      MPI_Comm colComm = MPI_COMM_NULL;

      Int mpirankRow, mpisizeRow, mpirankCol, mpisizeCol;

      //MPI_Comm_split( domain_.comm, mpirank / CommMat.nprow2D, mpirank, &rowComm );
      MPI_Comm_split( domain_.comm, mpirank % CommMat.nprow2D, mpirank, &colComm );

      //MPI_Comm_rank(rowComm, &mpirankRow);
      //MPI_Comm_size(rowComm, &mpisizeRow);

      MPI_Comm_rank(colComm, &mpirankCol);
      MPI_Comm_size(colComm, &mpisizeCol);

      #if 0
      if(CommMat.contxt2D >= 0){
        Cblacs_gridinfo(CommMat.contxt2D, &CommMat.nprow2D, &CommMat.npcol2D, &CommMat.myrow2D, &CommMat.mycol2D);
        nrows2D = SCALAPACK(numroc)(&m_MG, &m_MG2DBlocksize, &CommMat.myrow2D, &I_ZERO, &CommMat.nprow2D);
        ncols2D = SCALAPACK(numroc)(&n_MG, &n_MG2DBlocksize, &CommMat.mycol2D, &I_ZERO, &CommMat.npcol2D);
        lld2D = std::max( nrows2D, 1 );
      }

      SCALAPACK(descinit)(desc_MG2D, &m_MG, &n_MG, &m_MG2DBlocksize, &n_MG2DBlocksize, &I_ZERO, 
          &I_ZERO, &CommMat.contxt2D, &lld2D, &CommMat.info2D);
      #else
      // guofeng
      scalapack::Descriptor descMG2D;
      descMG2D.Init(m_MG, n_MG, m_MG2DBlocksize, n_MG2DBlocksize, I_ZERO, I_ZERO, CommMat.contxt2D);
      //m_MG2Local = nrows2D;
      //n_MG2Local = ncols2D;
      //GUOFENG
      m_MG2Local = SCALAPACK(numroc)(&m_MG, &m_MG2DBlocksize, &CommMat.myrow2D, &I_ZERO, &CommMat.nprow2D);
      n_MG2Local = SCALAPACK(numroc)(&n_MG, &n_MG2DBlocksize, &CommMat.mycol2D, &I_ZERO, &CommMat.npcol2D);
      #endif

      IntNumVec pivQRTmp1(ntotMG), pivQRTmp2(ntotMG), pivQRLocal(ntotMG);
      if( m_MG > ntot ){
        std::ostringstream msg;
        msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      // DiagR is only for debugging purpose
      //        DblNumVec diagRLocal( mb_MG );
      //        DblNumVec diagR( mb_MG );

      DblNumMat&  MG1D = MG;
      //DblNumMat  MG2D (m_MG2Local, n_MG2Local);
      //guofeng
      scalapack::ScaLAPACKMatrix<Real> MG2D;
      MG2D.SetDescriptor(descMG2D);

      #if 0
      SCALAPACK(pdgemr2d)(&m_MG, &n_MG, MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, 
          MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D, &CommMat.contxt1D );
      #else
      //guofeng
      SCALAPACK(pdgemr2d)(&m_MG, &n_MG, MG1D.Data(), &I_ONE, &I_ONE, descMG1D.Values(), 
          &MG2D.LocalMatrix()[0], &I_ONE, &I_ONE, descMG2D.Values(), &CommMat.contxt1D );
      #endif

      if(CommMat.contxt2D >= 0){

        Real timeQRCP1, timeQRCP2;
        GetTime( timeQRCP1 );

        SetValue( pivQRTmp1, 0 );
        #if 0
        scalapack::QRCPF( m_MG, n_MG, MG2D.Data(), desc_MG2D, pivQRTmp1.Data(), tau.Data() );
        #else
        //guofeng
        scalapack::QRCPF( m_MG, n_MG, MG2D.Data(), descMG2D.Values(), pivQRTmp1.Data(), tau.Data() );
        #endif

        //scalapack::QRCPR( m_MG, n_MG, numMu_, MG2D.Data(), desc_MG2D, pivQRTmp1.Data(), tau.Data(), BlockSizeScaLAPACK, 32);

        GetTime( timeQRCP2 );

#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time only for QRCP is " << timeQRCP2 - timeQRCP1 << " [s]" << std::endl << std::endl;
#endif

      }

      // Redistribute back eigenvectors
      //SetValue(MG1D, 0.0 );

      //SCALAPACK(pdgemr2d)( &m_MG, &n_MG, MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D,
      //    MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, &CommMat.contxt1D );

      // Combine the local pivQRTmp to global pivQR_
      SetValue( pivQRLocal, 0 );
      for( Int j = 0; j < n_MG2Local; j++ ){
        pivQRLocal[ (j / n_MG2DBlocksize) * n_MG2DBlocksize * CommMat.npcol2D + mpirankCol * n_MG2DBlocksize + j % n_MG2DBlocksize] = pivQRTmp1[j];
      }

      SetValue( pivQRTmp2, 0 );
      MPI_Allreduce( pivQRLocal.Data(), pivQRTmp2.Data(), 
          ntotMG, MPI_INT, MPI_SUM, colComm );

      SetValue( pivQR_, 0 );
      for( Int j = 0; j < ntotMG; j++ ){
        pivQR_(j) = MGIdx(pivQRTmp2(j));
      }

      //if( rowComm != MPI_COMM_NULL ) MPI_Comm_free( & rowComm );
      if( colComm != MPI_COMM_NULL ) MPI_Comm_free( & colComm );

      if(CommMat.contxt2D >= 0) {
        Cblacs_gridexit( CommMat.contxt2D );
      }

      if(CommMat.contxt1D >= 0) {
        Cblacs_gridexit( CommMat.contxt1D );
      }

    } // if(1) ScaLAPACL QRCP

    GetTime( timeQRCPEnd );

    //statusOFS << std::endl<< "All processors exit with abort in spinor.cpp." << std::endl;

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for QRCP alone is " <<
      timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for density fitting with QRCP is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    if (hybridDFType == "Kmeans+QRCP"){
      hybridDFType = "Kmeans";
    }
  } //  if( (isFixColumnDF == false) && (hybridDFType == "QRCP") )

  if( (isFixColumnDF == false) && (hybridDFType == "Kmeans") ){
   
    GetTime( timeSta );

    DblNumVec weight(ntot);
    Real* wp = weight.Data();
    Real timeW1,timeW2;
    Real timeKMEANSta, timeKMEANEnd;

    GetTime(timeW1); 
    DblNumVec phiW(ntot);
    SetValue(phiW,0.0);
    Real* phW = phiW.Data();
    Real* ph = phiCol.Data();

    for (int j = 0; j < numStateLocal; j++){
      for(int i = 0; i < ntot; i++){
        phW[i] += ph[i+j*ntot]*ph[i+j*ntot];
      }
    }
    MPI_Barrier(domain_.comm);
    MPI_Reduce(phW, wp, ntot, MPI_DOUBLE, MPI_SUM, 0, domain_.comm);
    MPI_Bcast(wp, ntot, MPI_DOUBLE, 0, domain_.comm);
    GetTime(timeW2);
    statusOFS << "Time for computing weight in Kmeans: " << timeW2-timeW1 << "[s]" << std::endl << std::endl;

    int rk = numMu_;
    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
    GetTime(timeKMEANSta);
    KMEAN(ntot, weight, rk, hybridDFKmeansTolerance, hybridDFKmeansMaxIter, hybridDFTolerance, domain_, pivQR_.Data());
    GetTime(timeKMEANEnd);
    statusOFS << "Time for Kmeans alone is " << timeKMEANEnd-timeKMEANSta << "[s]" << std::endl << std::endl;
 
    GetTime(timeEnd);
    
    statusOFS << "Time for density fitting with Kmeans is " << timeEnd-timeSta << "[s]" << std::endl << std::endl;


  } //  if( (isFixColumnDF == false) && (hybridDFType == "Kmeans") )


  // *********************************************************************
  // Compute the interpolation matrix via the density matrix formulation
  // *********************************************************************

  GetTime( timeSta );

  // PhiMu is scaled by the occupation number to reflect the "true" density matrix
  //IntNumVec pivMu(numMu_);
  IntNumVec pivMu1(numMu_);

  for( Int mu = 0; mu < numMu_; mu++ ){
    pivMu1(mu) = pivQR_(mu);
  }


  //if(ntot % mpisize != 0){
  //  for( Int mu = 0; mu < numMu_; mu++ ){
  //    Int k1 = (pivMu1(mu) / ntotLocalMG) - (ntot % mpisize);
  //    if(k1 > 0){
  //      pivMu1(mu) = pivQR_(mu) - k1;
  //    }
  //  }
  //}

  GetTime( timeSta1 );

  DblNumMat psiMuCol(numStateLocal, numMu_);
  DblNumMat phiMuCol(numStateLocal, numMu_);
  SetValue( psiMuCol, 0.0 );
  SetValue( phiMuCol, 0.0 );

  for (Int k=0; k<numStateLocal; k++) {
    for (Int mu=0; mu<numMu_; mu++) {
      psiMuCol(k, mu) = psiCol(pivMu1(mu),k);
      phiMuCol(k, mu) = phiCol(pivMu1(mu),k) * occupationRate[(k * mpisize + mpirank)];
    }
  }
  //guofeng
  #if 0
  DblNumMat psiMu2D(CommMat.nrowsNeNu2D, CommMat.ncolsNeNu2D);
  DblNumMat phiMu2D(CommMat.nrowsNeNu2D, CommMat.ncolsNeNu2D);
  SetValue( psiMu2D, 0.0 );
  SetValue( phiMu2D, 0.0 );
  #else
  scalapack::ScaLAPACKMatrix<Real>psiMu2D, phiMu2D;
  psiMu2D.SetDescriptor(descNeNu2D);
  phiMu2D.SetDescriptor(descNeNu2D);
  #endif
  
  #if 0
  SCALAPACK(pdgemr2d)(&Ne, &Nu, psiMuCol.Data(), &I_ONE, &I_ONE, CommMat.desc_NeNu1D, 
      psiMu2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NeNu2D, &CommMat.contxt11 );

  SCALAPACK(pdgemr2d)(&Ne, &Nu, phiMuCol.Data(), &I_ONE, &I_ONE, CommMat.desc_NeNu1D, 
      phiMu2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NeNu2D, &CommMat.contxt11 );
  #else
  SCALAPACK(pdgemr2d)(&Ne, &Nu, psiMuCol.Data(), &I_ONE, &I_ONE, descNeNu1D.Values(), 
      &psiMu2D.LocalMatrix()[0], &I_ONE, &I_ONE, descNeNu2D.Values(), &CommMat.contxt11 );

  SCALAPACK(pdgemr2d)(&Ne, &Nu, phiMuCol.Data(), &I_ONE, &I_ONE, descNeNu1D.Values(), 
      &phiMu2D.LocalMatrix()[0], &I_ONE, &I_ONE, descNeNu2D.Values(), &CommMat.contxt11 );
  #endif

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing psiMuRow and phiMuRow is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  GetTime( timeSta1 );

  //DblNumMat psi2D(CommMat.nrowsNgNe2D, CommMat.ncolsNgNe2D);
  //DblNumMat phi2D(CommMat.nrowsNgNe2D, CommMat.ncolsNgNe2D);
  //SetValue( psi2D, 0.0 );
  //SetValue( phi2D, 0.0 );
  scalapack::ScaLAPACKMatrix<Real> psi2D, phi2D;
  psi2D.SetDescriptor(descNgNe2D);
  phi2D.SetDescriptor(descNgNe2D);

  #if 0
  SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNe1DCol, 
      psi2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNe2D, &CommMat.contxt1 );
  #else
  SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, descNgNe1DCol.Values(), 
      &psi2D.LocalMatrix()[0], &I_ONE, &I_ONE, descNgNe2D.Values(), &CommMat.contxt1 );
  #endif
  //printf("descNgNe2D_2 = %d\n",CommMat.desc_NgNe2D);
  SCALAPACK(pdgemr2d)(&Ng, &Ne, phiCol.Data(), &I_ONE, &I_ONE, descNgNe1DCol.Values(), 
      &phi2D.LocalMatrix()[0], &I_ONE, &I_ONE, descNgNe2D.Values(), &CommMat.contxt1 );
  //printf("descNgNe2D_3 = %d\n",CommMat.desc_NgNe2D);

  DblNumMat PpsiMu2D(CommMat.nrowsNgNu2D, CommMat.ncolsNgNu2D);
  DblNumMat PphiMu2D(CommMat.nrowsNgNu2D, CommMat.ncolsNgNu2D);
  SetValue( PpsiMu2D, 0.0 );
  SetValue( PphiMu2D, 0.0 );

  SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Ne, 
      &D_ONE,
      psi2D.Data(), &I_ONE, &I_ONE, descNgNe2D.Values(),
      psiMu2D.Data(), &I_ONE, &I_ONE, descNeNu2D.Values(), 
      &D_ZERO,
      PpsiMu2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu2D);
  //printf("descNgNe2D_4 = %d\n",CommMat.desc_NgNe2D);

  SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Ne, 
      &D_ONE,
      phi2D.Data(), &I_ONE, &I_ONE, descNgNe2D.Values(),
      phiMu2D.Data(), &I_ONE, &I_ONE, descNeNu2D.Values(), 
      &D_ZERO,
      PphiMu2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu2D);
  //printf("descNgNe2D_5 = %d\n",CommMat.desc_NgNe2D);

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for PpsiMu and PphiMu GEMM is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  GetTime( timeSta1 );

  DblNumMat Xi2D(CommMat.nrowsNgNu2D, CommMat.ncolsNgNu2D);
  SetValue( Xi2D, 0.0 );

  Real* Xi2DPtr = Xi2D.Data();
  Real* PpsiMu2DPtr = PpsiMu2D.Data();
  Real* PphiMu2DPtr = PphiMu2D.Data();

  for( Int g = 0; g < CommMat.nrowsNgNu2D * CommMat.ncolsNgNu2D; g++ ){
    Xi2DPtr[g] = PpsiMu2DPtr[g] * PphiMu2DPtr[g];
  }

  DblNumMat Xi1D(CommMat.nrowsNgNu1D, CommMat.ncolsNuNu1D);
  SetValue( Xi1D, 0.0 );

  SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu2D, 
      Xi1D.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu1D, &CommMat.contxt2 );

  DblNumMat PMuNu1D(CommMat.nrowsNuNu1D, CommMat.ncolsNuNu1D);
  SetValue( PMuNu1D, 0.0 );

  for (Int mu=0; mu<CommMat.nrowsNuNu1D; mu++) {
    for (Int nu=0; nu<CommMat.ncolsNuNu1D; nu++) {
      PMuNu1D(mu, nu) = Xi1D(pivMu1(mu),nu);
    }
  }

  DblNumMat PMuNu2D(CommMat.nrowsNuNu2D, CommMat.ncolsNuNu2D);
  SetValue( PMuNu2D, 0.0 );

  SCALAPACK(pdgemr2d)(&Nu, &Nu, PMuNu1D.Data(), &I_ONE, &I_ONE, CommMat.desc_NuNu1D, 
      PMuNu2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NuNu2D, &CommMat.contxt1 );

  GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing PMuNu is " <<
    timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif


  //Method 2
  if(1){

    GetTime( timeSta1 );

    SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NuNu2D, &CommMat.info2);
    SCALAPACK(pdpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NuNu2D, &CommMat.info2);

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu Potrf is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeSta1 );

    DblNumMat PMuNu2DTemp(CommMat.nrowsNuNu2D, CommMat.ncolsNuNu2D);
    SetValue( PMuNu2DTemp, 0.0 );

    lapack::lacpy( lapack::MatrixType::General, CommMat.nrowsNuNu2D, CommMat.ncolsNuNu2D, PMuNu2D.Data(), CommMat.nrowsNuNu2D, PMuNu2DTemp.Data(), CommMat.nrowsNuNu2D );

    SCALAPACK(pdtradd)("U", "T", &Nu, &Nu, 
        &D_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NuNu2D, 
        &D_ZERO,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, CommMat.desc_NuNu2D);

    DblNumMat Xi2DTemp(CommMat.nrowsNgNu2D, CommMat.ncolsNgNu2D);
    SetValue( Xi2DTemp, 0.0 );

    SCALAPACK(pdgemm)("N", "N", &Ng, &Nu, &Nu, 
        &D_ONE,
        Xi2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu2D,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, CommMat.desc_NuNu2D, 
        &D_ZERO,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu2D);

    SetValue( Xi2D, 0.0 );
    lapack::lacpy( lapack::MatrixType::General, CommMat.nrowsNgNu2D, CommMat.ncolsNgNu2D, Xi2DTemp.Data(), CommMat.nrowsNgNu2D, Xi2D.Data(), CommMat.nrowsNgNu2D );

    GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for PMuNu and Xi pdgemm is " <<
      timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif

  }


  GetTime( timeEnd );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the interpolation vectors is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif



  // *********************************************************************
  // Solve the Poisson equations.
  // Store VXi separately. This is not the most memory efficient
  // implementation
  // *********************************************************************
  #if 1
  DblNumMat VXi2D(CommMat.nrowsNgNu2D, CommMat.ncolsNgNu2D);
  //guofeng
  #else
  scalapack::ScaLAPACKMatrix<Real> VXi2D;
  VXi2D.SetDescriptor(descNgNu2D);
  #endif

  {
    GetTime( timeSta );
    // XiCol used for both input and output
    DblNumMat XiCol(ntot, numMuLocal);
    SetValue(XiCol, 0.0 );

    SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu2D, 
        XiCol.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu1D, &CommMat.contxt2 );

    GetTime( timeSta );
    for( Int mu = 0; mu < numMuLocal; mu++ ){
      blas::copy( ntot,  XiCol.VecData(mu), 1, fft.inputVecR2C.Data(), 1 );

      FFTWExecute ( fft, fft.forwardPlanR2C );

      for( Int ig = 0; ig < ntotR2C; ig++ ){
        fft.outputVecR2C(ig) *= -exxFraction * exxgkkR2C(ig);
      }

      FFTWExecute ( fft, fft.backwardPlanR2C );

      blas::copy( ntot, fft.inputVecR2C.Data(), 1, XiCol.VecData(mu), 1 );

    } // for (mu)

    SetValue(VXi2D, 0.0 );
    #if 1
    SCALAPACK(pdgemr2d)(&Ng, &Nu, XiCol.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu1D, 
        VXi2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu2D, &CommMat.contxt1 );
    #else
    SCALAPACK(pdgemr2d)(&Ng, &Nu, XiCol.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu1D, 
        &VXi2D.LocalMatrix()[0], &I_ONE, &I_ONE, descNgNu2D.Values(), &CommMat.contxt1 );
    #endif


    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for solving Poisson-like equations is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  // Prepare for the computation of the M matrix
  GetTime( timeSta );

  DblNumMat MMatMuNu2D(CommMat.nrowsNuNu2D, CommMat.ncolsNuNu2D);
  SetValue(MMatMuNu2D, 0.0 );

  SCALAPACK(pdgemm)("T", "N", &Nu, &Nu, &Ng, 
      &D_MinusONE,
      Xi2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu2D,
      VXi2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu2D, 
      &D_ZERO,
      MMatMuNu2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NuNu2D);

  DblNumMat phiMuNu2D(CommMat.nrowsNuNu2D, CommMat.ncolsNuNu2D);
  SCALAPACK(pdgemm)("T", "N", &Nu, &Nu, &Ne, 
      &D_ONE,
      phiMu2D.Data(), &I_ONE, &I_ONE, descNeNu2D.Values(),
      phiMu2D.Data(), &I_ONE, &I_ONE, descNeNu2D.Values(), 
      &D_ZERO,
      phiMuNu2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NuNu2D);

  Real* MMat2DPtr = MMatMuNu2D.Data();
  Real* phi2DPtr  = phiMuNu2D.Data();

  for( Int g = 0; g < CommMat.nrowsNuNu2D * CommMat.ncolsNuNu2D; g++ ){
    MMat2DPtr[g] *= phi2DPtr[g];
  }

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for preparing the M matrix is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // *********************************************************************
  // Compute the exchange potential and the symmetrized inner product
  // *********************************************************************

  GetTime( timeSta );

  // Rewrite VXi by VXi.*PcolPhi
  #if 1
  Real* VXi2DPtr = VXi2D.Data();
  Real* PphiMu2DPtr1 = PphiMu2D.Data();
  for( Int g = 0; g < CommMat.nrowsNgNu2D * CommMat.ncolsNgNu2D; g++ ){
    VXi2DPtr[g] *= PphiMu2DPtr1[g];
  }
  #else
  // question: why not use scalapackr2d
  SCALAPACK(pdgemr2d)(&Ng, &Nu, PphiMu2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu2D, 
        &VXi2D.LocalMatrix()[0], &I_ONE, &I_ONE, CommMat.desc_NgNu2D, &CommMat.contxt2 );
  #endif

  // NOTE: Hpsi must be zero in order to compute the M matrix later
  // guofeng
  //DblNumMat Hpsi2D( CommMat.nrowsNgNe2D, CommMat.ncolsNgNe2D );
  //SetValue( Hpsi2D, 0.0 );

  
  //printf("LLDNgNe2D = %d\n",CommMat.lldNgNe2D);
  printf("descNgNe2D_Auto_Values = %d\n",descNgNe2D.Values());

  scalapack::ScaLAPACKMatrix<Real>  Hpsi2D, HpsiCol;
  Hpsi2D.SetDescriptor(descNgNe2D);
  HpsiCol.SetDescriptor(descNgNe1DCol);

  SCALAPACK(pdgemm)("N", "T", &Ng, &Ne, &Nu, 
      &D_ONE,
      VXi2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NgNu2D,
      psiMu2D.Data(), &I_ONE, &I_ONE, descNeNu2D.Values(), 
      &D_ZERO,
      &Hpsi2D.LocalMatrix()[0], &I_ONE, &I_ONE, Hpsi2D.Desc().Values());

  //blNumMat HpsiCol( ntot, numStateLocal );
  //SetValue(HpsiCol, 0.0 );
  //Gemr2d(Hpsi2D, HpsiCol);
  //auto desc_Hpsi2D = to_scalapacpp
  SCALAPACK(pdgemr2d)(&Ng, &Ne, Hpsi2D.Data(), &I_ONE, &I_ONE, Hpsi2D.Desc().Values(), 
      &HpsiCol.LocalMatrix()[0], &I_ONE, &I_ONE, HpsiCol.Desc().Values(), &CommMat.contxt2 );

  lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, HpsiCol.Data(), ntot, Hpsi.Data(), ntot );

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing the exchange potential is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // Compute the matrix VxMat = -Psi'* vexxPsi and symmetrize
  // vexxPsi (Hpsi) must be zero before entering this routine
  VxMat.Resize( numStateTotal, numStateTotal );

  GetTime( timeSta );

  if(1){

    //DblNumMat VxMat2D( CommMat.nrowsNeNe2D, CommMat.ncolsNeNe2D );
    //DblNumMat VxMatTemp2D( CommMat.nrowsNuNe2D, CommMat.ncolsNuNe2D );
    //guofeng


    scalapack::ScaLAPACKMatrix<Real> VxMatTemp2D, VxMat2D;
    VxMatTemp2D.SetDescriptor(descNuNe2D);
    VxMat2D.SetDescriptor(descNuNe2D);

    SCALAPACK(pdgemm)("N", "T", &Nu, &Ne, &Nu, 
        &D_ONE,
        MMatMuNu2D.Data(), &I_ONE, &I_ONE, CommMat.desc_NuNu2D,
        psiMu2D.Data(), &I_ONE, &I_ONE, descNeNu2D.Values(), 
        &D_ZERO,
        &VxMatTemp2D.LocalMatrix()[0], &I_ONE, &I_ONE, descNuNe2D.Values());

    SCALAPACK(pdgemm)("N", "N", &Ne, &Ne, &Nu, 
        &D_ONE,
        psiMu2D.Data(), &I_ONE, &I_ONE, descNeNu2D.Values(), 
        VxMatTemp2D.Data(), &I_ONE, &I_ONE, descNuNe2D.Values(), 
        &D_ZERO,
        &VxMat2D.LocalMatrix()[0], &I_ONE, &I_ONE, descNeNe2D.Values());

    SCALAPACK(pdgemr2d)(&Ne, &Ne, VxMat2D.Data(), &I_ONE, &I_ONE, VxMat2D.Desc().Values(), 
        VxMat.Data(), &I_ONE, &I_ONE, descNeNe0D.Values(), &CommMat.contxt2 );
    printf("new16\n");
    //if(mpirank == 0){
    //  MPI_Bcast( VxMat.Data(), Ne * Ne, MPI_DOUBLE, 0, domain_.comm );
    //}

  }
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing VxMat in the sym format is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  if(CommMat.contxt0 >= 0) {
    Cblacs_gridexit( CommMat.contxt0 );
  }

  if(CommMat.contxt1 >= 0) {
    Cblacs_gridexit( CommMat.contxt1 );
  }

  if(CommMat.contxt11 >= 0) {
    Cblacs_gridexit( CommMat.contxt11 );
  }

  if(CommMat.contxt2 >= 0) {
    Cblacs_gridexit( CommMat.contxt2 );
  }

  MPI_Barrier(domain_.comm);
  // guofeng
  printf("IN DF7\n");

  return ;
}        // -----  end of method Spinor::AddMultSpinorEXXDF7  ----- 


}  // namespace scales
