/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Weile Jia

This file is part of ScalES. All rights reserved.

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
/// @file device_spinor.cpp
/// @brief Spinor (wavefunction) for the global domain or extended
/// element.
/// @date 2020-08-12
#include  "spinor.hpp"
#include  "utility.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"

namespace scales{

using namespace scales::scalapack;

using namespace scales::PseudoComponent;

using namespace scales::esdf;

Spinor::Spinor ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    Int numStateLocal,
    const bool owndata, 
    Real* data, 
    bool isGPU )
{
  if(!isGPU || owndata)
    ErrorHandling(" GPU Spinor setup error.");

  // data is a GPU data.. 
  this->SetupGPU( dm, numComponent, numStateTotal, numStateLocal, owndata, data);

}         // -----  end of method Spinor::Spinor  ----- 

void Spinor::SetupGPU ( const Domain &dm, 
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

  cu_wavefun_  = deviceNumTns<Real>( dm.NumGridTotal(), numComponent, numStateLocal,
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
  for (Int i = 0; i < numStateLocal; i++){
    wavefunIdx_[i] = i * mpisize + mpirank ;
  }

}         // -----  end of method Spinor::Setup  ----- 

void
Spinor::AddTeterPrecond (Fourier* fftPtr, deviceNumTns<Real>& Hpsi)
{
  Fourier& fft = *fftPtr;
  if( !fftPtr->isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int ntot = cu_wavefun_.m();
  Int ncom = cu_wavefun_.n();
  Int nocc = cu_wavefun_.p();

  if( fftPtr->domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  Int ntothalf = fftPtr->numGridTotalR2C;

  deviceDblNumVec cu_psi(ntot);
  deviceDblNumVec cu_psi_out(2*ntothalf);
  //deviceDblNumVec cu_TeterPrecond(ntothalf);
  if( !teter_gpu_flag ) {
     // copy the Teter Preconditioner into GPU. only once. 
     dev_TeterPrecond = (double*) device_malloc( sizeof(Real) * ntothalf);
     device_memcpy_HOST2DEVICE(dev_TeterPrecond, fftPtr->TeterPrecondR2C.Data(), sizeof(Real)*ntothalf);
     //device_memcpy_HOST2DEVICE(cu_TeterPrecond.Data(), fftPtr->TeterPrecondR2C.Data(), sizeof(Real)*ntothalf);
     teter_gpu_flag = true;
  } 
  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
      device_memcpy_DEVICE2DEVICE(cu_psi.Data(), cu_wavefun_.VecData(j,k), sizeof(Real)*ntot);
      deviceFFTExecuteForward( fft, cuPlanR2C[0], 0, cu_psi, cu_psi_out);
      //device_teter( reinterpret_cast<cuDoubleComplex*>(cu_psi_out.Data()), cu_TeterPrecond.Data(), ntothalf);
      device_teter( reinterpret_cast<cuDoubleComplex*>(cu_psi_out.Data()), dev_TeterPrecond, ntothalf);

      deviceFFTExecuteInverse( fft, cuPlanC2R[0], 0, cu_psi_out, cu_psi);
      
      device_memcpy_DEVICE2DEVICE(Hpsi.VecData(j,k), cu_psi.Data(), ntot*sizeof(Real));
      // not a good style. first set Hpsi to zero, then do a axpy. should do copy
      //blas::Axpy( ntot, 1.0, fft.inputVecR2C.Data(), 1, Hpsi.VecData(j,k), 1 );
    }
  }

  return ;
}         // -----  end of method Spinor::AddTeterPrecond for the GPU code. ----- 

/*
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

  Int ntothalf = fftPtr->numGridTotalR2C;

  deviceDblNumVec cu_psi(ntot);
  deviceDblNumVec cu_psi_out(2*ntothalf);
  deviceDblNumVec cu_TeterPrecond(ntothalf);
  device_memcpy_HOST2DEVICE(cu_TeterPrecond.Data(), fftPtr->TeterPrecondR2C.Data(), sizeof(Real)*ntothalf);

  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
      device_memcpy_HOST2DEVICE(cu_psi.Data(), wavefun_.VecData(j,k), sizeof(Real)*ntot);
      deviceFFTExecuteForward( fft, cuPlanR2C[0], 0, cu_psi, cu_psi_out);
      device_teter( reinterpret_cast<cuDoubleComplex*>(cu_psi_out.Data()), cu_TeterPrecond.Data(), ntothalf);

      deviceFFTExecuteInverse( fft, cuPlanC2R[0], 0, cu_psi_out, cu_psi);
      
      device_memcpy_DEVICE2HOST(Hpsi.VecData(j,k), cu_psi.Data(), ntot*sizeof(Real));
      // not a good style. first set Hpsi to zero, then do a axpy. should do copy
      //blas::Axpy( ntot, 1.0, fft.inputVecR2C.Data(), 1, Hpsi.VecData(j,k), 1 );
    }
  }

  return ;
}         // -----  end of method Spinor::AddTeterPrecond for the GPU code. ----- 
*/
void
Spinor::AddMultSpinorR2C ( Fourier& fft, const DblNumVec& vtot, 
    const std::vector<PseudoPot>& pseudo, deviceNumTns<Real>& Hpsi )
{

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;
  Int ntot = cu_wavefun_.m();
  Int ncom = cu_wavefun_.n();
  Int numStateLocal = cu_wavefun_.p();
  Int ntotFine = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  Real timeSta, timeEnd;
  GetTime( timeSta );
  // Temporary variable for saving wavefunction on a fine grid
  DblNumVec psiFine(ntotFine);
  DblNumVec psiUpdateFine(ntotFine);
  deviceDblNumVec cu_psi(ntot);
  deviceDblNumVec cu_psi_out(2*ntotR2C);
  deviceDblNumVec cu_psi_fine(ntotFine);
  deviceDblNumVec cu_psi_fineUpdate(ntotFine);
  deviceDblNumVec cu_psi_fine_out(2*ntotR2CFine);
  
  if( NL_gpu_flag == false ) 
  {
    // get the total number of the nonlocal vector
     Int totNLNum = 0;
     totPart_gpu = 1;
     Int natm = pseudo.size();
     for (Int iatm=0; iatm<natm; iatm++) {
        Int nobt = pseudo[iatm].vnlList.size();
        totPart_gpu += nobt;
        for(Int iobt = 0; iobt < nobt; iobt++)
        {
              const SparseVec &vnlvecFine = pseudo[iatm].vnlList[iobt].first;
              const IntNumVec &ivFine = vnlvecFine.first;
              totNLNum += ivFine.m();
        }
    } 
    DblNumVec NLvecFine(totNLNum);
    IntNumVec NLindex(totNLNum);
    IntNumVec NLpart (totPart_gpu);
    DblNumVec atom_weight(totPart_gpu);
  
    Int index = 0;
    Int ipart = 0;
    for (Int iatm=0; iatm<natm; iatm++) {
        Int nobt = pseudo[iatm].vnlList.size();
        for(Int iobt = 0; iobt < nobt; iobt++)
        {
            const Real       vnlwgt = pseudo[iatm].vnlList[iobt].second;
            const SparseVec &vnlvecFine = pseudo[iatm].vnlList[iobt].first;
            const IntNumVec &ivFine = vnlvecFine.first;
            const DblNumMat &dvFine = vnlvecFine.second;
            const Int    *ivFineptr = ivFine.Data();
            const Real   *dvFineptr = dvFine.VecData(VAL);
            atom_weight(ipart) = vnlwgt *vol/Real(ntotFine);
  
            NLpart(ipart++) = index;
            for(Int i = 0; i < ivFine.m(); i++)
            {
               NLvecFine(index)  = *(dvFineptr++);
               NLindex(index++)  = *(ivFineptr++);
            }
        }
    }
    NLpart(ipart) = index;
    dev_NLvecFine   = ( double*) device_malloc ( sizeof(double) * totNLNum );
    dev_NLindex     = ( int*   ) device_malloc ( sizeof(int )   * totNLNum );
    dev_NLpart      = ( int*   ) device_malloc ( sizeof(int )   * totPart_gpu );
    dev_atom_weight = ( double*) device_malloc ( sizeof(double) * totPart_gpu );
    dev_temp_weight = ( double*) device_malloc ( sizeof(double) * totPart_gpu );

    dev_idxFineGridR2C = ( int*) device_malloc ( sizeof(int   ) * ntotR2C );
    dev_gkkR2C      = ( double*) device_malloc ( sizeof(double) * ntotR2C );
    dev_vtot        = ( double*) device_malloc ( sizeof(double) * ntotFine);

    device_memcpy_HOST2DEVICE( dev_NLvecFine,   NLvecFine.Data(),   totNLNum * sizeof(double) );
    device_memcpy_HOST2DEVICE( dev_atom_weight, atom_weight.Data(), totPart_gpu* sizeof(double) );
    device_memcpy_HOST2DEVICE( dev_NLindex,     NLindex.Data(),     totNLNum * sizeof(int) );
    device_memcpy_HOST2DEVICE( dev_NLpart ,     NLpart.Data(),      totPart_gpu  * sizeof(int) );

    device_memcpy_HOST2DEVICE(dev_idxFineGridR2C, fft.idxFineGridR2C.Data(), sizeof(Int) *ntotR2C); 
    device_memcpy_HOST2DEVICE(dev_gkkR2C, fft.gkkR2C.Data(), sizeof(Real) *ntotR2C); 
    device_memcpy_HOST2DEVICE(dev_vtot, vtot.Data(), sizeof(Real) *ntotFine); 

    NL_gpu_flag = true;
    vtot_gpu_flag = true;
/*
    deviceDblNumVec cu_NLvecFine(totNLNum);
    deviceIntNumVec cu_NLindex(totNLNum);
    deviceIntNumVec cu_NLpart (totPart_gpu);
    deviceDblNumVec cu_atom_weight( totPart_gpu);
    deviceDblNumVec cu_temp_weight( totPart_gpu);
  
    cu_NLvecFine.CopyFrom(NLvecFine);
    cu_NLindex.CopyFrom(NLindex);
    cu_NLpart.CopyFrom(NLpart);
    cu_atom_weight.CopyFrom(atom_weight);
    // cuda nonlocal vector created.
    //copy index into the GPU. note, this will be moved out of Hpsi. 
    deviceIntNumVec cu_idxFineGridR2C(ntotR2C);
    deviceDblNumVec cu_gkkR2C(ntotR2C);
    deviceDblNumVec cu_vtot(ntotFine);
    device_memcpy_HOST2DEVICE(cu_idxFineGridR2C.Data(), fft.idxFineGridR2C.Data(), sizeof(Int) *ntotR2C); 
    device_memcpy_HOST2DEVICE(cu_gkkR2C.Data(), fft.gkkR2C.Data(), sizeof(Real) *ntotR2C); 
    device_memcpy_HOST2DEVICE(cu_vtot.Data(), vtot.Data(), sizeof(Real) *ntotFine); 
*/
  }
  if( !vtot_gpu_flag) {
    dev_vtot        = ( double*) device_malloc ( sizeof(double) * ntotFine);
    device_memcpy_HOST2DEVICE(dev_vtot, vtot.Data(), sizeof(Real) *ntotFine); 
    vtot_gpu_flag = true;
  }
  Real timeSta1, timeEnd1;
  
  Real timeFFTCoarse = 0.0;
  Real timeFFTFine = 0.0;
  Real timeOther = 0.0;
  Int  iterFFTCoarse = 0;
  Int  iterFFTFine = 0;
  Int  iterOther = 0;
  Real timeNonlocal = 0.0;

  GetTime( timeSta1 );
 
  if(0)
  {
    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {

        SetValue( fft.inputVecR2C, 0.0 );
        SetValue( fft.outputVecR2C, Z_ZERO );

        blas::Copy( ntot, wavefun_.VecData(j,k), 1,
            fft.inputVecR2C.Data(), 1 );
        FFTWExecute ( fft, fft.forwardPlanR2C ); // So outputVecR2C contains the FFT result now


        for (Int i=0; i<ntotR2C; i++)
        {
          if(fft.gkkR2C(i) > 5.0)
            fft.outputVecR2C(i) = Z_ZERO;
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );
        blas::Copy( ntot,  fft.inputVecR2C.Data(), 1,
            wavefun_.VecData(j,k), 1 );

      }
    }
  }

  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << " Spinor overheader is : "<< timeEnd - timeSta <<  std::endl;
#endif

  for (Int k=0; k<numStateLocal; k++) {
    for (Int j=0; j<ncom; j++) {

      // R2C version
      // For c2r and r2c transforms, the default is to DESTROY the
      // input, therefore a copy of the original matrix is necessary. 
      GetTime( timeSta );
      device_memcpy_DEVICE2DEVICE(cu_psi.Data(), cu_wavefun_.VecData(j,k), sizeof(Real)*ntot);
      deviceFFTExecuteForward( fft, cuPlanR2C[0], 0, cu_psi, cu_psi_out);
      GetTime( timeEnd );
      iterFFTCoarse = iterFFTCoarse + 1;
      timeFFTCoarse = timeFFTCoarse + ( timeEnd - timeSta );

      // Interpolate wavefunction from coarse to fine grid
      SetValue(cu_psi_fine_out, 0.0);
      Real fac = sqrt( double(ntot) / double(ntotFine) );
      device_interpolate_wf_C2F( reinterpret_cast<cuDoubleComplex*>(cu_psi_out.Data()), 
                               reinterpret_cast<cuDoubleComplex*>(cu_psi_fine_out.Data()), 
                               dev_idxFineGridR2C,
                               ntotR2C, 
                               fac);

      GetTime( timeSta );
      deviceFFTExecuteInverse(fft, cuPlanC2RFine[0], 1, cu_psi_fine_out, cu_psi_fine);
      GetTime( timeEnd );
      iterFFTFine = iterFFTFine + 1;
      timeFFTFine = timeFFTFine + ( timeEnd - timeSta );

      // cuda local psedupotential , note, we need psi and psiUpdate for the next nonlocal calculation.
      device_memcpy_DEVICE2DEVICE(cu_psi_fineUpdate.Data(), cu_psi_fine.Data(), sizeof(Real)*ntotFine);
      device_vtot( cu_psi_fineUpdate.Data(),
                 dev_vtot,
                 ntotFine);

      // Add the contribution from nonlocal pseudopotential
      GetTime( timeSta );
      device_calculate_nonlocal(cu_psi_fineUpdate.Data(), 
                              cu_psi_fine.Data(),
                              dev_NLvecFine,
                              dev_NLindex,
                              dev_NLpart,
                              dev_atom_weight,
                              dev_temp_weight,
                              totPart_gpu-1);
      GetTime( timeEnd );
      timeNonlocal = timeNonlocal + ( timeEnd - timeSta );

      // Laplacian operator. Perform inverse Fourier transform in the end
      device_laplacian(  reinterpret_cast<cuDoubleComplex*>( cu_psi_out.Data()), 
                       dev_gkkR2C,
                       ntotR2C);
      // Fine to coarse grid
      // Note the update is important since the Laplacian contribution is already taken into account.
      // The computation order is also important
      GetTime( timeSta );

      device_memcpy_DEVICE2DEVICE(cu_psi_fine.Data(), cu_psi_fineUpdate.Data(), sizeof(Real)*ntotFine);
      deviceFFTExecuteForward(fft, cuPlanR2CFine[0], 1, cu_psi_fine, cu_psi_fine_out);

      GetTime( timeEnd );
      iterFFTFine = iterFFTFine + 1;
      timeFFTFine = timeFFTFine + ( timeEnd - timeSta );

      fac = sqrt( double(ntotFine) / double(ntot) );
      device_interpolate_wf_F2C( reinterpret_cast<cuDoubleComplex*>(cu_psi_fine_out.Data()), 
                               reinterpret_cast<cuDoubleComplex*>(cu_psi_out.Data()), 
                               dev_idxFineGridR2C,
                               ntotR2C, 
                               fac);

      GetTime( timeSta );

      //CUDA FFT inverse and copy back.
      deviceFFTExecuteInverse( fft, cuPlanC2R[0], 0, cu_psi_out, cu_psi);
      device_memcpy_DEVICE2DEVICE(Hpsi.VecData(j,k), cu_psi.Data(), sizeof(Real)*ntot);

      GetTime( timeEnd );
      iterFFTCoarse = iterFFTCoarse + 1;
      timeFFTCoarse = timeFFTCoarse + ( timeEnd - timeSta );

    } // j++
  } // k++

  GetTime( timeEnd1 );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd1 - timeSta1 ) - timeFFTCoarse - timeFFTFine;

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for iterFFTCoarse    = " << iterFFTCoarse       << "  timeFFTCoarse    = " << timeFFTCoarse << std::endl;
    statusOFS << "Time for iterFFTFine      = " << iterFFTFine         << "  timeFFTFine    = " << timeFFTFine << std::endl;
    statusOFS << "Time for iterOther        = " << iterOther           << "  timeOther        = " << timeOther << std::endl;
    statusOFS << "Time for nonlocal         = " << 1                   << "  timeNonlocal     = " << timeNonlocal<< std::endl;
#endif

  return ;
}

void Spinor::AddMultSpinorEXX ( Fourier& fft, 
    const NumTns<Real>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Real  numSpin,
    const DblNumVec& occupationRate,
    deviceNumTns<Real>& Hpsi )
{
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  Real timeSta, timeEnd;
  Real timeBcast = 0.0;
  Real timeBuf   = 0.0;
  Real timeMemcpy= 0.0;
  Real timeCompute = 0.0;
  Int  BcastTimes = 0;

  //MPI_Barrier(domain_.comm);
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
  deviceDblNumVec cu_phiTemp(ntot);
  deviceDblNumVec cu_psi(ntot);
  deviceDblNumVec cu_psi_out(2*ntotR2C);
  deviceDblNumVec cu_exxgkkR2C(ntotR2C);
  deviceDblNumMat cu_wave(ntot, numStateLocal);

  device_memcpy_HOST2DEVICE(cu_exxgkkR2C.Data(), exxgkkR2C.Data(), sizeof(Real)*ntotR2C);


  Int numStateLocalTemp;

  //MPI_Barrier(domain_.comm);

  // book keeping the old interface while do the GPU inside. 
  // 1st version code. 
  device_memcpy_HOST2DEVICE(cu_wave.Data(), wavefun_.Data(), sizeof(Real)* numStateLocal * ntot );
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

        //SetValue( phiTemp, 0.0 );

        GetTime( timeSta );
        if( iproc == mpirank )
        { 
          Real* phiPtr = phi.VecData(jphi, kphi);
          for( Int ir = 0; ir < ntot; ir++ ){
            phiTemp(ir) = phiPtr[ir];
          }
        }
        GetTime( timeEnd );
        timeBuf += timeEnd - timeSta;

        GetTime( timeSta );
        MPI_Bcast( phiTemp.Data(), ntot, MPI_DOUBLE, iproc, domain_.comm );
        BcastTimes ++;
        GetTime( timeEnd );
        timeBcast += timeEnd - timeSta;

        // version 1: only do the GPU for the inner most part. 
        // note that the ncom == 1; it is the spin.

        // copy the phiTemp to GPU.
        GetTime( timeSta );
        device_memcpy_HOST2DEVICE(cu_phiTemp.Data(), phiTemp.Data(), sizeof(Real)*ntot);
        Real fac = -exxFraction * occupationRate[wavefunIdxTemp(kphi)];  
        GetTime( timeEnd );
        timeMemcpy += timeEnd - timeSta;

        
        GetTime( timeSta );
        for (Int k=0; k<numStateLocal; k++) {
          for (Int j=0; j<ncom; j++) {

            //device_memcpy_DEVICE2DEVICE(cu_psi.Data(), & cu_wave(0,k), sizeof(Real)*ntot);
            device_set_vector( cu_psi.Data(), &cu_wave(0,k), ntot);

            // input vec = psi * phi
            device_vtot(cu_psi.Data(), cu_phiTemp.Data(), ntot);

            // exec the deviceFFT. 
            deviceFFTExecuteForward( fft, cuPlanR2C[0], 0, cu_psi, cu_psi_out);

            // Solve the Poisson-like problem for exchange
     	    // note, exxgkkR2C apply to psi exactly like teter or laplacian
            device_teter( reinterpret_cast<cuDoubleComplex*> (cu_psi_out.Data()), cu_exxgkkR2C.Data(), ntotR2C );

	    // exec the deviceFFT. 
            deviceFFTExecuteInverse( fft, cuPlanC2R[0], 0, cu_psi_out, cu_psi);

            // multiply by the occupationRate.
	    // multiply with fac.
            //Real *cu_HpsiPtr = Hpsi.VecData(j,k);
            //device_Axpyz( cu_HpsiPtr, 1.0, cu_psi.Data(), fac, cu_phiTemp.Data(), ntot);
            device_Axpyz( Hpsi.VecData(j,k), 1.0, cu_psi.Data(), fac, cu_phiTemp.Data(), ntot);
            
          } // for (j)
        } // for (k)
        GetTime( timeEnd );
        timeCompute += timeEnd - timeSta;


      } // for (jphi)
    } // for (kphi)

  } //iproc

 
  //MPI_Barrier(domain_.comm);

#if ( _DEBUGlevel_ >= 0 )
   statusOFS << "Time for SendBuf is " << timeBuf << " [s] Bcast time: " << 
     timeBcast << " [s]" <<  " Bcast times: " << BcastTimes 
     << " memCpy time: " << timeMemcpy  << " compute time: "
     << timeCompute << std::endl << std::endl;
   statusOFS << " time added are: " << timeBuf + timeBcast  + timeMemcpy + timeCompute << std::endl;
#endif

  return ;
}        // -----  end of method Spinor::AddMultSpinorEXX  ----- 



// This is density fitting formulation with symmetric implementation of
// the M matrix when combined with ACE, so that the POTRF does not fail as often
// Update: 1/10/2017
void Spinor::AddMultSpinorEXXDF3_GPU ( Fourier& fft, 
    const NumTns<Real>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Real  numSpin,
    const DblNumVec& occupationRate,
    const Real numMuFac,
    const Real numGaussianRandomFac,
    const Int numProcScaLAPACKPotrf,  
    const Int scaPotrfBlockSize,  
    deviceDblNumMat & cu_Hpsi, 
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

  GetTime( timeSta );
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


  if(1){ //For MPI

    // *********************************************************************
    // Perform interpolative separable density fitting
    // *********************************************************************

    //numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
    numMu_ = IRound(numStateTotal*numMuFac);

    Int numPre = IRound(std::sqrt(numMu_*numGaussianRandomFac));
    if( numPre > numStateTotal ){
      ErrorHandling("numMu is too large for interpolative separable density fitting!");
    }
    
    statusOFS << "numMu  = " << numMu_ << std::endl;
    statusOFS << "numPre*numPre = " << numPre * numPre << std::endl;

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

    DblNumMat localphiGRow( ntotLocal, numPre );
    DblNumMat localpsiGRow( ntotLocal, numPre );
    DblNumMat phiRow( ntotLocal, numStateTotal );
    DblNumMat psiRow( ntotLocal, numStateTotal );

    deviceDblNumMat cu_phiRow( ntotLocal, numStateTotal );
    deviceDblNumMat cu_psiRow( ntotLocal, numStateTotal );
    device_setValue( cu_phiRow.Data(), 0.0, ntotLocal*numStateTotal);
    device_setValue( cu_psiRow.Data(), 0.0, ntotLocal*numStateTotal);

    Real minus_one = -1.0;
    Real zero =  0.0;
    Real one  =  1.0;

    {
    //DblNumMat phiCol( ntot, numStateLocal );
    //DblNumMat psiCol( ntot, numStateLocal );
    deviceDblNumMat cu_phiCol( ntot, numStateLocal );
    deviceDblNumMat cu_psiCol( ntot, numStateLocal );
    
    //lapack::Lacpy( 'A', ntot, numStateLocal, phi.Data(), ntot, phiCol.Data(), ntot );
    //lapack::Lacpy( 'A', ntot, numStateLocal, wavefun_.Data(), ntot, psiCol.Data(), ntot );
    device_memcpy_HOST2DEVICE(cu_phiCol.Data(), phi.Data(),      sizeof(Real)*ntot*numStateLocal);
    device_memcpy_HOST2DEVICE(cu_psiCol.Data(), wavefun_.Data(), sizeof(Real)*ntot*numStateLocal);

    //AlltoallForward (phiCol, phiRow, domain_.comm);
    device_AlltoallForward (cu_phiCol, cu_phiRow, domain_.comm);
    device_AlltoallForward (cu_psiCol, cu_psiRow, domain_.comm);

    cu_phiRow.CopyTo(phiRow);
    cu_psiRow.CopyTo(psiRow);
    }
    // Computing the indices is optional

    Int ntotLocalMG, ntotMG;

    if( (ntot % mpisize) == 0 ){
      ntotLocalMG = ntotBlocksize;
    }
    else{
      ntotLocalMG = ntotBlocksize + 1;
    }
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for initilizations:              " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
   

    if( isFixColumnDF == false ){
      DblNumMat G(numStateTotal, numPre);
      SetValue( G, 0.0 );
      GetTime( timeSta );

      // Step 1: Pre-compression of the wavefunctions. This uses
      // multiplication with orthonormalized random Gaussian matrices
      if ( mpirank == 0) {
        GaussianRandom(G);
        lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
      }

      MPI_Bcast(G.Data(), numStateTotal * numPre, MPI_DOUBLE, 0, domain_.comm);

      deviceDblNumMat cu_localphiGRow( ntotLocal, numPre );
      deviceDblNumMat cu_G(numStateTotal, numPre);
      deviceDblNumMat cu_localpsiGRow( ntotLocal, numPre );

      cu_G.CopyFrom(G);
      //cu_localpsiGRow.CopyFrom(localpsiGRow);

      DEVICE_BLAS::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, &one, 
          cu_phiRow.Data(), ntotLocal, cu_G.Data(), numStateTotal, &zero,
          cu_localphiGRow.Data(), ntotLocal );

      DEVICE_BLAS::Gemm( 'N', 'N', ntotLocal, numPre, numStateTotal, &one, 
          cu_psiRow.Data(), ntotLocal, cu_G.Data(), numStateTotal, &zero,
          cu_localpsiGRow.Data(), ntotLocal );

      cu_localphiGRow.CopyTo(localphiGRow);
      cu_localpsiGRow.CopyTo(localpsiGRow);

      // Step 2: Pivoted QR decomposition  for the Hadamard product of
      // the compressed matrix. Transpose format for QRCP

      // NOTE: All processors should have the same ntotLocalMG
      ntotMG = ntotLocalMG * mpisize;

      DblNumMat MG( numPre*numPre, ntotLocalMG );
      SetValue( MG, 0.0 );
      for( Int j = 0; j < numPre; j++ ){
        for( Int i = 0; i < numPre; i++ ){
          for( Int ir = 0; ir < ntotLocal; ir++ ){
            MG(i+j*numPre,ir) = localphiGRow(ir,i) * localpsiGRow(ir,j);
          }
        }
      }

      DblNumVec tau(ntotMG);
      pivQR_.Resize(ntotMG);
      SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
      // Q factor does not need to be used

      Real timeQRCPSta, timeQRCPEnd;
      GetTime( timeQRCPSta );



      if(0){  
        lapack::QRCP( numPre*numPre, ntotMG, MG.Data(), numPre*numPre, 
            pivQR_.Data(), tau.Data() );
      }//


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

        Int mb_MG = numPre*numPre;
        Int nb_MG = ntotLocalMG;

        // FIXME The current routine does not actually allow ntotLocal to be different on different processors.
        // This must be fixed.
        SCALAPACK(descinit)(&desc_MG[0], &mb_MG, &ntotMG, &mb_MG, &nb_MG, &irsrc, 
            &icsrc, &contxt, &mb_MG, &info);

        IntNumVec pivQRTmp(ntotMG), pivQRLocal(ntotMG);
        if( mb_MG > ntot ){
          std::ostringstream msg;
          msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
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

      if(0){
        Real tolR = std::abs(MG(numMu_-1,numMu_-1)/MG(0,0));
        statusOFS << "numMu_ = " << numMu_ << std::endl;
        statusOFS << "|R(numMu-1,numMu-1)/R(0,0)| = " << tolR << std::endl;
      }

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for density fitting with QRCP is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Dump out pivQR_
      if(0){
        std::ostringstream muStream;
        serialize( pivQR_, muStream, NO_MASK );
        SharedWrite( "pivQR", muStream );
      }
    } //if isFixColumnDF == false 

    // Load pivQR_ file
    if(0){
      statusOFS << "Loading pivQR file.." << std::endl;
      std::istringstream muStream;
      SharedRead( "pivQR", muStream );
      deserialize( pivQR_, muStream, NO_MASK );
    }

    // *********************************************************************
    // Compute the interpolation matrix via the density matrix formulation
    // *********************************************************************

    GetTime( timeSta );
    
    DblNumMat XiRow(ntotLocal, numMu_);
    DblNumMat psiMu(numStateTotal, numMu_);

    deviceDblNumMat cu_psiMu(numStateTotal, numMu_);
    deviceDblNumMat cu_XiRow(ntotLocal, numMu_);

    // PhiMu is scaled by the occupation number to reflect the "true" density matrix
    DblNumMat PcolPhiMu(ntotLocal, numMu_);
    deviceDblNumMat cu_PcolPhiMu(ntotLocal, numMu_);

    IntNumVec pivMu(numMu_);
    DblNumMat phiMu(numStateTotal, numMu_);

    deviceDblNumMat cu_phiMu(numStateTotal, numMu_);

    {
    
      for( Int mu = 0; mu < numMu_; mu++ ){
        pivMu(mu) = pivQR_(mu);
      }

      // These three matrices are used only once. 
      // Used before reduce
      DblNumMat psiMuRow(numStateTotal, numMu_);
      DblNumMat phiMuRow(numStateTotal, numMu_);
      //DblNumMat PcolMuNuRow(numMu_, numMu_);
      //DblNumMat PcolPsiMuRow(ntotLocal, numMu_);

      // Collecting the matrices obtained from row partition
      DblNumMat PcolMuNu(numMu_, numMu_);
      DblNumMat PcolPsiMu(ntotLocal, numMu_);

      deviceDblNumMat cu_PcolPsiMu(ntotLocal, numMu_);
      deviceDblNumMat cu_PcolMuNu(numMu_, numMu_);

      SetValue( psiMuRow, 0.0 );
      SetValue( phiMuRow, 0.0 );
      //SetValue( PcolMuNuRow, 0.0 );
      //SetValue( PcolPsiMuRow, 0.0 );

      //SetValue( phiMu, 0.0 );
      //SetValue( PcolMuNu, 0.0 );
      //SetValue( PcolPsiMu, 0.0 );

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
      cu_psiMu.CopyFrom(psiMu);
      cu_phiMu.CopyFrom(phiMu);

      //cu_PcolPsiMu.CopyFrom(PcolPsiMu);
      //cu_PcolPhiMu.CopyFrom(PcolPhiMu);

      //cu_psiRow.CopyFrom(psiRow);
      //cu_phiRow.CopyFrom(phiRow);

      DEVICE_BLAS::Gemm( 'N', 'N', ntotLocal, numMu_, numStateTotal, &one, 
          cu_psiRow.Data(), ntotLocal, cu_psiMu.Data(), numStateTotal, &zero,
          cu_PcolPsiMu.Data(), ntotLocal );
      DEVICE_BLAS::Gemm( 'N', 'N', ntotLocal, numMu_, numStateTotal, &one, 
          cu_phiRow.Data(), ntotLocal, cu_phiMu.Data(), numStateTotal, &zero,
          cu_PcolPhiMu.Data(), ntotLocal );

      //cu_PcolPsiMu.CopyTo(PcolPsiMu);
      //cu_PcolPhiMu.CopyTo(PcolPhiMu);

      /*
      blas::Gemm( 'N', 'N', ntotLocal, numMu_, numStateTotal, 1.0, 
          psiRow.Data(), ntotLocal, psiMu.Data(), numStateTotal, 0.0,
          PcolPsiMu.Data(), ntotLocal );
      blas::Gemm( 'N', 'N', ntotLocal, numMu_, numStateTotal, 1.0, 
          phiRow.Data(), ntotLocal, phiMu.Data(), numStateTotal, 0.0,
          PcolPhiMu.Data(), ntotLocal );
      */
      GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for GEMM is " <<
        timeEnd1 - timeSta1 << " [s]" << std::endl << std::endl;
#endif
      
      GetTime( timeSta1 );
      /*
      Real* xiPtr = XiRow.Data();
      Real* PcolPsiMuPtr = PcolPsiMu.Data();
      Real* PcolPhiMuPtr = PcolPhiMu.Data();
      
      for( Int g = 0; g < ntotLocal * numMu_; g++ ){
        xiPtr[g] = PcolPsiMuPtr[g] * PcolPhiMuPtr[g];
      }
      */
      device_hadamard_product( cu_PcolPsiMu.Data(), cu_PcolPhiMu.Data(), cu_XiRow.Data(), ntotLocal * numMu_);
      cu_XiRow.CopyTo(XiRow);
      

      // 1/2/2017 Add extra weight to certain entries to the XiRow matrix
      // Currently only works for one processor
      /*
      if(0)
      {
        Real wgt = 10.0;
        // Correction for Diagonal entries
        for( Int mu = 0; mu < numMu_; mu++ ){
          xiPtr = XiRow.VecData(mu);
          for( Int i = 0; i < numStateTotal; i++ ){
            Real* phiPtr = phi.VecData(0, i);
            Real* psiPtr = wavefun_.VecData(0,i);
            Real  fac = phiMu(i,mu) * psiMu(i,mu) * wgt; 
            for( Int g = 0; g < ntotLocal; g++ ){
              xiPtr[g] += phiPtr[g] * psiPtr[g] * fac;
            } 
          }
        }
      }

      if(0)
      {
        Real wgt = 10.0;
        // Correction for HOMO 
        for( Int mu = 0; mu < numMu_; mu++ ){
          xiPtr = XiRow.VecData(mu);
          Real* phiPtr = phi.VecData(0, numStateTotal-1);
          for( Int i = 0; i < numStateTotal; i++ ){
            Real* psiPtr = wavefun_.VecData(0,i);
            Real  fac = phiMu(numStateTotal-1,mu) * psiMu(i,mu) * wgt; 
            for( Int g = 0; g < ntotLocal; g++ ){
              xiPtr[g] += phiPtr[g] * psiPtr[g] * fac;
            } 
          }
        }
      }
      */
      
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

      if(0){
        if ( mpirank == 0) {
          lapack::Potrf( 'L', numMu_, PcolMuNu.Data(), numMu_ );
        }
      } // if(0)

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

      cu_PcolMuNu.CopyFrom(PcolMuNu);
      //cu_XiRow.CopyFrom(XiRow);

      DEVICE_BLAS::Trsm( 'R', 'L', 'T', 'N', 
                    ntotLocal, numMu_, &one, cu_PcolMuNu.Data(), numMu_, cu_XiRow.Data(),ntotLocal);

      DEVICE_BLAS::Trsm( 'R', 'L', 'N', 'N', 
                    ntotLocal, numMu_, &one, cu_PcolMuNu.Data(), numMu_, cu_XiRow.Data(),ntotLocal);

      //cu_PcolMuNu.CopyTo(PcolMuNu);
      //cu_XiRow.CopyTo(XiRow);

      /*
      blas::Trsm( 'R', 'L', 'T', 'N', ntotLocal, numMu_, 1.0, 
          PcolMuNu.Data(), numMu_, XiRow.Data(), ntotLocal );

      blas::Trsm( 'R', 'L', 'N', 'N', ntotLocal, numMu_, 1.0, 
          PcolMuNu.Data(), numMu_, XiRow.Data(), ntotLocal );
      */
      GetTime( timeEnd1 );

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for Trsm is " <<
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

    GetTime( timeSta );
    DblNumMat VXiRow(ntotLocal, numMu_);
    deviceDblNumMat cu_VXiRow(ntotLocal, numMu_);

    if(1){
      // XiCol used for both input and output
      Real mpi_time;
      GetTime( timeSta1 );
      DblNumMat XiCol(ntot, numMuLocal);
      deviceDblNumMat cu_XiCol(ntot, numMuLocal);

      //cu_XiRow.CopyFrom( XiRow );
      device_AlltoallBackward (cu_XiRow, cu_XiCol, domain_.comm);
      //cu_XiCol.CopyFrom(XiCol);
      //cu_XiCol.CopyTo( XiCol );
      GetTime( timeEnd1 );
      mpi_time = timeEnd1 - timeSta1;

      deviceDblNumVec cu_psi(ntot);
      deviceDblNumVec cu_psi_out(2*ntotR2C);
      deviceDblNumVec cu_exxgkkR2C(ntotR2C);

      DblNumVec exxgkkR2CTemp(ntotR2C);
      /*
      for( Int ig = 0; ig < ntotR2C; ig++ ){
            exxgkkR2CTemp (ig) = -exxFraction * exxgkkR2C(ig);
      }

      device_memcpy_HOST2DEVICE(cu_exxgkkR2C.Data(), exxgkkR2CTemp.Data(), sizeof(Real)*ntotR2C);
      */
      Real temp = -exxFraction;
      device_memcpy_HOST2DEVICE(cu_exxgkkR2C.Data(), exxgkkR2C.Data(), sizeof(Real)*ntotR2C);
      DEVICE_BLAS::Scal( ntotR2C, &temp, cu_exxgkkR2C.Data(), 1);
     
      {
        for( Int mu = 0; mu < numMuLocal; mu++ ){

          // copy the WF to the buf.
          device_memcpy_DEVICE2DEVICE(cu_psi.Data(), & cu_XiCol(0,mu), sizeof(Real)*ntot);

          deviceFFTExecuteForward( fft, cuPlanR2C[0], 0, cu_psi, cu_psi_out);

          device_teter( reinterpret_cast<cuDoubleComplex*> (cu_psi_out.Data()), cu_exxgkkR2C.Data(), ntotR2C );

          deviceFFTExecuteInverse( fft, cuPlanC2R[0], 0, cu_psi_out, cu_psi);

          device_memcpy_DEVICE2DEVICE( & cu_XiCol(0,mu), cu_psi.Data(), sizeof(Real)*ntot);

        } // for (mu)

        //cu_XiCol.CopyTo(XiCol);

        GetTime( timeSta1 );

        device_AlltoallForward (cu_XiCol, cu_VXiRow, domain_.comm);
        //cu_VXiRow.CopyTo( VXiRow );    // copy the data back to VXiRow, used in the next steps.

        GetTime( timeEnd1 );
        mpi_time += timeEnd1 - timeSta1;

        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for solving Poisson-like equations is " <<
          timeEnd - timeSta << " [s]" << " MPI_Alltoall: " << mpi_time<< std::endl << std::endl;
#endif
      }
    }

    // Prepare for the computation of the M matrix

    GetTime( timeSta );

    DblNumMat MMatMuNu(numMu_, numMu_);
    deviceDblNumMat cu_MMatMuNu(numMu_, numMu_);
    {
      DblNumMat MMatMuNuTemp( numMu_, numMu_ );
      deviceDblNumMat cu_MMatMuNuTemp( numMu_, numMu_ );
      //cu_VXiRow.CopyFrom(VXiRow);
      //cu_XiRow.CopyFrom(XiRow);

      // Minus sign so that MMat is positive semidefinite
      DEVICE_BLAS::Gemm( 'T', 'N', numMu_, numMu_, ntotLocal, &minus_one,
          cu_XiRow.Data(), ntotLocal, cu_VXiRow.Data(), ntotLocal, &zero, 
          cu_MMatMuNuTemp.Data(), numMu_ );
      cu_MMatMuNuTemp.CopyTo(MMatMuNuTemp);

      /*
      blas::Gemm( 'T', 'N', numMu_, numMu_, ntotLocal, -1.0,
          XiRow.Data(), ntotLocal, VXiRow.Data(), ntotLocal, 0.0, 
          MMatMuNuTemp.Data(), numMu_ );
      */

      MPI_Allreduce( MMatMuNuTemp.Data(), MMatMuNu.Data(), numMu_ * numMu_, 
          MPI_DOUBLE, MPI_SUM, domain_.comm );
      cu_MMatMuNu.CopyFrom( MMatMuNu );
    }

    // Element-wise multiply with phiMuNu matrix (i.e. density matrix)
    {

      DblNumMat phiMuNu(numMu_, numMu_);
      deviceDblNumMat cu_phiMuNu(numMu_, numMu_);
      //cu_phiMu.CopyFrom(phiMu);

      DEVICE_BLAS::Gemm( 'T', 'N', numMu_, numMu_, numStateTotal, &one,
          cu_phiMu.Data(), numStateTotal, cu_phiMu.Data(), numStateTotal, &zero,
          cu_phiMuNu.Data(), numMu_ );
      device_hadamard_product( cu_MMatMuNu.Data(), cu_phiMuNu.Data(), cu_MMatMuNu.Data(), numMu_*numMu_);

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
    // cu_VXiRow.CopyFrom(VXiRow);
    device_hadamard_product( cu_VXiRow.Data(), cu_PcolPhiMu.Data(), cu_VXiRow.Data(), numMu_*ntotLocal );
    
    // NOTE: Hpsi must be zero in order to compute the M matrix later
    DblNumMat HpsiRow( ntotLocal, numStateTotal );
    device_setValue( cu_Hpsi.Data(), 0.0, ntotLocal*numStateTotal);

    cu_psiMu.CopyFrom(psiMu);

    DEVICE_BLAS::Gemm( 'N', 'T', ntotLocal, numStateTotal, numMu_, &one, 
        cu_VXiRow.Data(), ntotLocal, cu_psiMu.Data(), numStateTotal, &one,
        cu_Hpsi.Data(), ntotLocal ); 

    //cu_HpsiRow.CopyTo(HpsiRow);
    // there is no need in MPI_AlltoallBackward from Hpsirow to HpsiCol. 
    // cause in the next step, we will MPI_AlltoallForward them back to row parallel
   
    /*
    deviceDblNumMat cu_HpsiCol( ntot, numStateLocal );
    device_AlltoallBackward (cu_HpsiRow, cu_HpsiCol, domain_.comm);
    device_memcpy_DEVICE2HOST( Hpsi.Data(), cu_HpsiCol.Data(), sizeof(Real)*ntot*numStateLocal);
    */

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing the exchange potential is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // Compute the matrix VxMat = -Psi'* vexxPsi and symmetrize
    // vexxPsi (Hpsi) must be zero before entering this routine
    VxMat.Resize( numStateTotal, numStateTotal );
    GetTime( timeSta );
    /*
    if(0)
    {
      // Minus sign so that VxMat is positive semidefinite
      // NOTE: No measure factor vol / ntot due to the normalization
      // factor of psi
      DblNumMat VxMatTemp( numStateTotal, numStateTotal );
      SetValue( VxMatTemp, 0.0 );
      blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal, -1.0,
          psiRow.Data(), ntotLocal, HpsiRow.Data(), ntotLocal, 0.0, 
          VxMatTemp.Data(), numStateTotal );

      SetValue( VxMat, 0.0 );
      MPI_Allreduce( VxMatTemp.Data(), VxMat.Data(), numStateTotal * numStateTotal, 
          MPI_DOUBLE, MPI_SUM, domain_.comm );

      Symmetrize( VxMat );
     
    }
    */
    if(1){

      DblNumMat VxMatTemp( numMu_, numStateTotal );
      deviceDblNumMat cu_VxMatTemp( numMu_, numStateTotal );
      deviceDblNumMat cu_VxMat( numStateTotal, numStateTotal );

      cu_psiMu.CopyFrom(psiMu);
      //cu_MMatMuNu.CopyFrom(MMatMuNu);
    
      DEVICE_BLAS::Gemm( 'N', 'T', numMu_, numStateTotal, numMu_, &one, 
          cu_MMatMuNu.Data(), numMu_, cu_psiMu.Data(), numStateTotal, &zero,
          cu_VxMatTemp.Data(), numMu_ );
      DEVICE_BLAS::Gemm( 'N', 'N', numStateTotal, numStateTotal, numMu_, &one,
          cu_psiMu.Data(), numStateTotal, cu_VxMatTemp.Data(), numMu_, &zero,
          cu_VxMat.Data(), numStateTotal );

      cu_VxMat.CopyTo(VxMat);  // no need to copy them back to CPU. just keep them in GPU.

    }
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing VxMat in the sym format is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  } //if(1) for For MPI

  MPI_Barrier(domain_.comm);

  return ;
}        // -----  end of method Spinor::AddMultSpinorEXXDF3_GPU  ----- 





}  // namespace scales
