/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Weile Jia

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
/// @file device_hamiltonian.cpp
/// @brief device_hamiltonian class for planewave basis diagonalization method.
/// @date 2020-08-21
#include  "hamiltonian.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "device_utility.hpp"

namespace dgdft{

using namespace dgdft::PseudoComponent;
using namespace dgdft::DensityComponent;
using namespace dgdft::esdf;


// *********************************************************************
// KohnSham class
// *********************************************************************
#ifdef DEVICE
void
KohnSham::ACEOperator ( deviceDblNumMat& cu_psi, Fourier& fft, deviceDblNumMat& cu_Hpsi)
{

     // 1. the projector is in a Row Parallel fashion
     // 2. the projector is in GPU.
     // 3. the AX (H*psi) is in the GPU

     // in here we perform: 
     // M = W'*AX 
     // reduece M
     // AX = AX + W*M 
  if( isHybrid_ && isEXXActive_ ){

    if( esdfParam.isHybridACE ){ 
    int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
    int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

     Int ntot      = fft.domain.NumGridTotal();
     Int ntotFine  = fft.domain.NumGridTotalFine();
     Int numStateTotal = cu_psi.n();

     Int ntotBlocksize = ntot / mpisize;
     Int ntotLocal = ntotBlocksize;
     if(mpirank < (ntot % mpisize)){
       ntotLocal = ntotBlocksize + 1;
     }

     Real one = 1.0;
     Real minus_one = -1.0;
     Real zero = 0.0;

     DblNumMat MTemp( numStateTotal, numStateTotal );
     deviceDblNumMat cu_MTemp( numStateTotal, numStateTotal );

     DEVICE_BLAS::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal,
                   &one, cu_vexxProj_.Data(), ntotLocal, 
                   cu_psi.Data(), ntotLocal, &zero,
                   cu_MTemp.Data(), numStateTotal );
     device_memcpy_DEVICE2HOST( MTemp.Data(), cu_MTemp.Data(), numStateTotal*numStateTotal*sizeof(Real) );

     DblNumMat M(numStateTotal, numStateTotal);
     MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );
     device_memcpy_HOST2DEVICE(  cu_MTemp.Data(), M.Data(), numStateTotal*numStateTotal*sizeof(Real) );
     DEVICE_BLAS::Gemm( 'N', 'N', ntotLocal, numStateTotal, numStateTotal, 
                   &minus_one, cu_vexxProj_.Data(), ntotLocal, 
                   cu_MTemp.Data(), numStateTotal, &one, 
                   cu_Hpsi.Data(), ntotLocal );
    }
  }
}
void
KohnSham::MultSpinor_old    ( Spinor& psi, deviceNumTns<Real>& a3, Fourier& fft )
{

  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Real>& wavefun = psi.Wavefun();
  Int ncom = wavefun.n();

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

  // adding up the Hybrid part in the GPU
  // CHECK CHECK
  // Note now, the psi.data is the GPU data. and a3.data is also in GPU. 
  // also, a3 constains the Hpsi
  // need to do this in another subroutine.
  if(1)  
  if( isHybrid_ && isEXXActive_ ){

    GetTime( timeSta );

    if( esdfParam.isHybridACE ){ 

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

        // copy the GPU data to CPU.
        DblNumMat psiCol( ntot, numStateLocal );
        device_memcpy_DEVICE2HOST( psiCol.Data(), psi.cuWavefun().Data(), ntot*numStateLocal*sizeof(Real) );

        // for the Project VexxProj 
        DblNumMat vexxProjCol( ntot, numStateLocal );
        DblNumMat vexxProjRow( ntotLocal, numStateTotal );
        lapack::Lacpy( 'A', ntot, numStateLocal, vexxProj_.Data(), ntot, vexxProjCol.Data(), ntot );

        // MPI_Alltoall for the data redistribution.
        DblNumMat psiRow( ntotLocal, numStateTotal );
        AlltoallForward (psiCol, psiRow, domain_.comm);

        // MPI_Alltoall for data redistribution.
        AlltoallForward (vexxProjCol, vexxProjRow, domain_.comm);

        // GPU data for the G-para
        deviceDblNumMat cu_vexxProjRow ( ntotLocal, numStateTotal );
        deviceDblNumMat cu_psiRow ( ntotLocal, numStateTotal );
        deviceDblNumMat cu_MTemp( numStateTotal, numStateTotal );
        DblNumMat MTemp( numStateTotal, numStateTotal );

        // Copy data from CPU to GPU.
        device_memcpy_HOST2DEVICE( cu_psiRow.Data(), psiRow.Data(), numStateTotal*ntotLocal*sizeof(Real) );
        device_memcpy_HOST2DEVICE( cu_vexxProjRow.Data(), vexxProjRow.Data(), numStateTotal*ntotLocal*sizeof(Real) );

	Real one = 1.0;
	Real minus_one = -1.0;
	Real zero = 0.0;
        // GPU DGEMM calculation
        DEVICE_BLAS::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal,
                    &one, cu_vexxProjRow.Data(), ntotLocal, 
                    cu_psiRow.Data(), ntotLocal, &zero,
                    cu_MTemp.Data(), numStateTotal );

        device_memcpy_DEVICE2HOST( MTemp.Data(), cu_MTemp.Data(), numStateTotal*numStateTotal*sizeof(Real) );
        DblNumMat M(numStateTotal, numStateTotal);
        MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

	// copy from CPU to GPU
        device_memcpy_HOST2DEVICE(  cu_MTemp.Data(), M.Data(), numStateTotal*numStateTotal*sizeof(Real) );
        
        deviceDblNumMat cu_a3Row( ntotLocal, numStateTotal );
        DblNumMat a3Row( ntotLocal, numStateTotal );

        DEVICE_BLAS::Gemm( 'N', 'N', ntotLocal, numStateTotal, numStateTotal, 
                     &minus_one, cu_vexxProjRow.Data(), ntotLocal, 
                     cu_MTemp.Data(), numStateTotal, &zero, 
                     cu_a3Row.Data(), ntotLocal );

        device_memcpy_DEVICE2HOST( a3Row.Data(), cu_a3Row.Data(), numStateTotal*ntotLocal*sizeof(Real) );

        // a3Row to a3Col
        DblNumMat a3Col( ntot, numStateLocal );
        deviceDblNumMat cu_a3Col( ntot, numStateLocal );
        AlltoallBackward (a3Row, a3Col, domain_.comm);

	//Copy a3Col to GPU.
        device_memcpy_HOST2DEVICE( cu_a3Col.Data(), a3Col.Data(), numStateLocal*ntot*sizeof(Real) );

        // do the matrix addition.
	device_DMatrix_Add( a3.Data(), cu_a3Col.Data(), ntot, numStateLocal);

      } //if(1)

    }
    else{

      ErrorHandling(" GPU does not support normal HSE, try ACE");
      
    }

    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for updating hybrid Spinor is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//    statusOFS << "Time for Gemm is " <<
//      timeGemm << " [s]" << std::endl << std::endl;
//    statusOFS << "Time for Alltoallv is " <<
//      timeAlltoallv << " [s]" << std::endl << std::endl;
//    statusOFS << "Time for Allreduce is " <<
//      timeAllreduce << " [s]" << std::endl << std::endl;
//#endif


  }


  return ;
}         // -----  end of method KohnSham::MultSpinor  ----- 



void
KohnSham::MultSpinor    ( Spinor& psi, deviceNumTns<Real>& a3, Fourier& fft )
{

  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Real>& wavefun = psi.Wavefun();
  Int ncom = wavefun.n();

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

  // adding up the Hybrid part in the GPU
  // CHECK CHECK
  // Note now, the psi.data is the GPU data. and a3.data is also in GPU. 
  // also, a3 constains the Hpsi
  // need to do this in another subroutine.
  if(0)  // comment out the following parts.
  if( isHybrid_ && isEXXActive_ ){

    GetTime( timeSta );

    if( esdfParam.isHybridACE ){ 

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

        // copy the GPU data to CPU.
        DblNumMat psiCol( ntot, numStateLocal );
        device_memcpy_DEVICE2HOST( psiCol.Data(), psi.cuWavefun().Data(), ntot*numStateLocal*sizeof(Real) );

        // for the Project VexxProj 
        DblNumMat vexxProjCol( ntot, numStateLocal );
        DblNumMat vexxProjRow( ntotLocal, numStateTotal );
        lapack::Lacpy( 'A', ntot, numStateLocal, vexxProj_.Data(), ntot, vexxProjCol.Data(), ntot );

        // MPI_Alltoall for the data redistribution.
        DblNumMat psiRow( ntotLocal, numStateTotal );
        AlltoallForward (psiCol, psiRow, domain_.comm);

        // MPI_Alltoall for data redistribution.
        AlltoallForward (vexxProjCol, vexxProjRow, domain_.comm);

        // GPU data for the G-para
        deviceDblNumMat cu_vexxProjRow ( ntotLocal, numStateTotal );
        deviceDblNumMat cu_psiRow ( ntotLocal, numStateTotal );
        deviceDblNumMat cu_MTemp( numStateTotal, numStateTotal );
        DblNumMat MTemp( numStateTotal, numStateTotal );

        // Copy data from CPU to GPU.
        device_memcpy_HOST2DEVICE( cu_psiRow.Data(), psiRow.Data(), numStateTotal*ntotLocal*sizeof(Real) );
        device_memcpy_HOST2DEVICE( cu_vexxProjRow.Data(), vexxProjRow.Data(), numStateTotal*ntotLocal*sizeof(Real) );

	Real one = 1.0;
	Real minus_one = -1.0;
	Real zero = 0.0;
        // GPU DGEMM calculation
        DEVICE_BLAS::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal,
                    &one, cu_vexxProjRow.Data(), ntotLocal, 
                    cu_psiRow.Data(), ntotLocal, &zero,
                    cu_MTemp.Data(), numStateTotal );

        device_memcpy_DEVICE2HOST( MTemp.Data(), cu_MTemp.Data(), numStateTotal*numStateTotal*sizeof(Real) );
        DblNumMat M(numStateTotal, numStateTotal);
        MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

	// copy from CPU to GPU
        device_memcpy_HOST2DEVICE(  cu_MTemp.Data(), M.Data(), numStateTotal*numStateTotal*sizeof(Real) );
        
        deviceDblNumMat cu_a3Row( ntotLocal, numStateTotal );
        DblNumMat a3Row( ntotLocal, numStateTotal );

        DEVICE_BLAS::Gemm( 'N', 'N', ntotLocal, numStateTotal, numStateTotal, 
                     &minus_one, cu_vexxProjRow.Data(), ntotLocal, 
                     cu_MTemp.Data(), numStateTotal, &zero, 
                     cu_a3Row.Data(), ntotLocal );

        device_memcpy_DEVICE2HOST( a3Row.Data(), cu_a3Row.Data(), numStateTotal*ntotLocal*sizeof(Real) );

        // a3Row to a3Col
        DblNumMat a3Col( ntot, numStateLocal );
        deviceDblNumMat cu_a3Col( ntot, numStateLocal );
        AlltoallBackward (a3Row, a3Col, domain_.comm);

	//Copy a3Col to GPU.
        device_memcpy_HOST2DEVICE( cu_a3Col.Data(), a3Col.Data(), numStateLocal*ntot*sizeof(Real) );

        // do the matrix addition.
	device_DMatrix_Add( a3.Data(), cu_a3Col.Data(), ntot, numStateLocal);

      } //if(1)

    }
    else{

      ErrorHandling(" GPU does not support normal HSE, try ACE");
      
    }

    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for updating hybrid Spinor is " <<
//      timeEnd - timeSta << " [s]" << std::endl << std::endl;
//    statusOFS << "Time for Gemm is " <<
//      timeGemm << " [s]" << std::endl << std::endl;
//    statusOFS << "Time for Alltoallv is " <<
//      timeAlltoallv << " [s]" << std::endl << std::endl;
//    statusOFS << "Time for Allreduce is " <<
//      timeAllreduce << " [s]" << std::endl << std::endl;
//#endif


  }


  return ;
}         // -----  end of method KohnSham::MultSpinor  ----- 



void
KohnSham::CalculateVexxACEGPU ( Spinor& psi, Fourier& fft )
{
  // This assumes SetPhiEXX has been called so that phiEXX and psi
  // contain the same information. 

  // Since this is a projector, it should be done on the COARSE grid,
  // i.e. to the wavefunction directly

  //MPI_Barrier(domain_.comm);
  Real timeSta, timeEnd;
  GetTime( timeSta );

  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  // Only works for single processor
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  deviceNumTns<Real>  cu_vexxPsi( ntot, 1, numStateLocal );
  NumTns<Real>  vexxPsi( ntot, 1, numStateLocal );

  // VexxPsi = V_{exx}*Phi.
  //SetValue( vexxPsi, 0.0 );
  device_setValue( cu_vexxPsi.Data(), 0.0, ntot*numStateLocal);
  psi.AddMultSpinorEXX( fft, phiEXX_, exxgkkR2C_,
      exxFraction_,  numSpin_, occupationRate_, cu_vexxPsi );

  
  //device_memcpy_DEVICE2HOST(vexxPsi.Data(),cu_vexxPsi.Data(), sizeof(Real)*ntot*numStateLocal);
  // Implementation based on SVD
  DblNumMat  M(numStateTotal, numStateTotal);
  
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for AddMulSpinorEXX with GPU  is " <<
         timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  GetTime( timeSta );
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

    /*
    DblNumMat localPsiCol( ntot, numStateLocal );
    SetValue( localPsiCol, 0.0 );

    DblNumMat localVexxPsiCol( ntot, numStateLocal );
    SetValue( localVexxPsiCol, 0.0 );

    DblNumMat localPsiRow( ntotLocal, numStateTotal );
    SetValue( localPsiRow, 0.0 );

    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    SetValue( localVexxPsiRow, 0.0 );
    */
    DblNumMat localPsiRow( ntotLocal, numStateTotal );
    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    DblNumMat localPsiCol( ntot, numStateLocal );
    //DblNumMat localVexxPsiCol( ntot, numStateLocal );

    // Initialize
    //lapack::Lacpy( 'A', ntot, numStateLocal, vexxPsi.Data(), ntot, localVexxPsiCol.Data(), ntot );
    deviceDblNumMat cu_temp( ntot, numStateLocal, false, cu_vexxPsi.Data() );
    cu_vexxProj_.Resize( ntotLocal, numStateTotal );
    device_AlltoallForward (cu_temp, cu_vexxProj_, domain_.comm);

    //lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, localPsiCol.Data(), ntot );
    //AlltoallForward (localPsiCol, localPsiRow, domain_.comm);
    device_memcpy_HOST2DEVICE( cu_temp.Data(), psi.Wavefun().Data(), ntot*numStateLocal*sizeof(Real));
    deviceDblNumMat cu_localPsiRow( ntotLocal, numStateTotal);
    device_AlltoallForward (cu_temp, cu_localPsiRow, domain_.comm);
    //cu_localPsiRow.CopyFrom(localPsiRow);
    //AlltoallForward (localVexxPsiCol, localVexxPsiRow, domain_.comm);

    DblNumMat MTemp( numStateTotal, numStateTotal );
    //SetValue( MTemp, 0.0 );
    deviceDblNumMat cu_MTemp( numStateTotal, numStateTotal );
    //deviceDblNumMat cu_vexxProj_( ntotLocal, numStateTotal );

    //cu_vexxProj_.CopyFrom(localVexxPsiRow);

    Real minus_one = -1.0;
    Real zero =  0.0;
    Real one  =  1.0;

    DEVICE_BLAS::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal,
                  &minus_one, cu_localPsiRow.Data(), ntotLocal, 
                  cu_vexxProj_.Data(), ntotLocal, &zero,
                  cu_MTemp.Data(), numStateTotal );
    cu_MTemp.CopyTo(MTemp);

    MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );
    /*
    blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal,
        -1.0, localPsiRow.Data(), ntotLocal, 
        localVexxPsiRow.Data(), ntotLocal, 0.0,
        MTemp.Data(), numStateTotal );
    */
    //SetValue( M, 0.0 );
 
    //if ( mpirank == 0) {
    //  lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);
    //}
    //MPI_Bcast(M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, 0, domain_.comm);
    /*
    blas::Trsm( 'R', 'L', 'T', 'N', ntotLocal, numStateTotal, 1.0, 
        M.Data(), numStateTotal, localVexxPsiRow.Data(), ntotLocal );
    */

    cu_MTemp.CopyFrom(M);
#ifdef USE_MAGMA
    MAGMA::Potrf('L', numStateTotal, cu_MTemp.Data(), numStateTotal);
#else
    device_solver::Potrf('L', numStateTotal, cu_MTemp.Data(), numStateTotal);
#endif
    DEVICE_BLAS::Trsm( 'R', 'L', 'T', 'N', 
                  ntotLocal, numStateTotal, &one, cu_MTemp.Data(), numStateTotal, cu_vexxProj_.Data(),
                  ntotLocal);
    //cu_vexxProj_.CopyTo(localVexxPsiRow);
    vexxProj_.Resize( ntot, numStateLocal );
    cu_localPsiRow.Resize( ntot, numStateLocal ); // use this as a column distribution data.

    //AlltoallBackward (localVexxPsiRow, vexxProj_, domain_.comm);
    device_AlltoallBackward (cu_vexxProj_, cu_localPsiRow, domain_.comm);
    cu_localPsiRow.CopyTo( vexxProj_ );
  } //if(1)
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for GPU calculate vexxProjector  " <<
         timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

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
}         // -----  end of method KohnSham::CalculateVexxACEGPU  ----- 


void
KohnSham::CalculateVexxACEDFGPU ( Spinor& psi, Fourier& fft, bool isFixColumnDF )
{
  // This assumes SetPhiEXX has been called so that phiEXX and psi
  // contain the same information. 

  // Since this is a projector, it should be done on the COARSE grid,
  // i.e. to the wavefunction directly

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
  Real timeSta, timeEnd;

  GetTime( timeSta );
  // Only works for single processor
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  //NumTns<Real>  vexxPsi( ntot, 1, numStateLocal );

  Int ntotBlocksize = ntot / mpisize;
  Int ntotLocal = ntotBlocksize;
  if(mpirank < (ntot % mpisize)){
    ntotLocal = ntotBlocksize + 1;
  }

  deviceDblNumMat cu_vexxPsi( ntotLocal, numStateTotal );

  // VexxPsi = V_{exx}*Phi.
  DblNumMat  M(numStateTotal, numStateTotal);
  //SetValue( vexxPsi, 0.0 );
  //SetValue( M, 0.0 );

  // M = -Phi'*vexxPsi. The minus sign comes from vexx is a negative
  // semi-definite matrix.

  // why keep so many MPI_Alltoalls? while this can be easily avoided. 
  psi.AddMultSpinorEXXDF3_GPU( fft, phiEXX_, exxgkkR2C_, exxFraction_,  numSpin_, 
      occupationRate_, hybridDFNumMu_, hybridDFNumGaussianRandom_,
      hybridDFNumProcScaLAPACK_, BlockSizeScaLAPACK_,
      cu_vexxPsi, M, isFixColumnDF );

  GetTime( timeEnd );
  statusOFS << "GPU Time for AddMulSpinorEXXDF3_GPU  is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

  GetTime( timeSta );
  // Implementation based on Cholesky
  /*
  if(0){
    lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);

    blas::Trsm( 'R', 'L', 'T', 'N', ntot, numStateTotal, 1.0, 
        M.Data(), numStateTotal, vexxPsi.Data(), ntot );

    vexxProj_.Resize( ntot, numStateTotal );
    blas::Copy( ntot * numStateTotal, vexxPsi.Data(), 1, vexxProj_.Data(), 1 );
  }
  */
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
    //SetValue( localVexxPsiCol, 0.0 );

    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    //SetValue( localVexxPsiRow, 0.0 );

    // Initialize
    //lapack::Lacpy( 'A', ntot, numStateLocal, vexxPsi.Data(), ntot, localVexxPsiCol.Data(), ntot );

    //AlltoallForward (localVexxPsiCol, localVexxPsiRow, domain_.comm);
    
    /*
    if ( mpirank == 0) {
      lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);
    }

    MPI_Bcast(M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, 0, domain_.comm);
    blas::Trsm( 'R', 'L', 'T', 'N', ntotLocal, numStateTotal, 1.0, 
        M.Data(), numStateTotal, localVexxPsiRow.Data(), ntotLocal );
    */

    Real minus_one = -1.0;
    Real zero =  0.0;
    Real one  =  1.0;

    cu_vexxProj_.Resize( ntotLocal, numStateTotal );
    cu_vexxPsi.CopyTo( cu_vexxProj_);
    //cu_vexxProj_.CopyFrom(localVexxPsiRow);

    deviceDblNumMat cu_M( numStateTotal, numStateTotal );
    cu_M.CopyFrom(M);

#ifdef USE_MAGMA
    MAGMA::Potrf('L', numStateTotal, cu_M.Data(), numStateTotal);
#else
    device_solver::Potrf('L', numStateTotal, cu_M.Data(), numStateTotal);
#endif
    DEVICE_BLAS::Trsm( 'R', 'L', 'T', 'N', 
                  ntotLocal, numStateTotal, &one, cu_M.Data(), numStateTotal, cu_vexxProj_.Data(),
                  ntotLocal);

    cu_vexxProj_.CopyTo(localVexxPsiRow);

    vexxProj_.Resize( ntot, numStateLocal );

    AlltoallBackward (localVexxPsiRow, vexxProj_, domain_.comm);
  } //if(1)
  GetTime( timeEnd );
  statusOFS << "GPU Time for Vexx calculation is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
  return ;
}

#endif





} // namespace dgdft
