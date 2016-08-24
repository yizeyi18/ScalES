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
/// @file spinor.cpp
/// @brief Spinor (wavefunction) for the global domain or extended
/// element.
/// @date 2012-10-06
#include  "spinor.hpp"
#include  "blas.hpp"

namespace dgdft{

using namespace dgdft::PseudoComponent;

Spinor::Spinor () { }         

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

Spinor::~Spinor    () {}

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
Spinor::AddTeterPrecond (Fourier* fftPtr, NumTns<Real>& a3)
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
            blas::Copy( ntot, wavefun_.VecData(j,k), 1, 
                    reinterpret_cast<Real*>(fft.inputVecR2C.Data()), 1 );

            FFTWExecute ( fft, fft.forwardPlanR2C );
            //          fftw_execute_dft_r2c(
            //                  fftPtr->forwardPlanR2C, 
            //                  realInVec.Data(),
            //                  reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ));

            Real*    ptr1d   = fftPtr->TeterPrecondR2C.Data();
            Complex* ptr2    = fft.outputVecR2C.Data();
            for (Int i=0; i<ntothalf; i++) 
                *(ptr2++) *= *(ptr1d++);

            FFTWExecute ( fft, fft.backwardPlanR2C);
            //          fftw_execute_dft_c2r(
            //                  fftPtr->backwardPlanR2C,
            //                  reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ),
            //                  realInVec.Data() );
            //          blas::Axpy( ntot, 1.0 / Real(ntot), realInVec.Data(), 1, 
            //                  a3o.VecData(j, k), 1 );

            blas::Axpy( ntot, 1.0, fft.inputVecR2C.Data(), 1, a3.VecData(j,k), 1 );
        }
    }
    //#ifdef _USE_OPENMP_
    //  }
    //#endif


    return ;
}         // -----  end of method Spinor::AddTeterPrecond ----- 

void
Spinor::AddMultSpinorFine ( Fourier& fft, const DblNumVec& vtot, 
        const std::vector<PseudoPot>& pseudo, NumTns<Real>& a3 )
{
    // TODO Complex case

    if( !fft.isInitialized ){
        ErrorHandling("Fourier is not prepared.");
    }
    Int ntot = wavefun_.m();
    Int ncom = wavefun_.n();
    Int numStateLocal = wavefun_.p();
    Int ntotFine = domain_.NumGridTotalFine();
    Real vol = domain_.Volume();

    if( fft.domain.NumGridTotal() != ntot ){
        ErrorHandling("Domain size does not match.");
    }

    // Temporary variable for saving wavefunction on a fine grid
    DblNumVec psiFine(ntotFine);
    DblNumVec psiUpdateFine(ntotFine);

    for (Int k=0; k<numStateLocal; k++) {
        for (Int j=0; j<ncom; j++) {

            SetValue( psiFine, 0.0 );
            SetValue( psiUpdateFine, 0.0 );

            // Fourier transform

            //      for( Int i = 0; i < ntot; i++ ){
            //        fft.inputComplexVec(i) = Complex( wavefun_(i,j,k), 0.0 ); 
            //      }
            SetValue( fft.inputComplexVec, Z_ZERO );
            blas::Copy( ntot, wavefun_.VecData(j,k), 1,
                    reinterpret_cast<Real*>(fft.inputComplexVec.Data()), 2 );

            // Fourier transform of wavefunction saved in fft.outputComplexVec
            fftw_execute( fft.forwardPlan );

            // Interpolate wavefunction from coarse to fine grid
            {
                SetValue( fft.outputComplexVecFine, Z_ZERO ); 
                Int *idxPtr = fft.idxFineGrid.Data();
                Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
                Complex *fftOutPtr = fft.outputComplexVec.Data();
                for( Int i = 0; i < ntot; i++ ){
                    //          fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i);
                    fftOutFinePtr[*(idxPtr++)] = *(fftOutPtr++);
                }
            }
            fftw_execute( fft.backwardPlanFine );
            Real fac = 1.0 / std::sqrt( double(domain_.NumGridTotal())  *
                    double(domain_.NumGridTotalFine()) ); 
            //      for( Int i = 0; i < ntotFine; i++ ){
            //        psiFine(i) = fft.inputComplexVecFine(i).real() * fac; 
            //      }
            blas::Copy( ntotFine, reinterpret_cast<Real*>(fft.inputComplexVecFine.Data()),
                    2, psiFine.Data(), 1 );
            blas::Scal( ntotFine, fac, psiFine.Data(), 1 );

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
            if(1){
                Int natm = pseudo.size();
                for (Int iatm=0; iatm<natm; iatm++) {
                    Int nobt = pseudo[iatm].vnlListFine.size();
                    for (Int iobt=0; iobt<nobt; iobt++) {
                        const Real       vnlwgt = pseudo[iatm].vnlListFine[iobt].second;
                        const SparseVec &vnlvecFine = pseudo[iatm].vnlListFine[iobt].first;
                        const IntNumVec &ivFine = vnlvecFine.first;
                        const DblNumMat &dvFine = vnlvecFine.second;

                        Real    weight = 0.0; 
                        const Int    *ivFineptr = ivFine.Data();
                        const Real   *dvFineptr = dvFine.VecData(VAL);
                        for (Int i=0; i<ivFine.m(); i++) {
                            weight += (*(dvFineptr++)) * psiFine[*(ivFineptr++)];
                        }
                        weight *= vol/Real(ntotFine)*vnlwgt;

                        ivFineptr = ivFine.Data();
                        dvFineptr = dvFine.VecData(VAL);
                        for (Int i=0; i<ivFine.m(); i++) {
                            psiUpdateFine[*(ivFineptr++)] += (*(dvFineptr++)) * weight;
                        }
                    } // for (iobt)
                } // for (iatm)
            }


            // Laplacian operator. Perform inverse Fourier transform in the end
            {
                for (Int i=0; i<ntot; i++) 
                    fft.outputComplexVec(i) *= fft.gkk(i);
            }

            // Restrict psiUpdateFine from fine grid in the real space to
            // coarse grid in the Fourier space. Combine with the Laplacian contribution
            //      for( Int i = 0; i < ntotFine; i++ ){
            //        fft.inputComplexVecFine(i) = Complex( psiUpdateFine(i), 0.0 ); 
            //      }
            SetValue( fft.inputComplexVecFine, Z_ZERO );
            blas::Copy( ntotFine, psiUpdateFine.Data(), 1,
                    reinterpret_cast<Real*>(fft.inputComplexVecFine.Data()), 2 );

            // Fine to coarse grid
            // Note the update is important since the Laplacian contribution is already taken into account.
            // The computation order is also important
            fftw_execute( fft.forwardPlanFine );
            {
                Real fac = std::sqrt(Real(ntot) / (Real(ntotFine)));
                Int* idxPtr = fft.idxFineGrid.Data();
                Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
                Complex *fftOutPtr = fft.outputComplexVec.Data();

                for( Int i = 0; i < ntot; i++ ){
                    //          fft.outputComplexVec(i) += fft.outputComplexVecFine(fft.idxFineGrid(i)) * fac;
                    *(fftOutPtr++) += fftOutFinePtr[*(idxPtr++)] * fac;
                }
            }

            // Inverse Fourier transform to save back to the output vector
            fftw_execute( fft.backwardPlan );

            //      Real    *ptr1 = a3.VecData(j,k);
            //      for( Int i = 0; i < ntot; i++ ){
            //        ptr1[i] += fft.inputComplexVec(i).real() / Real(ntot);
            //      }
            blas::Axpy( ntot, 1.0 / Real(ntot), 
                    reinterpret_cast<Real*>(fft.inputComplexVec.Data()), 2,
                    a3.VecData(j,k), 1 );
        }
    }



    return ;
}        // -----  end of method Spinor::AddMultSpinorFine  ----- 

void
Spinor::AddMultSpinorFineR2C ( Fourier& fft, const DblNumVec& vtot, 
        const std::vector<PseudoPot>& pseudo, NumTns<Real>& a3 )
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
                // input, therefore a copy of the original matrix is necessary. 
                blas::Copy( ntot, wavefun_.VecData(j,k), 1, 
                        fft.inputVecR2C.Data(), 1 );

                FFTWExecute ( fft, fft.forwardPlanR2C );

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

                FFTWExecute ( fft, fft.backwardPlanR2CFine );

                blas::Copy( ntotFine, fft.inputVecR2CFine.Data(), 1, psiFine.Data(), 1 );

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
            if(1){
                Int natm = pseudo.size();
                for (Int iatm=0; iatm<natm; iatm++) {
                    Int nobt = pseudo[iatm].vnlListFine.size();
                    for (Int iobt=0; iobt<nobt; iobt++) {
                        const Real       vnlwgt = pseudo[iatm].vnlListFine[iobt].second;
                        const SparseVec &vnlvecFine = pseudo[iatm].vnlListFine[iobt].first;
                        const IntNumVec &ivFine = vnlvecFine.first;
                        const DblNumMat &dvFine = vnlvecFine.second;

                        Real    weight = 0.0; 
                        const Int    *ivFineptr = ivFine.Data();
                        const Real   *dvFineptr = dvFine.VecData(VAL);
                        for (Int i=0; i<ivFine.m(); i++) {
                            weight += (*(dvFineptr++)) * psiFine[*(ivFineptr++)];
                        }
                        weight *= vol/Real(ntotFine)*vnlwgt;

                        ivFineptr = ivFine.Data();
                        dvFineptr = dvFine.VecData(VAL);
                        for (Int i=0; i<ivFine.m(); i++) {
                            psiUpdateFine[*(ivFineptr++)] += (*(dvFineptr++)) * weight;
                        }
                    } // for (iobt)
                } // for (iatm)
            }


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
            blas::Copy( ntotFine, psiUpdateFine.Data(), 1, fft.inputVecR2CFine.Data(), 1 );

            // Fine to coarse grid
            // Note the update is important since the Laplacian contribution is already taken into account.
            // The computation order is also important
            // fftw_execute( fft.forwardPlanFine );
            FFTWExecute ( fft, fft.forwardPlanR2CFine );
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

            FFTWExecute ( fft, fft.backwardPlanR2C );

            // Inverse Fourier transform to save back to the output vector
            //fftw_execute( fft.backwardPlan );

            blas::Axpy( ntot, 1.0, fft.inputVecR2C.Data(), 1, a3.VecData(j,k), 1 );

        } // j++
    } // k++
    //#ifdef _USE_OPENMP_
    //    }
    //#endif

    if(0)
    {
        for (Int k=0; k<numStateLocal; k++) {
            for (Int j=0; j<ncom; j++) {

                SetValue( fft.inputVecR2C, 0.0 );
                SetValue( fft.outputVecR2C, Z_ZERO );

                blas::Copy( ntot, a3.VecData(j,k), 1,
                        fft.inputVecR2C.Data(), 1 );
                FFTWExecute ( fft, fft.forwardPlanR2C ); // So outputVecR2C contains the FFT result now


                for (Int i=0; i<ntotR2C; i++)
                {
                    if(fft.gkkR2C(i) > 5.0)
                        fft.outputVecR2C(i) = Z_ZERO;
                }

                FFTWExecute ( fft, fft.backwardPlanR2C );
                blas::Copy( ntot,  fft.inputVecR2C.Data(), 1,
                        a3.VecData(j,k), 1 );

            }
        }
    }



    return ;
}        // -----  end of method Spinor::AddMultSpinorFineR2C  ----- 

void Spinor::AddMultSpinorEXX ( Fourier& fft, 
        const NumTns<Real>& phi,
        const DblNumVec& exxgkkR2C,
        Real  exxFraction,
        Real  numSpin,
        const DblNumVec& occupationRate,
        NumTns<Real>& a3 )
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

                        Real* a3Ptr = a3.VecData(j,k);
                        Real fac = -exxFraction * occupationRate[wavefunIdxTemp(kphi)];  
                        for( Int ir = 0; ir < ntot; ir++ ){
                            a3Ptr[ir] += fft.inputVecR2C(ir) * phiTemp(ir) * fac;
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
        NumTns<Real>& a3, 
        NumMat<Real>& VxMat,
        bool isFixColumnDF )
{
    Real timeSta, timeEnd;

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

    // Computing the indices is optional
    if( isFixColumnDF == false ){
        GetTime( timeSta );
        numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
        
        // Step 1: Pre-compression of the wavefunctions. This uses
        // multiplication with orthonormalized random Gaussian matrices
        //
        /// @todo The factor 2.0 is hard coded.  The PhiG etc should in
        /// principle be a tensor, but only treated as matrix.
        Int numPre = std::min(IRound(std::sqrt(numMu_*2.0)), numStateTotal);
        //    Int numPre = std::min(IRound(std::sqrt(numMu_))+5, numStateTotal);
        DblNumMat phiG(ntot, numPre), psiG(ntot, numPre);
        {
            DblNumMat G(numStateTotal, numPre);
            // Generate orthonormal Gaussian random matrix 
            GaussianRandom(G);
            lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );

            blas::Gemm( 'N', 'N', ntot, numPre, numStateTotal, 1.0, 
                    phi.Data(), ntot, G.Data(), numStateTotal, 0.0,
                    phiG.Data(), ntot );

            GaussianRandom(G);
            lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );

            blas::Gemm( 'N', 'N', ntot, numPre, numStateTotal, 1.0, 
                    wavefun_.Data(), ntot, G.Data(), numStateTotal, 0.0,
                    psiG.Data(), ntot );
        }

        // Step 2: Pivoted QR decomposition  for the Hadamard product of
        // the compressed matrix. Transpose format for QRCP
        DblNumMat MG( numPre*numPre, ntot );
        for( Int j = 0; j < numPre; j++ ){
            for( Int i = 0; i < numPre; i++ ){
                for( Int ir = 0; ir < ntot; ir++ ){
                    MG(i+j*numPre,ir) = phiG(ir,i) * psiG(ir,j);
                }
            }
        }

        // IntNumVec pivQR_(ntot);

        DblNumVec tau(ntot);
        pivQR_.Resize(ntot);
        SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
        // Q factor does not need to be used
        Real timeQRCPSta, timeQRCPEnd;
        GetTime( timeQRCPSta );
        lapack::QRCP( numPre*numPre, ntot, MG.Data(), numPre*numPre, 
                pivQR_.Data(), tau.Data() );
        GetTime( timeQRCPEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for QRCP alone is " <<
            timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;
#endif


        if(1){
            Real tolR = std::abs(MG(numMu_-1,numMu_-1)/MG(0,0));
            statusOFS << "numMu_ = " << numMu_ << std::endl;
            statusOFS << "|R(numMu-1,numMu-1)/R(0,0)| = " << tolR << std::endl;
        }

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
    DblNumMat Xi(ntot, numMu_);
    DblNumMat psiMu(numStateTotal, numMu_);
    // PhiMu is scaled by the occupation number to reflect the "true" density matrix
    DblNumMat PcolPhiMu(ntot, numMu_);
    IntNumVec pivMu(numMu_);
    
    {
        GetTime( timeSta );
        for( Int mu = 0; mu < numMu_; mu++ ){
            pivMu(mu) = pivQR_(mu);
        }
        
        // These three matrices are used only once
        DblNumMat phiMu(numStateTotal, numMu_);
        DblNumMat PcolMuNu(numMu_, numMu_);
        DblNumMat PcolPsiMu(ntot, numMu_);

        for( Int mu = 0; mu < numMu_; mu++ ){
            Int muInd = pivMu(mu);
            for (Int k=0; k<numStateTotal; k++) {
                psiMu(k, mu) = wavefun_(muInd,0,k);
                phiMu(k, mu) = phi(muInd,0,k) * occupationRate[k];
            }
        }
        
        blas::Gemm( 'N', 'N', ntot, numMu_, numStateTotal, 1.0, 
                wavefun_.Data(), ntot, psiMu.Data(), numStateTotal, 0.0,
                PcolPsiMu.Data(), ntot );
        blas::Gemm( 'N', 'N', ntot, numMu_, numStateTotal, 1.0, 
                phi.Data(), ntot, phiMu.Data(), numStateTotal, 0.0,
                PcolPhiMu.Data(), ntot );
        
        Real* xiPtr = Xi.Data();
        Real* PcolPsiMuPtr = PcolPsiMu.Data();
        Real* PcolPhiMuPtr = PcolPhiMu.Data();

        for( Int g = 0; g < ntot * numMu_; g++ ){
            xiPtr[g] = PcolPsiMuPtr[g] * PcolPhiMuPtr[g];
        }

        for( Int mu = 0; mu < numMu_; mu++ ){
            Int muInd = pivMu(mu);
            for (Int nu=0; nu < numMu_; nu++) {
                PcolMuNu( mu, nu ) = Xi( muInd, nu );
            }
        }

//        statusOFS << "PcolMuNu = " << PcolMuNu << std::endl;

        // Inversion based on Cholesky factorization
        // Xi <- Xi * L^{-T} L^{-1}
        // If overflow / underflow, reduce numMu_
        lapack::Potrf( 'L', numMu_, PcolMuNu.Data(), numMu_ );

        blas::Trsm( 'R', 'L', 'T', 'N', ntot, numMu_, 1.0, 
                PcolMuNu.Data(), numMu_, Xi.Data(), ntot );

        blas::Trsm( 'R', 'L', 'N', 'N', ntot, numMu_, 1.0, 
                PcolMuNu.Data(), numMu_, Xi.Data(), ntot );

        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing the interpolation vectors is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }
    
    
    // *********************************************************************
    // Solve the Poisson equations
    // Rewrite Xi by the potential of Xi
    // *********************************************************************

    {
        GetTime( timeSta );
        for( Int mu = 0; mu < numMu_; mu++ ){
            blas::Copy( ntot,  Xi.VecData(mu), 1, fft.inputVecR2C.Data(), 1 );

            FFTWExecute ( fft, fft.forwardPlanR2C );

            for( Int ig = 0; ig < ntotR2C; ig++ ){
                fft.outputVecR2C(ig) *= -exxFraction * exxgkkR2C(ig);
            }

            FFTWExecute ( fft, fft.backwardPlanR2C );

            blas::Copy( ntot, fft.inputVecR2C.Data(), 1, Xi.VecData(mu), 1 );
        } // for (mu)

        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for solving Poisson-like equations is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }
    
    // *********************************************************************
    // Compute the exchange potential and the symmetrized inner product
    // *********************************************************************

    {
        GetTime( timeSta );
        // Rewrite Xi by Xi.*PcolPhi
        Real* xiPtr = Xi.Data();
        Real* PcolPhiMuPtr = PcolPhiMu.Data();
        for( Int g = 0; g < ntot * numMu_; g++ ){
            xiPtr[g] *= PcolPhiMuPtr[g];
        }
        
        // NOTE: a3 must be zero in order to compute the M matrix later
        blas::Gemm( 'N', 'T', ntot, numStateTotal, numMu_, 1.0, 
                Xi.Data(), ntot, psiMu.Data(), numStateTotal, 1.0,
                a3.Data(), ntot ); 
    
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing the exchange potential is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

    // Compute the matrix VxMat = -Psi'* vexxPsi and symmetrize
    // vexxPsi (a3) must be zero before entering this routine
    VxMat.Resize( numStateTotal, numStateTotal );
    {
        // Minus sign so that VxMat is positive semidefinite
        // NOTE: No measure factor vol / ntot due to the normalization
        // factor of psi
        GetTime( timeSta );
        blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, -1.0,
                wavefun_.Data(), ntot, a3.Data(), ntot, 0.0, 
                VxMat.Data(), numStateTotal );

//        statusOFS << "VxMat = " << VxMat << std::endl;

        Symmetrize( VxMat );
        
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing VxMat in the sym format is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }


    MPI_Barrier(domain_.comm);

    return ;
}        // -----  end of method Spinor::AddMultSpinorEXXDF  ----- 


// Old implementation always with QR. The saving of QR compared to QRCP
// is not very large.
// 
//void Spinor::AddMultSpinorEXXDF ( Fourier& fft, 
//        const NumTns<Real>& phi,
//        const DblNumVec& exxgkkR2C,
//        Real  exxFraction,
//        Real  numSpin,
//        const DblNumVec& occupationRate,
//        const Real numMuFac,
//        NumTns<Real>& a3, 
//        NumMat<Real>& VxMat,
//        bool isFixColumnDF )
//{
//    Real timeSta, timeEnd;
//
//    if( !fft.isInitialized ){
//        ErrorHandling("Fourier is not prepared.");
//    }
//
//    MPI_Barrier(domain_.comm);
//    int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
//    int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
//
//    Index3& numGrid = domain_.numGrid;
//    Index3& numGridFine = domain_.numGridFine;
//
//    Int ntot     = domain_.NumGridTotal();
//    Int ntotFine = domain_.NumGridTotalFine();
//    Int ntotR2C = fft.numGridTotalR2C;
//    Int ntotR2CFine = fft.numGridTotalR2CFine;
//    Int ncom = wavefun_.n();
//    Int numStateLocal = wavefun_.p();
//    Int numStateTotal = numStateTotal_;
//
//    Int ncomPhi = phi.n();
//
//    Real vol = domain_.Volume();
//
//    if( ncomPhi != 1 || ncom != 1 ){
//        ErrorHandling("Spin polarized case not implemented.");
//    }
//
//    if( fft.domain.NumGridTotal() != ntot ){
//        ErrorHandling("Domain size does not match.");
//    }
//
//
//    // *********************************************************************
//    // Perform interpolative separable density fitting
//    // *********************************************************************
//    GetTime( timeSta );
//    // Perform QRCP 
//    IntNumVec pivMu;
//    DblNumMat Xi;
//    if( isFixColumnDF == false ){
//        numMu_ = std::min(IRound(numStateTotal*numMuFac), ntot);
//        
//        // Step 1: Pre-compression of the wavefunctions. This uses
//        // multiplication with orthonormalized random Gaussian matrices
//        //
//
//        /// @todo The factor 2.0 is hard coded.  The PhiG etc should in
//        /// principle be a tensor, but only treated as matrix.
//        Int numPre = std::min(IRound(std::sqrt(numMu_*2.0)), numStateTotal);
//        //    Int numPre = std::min(IRound(std::sqrt(numMu_))+5, numStateTotal);
//        DblNumMat phiG(ntot, numPre), psiG(ntot, numPre);
//        if(1)
//        {
//            DblNumMat G(numStateTotal, numPre);
//            // Generate orthonormal Gaussian random matrix 
//            GaussianRandom(G);
//            lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
//
//            blas::Gemm( 'N', 'N', ntot, numPre, numStateTotal, 1.0, 
//                    phi.Data(), ntot, G.Data(), numStateTotal, 0.0,
//                    phiG.Data(), ntot );
//
//            GaussianRandom(G);
//            lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
//
//            if(0){
//                DblNumMat OverG(numPre,numPre);
//                blas::Gemm( 'T', 'N', numPre, numPre, numStateTotal, 1.0,
//                        G.Data(), numStateTotal, G.Data(), numStateTotal, 0.0,
//                        OverG.Data(), numPre );
//                statusOFS << "OverG = " << OverG << std::endl;
//            }
//
//            blas::Gemm( 'N', 'N', ntot, numPre, numStateTotal, 1.0, 
//                    wavefun_.Data(), ntot, G.Data(), numStateTotal, 0.0,
//                    psiG.Data(), ntot );
//        }
//        //    if(0){
//        //        blas::Copy( ntot*numStateTotal, phi.Data(), 1, phiG.Data(), 1 );
//        //        blas::Copy( ntot*numStateTotal, wavefun_.Data(), 1, psiG.Data(), 1 );
//        //    }
//
//        // Step 2: Pivoted QR decomposition  for the Hadamard product of
//        // the compressed matrix. Transpose format for QRCP
//        DblNumMat MG( numPre*numPre, ntot );
//        for( Int j = 0; j < numPre; j++ ){
//            for( Int i = 0; i < numPre; i++ ){
//                for( Int ir = 0; ir < ntot; ir++ ){
//                    MG(i+j*numPre,ir) = phiG(ir,i) * psiG(ir,j);
//                }
//            }
//        }
////        IntNumVec pivQR_(ntot);
//
//        DblNumVec tau(ntot);
//        pivQR_.Resize(ntot);
//        SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
//        // Q factor does not need to be used
//        Real timeQRCPSta, timeQRCPEnd;
//        GetTime( timeQRCPSta );
//        lapack::QRCP( numPre*numPre, ntot, MG.Data(), numPre*numPre, 
//                pivQR_.Data(), tau.Data() );
//        GetTime( timeQRCPEnd );
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for QRCP is " <<
//            timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;
//#endif
//
//        // Important: eliminate the Q part in MG for equation solving
//        for( Int mu = 0; mu < numMu_; mu++ ){
//            for( Int i = mu+1; i < numMu_; i++ ){
//                MG(i,mu) = 0.0;
//            }
//        }
//
//
//        // Step 3: Construct the interpolation matrix
//        pivMu.Resize(numMu_);
//        for( Int mu = 0; mu < numMu_; mu++ ){
//            pivMu(mu) = pivQR_(mu);
//        }
//        Real tolR = std::abs(MG(numMu_-1,numMu_-1)/MG(0,0));
//        statusOFS << "numMu_ = " << numMu_ << std::endl;
//        statusOFS << "|R(numMu-1,numMu-1)/R(0,0)| = " << tolR << std::endl;
//        if(1){
//            Int numDiag = std::min(numPre*numPre, ntot);
//            DblNumVec diagR(numDiag);
//            for( Int i = 0; i < numDiag; i++ ){
//                diagR(i) = MG(i,i);
//            }
//            statusOFS << "diagR = " << diagR << std::endl;
////            statusOFS << "piv = " << pivQR_ << std::endl;
//        }
//
//        // Solve R_1^{-1} [R_1 R_2]
//        DblNumMat R1(numMu_, numMu_);
//        lapack::Lacpy('U', numMu_, numMu_, MG.Data(), numPre*numPre,
//                R1.Data(), numMu_);
//        blas::Trsm( 'L', 'U', 'N', 'N', numMu_, ntot, 1.0, 
//                R1.Data(), numMu_, MG.Data(), numPre*numPre );
//
//        // Store info from the first numMu_ rows into the MG into Xi
//        // after permutation
//        Xi.Resize(ntot, numMu_);
//        for( Int mu = 0; mu < numMu_; mu++ ){
//            for( Int ir = 0; ir < ntot; ir++ ){
//                Xi(pivQR_(ir),mu) = MG(mu,ir);
//            }
//        }
//        if(0){
//            DblNumVec diagR(numMu_);
//            for( Int i = 0; i < numMu_; i++ ){
//                diagR(i) = MG(i,i);
//            }
//            statusOFS << "diagR (xi) = " << diagR << std::endl;
//        }
//
//        GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for density fitting with QRCP is " <<
//            timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//    }
//    else{
//        // Reuse the information from pivQR_ and numMu_;
//        pivMu.Resize(numMu_);
//        for( Int mu = 0; mu < numMu_; mu++ ){
//            pivMu(mu) = pivQR_(mu);
//        }
//
//        // Step 1: Pre-compression of the wavefunctions. This uses
//        // multiplication with orthonormalized random Gaussian matrices
//
//        /// @todo The factor 2.0 is hard coded.  The PhiG etc should in
//        /// principle be a tensor, but only treated as matrix.
//        Int numPre = std::min(IRound(std::sqrt(numMu_*2.0)), numStateTotal);
//        //    Int numPre = std::min(IRound(std::sqrt(numMu_))+5, numStateTotal);
//        DblNumMat phiG(ntot, numPre), psiG(ntot, numPre);
//        if(1)
//        {
//            DblNumMat G(numStateTotal, numPre);
//            // Generate orthonormal Gaussian random matrix 
//            GaussianRandom(G);
//            lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
//
//            blas::Gemm( 'N', 'N', ntot, numPre, numStateTotal, 1.0, 
//                    phi.Data(), ntot, G.Data(), numStateTotal, 0.0,
//                    phiG.Data(), ntot );
//
//            GaussianRandom(G);
//            lapack::Orth( numStateTotal, numPre, G.Data(), numStateTotal );
//
//            blas::Gemm( 'N', 'N', ntot, numPre, numStateTotal, 1.0, 
//                    wavefun_.Data(), ntot, G.Data(), numStateTotal, 0.0,
//                    psiG.Data(), ntot );
//        }
//
//        // Step 2: QR decomposition for the pivoted matrix
//        DblNumMat MG( numPre*numPre, ntot );
//        for( Int j = 0; j < numPre; j++ ){
//            for( Int i = 0; i < numPre; i++ ){
//                for( Int ir = 0; ir < ntot; ir++ ){
//                    MG(i+j*numPre,ir) = phiG(pivQR_(ir),i) * psiG(pivQR_(ir),j);
//                }
//            }
//        }
//        DblNumVec tau(ntot);
//        // Q factor does not need to be used
//        Real timeQRSta, timeQREnd;
//        GetTime( timeQRSta );
//        lapack::QR( numPre*numPre, ntot, MG.Data(), numPre*numPre, tau.Data() );
//        GetTime( timeQREnd );
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for QR is " <<
//            timeQREnd - timeQRSta << " [s]" << std::endl << std::endl;
//#endif
//
//        // Important: eliminate the Q part in MG for equation solving
//        for( Int mu = 0; mu < numMu_; mu++ ){
//            for( Int i = mu+1; i < numMu_; i++ ){
//                MG(i,mu) = 0.0;
//            }
//        }
//
//
//        // Step 3: Construct the interpolation matrix
//        Real tolR = std::abs(MG(numMu_-1,numMu_-1)/MG(0,0));
//        statusOFS << "numMu_ = " << numMu_ << std::endl;
//        statusOFS << "|R(numMu-1,numMu-1)/R(0,0)| = " << tolR << std::endl;
//        if(1){
//            Int numDiag = std::min(numPre*numPre, ntot);
//            DblNumVec diagR(numDiag);
//            for( Int i = 0; i < numDiag; i++ ){
//                diagR(i) = MG(i,i);
//            }
//            statusOFS << "diagR = " << diagR << std::endl;
////            statusOFS << "piv = " << pivQR_ << std::endl;
//        }
//
//        // Solve R_1^{-1} [R_1 R_2]
//        DblNumMat R1(numMu_, numMu_);
//        lapack::Lacpy('U', numMu_, numMu_, MG.Data(), numPre*numPre,
//                R1.Data(), numMu_);
//        blas::Trsm( 'L', 'U', 'N', 'N', numMu_, ntot, 1.0, 
//                R1.Data(), numMu_, MG.Data(), numPre*numPre );
//
//        // Store info from the first numMu_ rows into the MG into Xi
//        // after permutation
//        Xi.Resize(ntot, numMu_);
//        for( Int mu = 0; mu < numMu_; mu++ ){
//            for( Int ir = 0; ir < ntot; ir++ ){
//                Xi(pivQR_(ir),mu) = MG(mu,ir);
//            }
//        }
//        if(0){
//            DblNumVec diagR(numMu_);
//            for( Int i = 0; i < numMu_; i++ ){
//                diagR(i) = MG(i,i);
//            }
//            statusOFS << "diagR (xi) = " << diagR << std::endl;
//        }
//
//        GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for density fitting with QR is " <<
//            timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//    }
//    // *********************************************************************
//    // Solve the Poisson equations
//    // Also accumulate the VxMat matrix
//    // VxMat = Psi'*Vx[Phi]*Psi
//    // This is performed using the symmetric format with density fitting
//    // being performed symmetrically
//    // 
//    // This is not a memory efficient implementation
//    // *********************************************************************
//
//    // Step 1: Solve the Poisson-like problem for exchange
//    GetTime( timeSta );
//    DblNumMat XiPot(ntot, numMu_);
//    for( Int mu = 0; mu < numMu_; mu++ ){
//        blas::Copy( ntot,  Xi.VecData(mu), 1, fft.inputVecR2C.Data(), 1 );
//
//        FFTWExecute ( fft, fft.forwardPlanR2C );
//
//        for( Int ig = 0; ig < ntotR2C; ig++ ){
//            fft.outputVecR2C(ig) *= -exxFraction * exxgkkR2C(ig);
//        }
//
//        FFTWExecute ( fft, fft.backwardPlanR2C );
//
//        blas::Copy( ntot, fft.inputVecR2C.Data(), 1, XiPot.VecData(mu), 1 );
//    } // for (mu)
//    
//    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for solving Poisson-like equations is " <<
//        timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//    
//
//    // Step 2: accumulate to the matrix vector multiplication
//    DblNumMat phiMod(ntot, numMu_);
//    SetValue( phiMod, 0.0 );
//    // accumulate \sum_j f_j \varphi_j (r) \varphi_j(r_mu) 
//    // can be done with gemv
//    if(0){
//        GetTime( timeSta );
//        for( Int mu = 0; mu < numMu_; mu++ ){
//            for( Int k = 0; k < numStateTotal; k++ ){
//                blas::Axpy(ntot, phi(pivMu(mu), 0, k) * occupationRate[k], 
//                        phi.VecData(0, k), 1, phiMod.VecData(mu), 1 );
//            }
//        }
//        GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for computing phiMod via Axpy is " <<
//            timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//    }
//    if(1){
//        DblNumMat phiWeight(numStateTotal, numMu_);
//        for( Int mu = 0; mu < numMu_; mu++ ){
//            Int muInd = pivMu(mu);
//            for( Int k = 0; k < numStateTotal; k++ ){
//                phiWeight(k,mu) = phi(muInd, 0, k) * occupationRate[k];
//            }
//        }
//        blas::Gemm( 'N', 'N', ntot, numMu_, numStateTotal, 1.0,
//                phi.Data(), ntot, phiWeight.Data(), numStateTotal, 0.0,
//                phiMod.Data(), ntot );
//        GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for computing phiMod via Gemm is " <<
//            timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//    }
//
//    DblNumMat psiMu(numMu_, numStateTotal);
//    if(0){
//        GetTime( timeSta );
//        for (Int k=0; k<numStateTotal; k++) {
//            for( Int mu = 0; mu < numMu_; mu++ ){
//                psiMu(mu,k) = wavefun_(pivMu(mu),0,k);
//                //            for( Int ir = 0; ir < ntot; ir++ ){
//                //                a3(ir, 0, k) += psiMu(mu,k) * XiPot(ir,mu) * phiMod(ir,mu);
//                //            }
//                // Very significant improvement of performance 
//                Real* a3Ptr = a3.VecData(0,k);
//                Real* xiPotPtr = XiPot.VecData(mu);
//                Real* phiModPtr = phiMod.VecData(mu);
//                Real fac = psiMu(mu,k);
//                for( Int ir = 0; ir < ntot; ir++ ){
//                    a3Ptr[ir] += fac * xiPotPtr[ir] * phiModPtr[ir];
//                }
//            }
//        } // for (k)
//        GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for computing V_x[Phi] Psi via BLAS 1 is " <<
//            timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//    }
//    if(1){
//        GetTime( timeSta );
//        for( Int mu = 0; mu < numMu_; mu++ ){
//            Int muInd = pivMu(mu);
//            for (Int k=0; k<numStateTotal; k++) {
//                psiMu(mu,k) = wavefun_(muInd,0,k);
//            }
//        }
//        DblNumMat XiPhi( ntot, numMu_ );
//        Real* xiPotPtr = XiPot.Data();
//        Real* phiModPtr = phiMod.Data();
//        Real* xiPhiPtr = XiPhi.Data();
//        // XiPhi = XiPot .* phiMod  
//        for( Int g = 0; g < ntot * numMu_; g++ ){
//            *(xiPhiPtr++) = (*(xiPotPtr++)) * (*(phiModPtr++)); 
//        }
//        blas::Gemm( 'N', 'N', ntot, numStateTotal, numMu_, 1.0, 
//                XiPhi.Data(), ntot, psiMu.Data(), numMu_, 1.0,
//                a3.Data(), ntot ); 
//        GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//        statusOFS << "Time for computing V_x[Phi] Psi via GEMM is " <<
//            timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//    }
//
//    // Step 3: Compute the matrix VxMat = -Psi'* Vx[Phi] * Psi in the
//    // density fitting format
//    GetTime( timeSta );
//    VxMat.Resize( numStateTotal, numStateTotal );
//    {
//        DblNumMat OverMat( numMu_, numMu_ ); 
//        // Minus sign so that VxMat is positive semidefinite
//        // NOTE: No measure factor vol / ntot due to the normalization
//        // factor of psi
//        blas::Gemm( 'T', 'N', numMu_, numMu_, ntot, -1.0,
//                Xi.Data(), ntot, XiPot.Data(), ntot, 0.0, 
//                OverMat.Data(), numMu_ );
//        for( Int mu = 0; mu < numMu_; mu++ ){
//            for( Int nu = 0; nu < numMu_; nu++ ){
//                OverMat(mu,nu) *= phiMod(pivQR_(mu), nu);
//            }
//        }
//        
//        DblNumMat TempMat1(numStateTotal, numMu_);
//        blas::Gemm( 'T', 'N', numStateTotal, numMu_, numMu_, 1.0,
//                psiMu.Data(), numMu_, OverMat.Data(), numMu_, 0.0,
//                TempMat1.Data(), numStateTotal );
//
//        blas::Gemm( 'N', 'N', numStateTotal, numStateTotal, numMu_, 1.0,
//                TempMat1.Data(), numStateTotal, psiMu.Data(), numMu_,
//                0.0, VxMat.Data(), numStateTotal );
//    }
//
//    GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 0 )
//    statusOFS << "Time for computing VxMat in the sym format is " <<
//        timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//    MPI_Barrier(domain_.comm);
//
//
//    return ;
//}        // -----  end of method Spinor::AddMultSpinorEXXDF  ----- 

}  // namespace dgdft
