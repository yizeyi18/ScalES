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
#ifndef _RELEASE_
	PushCallStack("Spinor::Spinor");
#endif  // ifndef _RELEASE_
	this->Setup( dm, numComponent, numStateTotal, numStateLocal, val );

#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_
} 		// -----  end of method Spinor::Spinor  ----- 

Spinor::Spinor ( const Domain &dm, 
		const Int numComponent, 
		const Int numStateTotal,
          Int numStateLocal,
		const bool owndata, 
		Real* data )
{
#ifndef _RELEASE_
	PushCallStack("Spinor::Spinor");
#endif  // ifndef _RELEASE_
	this->Setup( dm, numComponent, numStateTotal, numStateLocal, owndata, data );
#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_

} 		// -----  end of method Spinor::Spinor  ----- 

Spinor::~Spinor	() {}

void Spinor::Setup ( 
		const Domain &dm, 
		const Int     numComponent,
		const Int     numStateTotal,
          Int     numStateLocal,
		const Real  val ) {
#ifndef _RELEASE_
	PushCallStack("Spinor::Setup ");
#endif  // ifndef _RELEASE_
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
  
#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_
} 		// -----  end of method Spinor::Setup  ----- 

void Spinor::Setup ( const Domain &dm, 
		const Int numComponent, 
		const Int numStateTotal,
          Int numStateLocal,
		const bool owndata, 
		Real* data )
{
#ifndef _RELEASE_
  PushCallStack("Spinor::Setup");
#endif  // ifndef _RELEASE_

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  domain_       = dm;

  wavefun_      = NumTns<Real>( dm.NumGridTotal(), numComponent, numStateLocal,
      owndata, data );

  wavefunIdx_.Resize( numStateLocal );
  SetValue( wavefunIdx_, 0 );
  for (Int i = 0; i < numStateLocal; i++){
    wavefunIdx_[i] = i * mpisize + mpirank ;
  }

#ifndef _RELEASE_
  PopCallStack();
#endif  // ifndef _RELEASE_
} 		// -----  end of method Spinor::Setup  ----- 

void
Spinor::Normalize	( )
{
#ifndef _RELEASE_
  PushCallStack("Spinor::Normalize");
#endif
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
#ifndef _RELEASE_
	PopCallStack();
#endif
	return ;
} 		// -----  end of method Spinor::Normalize  ----- 

void
Spinor::AddRealDiag	(Int iocc, const DblNumVec &val, NumMat<Real>& y)
{
#ifndef _RELEASE_
	PushCallStack("Spinor::AddRealDiag");
#endif
	if( val.m() == 0 || val.m() != wavefun_.m() ){
		throw std::logic_error("Vector dimension does not match.");
	}

	Int ntot = wavefun_.m();
	Int ncom = wavefun_.n();
	Int nocc = wavefun_.p();

	if( iocc < 0 || iocc >= nocc ){
		throw std::logic_error("iocc is out of bound.");
	}	


	Int k = iocc;
	for (Int j=0; j<ncom; j++) {
		Real *p1 = wavefun_.VecData(j, k);
		Real   *p2 = val.Data();
		Real *p3 = y.VecData(j);
    for (Int i=0; i<ntot; i++) { 
      *(p3) += (*p1) * (*p2); p3++; p1++; p2++; 
    }
	}

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method Spinor::AddRealDiag  ----- 

void Spinor::AddRealDiag	(const DblNumVec &val, NumTns<Real> &a3)
{
#ifndef _RELEASE_
	PushCallStack("Spinor::AddRealDiag");
#endif
	if( val.m() == 0 || val.m() != wavefun_.m() ){
		throw std::logic_error("Vector dimension does not match.");
	}

	Int ntot = wavefun_.m();
	Int ncom = wavefun_.n();
	Int nocc = wavefun_.p();

#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) nowait
#endif
  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
      Real *p1 = wavefun_.VecData(j, k);
      Real   *p2 = val.Data();
      Real *p3 = a3.VecData(j, k);
      for (Int i=0; i<ntot; i++) { 
        *(p3++) += (*p1++) * (*p2++); 
      }
    }
  }

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method Spinor::AddRealDiag  ----- 

void
Spinor::AddLaplacian (Int iocc, Fourier* fftPtr, NumMat<Real>& y)
{
#ifndef _RELEASE_
	PushCallStack("Spinor::AddLaplacian");
#endif
	if( !fftPtr->isInitialized ){
		throw std::runtime_error("Fourier is not prepared.");
	}
	Int ntot = wavefun_.m();
	Int ncom = wavefun_.n();
	Int nocc = wavefun_.p();

	if( fftPtr->domain.NumGridTotal() != ntot ){
		throw std::logic_error("Domain size does not match.");
	}

#ifndef _USE_COMPLEX_ // Real case
	Int ntothalf = fftPtr->numGridTotalR2C;
	{
    // These two are private variables in the OpenMP context
    DblNumVec realInVec(ntot);
		CpxNumVec cpxOutVec(ntothalf);

    Int k = iocc;
    for (Int j=0; j<ncom; j++) {
      Real* p3 = y.VecData(j);
      // For c2r and r2c transforms, the default is to DESTROY the
      // input, therefore a copy of the original matrix is necessary. 
      blas::Copy( ntot, wavefun_.VecData(j, k), 1, 
          realInVec.Data(), 1 );
      fftw_execute_dft_r2c(
          fftPtr->forwardPlanR2C, 
          realInVec.Data(),
          reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ));

      Real*    ptr1d   = fftPtr->gkkR2C.Data();
      Complex* ptr2    = cpxOutVec.Data();
      for (Int i=0; i<ntothalf; i++) 
        *(ptr2++) *= *(ptr1d++);

      fftw_execute_dft_c2r(
          fftPtr->backwardPlanR2C,
          reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ),
          realInVec.Data() );

      blas::Axpy( ntot, 1.0 / Real(ntot), realInVec.Data(), 1, 
          p3, 1 );
    }
	}
#endif

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method Spinor::AddLaplacian  ----- 


void
Spinor::AddLaplacian (Fourier* fftPtr, NumTns<Real>& a3)
{
#ifndef _RELEASE_
	PushCallStack("Spinor::AddLaplacian");
#endif
	if( !fftPtr->isInitialized ){
		throw std::runtime_error("Fourier is not prepared.");
	}
	Int ntot = wavefun_.m();
	Int ncom = wavefun_.n();
	Int nocc = wavefun_.p();

	if( fftPtr->domain.NumGridTotal() != ntot ){
		throw std::logic_error("Domain size does not match.");
	}

#ifndef _USE_COMPLEX_ // Real case
	Int ntothalf = fftPtr->numGridTotalR2C;
	{
    // These two are private variables in the OpenMP context
    DblNumVec realInVec(ntot);
		CpxNumVec cpxOutVec(ntothalf);

#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) nowait
#endif
		for (Int k=0; k<nocc; k++) {
			for (Int j=0; j<ncom; j++) {
        // For c2r and r2c transforms, the default is to DESTROY the
        // input, therefore a copy of the original matrix is necessary. 
        blas::Copy( ntot, wavefun_.VecData(j, k), 1, 
            realInVec.Data(), 1 );
				fftw_execute_dft_r2c(
						fftPtr->forwardPlanR2C, 
						realInVec.Data(),
						reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ));

				Real*    ptr1d   = fftPtr->gkkR2C.Data();
				Complex* ptr2    = cpxOutVec.Data();
				for (Int i=0; i<ntothalf; i++) 
					*(ptr2++) *= *(ptr1d++);

				fftw_execute_dft_c2r(
						fftPtr->backwardPlanR2C,
						reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ),
						realInVec.Data() );

        blas::Axpy( ntot, 1.0 / Real(ntot), realInVec.Data(), 1, 
            a3.VecData(j, k), 1 );
			}
		}
	}
#else // Complex case
  // TODO OpenMP implementation
	for (Int k=0; k<nocc; k++) {
		for (Int j=0; j<ncom; j++) {
			Complex *ptr0 = wavefun_.VecData(j, k);
			fftw_execute_dft(fftPtr->forwardPlan, reinterpret_cast<fftw_complex*>(ptr0), 
					reinterpret_cast<fftw_complex*>(fftPtr->outputComplexVec.Data() ));
			
			Real *ptr1d = fftPtr->gkk.Data();
			ptr0 = fftPtr->outputComplexVec.Data();
			for (Int i=0; i<ntot; i++) 
				*(ptr0++) *= *(ptr1d++);


			fftw_execute(fftPtr->backwardPlan);
			Complex *ptr1 = a3.VecData(j, k);
			ptr0 = fftPtr->inputComplexVec.Data();
			for (Int i=0; i<ntot; i++) *(ptr1++) += *(ptr0++) / Real(ntot);
		}
	}
#endif

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method Spinor::AddLaplacian  ----- 


void
Spinor::AddNonlocalPP	(Int iocc, const std::vector<PseudoPot>& pseudo, 
    NumMat<Real>& y)
{
#ifndef _RELEASE_
	PushCallStack("Spinor::AddNonlocalPP");
#endif
	Int ntot = wavefun_.m(); 
	Int ncom = wavefun_.n();
	Int nocc = wavefun_.p();
	Real vol = domain_.Volume();

  Int k = iocc;
  for (Int j=0; j<ncom; j++) {
    Real    *ptr0 = wavefun_.VecData(j,k);
    Real    *ptr1 = y.VecData(j);
    Int natm = pseudo.size();
    for (Int iatm=0; iatm<natm; iatm++) {
      Int nobt = pseudo[iatm].vnlList.size();
      for (Int iobt=0; iobt<nobt; iobt++) {
        const SparseVec &vnlvec = pseudo[iatm].vnlList[iobt].first;
        const Real       vnlwgt = pseudo[iatm].vnlList[iobt].second;
        const IntNumVec &iv = vnlvec.first;
        const DblNumMat &dv = vnlvec.second;

        Real    weight = 0.0; 
        const Int    *ivptr = iv.Data();
        const Real   *dvptr = dv.VecData(VAL);
        for (Int i=0; i<iv.m(); i++) {
          weight += (*(dvptr++)) * ptr0[*(ivptr++)];
        }
        weight *= vol/Real(ntot)*vnlwgt;

        ivptr = iv.Data();
        dvptr = dv.VecData(VAL);
        for (Int i=0; i<iv.m(); i++) {
          ptr1[*(ivptr++)] += (*(dvptr++)) * weight;
        }
      }
    }
	}

#ifndef _RELEASE_
	PopCallStack();
#endif
	return ;
} 		// -----  end of method Spinor::AddNonlocalPP  ----- 


void
Spinor::AddNonlocalPP	(const std::vector<PseudoPot>& pseudo, NumTns<Real> &a3)
{
#ifndef _RELEASE_
	PushCallStack("Spinor::AddNonlocalPP");
#endif
	Int ntot = wavefun_.m(); 
	Int ncom = wavefun_.n();
	Int nocc = wavefun_.p();
	Real vol = domain_.Volume();

#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) nowait
#endif
	for (Int k=0; k<nocc; k++) {
		for (Int j=0; j<ncom; j++) {
			Real    *ptr0 = wavefun_.VecData(j,k);
			Real    *ptr1 = a3.VecData(j,k);
			Int natm = pseudo.size();
			for (Int iatm=0; iatm<natm; iatm++) {
				Int nobt = pseudo[iatm].vnlList.size();
				for (Int iobt=0; iobt<nobt; iobt++) {
					const SparseVec &vnlvec = pseudo[iatm].vnlList[iobt].first;
					const Real       vnlwgt = pseudo[iatm].vnlList[iobt].second;
					const IntNumVec &iv = vnlvec.first;
					const DblNumMat &dv = vnlvec.second;

					Real    weight = 0.0; 
					const Int    *ivptr = iv.Data();
					const Real   *dvptr = dv.VecData(VAL);
					for (Int i=0; i<iv.m(); i++) {
						weight += (*(dvptr++)) * ptr0[*(ivptr++)];
					}
					weight *= vol/Real(ntot)*vnlwgt;

					ivptr = iv.Data();
					dvptr = dv.VecData(VAL);
					for (Int i=0; i<iv.m(); i++) {
						ptr1[*(ivptr++)] += (*(dvptr++)) * weight;
					}
				}
			}
		}
	}

#ifndef _RELEASE_
	PopCallStack();
#endif
	return ;
} 		// -----  end of method Spinor::AddNonlocalPP  ----- 


void
Spinor::AddTeterPrecond ( Int iocc, Fourier* fftPtr, NumTns<Real>& a3)
{
#ifndef _RELEASE_
	PushCallStack("Spinor::AddTeterPrecond");
#endif
	if( !fftPtr->isInitialized ){
		throw std::runtime_error("Fourier is not prepared.");
	}
	Int ntot = wavefun_.m();
	Int ncom = wavefun_.n();
	Int nocc = wavefun_.p();

	if( fftPtr->domain.NumGridTotal() != ntot ){
		throw std::logic_error("Domain size does not match.");
	}

  // For convenience
  NumTns<Real>& a3i = wavefun_; 
  NumTns<Real>& a3o = a3;

//#ifndef _USE_COMPLEX_ // Real case
    Int ntothalf = fftPtr->numGridTotalR2C;
    // These two are private variables in the OpenMP context
    DblNumVec realInVec(ntot);
		CpxNumVec cpxOutVec(ntothalf);

    Int k = iocc; 
      for (Int j=0; j<ncom; j++) {
        // For c2r and r2c transforms, the default is to DESTROY the
        // input, therefore a copy of the original matrix is necessary. 
        blas::Copy( ntot, a3i.VecData(j,k), 1, 
            realInVec.Data(), 1 );

				fftw_execute_dft_r2c(
						fftPtr->forwardPlanR2C, 
						realInVec.Data(),
						reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ));

        Real*    ptr1d   = fftPtr->TeterPrecondR2C.Data();
				Complex* ptr2    = cpxOutVec.Data();
        for (Int i=0; i<ntothalf; i++) 
					*(ptr2++) *= *(ptr1d++);

				fftw_execute_dft_c2r(
						fftPtr->backwardPlanR2C,
						reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ),
						realInVec.Data() );

        blas::Axpy( ntot, 1.0 / Real(ntot), realInVec.Data(), 1, 
            a3o.VecData(j, k), 1 );
      }

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method Spinor::AddTeterPrecond ----- 

void
Spinor::AddTeterPrecond (Fourier* fftPtr, NumTns<Real>& a3)
{
#ifndef _RELEASE_
	PushCallStack("Spinor::AddTeterPrecond");
#endif
	if( !fftPtr->isInitialized ){
		throw std::runtime_error("Fourier is not prepared.");
	}
	Int ntot = wavefun_.m();
	Int ncom = wavefun_.n();
	Int nocc = wavefun_.p();

	if( fftPtr->domain.NumGridTotal() != ntot ){
		throw std::logic_error("Domain size does not match.");
	}

  // For convenience
  NumTns<Real>& a3i = wavefun_; 
  NumTns<Real>& a3o = a3;

#ifdef _USE_OPENMP_
#pragma omp parallel
  {
#endif
#ifndef _USE_COMPLEX_ // Real case
    Int ntothalf = fftPtr->numGridTotalR2C;
    // These two are private variables in the OpenMP context
    DblNumVec realInVec(ntot);
		CpxNumVec cpxOutVec(ntothalf);

#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) nowait
#endif
    for (Int k=0; k<nocc; k++) {
      for (Int j=0; j<ncom; j++) {
        // For c2r and r2c transforms, the default is to DESTROY the
        // input, therefore a copy of the original matrix is necessary. 
        blas::Copy( ntot, a3i.VecData(j,k), 1, 
            realInVec.Data(), 1 );

				fftw_execute_dft_r2c(
						fftPtr->forwardPlanR2C, 
						realInVec.Data(),
						reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ));

        Real*    ptr1d   = fftPtr->TeterPrecondR2C.Data();
				Complex* ptr2    = cpxOutVec.Data();
        for (Int i=0; i<ntothalf; i++) 
					*(ptr2++) *= *(ptr1d++);

				fftw_execute_dft_c2r(
						fftPtr->backwardPlanR2C,
						reinterpret_cast<fftw_complex*>(cpxOutVec.Data() ),
						realInVec.Data() );

        blas::Axpy( ntot, 1.0 / Real(ntot), realInVec.Data(), 1, 
            a3o.VecData(j, k), 1 );
      }
    }
#else // Complex case
  // TODO OpenMP implementation
    for (Int k=0; k<nocc; k++) {
      for (Int j=0; j<ncom; j++) {
        Complex *ptr0  = a3i.VecData(j,k);

        fftw_execute_dft(fftPtr>forwardPlan, reinterpret_cast<fftw_complex*>(ptr0), 
            reinterpret_cast<fftw_complex*>(fftPtr->outputComplexVec.Data() ));
        Real *ptr1d = fftPtr->TeterPrecond.Data();
        ptr0 = fftPtr->outputComplexVec.Data();
        for (Int i=0; i<ntot; i++) 
          *(ptr0++) *= *(ptr1d++);

        fftw_execute(fftPtr->backwardPlan);
        ptr0 = fftPtr->inputComplexVec.Data();

        Complex *ptra3o = a3o.VecData(j, k); 
        for (Int i=0; i<ntot; i++) 
          *(ptra3o++) = *(ptr0++) / Real(ntot);
      }
    }
#endif
#ifdef _USE_OPENMP_
  }
#endif

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method Spinor::AddTeterPrecond ----- 

void
Spinor::AddMultSpinorFine ( Fourier& fft, const DblNumVec& vtot, 
    const std::vector<PseudoPot>& pseudo, NumTns<Real>& a3 )
{
#ifndef _RELEASE_
	PushCallStack("Spinor::AddMultSpinorFine");
#endif
  // TODO Complex case

	if( !fft.isInitialized ){
		throw std::runtime_error("Fourier is not prepared.");
	}
	Int ntot = wavefun_.m();
	Int ncom = wavefun_.n();
	Int numStateLocal = wavefun_.p();
  Int ntotFine = domain_.NumGridTotalFine();
	Real vol = domain_.Volume();

	if( fft.domain.NumGridTotal() != ntot ){
		throw std::logic_error("Domain size does not match.");
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
  

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
}		// -----  end of method Spinor::AddMultSpinorFine  ----- 

void
Spinor::AddMultSpinorFineR2C ( Fourier& fft, const DblNumVec& vtot, 
    const std::vector<PseudoPot>& pseudo, NumTns<Real>& a3 )
{
#ifndef _RELEASE_
  PushCallStack("Spinor::AddMultSpinorFineR2C");
#endif
  // TODO Complex case

  if( !fft.isInitialized ){
    throw std::runtime_error("Fourier is not prepared.");
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
    throw std::logic_error("Domain size does not match.");
  }

  // Temporary variable for saving wavefunction on a fine grid
  DblNumVec psiFine(ntotFine);
  DblNumVec psiUpdateFine(ntotFine);

  // FIXME OpenMP does not work since all variables are shared
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

#ifndef _RELEASE_
  PopCallStack();
#endif

  return ;
}		// -----  end of method Spinor::AddMultSpinorFineR2C  ----- 

// EXX: Spinor with exact exchange. Will be merged later.
// However, keeping the names separate is good for now, since the new
// algorithm requires a different set of input parameters for AddMultSpinor
// 
// NOTE 
// Currently there is no parallelization over the phi tensor
//
void Spinor::AddMultSpinorEXX ( Fourier& fft, 
    const NumTns<Real>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Real  numSpin,
    const DblNumVec& occupationRate,
    NumTns<Real>& a3 )
{
#ifndef _RELEASE_
  PushCallStack("Spinor::AddMultSpinorEXX");
#endif
  if( !fft.isInitialized ){
    throw std::runtime_error("Fourier is not prepared.");
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
    throw std::logic_error("Spin polarized case not implemented.");
  }

  if( fft.domain.NumGridTotal() != ntot ){
    throw std::logic_error("Domain size does not match.");
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

#ifndef _RELEASE_
  PopCallStack();
#endif

  return ;
}		// -----  end of method Spinor::AddMultSpinorEXX  ----- 

}  // namespace dgdft
