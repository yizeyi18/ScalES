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
		const Scalar  val ) {
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
		Scalar* data )
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
    const Int     numStateLocal,
		const Scalar  val ) {
#ifndef _RELEASE_
	PushCallStack("Spinor::Setup ");
#endif  // ifndef _RELEASE_
	domain_       = dm;

  // huwei
  
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

  // Check Sum{numStateLocal} = numStateTotal
  Int numStateTotalTest; 
  Int numState = numStateLocal;

  mpi::Allreduce( &numState, &numStateTotalTest, 1, MPI_SUM, domain_.comm );

//  if( numStateTotalTest != numStateTotal ){
//    throw std::logic_error("Sum{numStateLocal} = numStateTotal does not match.");
//  }
 
  // huwei

  wavefun_.Resize( dm.NumGridTotal(), numComponent, numStateLocal );
	SetValue( wavefun_, val );

#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_
} 		// -----  end of method Spinor::Setup  ----- 

void Spinor::Setup ( const Domain &dm, 
		const Int numComponent, 
		const Int numStateTotal,
    const Int numStateLocal,
		const bool owndata, 
		Scalar* data )
{
#ifndef _RELEASE_
	PushCallStack("Spinor::Setup");
#endif  // ifndef _RELEASE_
	domain_       = dm;
  // FIXME Partition the spinor here
	wavefun_      = NumTns<Scalar>( dm.NumGridTotal(), numComponent, numStateLocal,
      owndata, data );

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
		Scalar *ptr = wavefun_.MatData(k);
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
Spinor::AddScalarDiag	(Int iocc, const DblNumVec &val, NumMat<Scalar>& y)
{
#ifndef _RELEASE_
	PushCallStack("Spinor::AddScalarDiag");
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
		Scalar *p1 = wavefun_.VecData(j, k);
		Real   *p2 = val.Data();
		Scalar *p3 = y.VecData(j);
    for (Int i=0; i<ntot; i++) { 
      *(p3) += (*p1) * (*p2); p3++; p1++; p2++; 
    }
	}

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method Spinor::AddScalarDiag  ----- 

void Spinor::AddScalarDiag	(const DblNumVec &val, NumTns<Scalar> &a3)
{
#ifndef _RELEASE_
	PushCallStack("Spinor::AddScalarDiag");
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
      Scalar *p1 = wavefun_.VecData(j, k);
      Real   *p2 = val.Data();
      Scalar *p3 = a3.VecData(j, k);
      for (Int i=0; i<ntot; i++) { 
        *(p3++) += (*p1++) * (*p2++); 
      }
    }
  }

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method Spinor::AddScalarDiag  ----- 

void
Spinor::AddLaplacian (Int iocc, Fourier* fftPtr, NumMat<Scalar>& y)
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
      Scalar* p3 = y.VecData(j);
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
Spinor::AddLaplacian (Fourier* fftPtr, NumTns<Scalar>& a3)
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
    NumMat<Scalar>& y)
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
    Scalar    *ptr0 = wavefun_.VecData(j,k);
    Scalar    *ptr1 = y.VecData(j);
    Int natm = pseudo.size();
    for (Int iatm=0; iatm<natm; iatm++) {
      Int nobt = pseudo[iatm].vnlList.size();
      for (Int iobt=0; iobt<nobt; iobt++) {
        const SparseVec &vnlvec = pseudo[iatm].vnlList[iobt].first;
        const Real       vnlwgt = pseudo[iatm].vnlList[iobt].second;
        const IntNumVec &iv = vnlvec.first;
        const DblNumMat &dv = vnlvec.second;

        Scalar    weight = SCALAR_ZERO; 
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
Spinor::AddNonlocalPP	(const std::vector<PseudoPot>& pseudo, NumTns<Scalar> &a3)
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
			Scalar    *ptr0 = wavefun_.VecData(j,k);
			Scalar    *ptr1 = a3.VecData(j,k);
			Int natm = pseudo.size();
			for (Int iatm=0; iatm<natm; iatm++) {
				Int nobt = pseudo[iatm].vnlList.size();
				for (Int iobt=0; iobt<nobt; iobt++) {
					const SparseVec &vnlvec = pseudo[iatm].vnlList[iobt].first;
					const Real       vnlwgt = pseudo[iatm].vnlList[iobt].second;
					const IntNumVec &iv = vnlvec.first;
					const DblNumMat &dv = vnlvec.second;

					Scalar    weight = SCALAR_ZERO; 
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
Spinor::AddTeterPrecond ( Int iocc, Fourier* fftPtr, NumTns<Scalar>& a3)
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
  NumTns<Scalar>& a3i = wavefun_; 
  NumTns<Scalar>& a3o = a3;

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
Spinor::AddTeterPrecond (Fourier* fftPtr, NumTns<Scalar>& a3)
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
  NumTns<Scalar>& a3i = wavefun_; 
  NumTns<Scalar>& a3o = a3;

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








//int Spinor::add_nonlocalPS 
//(vector< vector< pair<SparseVec,double> > > &val, CpxNumTns &a3) {
//	int ntot = _wavefun._m;
//	int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//	double vol = _domain._Ls[0] * _domain._Ls[1] * _domain._Ls[2];
//
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j++) {
//			cpx    *ptr0 = _wavefun.clmdata(j,k);
//			cpx    *ptr1 = a3.clmdata(j,k);
//			int natm = val.size();
//			for (int iatm=0; iatm<natm; iatm++) {
//				int nobt = val[iatm].size();
//				for (int iobt=0; iobt<nobt; iobt++) {
//					SparseVec &vnlvec = val[iatm][iobt].first;
//					double vnlwgt = val[iatm][iobt].second;
//					IntNumVec &iv = vnlvec.first;
//					DblNumMat &dv = vnlvec.second;
//
//
//					cpx    weight = cpx(0.0, 0.0); 
//					int dvm = dv.m(); 
//					// dvm = 4 here to represent the function value, 
//					// and its derivatives along x,y,z directions.
//					int *ivptr = iv.Data();
//					double *dvptr = dv.Data();
//					for (int i=0; i<iv.m(); i++) {
//						weight += (*dvptr) * ptr0[*ivptr];
//						ivptr++; dvptr += dvm;
//					}
//					weight *= vol/double(ntot)*vnlwgt;
//
//					ivptr = iv.Data();
//					dvptr = dv.Data();
//					for (int i=0; i<iv.m(); i++) {
//						ptr1[*ivptr] += (*dvptr) * weight;
//						ivptr++; dvptr +=dvm;
//					}
//				}
//			}
//		}
//	}
//	return 0;
//};
//
//
//int Spinor::add_D2_c2c (CpxNumTns &a3, FFTPrepare &fp) {
//	// should be noted here D2 == \nabla^2/2.
//	// make sure the FFTW is ready
//	if (!fp._is_prepared) {
//		fp.setup_xyz(_domain._Ns[0], _domain._Ns[1], _domain._Ns[2], 
//				_domain._Ls[0], _domain._Ls[1], _domain._Ls[2]);
//	}
//
//	int ntot     = fp._size;
//	int nocc     = _wavefun._p;
//	int ncom     = _wavefun._n;
//
//	iA (ntot == _wavefun._m);
//
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j++) {
//			cpx    *ptr0 = _wavefun.clmdata(j,k);
//			fftw_execute_dft(fp._forward, reinterpret_cast<fftw_complex*>(ptr0), 
//					reinterpret_cast<fftw_complex*>(fp._out_cpx));
//
//			double *ptr1d = fp._gkk.Data();
//			ptr0 = fp._out_cpx;
//			for (int i=0; i<ntot; i++) *(ptr0++) *= *(ptr1d++);
//
//			//      ptr0 = fp._out_cpx;
//			//      fftw_execute_dft(fp._backward, reinterpret_cast<fftw_complex*>(ptr0), 
//			//	reinterpret_cast<fftw_complex*>(fp._in_cpx));
//
//			fftw_execute(fp._backward);
//			cpx   *ptr1 = a3.clmdata(j,k);
//			ptr0 = fp._in_cpx;
//			for (int i=0; i<ntot; i++) *(ptr1++) += *(ptr0++) / double(ntot);
//
//		}
//	}
//	return 0;
//};
//
//int Spinor::get_D2_c2c (CpxNumTns &a3, FFTPrepare &fp) {
//	// should be noted here D2 == \nabla^2/2.
//	// make sure the FFTW is ready
//	if (!fp._is_prepared) {
//		fp.setup_xyz(_domain._Ns[0], _domain._Ns[1], _domain._Ns[2], 
//				_domain._Ls[0], _domain._Ls[1], _domain._Ls[2]);
//	}
//
//	int ntot     = fp._size;
//	int nocc     = _wavefun._p;
//	int ncom     = _wavefun._n;
//
//	iA (ntot == _wavefun._m);
//
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j++) {
//			cpx    *ptr0 = _wavefun.clmdata(j,k);
//			fftw_execute_dft(fp._forward, reinterpret_cast<fftw_complex*>(ptr0), 
//					reinterpret_cast<fftw_complex*>(fp._out_cpx));
//
//			double *ptr1d = fp._gkk.Data();
//			ptr0 = fp._out_cpx;
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) *= *(ptr1d++) / ntot; //divide by ntot
//			}
//
//			ptr0 = a3.clmdata(j,k);
//			fftw_execute_dft(fp._backward, 
//					reinterpret_cast<fftw_complex*>(fp._out_cpx), 
//					reinterpret_cast<fftw_complex*>(ptr0));
//
//		}
//	}
//
//	//  int size = ntot * ncom * nocc;
//	//  cpx *ptr0 = a3.Data();
//	//  for (int i=0; i<size; i++) *(ptr0++) /= ntot;
//
//	return 0;
//};
//
//int Spinor::add_sigma_x (DblNumVec &a1, CpxNumTns &a3) {
//	int ntot = _wavefun._m;
//	int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j+=2) {
//			double tsig = - 1.0;
//			if (j<2) tsig = 1.0;
//
//			cpx    *ptr0 = a3.clmdata(j,k);
//			double *ptrd = a1.Data();
//			cpx    *ptr1 = _wavefun.clmdata(j+1,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += (*(ptr1++)) * (*(ptrd++)) * tsig;
//			}
//		}
//		for (int j=1; j<ncom; j+=2) {
//			double tsig = - 1.0;
//			if (j<2) tsig = 1.0;
//			cpx    *ptr0 = a3.clmdata(j,k);
//
//			double *ptrd = a1.Data();
//			cpx    *ptr1 = _wavefun.clmdata(j-1,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += (*(ptr1++)) * (*(ptrd++)) * tsig;
//			}
//		}
//	}
//	return 0;
//};
//
//int Spinor::add_sigma_y (DblNumVec &a1, CpxNumTns &a3) {
//	int ntot = _wavefun._m;
//	int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j+=2) {
//			double tsig = - 1.0;
//			if (j<2) tsig = 1.0;
//
//			cpx    *ptr0 = a3.clmdata(j,k);
//			double *ptrd = a1.Data();
//			cpx    *ptr1 = _wavefun.clmdata(j+1,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) -= cpx(0.0, 1.0) * (*(ptr1++)) * (*(ptrd++)) * tsig;
//			}
//		}
//		for (int j=1; j<ncom; j+=2) {
//			double tsig = - 1.0;
//			if (j<2) tsig = 1.0;
//
//			cpx    *ptr0 = a3.clmdata(j,k);
//			double *ptrd = a1.Data();
//			cpx    *ptr1 = _wavefun.clmdata(j-1,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += cpx(0.0, 1.0) * (*(ptr1++)) * (*(ptrd++)) * tsig;
//			}
//		}
//	}
//	return 0;
//};
//
//int Spinor::add_sigma_z (DblNumVec &a1, CpxNumTns &a3) {
//	int ntot = _wavefun._m;
//	int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j+=2) {
//			double tsig = - 1.0;
//			if (j<2) tsig = 1.0;
//
//			cpx    *ptr0 = a3.clmdata(j,k);
//			double *ptrd = a1.Data();
//			cpx    *ptr1 = _wavefun.clmdata(j,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += (*(ptr1++)) * (*(ptrd++)) * tsig;
//			}
//		}
//		for (int j=1; j<ncom; j+=2) {
//			double tsig = - 1.0;
//			if (j<2) tsig = 1.0;
//
//			cpx    *ptr0 = a3.clmdata(j,k);
//			double *ptrd = a1.Data();
//			cpx    *ptr1 = _wavefun.clmdata(j,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) -= (*(ptr1++)) * (*(ptrd++)) * tsig;
//			}
//		}
//	}
//	return 0;
//};
//
//int Spinor::add_nonlocalPS_SOC 
//(vector< vector< pair< SparseVec,double> > > &val, 
// vector<Atom> &atomvec, vector<DblNumVec> &grid, 
// CpxNumTns &a3, FFTPrepare &fp) {
//	int ntot = _wavefun._m;
//	int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//	double vol = _domain._Ls[0] * _domain._Ls[1] * _domain._Ls[2];
//
//	int nx = _domain._Ns(0);
//	int ny = _domain._Ns(1);
//	int nz = _domain._Ns(2);
//
//	CpxNumVec psix, psiy, psiz;
//	psix.resize(ntot);
//	psiy.resize(ntot);
//	psiz.resize(ntot);
//	cpx *ptr0, *ptr1, *ptr2;
//
//	CpxNumVec lx, ly, lz;
//	lx.resize(ntot); 
//	ly.resize(ntot); 
//	lz.resize(ntot); 
//	cpx *ptr_lx, *ptr_ly, *ptr_lz;
//
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j++) {
//
//			// get the moment
//			ptr0 = _wavefun.clmdata(j,k);
//			fftw_execute_dft(fp._forward, 
//					reinterpret_cast<fftw_complex*>(ptr0), 
//					reinterpret_cast<fftw_complex*>(fp._out_cpx));
//			// px
//			ptr0 = fp._out_cpx;
//			ptr1 = fp._ik.clmdata(0);
//			ptr2 = fp._in_cpx;
//			for (int i=0; i<ntot; i++) { // divide by ntot
//				*(ptr2++) = *(ptr0++) * (*(ptr1++)/double(ntot)) * cpx(0.0, -1.0);
//			}
//			ptr0 = psix.Data();
//			fftw_execute_dft(fp._backward, 
//					reinterpret_cast<fftw_complex*>(fp._in_cpx), 
//					reinterpret_cast<fftw_complex*>(ptr0));
//			// py
//			ptr0 = fp._out_cpx;
//			ptr1 = fp._ik.clmdata(1);
//			ptr2 = fp._in_cpx;
//			for (int i=0; i<ntot; i++) {
//				*(ptr2++) = *(ptr0++) * (*(ptr1++)/double(ntot)) * cpx(0.0, -1.0);
//			}
//			ptr0 = psiy.Data();
//			fftw_execute_dft(fp._backward, 
//					reinterpret_cast<fftw_complex*>(fp._in_cpx), 
//					reinterpret_cast<fftw_complex*>(ptr0));
//			// pz
//			ptr0 = fp._out_cpx;
//			ptr1 = fp._ik.clmdata(2);
//			ptr2 = fp._in_cpx;
//			for (int i=0; i<ntot; i++) {
//				*(ptr2++) = *(ptr0++) * (*(ptr1++)/double(ntot)) * cpx(0.0, -1.0);
//			}
//			ptr0 = psiz.Data();
//			fftw_execute_dft(fp._backward, 
//					reinterpret_cast<fftw_complex*>(fp._in_cpx), 
//					reinterpret_cast<fftw_complex*>(ptr0));
//			// end of geting the moment
//
//			ptr0 = psix.Data();
//			ptr1 = psiy.Data();
//			ptr2 = psiz.Data();
//
//			setvalue(lx, cpx(0.0, 0.0));
//			setvalue(ly, cpx(0.0, 0.0));
//			setvalue(lz, cpx(0.0, 0.0));
//			ptr_lx = lx.Data();
//			ptr_ly = ly.Data();
//			ptr_lz = lz.Data();
//
//			int natm = val.size();
//			for (int iatm=0; iatm<natm; iatm++) {
//				Point3 coord = atomvec[iatm].coord();
//				int nobt = val[iatm].size();
//				for (int iobt=0; iobt<nobt; iobt++) {
//					SparseVec &vnlvec = val[iatm][iobt].first;
//					double vnlwgt = val[iatm][iobt].second;
//					IntNumVec &iv = vnlvec.first;
//					DblNumMat &dv = vnlvec.second;
//					cpx weight0 = cpx(0.0, 0.0);
//					cpx weight1 = cpx(0.0, 0.0);
//					cpx weight2 = cpx(0.0, 0.0);
//					int dvm = dv.m(); 
//					int *ivptr = iv.Data();
//					double *dvptr = dv.Data();
//					//	  for (int i=0; i<iv.m(); i++) {
//					//	    weight0 += (*dvptr) * ptr0[*ivptr];
//					//	    weight1 += (*dvptr) * ptr1[*ivptr];
//					//	    weight2 += (*dvptr) * ptr2[*ivptr];
//					//	    ivptr++; dvptr += dvm;
//					//	  }
//					for (int i=0; i<iv.m(); i++) {
//						int itmp = *ivptr;
//						Index3 icoord;
//						icoord(2) = itmp/(nx*ny);
//						icoord(1) = (itmp - icoord(2)*nx*ny)/nx;
//						icoord(0) = (itmp - icoord(2)*nx*ny - icoord(1)*nx);
//						double dtmp = *dvptr;
//						double rel0 = grid[0](icoord(0))-coord(0);
//						double rel1 = grid[1](icoord(1))-coord(1);
//						double rel2 = grid[2](icoord(2))-coord(2);
//						// Shifting: VERY IMPORTANT
//						rel0 = rel0 - IRound(rel0 / _domain._Ls[0]) * _domain._Ls[0];
//						rel1 = rel1 - IRound(rel1 / _domain._Ls[1]) * _domain._Ls[1];
//						rel2 = rel2 - IRound(rel2 / _domain._Ls[2]) * _domain._Ls[2];
//
//						weight0 += dtmp * (rel1*ptr2[itmp]-rel2*ptr1[itmp]);
//						weight1 += dtmp * (rel2*ptr0[itmp]-rel0*ptr2[itmp]);
//						weight2 += dtmp * (rel0*ptr1[itmp]-rel1*ptr0[itmp]);
//						ivptr++; dvptr += dvm;
//					}
//					weight0 *= vol/double(ntot)*vnlwgt;
//					weight1 *= vol/double(ntot)*vnlwgt;
//					weight2 *= vol/double(ntot)*vnlwgt;
//
//					ivptr = iv.Data();
//					dvptr = dv.Data();
//					for (int i=0; i<iv.m(); i++) {
//						int itmp = *ivptr;
//						double dtmp = *dvptr;
//						ptr_lx[itmp] += dtmp * weight0;
//						ptr_ly[itmp] += dtmp * weight1;
//						ptr_lz[itmp] += dtmp * weight2;
//						ivptr++; dvptr +=dvm;
//					}
//				}
//			}
//
//			ptr_lx = lx.Data();
//			ptr_ly = ly.Data();
//			cpx sign0;
//			double sign1;
//			if (j%2 == 0) { 
//				ptr0 = a3.clmdata(j+1,k); 
//				sign0 = cpx(0.0, +1.0); 
//				sign1 = +1.0; 
//			}
//			else          { 
//				ptr0 = a3.clmdata(j-1,k); 
//				sign0 = cpx(0.0, -1.0); 
//				sign1 = -1.0; 
//			}
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += 0.5 * ( *(ptr_lx++) + sign0 * (*(ptr_ly++)) );
//			}
//
//			ptr_lz = lz.Data();
//			ptr0   = a3.clmdata(j,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += 0.5 * ( *(ptr_lz++) * sign1 );
//			}
//		}
//	}
//
//	return 0;
//};
//
//int Spinor::add_matrix_ij(int ir, int jc, DblNumVec &a1, CpxNumTns &a3) {
//
//	int ntot = _wavefun._m;
//	// int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//
//	for (int k=0; k<nocc; k++) {
//		cpx    *ptr0 = a3.clmdata(ir,k);
//		double *ptr1 = a1.Data();
//		cpx    *ptr2 = _wavefun.clmdata(jc,k);
//		for (int i=0; i<ntot; i++) {
//			*(ptr0++) += *(ptr1++) * (*(ptr2++));
//		}
//	}
//	return 0;
//};
//int Spinor::add_matrix_ij(int ir, int jc, double  *ptr1, CpxNumTns &a3) {
//
//	int ntot = _wavefun._m;
//	int nocc = _wavefun._p;
//
//	for (int k=0; k<nocc; k++) {
//		cpx    *ptr0 = a3.clmdata(ir,k);
//		// double *ptr1 = a1.Data();
//		cpx    *ptr2 = _wavefun.clmdata(jc,k);
//		for (int i=0; i<ntot; i++) {
//			*(ptr0++) += *(ptr1++) * (*(ptr2++));
//		}
//	}
//	return 0;
//};
//
//int Spinor::get_DKS (DblNumVec &vtot, DblNumMat &vxc,
//		vector< vector< pair<SparseVec,double> > > &vnlss,
//		vector< vector< pair<SparseVec,double> > > &vnlso,
//		vector<Atom> &atomvec, CpxNumTns &a3, FFTPrepare &fp,  
//		vector<DblNumVec> &grid) {
//
//	int ntot = _wavefun._m;
//	int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//	double vol = _domain._Ls[0] * _domain._Ls[1] * _domain._Ls[2];
//
//	int nx = _domain._Ns(0);
//	int ny = _domain._Ns(1);
//	int nz = _domain._Ns(2);
//
//	// get vtot only for four component
//	double energyShift = 2.0 * pow(SPEED_OF_LIGHT, 2.0);
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<2; j++) {
//			cpx    *p1 = _wavefun.clmdata(j,k);
//			cpx    *p3 =      a3.clmdata(j,k);
//			double *p2 =    vtot.Data();
//			for (int i=0; i<ntot; i++) { 
//				*(p3) = (*p1) * (*p2); 
//				p3++; p1++; p2++; 
//			}
//		}
//		for (int j=2; j<ncom; j++) {
//			cpx    *p1 = _wavefun.clmdata(j,k);
//			cpx    *p3 =      a3.clmdata(j,k);
//			double *p2 =    vtot.Data();
//			for (int i=0; i<ntot; i++) { 
//				*(p3) = (*p1) * (*p2 - energyShift); 
//				p3++; p1++; p2++; 
//			}
//		}
//	} // end of vtot multiplication
//
//	CpxNumVec psix, psiy, psiz;
//	psix.resize(ntot);
//	psiy.resize(ntot);
//	psiz.resize(ntot);
//	cpx *ptr0, *ptr1, *ptr2;
//
//	CpxNumVec lx, ly, lz;
//	lx.resize(ntot); 
//	ly.resize(ntot); 
//	lz.resize(ntot); 
//	cpx *ptr_lx, *ptr_ly, *ptr_lz;
//
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j++) {
//
//			// get the moment
//			ptr0 = _wavefun.clmdata(j,k);
//			fftw_execute_dft(fp._forward, 
//					reinterpret_cast<fftw_complex*>(ptr0), 
//					reinterpret_cast<fftw_complex*>(fp._out_cpx));
//			// px
//			ptr0 = fp._out_cpx;
//			ptr1 = fp._ik.clmdata(0);
//			ptr2 = fp._in_cpx;
//			for (int i=0; i<ntot; i++) { // divide by ntot
//				*(ptr2++) = *(ptr0++) * (*(ptr1++)/double(ntot)) * cpx(0.0, -1.0);
//			}
//			ptr0 = psix.Data();
//			fftw_execute_dft(fp._backward, 
//					reinterpret_cast<fftw_complex*>(fp._in_cpx), 
//					reinterpret_cast<fftw_complex*>(ptr0));
//			// py
//			ptr0 = fp._out_cpx;
//			ptr1 = fp._ik.clmdata(1);
//			ptr2 = fp._in_cpx;
//			for (int i=0; i<ntot; i++) {
//				*(ptr2++) = *(ptr0++) * (*(ptr1++)/double(ntot)) * cpx(0.0, -1.0);
//			}
//			ptr0 = psiy.Data();
//			fftw_execute_dft(fp._backward, 
//					reinterpret_cast<fftw_complex*>(fp._in_cpx), 
//					reinterpret_cast<fftw_complex*>(ptr0));
//			// pz
//			ptr0 = fp._out_cpx;
//			ptr1 = fp._ik.clmdata(2);
//			ptr2 = fp._in_cpx;
//			for (int i=0; i<ntot; i++) {
//				*(ptr2++) = *(ptr0++) * (*(ptr1++)/double(ntot)) * cpx(0.0, -1.0);
//			}
//			ptr0 = psiz.Data();
//			fftw_execute_dft(fp._backward, 
//					reinterpret_cast<fftw_complex*>(fp._in_cpx), 
//					reinterpret_cast<fftw_complex*>(ptr0));
//			// end of geting the moment
//
//			cpx sign0;
//			double sign1;
//			int ito;
//
//			if (j%2 == 0) {
//				sign0 = cpx(0.0, +1.0); 
//				sign1 = +1.0; 
//			}
//			else          { 
//				sign0 = cpx(0.0, -1.0); 
//				sign1 = -1.0; 
//			}
//
//			// start $c \vec{\sigma}\cdot\vec{p}$ 
//			ptr_lx = psix.Data();
//			ptr_ly = psiy.Data();
//			ito = ncom-1-j;
//			ptr0 = a3.clmdata(ito,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += SPEED_OF_LIGHT * ( *(ptr_lx++) + sign0 * (*(ptr_ly++)) );
//			}
//
//			ptr_lz = psiz.Data();
//			if (j<2) { ito = (j+2); } // %ncom for two component
//			else     { ito = (j-2); }
//			ptr0   = a3.clmdata(ito,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += SPEED_OF_LIGHT * ( *(ptr_lz++) * sign1 );
//			}// end of $c \vec{\sigma} \cdot \vec{p}$
//
//			// start PS-SOC
//			ptr0 = psix.Data();
//			ptr1 = psiy.Data();
//			ptr2 = psiz.Data();
//
//			setvalue(lx, cpx(0.0, 0.0));
//			setvalue(ly, cpx(0.0, 0.0));
//			setvalue(lz, cpx(0.0, 0.0));
//			ptr_lx = lx.Data();
//			ptr_ly = ly.Data();
//			ptr_lz = lz.Data();
//
//			// get the angular momentum $L_{x,y,z}$
//			int natm = vnlso.size();
//			for (int iatm=0; iatm<natm; iatm++) {
//				Point3 coord = atomvec[iatm].coord();
//				int nobt = vnlso[iatm].size();
//				for (int iobt=0; iobt<nobt; iobt++) {
//					SparseVec &vnlvec = vnlso[iatm][iobt].first;
//					double vnlwgt = vnlso[iatm][iobt].second;
//					IntNumVec &iv = vnlvec.first;
//					DblNumMat &dv = vnlvec.second;
//					cpx weight0 = cpx(0.0, 0.0);
//					cpx weight1 = cpx(0.0, 0.0);
//					cpx weight2 = cpx(0.0, 0.0);
//					int dvm = dv.m(); 
//					int *ivptr = iv.Data();
//					double *dvptr = dv.Data();
//					for (int i=0; i<iv.m(); i++) {
//						int itmp = *ivptr;
//						Index3 icoord;
//						icoord(2) = itmp/(nx*ny);
//						icoord(1) = (itmp - icoord(2)*nx*ny)/nx;
//						icoord(0) = (itmp - icoord(2)*nx*ny - icoord(1)*nx);
//						double dtmp = *dvptr;
//						double rel0 = grid[0](icoord(0))-coord(0);
//						double rel1 = grid[1](icoord(1))-coord(1);
//						double rel2 = grid[2](icoord(2))-coord(2);
//						// Shifting: VERY IMPORTANT
//						rel0 = rel0 - IRound(rel0 / _domain._Ls[0]) * _domain._Ls[0];
//						rel1 = rel1 - IRound(rel1 / _domain._Ls[1]) * _domain._Ls[1];
//						rel2 = rel2 - IRound(rel2 / _domain._Ls[2]) * _domain._Ls[2];
//
//						weight0 += dtmp * (rel1*ptr2[itmp]-rel2*ptr1[itmp]);
//						weight1 += dtmp * (rel2*ptr0[itmp]-rel0*ptr2[itmp]);
//						weight2 += dtmp * (rel0*ptr1[itmp]-rel1*ptr0[itmp]);
//						ivptr++; dvptr += dvm;
//					}
//					weight0 *= vol/double(ntot)*vnlwgt;
//					weight1 *= vol/double(ntot)*vnlwgt;
//					weight2 *= vol/double(ntot)*vnlwgt;
//
//					ivptr = iv.Data();
//					dvptr = dv.Data();
//					for (int i=0; i<iv.m(); i++) {
//						int itmp = *ivptr;
//						double dtmp = *dvptr;
//						ptr_lx[itmp] += dtmp * weight0;
//						ptr_ly[itmp] += dtmp * weight1;
//						ptr_lz[itmp] += dtmp * weight2;
//						ivptr++; dvptr +=dvm;
//					}
//				}
//			}// end of $L_{x,y,z}$
//
//			ptr_lx = lx.Data();
//			ptr_ly = ly.Data();
//			if (j<2) { ito = 1-j; } // %ncom for two component
//			else     { ito = 5-j; }
//			ptr0 = a3.clmdata(ito,k); 
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += 0.5 * ( *(ptr_lx++) + sign0 * (*(ptr_ly++)) );
//			}
//
//			ptr_lz = lz.Data();
//			ito = j;
//			ptr0   = a3.clmdata(ito,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += 0.5 * ( *(ptr_lz++) * sign1 );
//			}//end of the PS-SOC part
//		}
//	}
//
//	// add nonlocal PS
//	add_nonlocalPS(vnlss, a3);
//
//	// Magnetic part of Vxc
//	DblNumVec BxcX = DblNumVec(ntot, false, vxc.clmdata(MAGX));
//	DblNumVec BxcY = DblNumVec(ntot, false, vxc.clmdata(MAGY));
//	DblNumVec BxcZ = DblNumVec(ntot, false, vxc.clmdata(MAGZ));
//	add_sigma_x(BxcX, a3);
//	add_sigma_y(BxcY, a3);
//	add_sigma_z(BxcZ, a3);
//
//	return 0;
//};
}  // namespace dgdft
