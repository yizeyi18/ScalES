#include	"eigensolver.hpp"

namespace dgdft{

EigenSolver::EigenSolver() {}

EigenSolver::~EigenSolver() {}

EigenSolver::EigenSolver( 
		const Hamiltonian& ham,
		const Spinor& psi,
		const Fourier& fft,
		const Int maxIter,
		const Real absTol, 
		const Real relTol) 
	: hamPtr_(&ham), psiPtr_(&psi), fftPtr_(&fft), 
	maxIter_(maxIter), absTol_(absTol), relTol_(relTol) 
	{
#ifndef _RELEASE_
		PushCallStack("EigenSolver::EigenSolver");
#endif  // ifndef _RELEASE_
		eigVal_.Resize(psiPtr->NumState());  SetValue(eigVal_, 0.0);
		resVal_.Resize(psiPtr->NumState());  SetValue(resVal_, 0.0);
#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_
	} 		// -----  end of method EigenSolver::EigenSolver  ----- 

BlopexInt EigenSolver::HamiltonianMult
(serial_Multi_Vector *x, serial_Multi_Vector *y) {
#ifndef _RELEASE_
	PushCallStack("EigenSolver::HamiltonianMult");
#endif
  Int ntot = psiPtr_->NumGridTotal();
  Int ncom = psiPtr_->NumComponent();
  Int nocc = psiptr_->NumState();

	if( (x->size * x->num_vectors) != ntot*ncom*nocc ) {
		throw std::logic_error("Vector size does not match.");
	}

  Spinor psitemp(fftPtr_->domain, ncom, nocc, false, x->data);
  NumTns<Scalar> a3(ntot, ncom, nocc, false, y->data);
  hamptr_->MultSpinor(psitemp, a3, *fftptr_);
#ifndef _RELEASE_
	PopCallStack();
#endif  
  return 0;
}

BlopexInt EigenSolver::PrecondMult
(serial_Multi_Vector *x, serial_Multi_Vector *y) {
#ifndef _RELEASE_
	PushCallStack("EigenSolver::PrecondMult");
#endif
	if( !fftPtr->isPrepared ){
		throw std::runtime_error("Fourier is not prepared.");
	}
  Int ntot = psiPtr_->NumGridTotal();
  Int ncom = psiPtr_->NumComponent();
  Int nocc = psiptr_->NumState();
	
	if( fftPtr->domain.NumGridTotal() != ntot ){
		throw std::logic_error("Domain size does not match.");
	}

  NumTns<Scalar> a3i(ntot, ncom, nocc, false, x->data);
  NumTns<Scalar> a3o(ntot, ncom, nocc, false, y->data);

#ifndef _USE_COMPLEX_ // Real case
	CpxNumVec   cpxtmp(ntot);
  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
			Real   *ptra3i = a3i.MatData(j,k);
			for(Int i = 0; i < ntot; i++){
				cpxtmp[i] = Complex(ptra3i[i], 0.0);
			}
      Complex *ptr0 = cpxtmp.Data();

			fftw_execute_dft(fftPtr->forwardPlan, reinterpret_cast<fftw_complex*>(ptr0), 
					reinterpret_cast<fftw_complex*>(fftPtr->outputComplexVec));
			Real *ptr1d = fftPtr_->TeterPrecond.Data();
			ptr0 = fftPtr_->outputComplexVec;
			for (Int i=0; i<ntot; i++) 
				*(ptr0++) *= *(ptr1d++);
      
			fftw_execute(fftPtr->backwardPlan);
			ptr0 = fftPtr->inputComplexVec;

			Real *ptra3o = a3o.VecData(j, k); 
			for (Int i=0; i<ntot; i++) 
				*(ptra3o++) = (*(ptr0++)).real() / Real(ntot);
    }
  }
#else // Complex case
  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
			Complex *ptr0  = a3i.MatData(j,k);
			
			fftw_execute_dft(fftPtr->forwardPlan, reinterpret_cast<fftw_complex*>(ptr0), 
					reinterpret_cast<fftw_complex*>(fftPtr->outputComplexVec));
			Real *ptr1d = fftPtr_->TeterPrecond.Data();
			ptr0 = fftPtr_->outputComplexVec;
			for (Int i=0; i<ntot; i++) 
				*(ptr0++) *= *(ptr1d++);
      
			fftw_execute(fftPtr->backwardPlan);
			ptr0 = fftPtr->inputComplexVec;

			Complex *ptra3o = a3o.VecData(j, k); 
			for (Int i=0; i<ntot; i++) 
				*(ptra3o++) = *(ptr0++) / Real(ntot);
    }
  }
#endif

#ifndef _RELEASE_
	PopCallStack();
#endif  
  return 0;
}

void EigenSolver::LOBPCGHamiltonianMult(void *A, void *X, void *AX) {
  iC(((EigenSolver*)A)->HamiltonianMult((serial_Multi_Vector*)X,
      (serial_Multi_Vector*)AX));
  return;
};

void EigenSolver::LOBPCGPrecondMult(void *A, void *X, void *AX) {
  iC(((EigenSolver*)A)->PrecondMult((serial_Multi_Vector*)X,
      (serial_Multi_Vector*)AX));
  return;
};

void
EigenSolver::Solve	()
{
#ifndef _RELEASE_
	PushCallStack("EigenSolver::Solve");
#endif 
  
  serial_Multi_Vector *x;
  x = (serial_Multi_Vector*)malloc(sizeof(serial_Multi_Vector));

  x->data = psiPtr_->Wavefun().Data();
  x->owns_data = 0;
  x->size = psiPtr_->NumGridTotal() * psiPtr_->NumComponent();
  x->num_vectors = psiPtr_->NumState();
  x->num_active_vectors = x->num_vectors;
  x->active_indices = (BlopexInt*)malloc(sizeof(BlopexInt)*x->num_active_vectors);
  for (Int i=0; i<x->num_active_vectors; i++) x->active_indices[i] = i;

  mv_MultiVectorPtr xx;
  mv_InterfaceInterpreter ii;
  lobpcg_Tolerance lobpcg_tol;
  lobpcg_BLASLAPACKFunctions blap_fn;

  lobpcg_tol.absolute = absTol_;
  lobpcg_tol.relative = relTol_;

  SerialSetupInterpreter ( &ii );
  xx = mv_MultiVectorWrap( &ii, x, 0);

  Int iterations;
	Real  timeSolveStart, timeSolveEnd;
	
	timeSolveStart = MPI_Wtime();

#ifndef _USE_COMPLEX_ // Real case
  blap_fn.dpotrf = dpotrf_;
  blap_fn.dsygv  = dsygv_;
	std::cout<<"Call lobpcg_double"<<std::endl;
	lobpcg_solve_double ( 
			xx,
			this,
			LOBPCGHamiltonianMult,
			NULL,
			NULL,
			this,
			LOBPCGPrecondMult,
			NULL,
			blap_fn,
			lobpcg_tol,
			maxIter_,
			0,
			&iterations,
			eigVal_.data(),
			NULL,
			0,
			resVal_.data(),
			NULL,
			0);
#else // Complex case
  blap_fn.zpotrf = zpotrf_;
  blap_fn.zhegv  = zhegv_;
	std::cout<<"Call lobpcg_complex"<<std::endl;
  CpxNumVec cpxEigVal;
  cpxEigVal.Resize(eigVal_.m());  
  SetValue(cpxEigVal, Z_ZERO);
	lobpcg_solve_complex( 
			xx,
			this,
			LOBPCGHamiltonianMult,
			NULL,
			NULL,
			this,
			LOBPCGPrecondMult,
			NULL,
			blap_fn,
			lobpcg_tol,
			maxIter_,
			0,
			&iterations,
			(komplex*)(cpxEigVal.Data()),
			NULL,
			0,
			resVal_.Data(),
			NULL,
			0);
  for(Int i = 0; i < eigVal_.m(); i++)
    eigVal_(i) = cpxEigVal(i).real();
#endif
  
	timeSolveEnd   = MPI_Wtime();

	std::cout << "Solution time: " << timeSolveEnd - timeSolveStart
		<< std::endl;

  serial_Multi_VectorDestroy(x);
  mv_MultiVectorDestroy(xx);

#ifndef _RELEASE_
	PopCallStack();
#endif  

	return ;
} 		// -----  end of method EigenSolver::Solve  ----- 



void
EigenSolver::PostProcessing	(  )
{
#ifndef _RELEASE_
	PushCallStack("EigenSolver::PostProcessing");
#endif  // ifndef _RELEASE_
	
	//TODO get the sorted eigenvalues into Hamiltonian
	//TODO get the eigenfunctions into spinor
	

#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_

	return ;
} 		// -----  end of method EigenSolver::PostProcessing  ----- 


} // namespace dgdft
