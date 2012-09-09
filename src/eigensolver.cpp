#include	"eigensolver.hpp"

namespace dgdft{

EigenSolver::EigenSolver(): isPrepared_(NULL){}

EigenSolver::EigenSolver( 
		const Hamiltonian& ham,
		const Spinor& psi,
		const Fourier& fft )
{
#ifndef _RELEASE_
	PushCallStack("EigenSolver::EigenSolver");
#endif  // ifndef _RELEASE_
	SetEigenSolverData( ham, psi, fft );
#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_
} 		// -----  end of method EigenSolver::EigenSolver  ----- 


EigenSolver::~EigenSolver	(  )
{
#ifndef _RELEASE_
	PushCallStack("EigenSolver::~EigenSolver");
#endif  // ifndef _RELEASE_
	if( isPrepared_ ){
		EPSDestroy( &eps_ );
		MatDestroy( &hamiltonianShell_ );
    MatDestroy( &precondShell_ );
	}
	// TODO Destroy packedWavefunPtr in future
	
#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_
} 		// -----  end of method EigenSolver::~EigenSolver  ----- 

void
EigenSolver::Setup	(  )
{
#ifndef _RELEASE_
	PushCallStack("EigenSolver::Setup");
#endif  // ifndef _RELEASE_
	PetscErrorCode   ierr;

	if( solverData_.hamPtr == NULL || 
			solverData_.fftPtr == NULL || 
			solverData_.psiPtr == NULL ){
		throw std::logic_error( "EigenSolverData has not been prepared!" );
	}
	//FIXME TODO decide nev, maxit, tolerance
	Int nev = 10;
	Real tol = 1e-3, maxits = 3;

	// FIXME For the time being, use the domain provided by fftPtr;

	Domain dm = solverData_.fftPtr->domain;

	// Setup the packedWavefunPtr that wraps up all the components.  For
	// the moment. Just set them to be the zeroth component.
	
	// TODO: Change the dimension of the eigensolver to be gridsize *
	// component for hamiltonian matrix and preconditioner
	
	// TODO Add allocation/ deallocation for the new packedWavefunPtr, the
	// data comes from localWavefun in Spinor.
	
	// FIXME This is temporary
	packedWavefunPtr_.resize( nev );
	for( Int i = 0; i < nev; i++ ){
		packedWavefunPtr_[i] = &( (solverData_.psiPtr)->Wavefun(0, i) );
	}


	// Setup for operators
	ierr = MatCreateShell(
			dm.comm,
			solverData_.fftPtr->numGridLocal,
			solverData_.fftPtr->numGridLocal,
			dm.NumGridTotal(),
			dm.NumGridTotal(),
			(void*) (&solverData_), 
			&hamiltonianShell_ ); if( ierr ) throw ierr;

	ierr = MatShellSetOperation( hamiltonianShell_, 
			MATOP_MULT, (void(*)()) ApplyHamiltonian ); if( ierr ) throw ierr;

	ierr = MatCreateShell(
			dm.comm,
			solverData_.fftPtr->numGridLocal,
			solverData_.fftPtr->numGridLocal,
			dm.NumGridTotal(),
			dm.NumGridTotal(),
			(void*) (&solverData_), 
			&precondShell_ ); if( ierr ) throw ierr;
	
	ierr = MatShellSetOperation( precondShell_, 
			MATOP_MULT, (void(*)()) ApplyPrecond ); if( ierr ) throw ierr;


	// Register for the modified blopex solver
	EPSRegister( "modified_blopex", 0, "EPSCreate_MODIFIED_BLOPEX", 
			EPSCreate_MODIFIED_BLOPEX );


  ierr = EPSCreate( dm.comm, &eps_ );  if( ierr ) throw ierr;
  ierr = EPSSetOperators( eps_, hamiltonianShell_, PETSC_NULL ); if( ierr ) throw ierr;
	ierr = EPSSetProblemType( eps_, EPS_HEP ); if( ierr ) throw ierr;
	ierr = EPSSetWhichEigenpairs( eps_,EPS_SMALLEST_REAL ); if( ierr ) throw ierr; 
	ierr = EPSSetType( eps_, EPSMODIFIEDBLOPEX ); if( ierr ) throw ierr;
	//FIXME 
	ierr = EPSSetDimensions( eps_, nev, nev, PETSC_DECIDE ); if( ierr ) throw ierr;
	ierr = EPSSetTolerances( eps_, tol, maxits );

	eigVal_.Resize( nev );
	resVal_.Resize( nev );

  ierr = EPSGetST( eps_, &st_ );  if( ierr ) throw ierr;
	ierr = STPrecondSetMatForPC( st_, precondShell_ ); if( ierr ) throw ierr;

	// Finalize
	isPrepared_ = true;

#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_

	return ;
} 		// -----  end of method EigenSolver::Setup  ----- 


void
EigenSolver::Solve	( const bool isInitialDataUsed  )
{
#ifndef _RELEASE_
	PushCallStack("EigenSolver::Solve");
#endif  // ifndef _RELEASE_
	PetscErrorCode   ierr;
	Real  timeSolveStart, timeSolveEnd;

	if( isPrepared_ == false ){
		throw std::logic_error( "EigenSolver has not been prepared.  Setup the eigensolver first." );
	}


	// TODO: Maybe change Wavefun in Spinor from std::vector<Vec> to
	// std::vector<Vec*>.  The management should be similar.
	
	// FIXME: nev
	Int  nev = 10;

  if( isInitialDataUsed )
		ierr = EPSSetInitialSpace( eps_, nev, packedWavefunPtr_[0] );
	
	
	timeSolveStart = MPI_Wtime();
	ierr = EPSSolve( eps_ ); if( ierr ) throw ierr;
	timeSolveEnd   = MPI_Wtime();

	Int mpirank;
 	MPI_Comm_size( ((solverData_.fftPtr)->domain).comm, &mpirank );
	if( mpirank == 0 ){
		std::cout << "Solution time: " << timeSolveEnd - timeSolveStart
			<< std::endl;
	}

	// Get the invariant subspace into spinor, without evaluating the
	// eigenvalues and eigenfunctions
	ierr = EPSGetInvariantSubspace( eps_, packedWavefunPtr_[0] );

#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_

	return ;
} 		// -----  end of method EigenSolver::Solve  ----- 



void
EigenSolver::PostProcessing	(  )
{
#ifndef _RELEASE_
	PushCallStack("EigenSolver::PostProcessing");
#endif  // ifndef _RELEASE_
	PetscErrorCode   ierr;
	
	//TODO get the sorted eigenvalues into Hamiltonian
	//TODO get the eigenfunctions into spinor
	
	ierr = EPSPrintSolution(eps_, PETSC_NULL); if( ierr ) throw ierr;

#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_

	return ;
} 		// -----  end of method EigenSolver::PostProcessing  ----- 


void
EigenSolver::SetEigenSolverData	(
		const Hamiltonian& ham,
		const Spinor& psi,
		const Fourier& fft )
{
#ifndef _RELEASE_
	PushCallStack("EigenSolver::SetEigenSolverData");
#endif  // ifndef _RELEASE_
	solverData_.hamPtr  = const_cast<Hamiltonian*>( &ham );
	solverData_.psiPtr  = const_cast<Spinor*>( &psi );
	solverData_.fftPtr  = const_cast<Fourier*>( &fft );
	
	// TODO: Assert that ham, psi, fft share the same DOMAIN
#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_

	return ;
} 		// -----  end of method EigenSolver::SetEigenSolverData  ----- 

PetscErrorCode
EigenSolver::ApplyHamiltonian	( Mat H, Vec x, Vec y )
{
#ifndef _RELEASE_
	PushCallStack("EigenSolver::ApplyHamiltonian");
#endif  // ifndef _RELEASE_
  PetscErrorCode     ierr;
	Int                numGridTotal, 
						         numGridLocal;
	Scalar*            xArray;
	Scalar*            yArray;       // Local arrays for x and y vectors
	Int                localSize;    // Local sizes of x and y vectors
	Int                ownLB, ownUB; // Ownership range of the x and y vectors
	
	EigenSolverData*   solverDataPtr;
	Fourier*           fftPtr;

  ierr = MatShellGetContext(H,(void**)(&solverDataPtr)); if( ierr ) throw ierr;
	fftPtr = solverDataPtr -> fftPtr;

	// TODO: Hamiltonian apply
	
	// TODO: Different component number

	numGridTotal = fftPtr->domain.NumGridTotal();
	numGridLocal = fftPtr->numGridLocal;
	
	// Local size
	{
		Int xLocalSize, yLocalSize; 
		ierr = VecGetLocalSize( x, &xLocalSize ); if( ierr ) throw ierr;
		ierr = VecGetLocalSize( y, &yLocalSize ); if( ierr ) throw ierr;
		if( xLocalSize != yLocalSize ||
				xLocalSize != numGridLocal ||
				yLocalSize != numGridLocal ){
			std::ostringstream msg;
			msg 
				<< "Local sizes do not match." << std::endl
				<< " x is of size " << xLocalSize << std::endl
				<< " y is of size " << yLocalSize << std::endl;
			
			throw std::logic_error( msg.str().c_str() );
		}
		localSize = xLocalSize;
	}
	

	// ownership range
	{
		Int xOwnLB, xOwnUB, yOwnLB, yOwnUB;
		ierr = VecGetOwnershipRange( x, &xOwnLB, &xOwnUB ); if( ierr ) throw ierr;
		ierr = VecGetOwnershipRange( y, &yOwnLB, &yOwnUB ); if( ierr ) throw ierr;
		if( xOwnLB != yOwnLB ||
				xOwnUB != yOwnUB ){
			std::ostringstream msg;
			msg 
				<< "Ownership ranges do not match." << std::endl
				<< " x ~ [ " << xOwnLB << " , " << xOwnUB << " ) " << std::endl
				<< " y ~ [ " << xOwnLB << " , " << yOwnUB << " ) " << std::endl;
			throw std::logic_error( msg.str().c_str() );
		}
		ownLB = xOwnLB;
		ownUB = yOwnUB;
	}


	// Get arrays

	{
		VecGetArray( x, reinterpret_cast<PetscScalar**>(&xArray));
		VecGetArray( y, reinterpret_cast<PetscScalar**>(&yArray));
	}

	fftw_mpi_execute_dft( fftPtr->forwardPlan, 
			reinterpret_cast<fftw_complex*>( xArray ),  
			reinterpret_cast<fftw_complex*>( fftPtr->outputComplexVec.Data() ) );

	// gkk is owned globally
	// TODO own gkk locally, as should be done for density
	for( Int i = ownLB; i < ownUB; i++ ){
    fftPtr->outputComplexVec[ i - ownLB ] 
			*= fftPtr->gkk( i );
	}
	
	fftw_mpi_execute_dft( fftPtr->backwardPlan, 
			reinterpret_cast<fftw_complex*>( fftPtr->outputComplexVec.Data() ),  
			reinterpret_cast<fftw_complex*>( yArray ) );

  VecRestoreArray( x, reinterpret_cast<PetscScalar**>(&xArray));
	VecRestoreArray( y, reinterpret_cast<PetscScalar**>(&yArray));
  ierr = VecScale(y, 1.0 / numGridTotal);	 if( ierr ) throw ierr;

	// FIXME Add the identity operator
	ierr = VecAXPY(y, 1.0, x); if( ierr ) throw ierr;

#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_
  PetscFunctionReturn(0);
} 		// -----  end of method EigenSolver::ApplyHamiltonian  ----- 

PetscErrorCode
EigenSolver::ApplyPrecond ( Mat P, Vec x, Vec y )
{
#ifndef _RELEASE_
	PushCallStack("EigenSolver::ApplyPrecond");
#endif  // ifndef _RELEASE_
  PetscErrorCode     ierr;
	Int                numGridTotal, 
						         numGridLocal;
	Scalar*            xArray;
	Scalar*            yArray;       // Local arrays for x and y vectors
	Int                localSize;    // Local sizes of x and y vectors
	Int                ownLB, ownUB; // Ownership range of the x and y vectors
	
	EigenSolverData*   solverDataPtr;
	Fourier*           fftPtr;

  ierr = MatShellGetContext(P,(void**)(&solverDataPtr)); if( ierr ) throw ierr;
	fftPtr = solverDataPtr -> fftPtr;

	
	// TODO: Different component number

	numGridTotal = fftPtr->domain.NumGridTotal();
	numGridLocal = fftPtr->numGridLocal;
	
	// Local size
	{
		Int xLocalSize, yLocalSize; 
		ierr = VecGetLocalSize( x, &xLocalSize ); if( ierr ) throw ierr;
		ierr = VecGetLocalSize( y, &yLocalSize ); if( ierr ) throw ierr;
		if( xLocalSize != yLocalSize ||
				xLocalSize != numGridLocal ||
				yLocalSize != numGridLocal ){
			std::ostringstream msg;
			msg 
				<< "Local sizes do not match." << std::endl
				<< " x is of size " << xLocalSize << std::endl
				<< " y is of size " << yLocalSize << std::endl;
			
			throw std::logic_error( msg.str().c_str() );
		}
		localSize = xLocalSize;
	}
	

	// ownership range
	{
		Int xOwnLB, xOwnUB, yOwnLB, yOwnUB;
		ierr = VecGetOwnershipRange( x, &xOwnLB, &xOwnUB ); if( ierr ) throw ierr;
		ierr = VecGetOwnershipRange( y, &yOwnLB, &yOwnUB ); if( ierr ) throw ierr;
		if( xOwnLB != yOwnLB ||
				xOwnUB != yOwnUB ){
			std::ostringstream msg;
			msg 
				<< "Ownership ranges do not match." << std::endl
				<< " x ~ [ " << xOwnLB << " , " << xOwnUB << " ) " << std::endl
				<< " y ~ [ " << xOwnLB << " , " << yOwnUB << " ) " << std::endl;
			throw std::logic_error( msg.str().c_str() );
		}
		ownLB = xOwnLB;
		ownUB = yOwnUB;
	}


	// Get arrays

	VecGetArray( x, reinterpret_cast<PetscScalar**>(&xArray) );
	VecGetArray( y, reinterpret_cast<PetscScalar**>(&yArray) );

	fftw_mpi_execute_dft( fftPtr->forwardPlan, 
		reinterpret_cast<fftw_complex*>( xArray ),  
			reinterpret_cast<fftw_complex*>( fftPtr->outputComplexVec.Data() ) );

	// TeterPrecond is owned globally
	// TODO own TeterPrecond locally, as should be done for density
	for( Int i = ownLB; i < ownUB; i++ ){
    fftPtr->outputComplexVec[ i - ownLB ] 
			*= fftPtr->TeterPrecond( i );
	}
	
	fftw_mpi_execute_dft( fftPtr->backwardPlan, 
			reinterpret_cast<fftw_complex*>( fftPtr->outputComplexVec.Data() ),  
			reinterpret_cast<fftw_complex*>( yArray ) );

  VecRestoreArray( x, reinterpret_cast<PetscScalar**>(&xArray));
	VecRestoreArray( y, reinterpret_cast<PetscScalar**>(&yArray));
  ierr = VecScale(y, 1.0 / numGridTotal);	 if( ierr ) throw ierr;

#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_
  PetscFunctionReturn(0);
} 		// -----  end of method EigenSolver::ApplyPrecond  ----- 


} // namespace dgdft
