#ifndef _EIGENSOLVER_HPP_
#define _EIGENSOLVER_HPP_

#include  "environment_impl.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "hamiltonian.hpp"
#include  "spinor.hpp"


namespace dgdft{

#ifndef EPSMODIFIEDBLOPEX
#define EPSMODIFIEDBLOPEX      "modified_blopex"
#endif

extern "C"{
extern PetscErrorCode EPSCreate_MODIFIED_BLOPEX(EPS eps);
}


class EigenSolver
{
private:
	struct EigenSolverData{
		Hamiltonian*      hamPtr;
		Fourier*          fftPtr;
		Spinor*           psiPtr;

		EigenSolverData(): hamPtr(NULL), fftPtr(NULL), psiPtr(NULL){}
		~EigenSolverData() {}
	};


	EigenSolverData     solverData_;
	DblNumVec           eigVal_;
	DblNumVec           resVal_;
	bool                isPrepared_;
	std::vector<Vec*>   packedWavefunPtr_;            // Spinor vectors with all components packed together

	// PETSc/SLEPc related
	Mat               hamiltonianShell_;
	Mat               precondShell_;
	EPS               eps_;
	ST                st_;

	static PetscErrorCode    ApplyHamiltonian (Mat, Vec, Vec);
	static PetscErrorCode    ApplyPrecond     (Mat, Vec, Vec);

public:

	// ********************  LIFECYCLE   *******************************

	EigenSolver ();
	EigenSolver( const Hamiltonian& ham,
			const Spinor& psi,
			const Fourier& fft );

	~EigenSolver();

	// ********************  OPERATORS   *******************************

	// ********************  OPERATIONS  *******************************
	// Setup the SLEPc eigensolver
	void Setup();                                 

	// Solve the eigenvalue problem using SLEPc.
	void Solve( const bool isInitialDataUsed = false );

	// Obtaining eigenvalues, eigenvectors etc.
	void PostProcessing();

	// ********************  ACCESS      *******************************
	void SetEigenSolverData( 
			const Hamiltonian& ham,
			const Spinor& psi,
			const Fourier& fft );

	// TODO: Set type, tolerance, maxit

	// ********************  INQUIRY     *******************************

	// TODO: Get EPS context

}; // -----  end of class  EigenSolver  ----- 

} // namespace dgdft
#endif

