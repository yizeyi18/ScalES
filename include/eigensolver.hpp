#ifndef _EIGENSOLVER_HPP_
#define _EIGENSOLVER_HPP_

#include  "environment_impl.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "hamiltonian.hpp"
#include  "spinor.hpp"
#include  "lobpcg++.hpp"
#include  "lapack.hpp"

namespace dgdft{

using namespace dgdft::LOBPCG;

class EigenSolver
{
private:

	Hamiltonian*        hamPtr_;
	Fourier*            fftPtr_;
	Spinor*             psiPtr_;

	Int                 maxIter_;
	Real                absTol_;
	Real                relTol_;

	DblNumVec           eigVal_;
	DblNumVec           resVal_;

public:

	// ********************  LIFECYCLE   *******************************

	EigenSolver ();
	EigenSolver( Hamiltonian& ham,
			Spinor& psi,
			Fourier& fft,
		  const Int maxIter,
			const Real absTol, 
			const Real relTol );

	~EigenSolver();

	// ********************  OPERATORS   *******************************
	static void LOBPCGHamiltonianMult(void *A, void *X, void *AX);
	static void LOBPCGPrecondMult    (void *A, void *X, void *AX);

	BlopexInt HamiltonianMult (serial_Multi_Vector *x, serial_Multi_Vector *y);
	BlopexInt PrecondMult     (serial_Multi_Vector *x, serial_Multi_Vector *y);

	// Specific for DiracKohnSham
//	static void lobpcg_apply_preconditioner_DKS  (void *A, void *X, void *AX);
//	BlopexInt apply_preconditioner_DKS  (serial_Multi_Vector *x, serial_Multi_Vector *y);
//	int solve_DKS(); 
//	int prune_spinor(CpxNumTns& X);


	// ********************  OPERATIONS  *******************************
	// Solve the eigenvalue problem using BLOPEX.
	void Solve();

	// ********************  ACCESS      *******************************
	DblNumVec& EigVal() { return eigVal_; }
	DblNumVec& ResVal() { return resVal_; }

	// ********************  INQUIRY     *******************************

}; // -----  end of class  EigenSolver  ----- 

} // namespace dgdft
#endif // _EIGENSOLVER_HPP_
