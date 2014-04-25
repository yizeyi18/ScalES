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
/// @file eigensolver.hpp
/// @brief Eigensolver in the global domain or extended element.
/// @date 2012-11-20 Original version
/// @date 2014-04-25 Parallel eigensolver.
#ifndef _EIGENSOLVER_HPP_
#define _EIGENSOLVER_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "hamiltonian.hpp"
#include  "spinor.hpp"
#include  "lobpcg++.hpp"
#include  "plobpcg++.hpp"
#include  "esdf.hpp"

namespace dgdft{

using namespace dgdft::LOBPCG;


class EigenSolver
{
protected:

	Hamiltonian*        hamPtr_;
	Fourier*            fftPtr_;
	Spinor*             psiPtr_;

	Int                 eigMaxIter_;
	Real                eigTolerance_;

	DblNumVec           eigVal_;
	DblNumVec           resVal_;

  Index3  numGridWavefunctionElem_;
  Index3  numGridDensityElem_;


public:

	// ********************  LIFECYCLE   *******************************

	EigenSolver () {};

	virtual ~EigenSolver() {};

	// ********************  OPERATORS   *******************************

	virtual void Setup(
			const esdf::ESDFInputParam& esdfParam,
			Hamiltonian& ham,
			Spinor& psi,
			Fourier& fft ) {};

	virtual BlopexInt HamiltonianMult (serial_Multi_Vector *x, serial_Multi_Vector *y) {};
	virtual BlopexInt PrecondMult     (serial_Multi_Vector *x, serial_Multi_Vector *y) {};

	// Specific for DiracKohnSham
//	static void lobpcg_apply_preconditioner_DKS  (void *A, void *X, void *AX);
//	BlopexInt apply_preconditioner_DKS  (serial_Multi_Vector *x, serial_Multi_Vector *y);
//	int solve_DKS(); 
//	int prune_spinor(CpxNumTns& X);


	// ********************  OPERATIONS  *******************************
	// Solve the eigenvalue problem using BLOPEX.
	virtual void Solve() {};

	// ********************  ACCESS      *******************************
	DblNumVec& EigVal() { return eigVal_; }
	DblNumVec& ResVal() { return resVal_; }

	Hamiltonian& Ham()  {return *hamPtr_;}
	Spinor&      Psi()  {return *psiPtr_;}
	Fourier&     FFT()  {return *fftPtr_;}

	// ********************  INQUIRY     *******************************

}; // -----  end of class  EigenSolver  ----- 


// *********************************************************************
// Sequential eigensolver
// *********************************************************************

class SEigenSolver: public EigenSolver
{
public:

	// ********************  LIFECYCLE   *******************************

	SEigenSolver () {};

	~SEigenSolver() {};

	// ********************  OPERATORS   *******************************

	virtual void Setup(
			const esdf::ESDFInputParam& esdfParam,
			Hamiltonian& ham,
			Spinor& psi,
			Fourier& fft );

	static void LOBPCGHamiltonianMult(void *A, void *X, void *AX);
	static void LOBPCGPrecondMult    (void *A, void *X, void *AX);

	virtual BlopexInt HamiltonianMult (serial_Multi_Vector *x, serial_Multi_Vector *y);
	virtual BlopexInt PrecondMult     (serial_Multi_Vector *x, serial_Multi_Vector *y);

	// ********************  OPERATIONS  *******************************
	// Solve the eigenvalue problem using BLOPEX.
	virtual void Solve();


}; // -----  end of class  SEigenSolver  ----- 



// *********************************************************************
// Parallel eigensolver
// *********************************************************************

class PEigenSolver: public EigenSolver
{
public:

	// ********************  LIFECYCLE   *******************************

	PEigenSolver ();

	~PEigenSolver();

	// ********************  OPERATORS   *******************************

	virtual void Setup(
			const esdf::ESDFInputParam& esdfParam,
			Hamiltonian& ham,
			Spinor& psi,
			Fourier& fft );

	static void LOBPCGHamiltonianMult(void *A, void *X, void *AX);
	static void LOBPCGPrecondMult    (void *A, void *X, void *AX);

	virtual BlopexInt HamiltonianMult (parallel_Multi_Vector *x, parallel_Multi_Vector *y);
	virtual BlopexInt PrecondMult     (parallel_Multi_Vector *x, parallel_Multi_Vector *y);

	// ********************  OPERATIONS  *******************************
	// Solve the eigenvalue problem using BLOPEX.
	virtual void Solve();


}; // -----  end of class  PEigenSolver  ----- 


} // namespace dgdft
#endif // _EIGENSOLVER_HPP_
