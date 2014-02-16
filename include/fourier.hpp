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
/// @file fourier.hpp
/// @brief Sequential and Distributed Fourier wrapper.
/// @date 2011-11-01
#ifndef _FOURIER_HPP_
#define _FOURIER_HPP_

#include  "environment.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"

namespace dgdft{

// *********************************************************************
// Sequential FFTW interface
// *********************************************************************

/// @struct Fourier
/// @brief Sequential FFTW interface.
struct Fourier {
  /// @brief Domain for the Fourier transform.
	Domain           domain;
  /// @brief Whether the Fourier structure is initialized.
  bool             isInitialized;
  /// @brief The total number of grid points in the domain.
	Int              numGridTotal;

  // plans
  /// @brief Plan for backward Fourier transform
  fftw_plan backwardPlan;
  /// @brief Plan for forward Fourier transform
  fftw_plan forwardPlan;

  /// @brief Mode for executing FFTW.
  unsigned  plannerFlag;

	// Laplacian operator related
  /// @brief 1/2 k^2 in the 3D domain, for the Laplacian operator
  /// \f$-\frac12 \Delta\f$.
  DblNumVec                gkk;
  /// @brief ikx, iky, ikz in the 3D domain, for the partial
  /// differential operator \f$\partial_x, \partial_y, \partial_z\f$.
	std::vector<CpxNumVec>   ik;
  /// @brief T(k) for Teter's preconditioner for eigenvalue computation.
  DblNumVec                TeterPrecond;

	// Temporary vectors that can also be used globally
  /// @brief Temporary input vector for FFTW.
	CpxNumVec                inputComplexVec;     
  /// @brief Temporary output vector for FFTW.
	CpxNumVec                outputComplexVec;     

	// Real data Fourier transform
	Int       numGridTotalR2C;
  fftw_plan backwardPlanR2C;
  fftw_plan forwardPlanR2C;


  DblNumVec                gkkR2C;
	std::vector<CpxNumVec>   ikR2C;
  DblNumVec                TeterPrecondR2C;

	// Temporary vectors that can also be used globally
	DblNumVec                inputVecR2C;     
	CpxNumVec                outputVecR2C;     


	Fourier();
	~Fourier();

	void Initialize( const Domain& dm );
};


// *********************************************************************
// Parallel FFTW interface
// *********************************************************************

/// @struct DistFourier
/// @brief Distributed memory (MPI only) parallel FFTW interface.
struct DistFourier {
	Domain           domain;                      
  bool             isInitialized;
	Int              numGridTotal;
  Int              numGridLocal;	
	ptrdiff_t        localNz;
	ptrdiff_t        localNzStart;
	// numAllocLocal is the size for the FFTW vectors.
	// numAllocLocal is close but may not be exactly the same as numGridLocal, which is
	// localNz * Ny * Nx.  This is because FFTW may need some intermediate
	// space.  For more information see FFTW's manual.
	ptrdiff_t        numAllocLocal;
	
	/// @brief Whether the processor according to the rank in domain
	/// participate in the FFTW calculation.
	bool             isInGrid;
	/// @brief The communicator used by parallel FFTW, should be
	/// consistent with inGrid.
	MPI_Comm         comm;

  // plans
  fftw_plan        backwardPlan;
  fftw_plan        forwardPlan;

  unsigned         plannerFlag;

	// Laplacian operator related
  DblNumVec                gkkLocal;
	std::vector<CpxNumVec>   ikLocal;
  DblNumVec                TeterPrecondLocal;

	// Temporary vectors that can also be used globally
	CpxNumVec                inputComplexVecLocal;     
	CpxNumVec                outputComplexVecLocal;     

	DistFourier();
	~DistFourier();

	/// @brief Initialize the FFTW variables.
	/// 
	/// @param[in] dm Domain for the FFTW calculation. 
	/// @param[in] numProc The number of processors actually participate
	/// in the FFTW calculation.  A processor participates in the FFTW
	/// calculation if mpirank(dm.comm) < numProc.
	void Initialize( const Domain& dm, Int numProc );
};

} // namespace dgdft


#endif // _FOURIER_HPP_
