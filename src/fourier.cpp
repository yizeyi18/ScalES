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
/// @file fourier.cpp
/// @brief Sequential and Distributed Fourier wrapper.
/// @date 2011-11-01
/// @date 2014-02-01 Dual grid implementation.
#include  "fourier.hpp"

namespace dgdft{


// *********************************************************************
// Sequential FFTW
// *********************************************************************

Fourier::Fourier () : 
	isInitialized(false),
	numGridTotal(0),
  numGridTotalFine(0),
	// plannerFlag(FFTW_MEASURE)
	plannerFlag(FFTW_ESTIMATE)
	{
		backwardPlan  = NULL;
		forwardPlan   = NULL;
		backwardPlanR2C  = NULL;
		forwardPlanR2C   = NULL;
		backwardPlanFine  = NULL;
		forwardPlanFine   = NULL;
		backwardPlanR2CFine  = NULL;
		forwardPlanR2CFine   = NULL;
    
	}

Fourier::~Fourier () 
{
	if( backwardPlan ) fftw_destroy_plan( backwardPlan );
	if( forwardPlan  ) fftw_destroy_plan( forwardPlan );
	if( backwardPlanR2C  ) fftw_destroy_plan( backwardPlanR2C );
	if( forwardPlanR2C   ) fftw_destroy_plan( forwardPlanR2C );
	if( backwardPlanFine ) fftw_destroy_plan( backwardPlanFine );
	if( forwardPlanFine  ) fftw_destroy_plan( forwardPlanFine );
	if( backwardPlanR2CFine  ) fftw_destroy_plan( backwardPlanR2CFine );
	if( forwardPlanR2CFine   ) fftw_destroy_plan( forwardPlanR2CFine );
}

void Fourier::Initialize ( const Domain& dm )
{
#ifndef _RELEASE_
	PushCallStack("Fourier::Initialize");
#endif  // ifndef _RELEASE_

	if( isInitialized ) {
		throw std::logic_error("Fourier has been initialized.");
	}

	domain = dm;
	Index3& numGrid = domain.numGrid;
	Point3& length  = domain.length;

	numGridTotal = domain.NumGridTotal();

	inputComplexVec.Resize( numGridTotal );
	outputComplexVec.Resize( numGridTotal );

	forwardPlan = fftw_plan_dft_3d( 
			numGrid[2], numGrid[1], numGrid[0], 
			reinterpret_cast<fftw_complex*>( &inputComplexVec[0] ), 
			reinterpret_cast<fftw_complex*>( &outputComplexVec[0] ),
			FFTW_FORWARD, plannerFlag );

	backwardPlan = fftw_plan_dft_3d(
			numGrid[2], numGrid[1], numGrid[0],
			reinterpret_cast<fftw_complex*>( &outputComplexVec[0] ),
			reinterpret_cast<fftw_complex*>( &inputComplexVec[0] ),
			FFTW_BACKWARD, plannerFlag);

	std::vector<DblNumVec>  KGrid(DIM);                // Fourier grid
	for( Int idim = 0; idim < DIM; idim++ ){
		KGrid[idim].Resize( numGrid[idim] );
		for( Int i = 0; i <= numGrid[idim] / 2; i++ ){
			KGrid[idim](i) = i * 2.0 * PI / length[idim];
		}
		for( Int i = numGrid[idim] / 2 + 1; i < numGrid[idim]; i++ ){
			KGrid[idim](i) = ( i - numGrid[idim] ) * 2.0 * PI / length[idim];
		}
	}

	gkk.Resize( dm.NumGridTotal() );
	TeterPrecond.Resize( dm.NumGridTotal() );
	ik.resize(DIM);
	ik[0].Resize( dm.NumGridTotal() );
	ik[1].Resize( dm.NumGridTotal() );
	ik[2].Resize( dm.NumGridTotal() );

	Real*     gkkPtr = gkk.Data();
	Complex*  ikXPtr = ik[0].Data();
	Complex*  ikYPtr = ik[1].Data();
	Complex*  ikZPtr = ik[2].Data();



	for( Int k = 0; k < numGrid[2]; k++ ){
		for( Int j = 0; j < numGrid[1]; j++ ){
			for( Int i = 0; i < numGrid[0]; i++ ){
				*(gkkPtr++) = 
					( KGrid[0](i) * KGrid[0](i) +
						KGrid[1](j) * KGrid[1](j) +
						KGrid[2](k) * KGrid[2](k) ) / 2.0;

				*(ikXPtr++) = Complex( 0.0, KGrid[0](i) );
				*(ikYPtr++) = Complex( 0.0, KGrid[1](j) );
				*(ikZPtr++) = Complex( 0.0, KGrid[2](k) );

			}
		}
	}

	// TeterPreconditioner
	Real  a, b;
	for( Int i = 0; i < domain.NumGridTotal(); i++ ){
		a = gkk[i] * 2.0;
		b = 27.0 + a * (18.0 + a * (12.0 + a * 8.0) );
		TeterPrecond[i] = b / ( b + 16.0 * pow(a, 4.0) );
	}


	// R2C transform
	numGridTotalR2C = (numGrid[0]/2+1) * numGrid[1] * numGrid[2];

	inputVecR2C.Resize( numGridTotal );
	outputVecR2C.Resize( numGridTotalR2C );

	forwardPlanR2C = fftw_plan_dft_r2c_3d( 
			numGrid[2], numGrid[1], numGrid[0], 
			( &inputVecR2C[0] ), 
			reinterpret_cast<fftw_complex*>( &outputVecR2C[0] ),
			plannerFlag );

	backwardPlanR2C = fftw_plan_dft_c2r_3d(
			numGrid[2], numGrid[1], numGrid[0],
			reinterpret_cast<fftw_complex*>( &outputVecR2C[0] ),
			&inputVecR2C[0],
			plannerFlag);
  

	// -1/2 \Delta  and Teter preconditioner in R2C
	gkkR2C.Resize( numGridTotalR2C );
	TeterPrecondR2C.Resize( numGridTotalR2C );
	ikR2C.resize(DIM);
	ikR2C[0].Resize( numGridTotalR2C );
	ikR2C[1].Resize( numGridTotalR2C );
	ikR2C[2].Resize( numGridTotalR2C );


	Real*  gkkR2CPtr = gkkR2C.Data();
	Complex*  ikXR2CPtr = ikR2C[0].Data();
	Complex*  ikYR2CPtr = ikR2C[1].Data();
	Complex*  ikZR2CPtr = ikR2C[2].Data();
	for( Int k = 0; k < numGrid[2]; k++ ){
		for( Int j = 0; j < numGrid[1]; j++ ){
			for( Int i = 0; i < numGrid[0]/2+1; i++ ){
				*(gkkR2CPtr++) = 
					( KGrid[0](i) * KGrid[0](i) +
						KGrid[1](j) * KGrid[1](j) +
						KGrid[2](k) * KGrid[2](k) ) / 2.0;

				*(ikXR2CPtr++) = Complex( 0.0, KGrid[0](i) );
				*(ikYR2CPtr++) = Complex( 0.0, KGrid[1](j) );
				*(ikZR2CPtr++) = Complex( 0.0, KGrid[2](k) );
			}
		}
	}


	// TeterPreconditioner
	for( Int i = 0; i < numGridTotalR2C; i++ ){
		a = gkkR2C[i] * 2.0;
		b = 27.0 + a * (18.0 + a * (12.0 + a * 8.0) );
		TeterPrecondR2C[i] = b / ( b + 16.0 * pow(a, 4.0) );
	}

	// Mark Fourier to be initialized
	isInitialized = true;

#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_

	return ;
}		// -----  end of function Fourier::Initialize  ----- 









void Fourier::InitializeFine ( const Domain& dm )
{
#ifndef _RELEASE_
	PushCallStack("Fourier::Initialize");
#endif  // ifndef _RELEASE_

//  if( isInitialized ) {
//		throw std::logic_error("Fourier has been prepared.");
//	}

	domain = dm;
	Index3& numGrid = domain.numGridFine;
	Point3& length  = domain.length;

	numGridTotalFine = domain.NumGridTotalFine();

	inputComplexVecFine.Resize( numGridTotalFine );
	outputComplexVecFine.Resize( numGridTotalFine );

	forwardPlanFine = fftw_plan_dft_3d( 
			numGrid[2], numGrid[1], numGrid[0], 
			reinterpret_cast<fftw_complex*>( &inputComplexVecFine[0] ), 
			reinterpret_cast<fftw_complex*>( &outputComplexVecFine[0] ),
			FFTW_FORWARD, plannerFlag );

	backwardPlanFine = fftw_plan_dft_3d(
			numGrid[2], numGrid[1], numGrid[0],
			reinterpret_cast<fftw_complex*>( &outputComplexVecFine[0] ),
			reinterpret_cast<fftw_complex*>( &inputComplexVecFine[0] ),
			FFTW_BACKWARD, plannerFlag);

	std::vector<DblNumVec>  KGrid(DIM);                // Fourier grid
	for( Int idim = 0; idim < DIM; idim++ ){
		KGrid[idim].Resize( numGrid[idim] );
		for( Int i = 0; i <= numGrid[idim] / 2; i++ ){
			KGrid[idim](i) = i * 2.0 * PI / length[idim];
		}
		for( Int i = numGrid[idim] / 2 + 1; i < numGrid[idim]; i++ ){
			KGrid[idim](i) = ( i - numGrid[idim] ) * 2.0 * PI / length[idim];
		}
	}

	gkkFine.Resize( dm.NumGridTotalFine() );
	TeterPrecondFine.Resize( dm.NumGridTotalFine() );
	ikFine.resize(DIM);
	ikFine[0].Resize( dm.NumGridTotalFine() );
	ikFine[1].Resize( dm.NumGridTotalFine() );
	ikFine[2].Resize( dm.NumGridTotalFine() );

	Real*     gkkPtr = gkkFine.Data();
	Complex*  ikXPtr = ikFine[0].Data();
	Complex*  ikYPtr = ikFine[1].Data();
	Complex*  ikZPtr = ikFine[2].Data();

	for( Int k = 0; k < numGrid[2]; k++ ){
		for( Int j = 0; j < numGrid[1]; j++ ){
			for( Int i = 0; i < numGrid[0]; i++ ){
				*(gkkPtr++) = 
					( KGrid[0](i) * KGrid[0](i) +
						KGrid[1](j) * KGrid[1](j) +
						KGrid[2](k) * KGrid[2](k) ) / 2.0;

				*(ikXPtr++) = Complex( 0.0, KGrid[0](i) );
				*(ikYPtr++) = Complex( 0.0, KGrid[1](j) );
				*(ikZPtr++) = Complex( 0.0, KGrid[2](k) );

			}
		}
	}


	// TeterPreconditioner
	Real  a, b;
	for( Int i = 0; i < domain.NumGridTotalFine(); i++ ){
		a = gkkFine[i] * 2.0;
		b = 27.0 + a * (18.0 + a * (12.0 + a * 8.0) );
		TeterPrecondFine[i] = b / ( b + 16.0 * pow(a, 4.0) );
	}

	// R2C transform
	numGridTotalR2CFine = (numGrid[0]/2+1) * numGrid[1] * numGrid[2];

	inputVecR2CFine.Resize( numGridTotalFine );
	outputVecR2CFine.Resize( numGridTotalR2CFine );

	forwardPlanR2CFine = fftw_plan_dft_r2c_3d( 
			numGrid[2], numGrid[1], numGrid[0], 
			( &inputVecR2CFine[0] ), 
			reinterpret_cast<fftw_complex*>( &outputVecR2CFine[0] ),
			plannerFlag );

	backwardPlanR2CFine = fftw_plan_dft_c2r_3d(
			numGrid[2], numGrid[1], numGrid[0],
			reinterpret_cast<fftw_complex*>( &outputVecR2CFine[0] ),
			&inputVecR2CFine[0],
			plannerFlag);

	// -1/2 \Delta  and Teter preconditioner in R2C
	gkkR2CFine.Resize( numGridTotalR2CFine );
	TeterPrecondR2CFine.Resize( numGridTotalR2CFine );

	Real*  gkkR2CPtr = gkkR2CFine.Data();

  for( Int k = 0; k < numGrid[2]; k++ ){
		for( Int j = 0; j < numGrid[1]; j++ ){
			for( Int i = 0; i < numGrid[0]/2+1; i++ ){
				*(gkkR2CPtr++) = 
					( KGrid[0](i) * KGrid[0](i) +
						KGrid[1](j) * KGrid[1](j) +
						KGrid[2](k) * KGrid[2](k) ) / 2.0;
			}
		}
	}

	// TeterPreconditioner
	for( Int i = 0; i < numGridTotalR2CFine; i++ ){
		a = gkkR2CFine[i] * 2.0;
		b = 27.0 + a * (18.0 + a * (12.0 + a * 8.0) );
		TeterPrecondR2CFine[i] = b / ( b + 16.0 * pow(a, 4.0) );
	}

  // Compute the index for mapping coarse to find grid
  idxFineGrid.Resize(domain.NumGridTotal());
  SetValue( idxFineGrid, 0 );
  {
    Int PtrC, PtrF, iF, jF, kF;
    for( Int kk = 0; kk < domain.numGrid[2]; kk++ ){
      for( Int jj = 0; jj < domain.numGrid[1]; jj++ ){
        for( Int ii = 0; ii < domain.numGrid[0]; ii++ ){

          PtrC = ii + jj * domain.numGrid[0] + kk * domain.numGrid[0] * domain.numGrid[1];

          if ( (0 <= ii) && (ii <= domain.numGrid[0] / 2) ) { iF = ii; } 
          else { iF = domain.numGridFine[0] - domain.numGrid[0] + ii; } 

          if ( (0 <= jj) && (jj <= domain.numGrid[1] / 2) ) { jF = jj; } 
          else { jF = domain.numGridFine[1] - domain.numGrid[1] + jj; } 

          if ( (0 <= kk) && (kk <= domain.numGrid[2] / 2) ) { kF = kk; } 
          else { kF = domain.numGridFine[2] - domain.numGrid[2] + kk; } 

          PtrF = iF + jF * domain.numGridFine[0] + kF * domain.numGridFine[0] * domain.numGridFine[1];

          idxFineGrid[PtrC] = PtrF;
        } 
      }
    }
  }



	// Mark Fourier to be initialized
//	isInitialized = true;

#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_

	return ;
}		// -----  end of function Fourier::InitializeFine  ----- 








// *********************************************************************
// Parallel FFTW
// *********************************************************************

DistFourier::DistFourier () : 
  isInitialized(false),
	numGridTotal(0),
	numGridLocal(0),
	localNz(0),
	localNzStart(0),
	numAllocLocal(0),
	isInGrid(false),
	// plannerFlag(FFTW_MEASURE),
	plannerFlag(FFTW_ESTIMATE),
	comm(MPI_COMM_NULL),
	forwardPlan(NULL),
	backwardPlan(NULL)
{ }

DistFourier::~DistFourier () 
{
	if( backwardPlan ) fftw_destroy_plan( backwardPlan );
	if( forwardPlan  ) fftw_destroy_plan( forwardPlan );
	if( comm != MPI_COMM_NULL ) MPI_Comm_free( & comm );
}

void DistFourier::Initialize ( const Domain& dm, Int numProc )
{
#ifndef _RELEASE_
	PushCallStack("DistFourier::Initialize");
#endif  // ifndef _RELEASE_
	if( isInitialized ) {
		throw std::logic_error("Fourier has been prepared.");
	}

	domain = dm;
	Index3& numGrid = domain.numGridFine;
	Point3& length  = domain.length;

	numGridTotal = domain.NumGridTotalFine();

	// Create the new communicator
	{
		Int mpirankDomain, mpisizeDomain;
		MPI_Comm_rank( dm.colComm, &mpirankDomain );
		MPI_Comm_size( dm.colComm, &mpisizeDomain );
    
    Int mpirank, mpisize;
    MPI_Comm_rank( dm.comm, &mpirank );
    MPI_Comm_size( dm.comm, &mpisize );
    
    MPI_Barrier(dm.rowComm);
    Int mpirankRow;  MPI_Comm_rank(dm.rowComm, &mpirankRow);
    Int mpisizeRow;  MPI_Comm_size(dm.rowComm, &mpisizeRow);

    MPI_Barrier(dm.colComm);
    Int mpirankCol;  MPI_Comm_rank(dm.colComm, &mpirankCol);
    Int mpisizeCol;  MPI_Comm_size(dm.colComm, &mpisizeCol);

    //numProc = mpisizeCol;

    if( numProc > mpisizeDomain ){
			std::ostringstream msg;
			msg << "numProc cannot exceed mpisize."  << std::endl
				<< "numProc ~ " << numProc << std::endl
				<< "mpisize = " << mpisizeDomain << std::endl;
			throw std::runtime_error( msg.str().c_str() );
		}
		if( mpirankDomain < numProc )
			isInGrid = true;
		else
			isInGrid = false;

		MPI_Comm_split( dm.colComm, isInGrid, mpirankDomain, &comm );
	}

	if( isInGrid ){
	
		// Rank and size of the processor group participating in FFT calculation.
		Int mpirankFFT, mpisizeFFT;
		MPI_Comm_rank( comm, &mpirankFFT );
		MPI_Comm_size( comm, &mpisizeFFT );

		if( numGrid[2] < mpisizeFFT ){
			std::ostringstream msg;
			msg << "numGrid[2] > mpisizeFFT. FFTW initialization failed. "  << std::endl
				<< "numGrid    = " << numGrid << std::endl
				<< "mpisizeFFT = " << mpisizeFFT << std::endl;
			throw std::runtime_error( msg.str().c_str() );
		}

		// IMPORTANT: the order of numGrid. This is because the FFTW arrays
		// are row-major ordered.
		numAllocLocal =  fftw_mpi_local_size_3d(
				numGrid[2], numGrid[1], numGrid[0], comm, 
				&localNz, &localNzStart );

		numGridLocal = numGrid[0] * numGrid[1] * localNz;

#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "localNz        = " << localNz << std::endl;
		statusOFS << "localNzStart   = " << localNzStart << std::endl;
		statusOFS << "numAllocLocal  = " << numAllocLocal << std::endl;
		statusOFS << "numGridLocal   = " << numGridLocal << std::endl;
		statusOFS << "numGridTotal   = " << numGridTotal << std::endl;
#endif

		inputComplexVecLocal.Resize( numAllocLocal );
		outputComplexVecLocal.Resize( numAllocLocal );

		// IMPORTANT: the order of numGrid. This is because the FFTW arrays
		// are row-major ordered.
		forwardPlan = fftw_mpi_plan_dft_3d( 
				numGrid[2], numGrid[1], numGrid[0], 
				reinterpret_cast<fftw_complex*>( &inputComplexVecLocal[0] ), 
				reinterpret_cast<fftw_complex*>( &outputComplexVecLocal[0] ),
				comm, FFTW_FORWARD, plannerFlag );


		backwardPlan = fftw_mpi_plan_dft_3d(
				numGrid[2], numGrid[1], numGrid[0],
				reinterpret_cast<fftw_complex*>( &outputComplexVecLocal[0] ),
				reinterpret_cast<fftw_complex*>( &inputComplexVecLocal[0] ),
				comm, FFTW_BACKWARD, plannerFlag);

		std::vector<DblNumVec>  KGrid(DIM);                // Fourier grid
		for( Int idim = 0; idim < DIM; idim++ ){
			KGrid[idim].Resize( numGrid[idim] );
			for( Int i = 0; i <= numGrid[idim] / 2; i++ ){
				KGrid[idim](i) = i * 2.0 * PI / length[idim];
			}
			for( Int i = numGrid[idim] / 2 + 1; i < numGrid[idim]; i++ ){
				KGrid[idim](i) = ( i - numGrid[idim] ) * 2.0 * PI / length[idim];
			}
		}

		gkkLocal.Resize( numGridLocal );
		TeterPrecondLocal.Resize( numGridLocal );
		ikLocal.resize(DIM);
		ikLocal[0].Resize( numGridLocal );
		ikLocal[1].Resize( numGridLocal );
		ikLocal[2].Resize( numGridLocal );

		Real*     gkkPtr = gkkLocal.Data();
		Complex*  ikXPtr = ikLocal[0].Data();
		Complex*  ikYPtr = ikLocal[1].Data();
		Complex*  ikZPtr = ikLocal[2].Data();

		for( Int k = localNzStart; k < localNzStart + localNz; k++ ){
			for( Int j = 0; j < numGrid[1]; j++ ){
				for( Int i = 0; i < numGrid[0]; i++ ){
					*(gkkPtr++) = 
						( KGrid[0](i) * KGrid[0](i) +
							KGrid[1](j) * KGrid[1](j) +
							KGrid[2](k) * KGrid[2](k) ) / 2.0;

					*(ikXPtr++) = Complex( 0.0, KGrid[0](i) );
					*(ikYPtr++) = Complex( 0.0, KGrid[1](j) );
					*(ikZPtr++) = Complex( 0.0, KGrid[2](k) );

				}
			}
		}

		// TeterPreconditioner
		Real  a, b;
		for( Int i = 0; i < numGridLocal; i++ ){
			a = gkkLocal[i] * 2.0;
			b = 27.0 + a * (18.0 + a * (12.0 + a * 8.0) );
			TeterPrecondLocal[i] = b / ( b + 16.0 * pow(a, 4.0) );
		}
	} // if (isInGrid)

	isInitialized = true;

#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_

	return ;
}		// -----  end of function DistFourier::Initialize  ----- 


} // namespace dgdft
