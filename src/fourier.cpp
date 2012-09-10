#include  "fourier.hpp"

namespace dgdft{
Fourier::Fourier () : 
isPrepared(false),
	numGridTotal(0),
	plannerFlag(FFTW_MEASURE)
{
	backwardPlan  = NULL;
	forwardPlan   = NULL;
}

Fourier::~Fourier () 
{
	if( backwardPlan ) fftw_destroy_plan( backwardPlan );
	if( forwardPlan  ) fftw_destroy_plan( forwardPlan );
}

void
PrepareFourier ( Fourier& fft, const Domain& dm )
{
#ifndef _RELEASE_
	PushCallStack("PrepareFourier");
#endif  // ifndef _RELEASE_
	if( fft.isPrepared ) {
		throw std::logic_error("Fourier has been prepared.");
	}

	fft.domain = dm;
	ptrdiff_t localN0, localN0Start;
	Index3& numGrid = fft.domain.numGrid;
	Point3& length  = fft.domain.length;
	

	// TODO MPI
	// IMPORTANT: the order of numGrid. This is because the FFTW arrays
	// are row-major ordered.
//  fft.numGridLocal = (Int) fftw_mpi_local_size_3d(
//      numGrid[2], numGrid[1], numGrid[0], dm.comm, 
//      &localN0, 
//      &localN0Start);

	// IMPORTANT: the static cast here is necessary. Do not remove.
//	fft.localN0      = static_cast<Int>(localN0);
//	fft.localN0Start = static_cast<Int>(localN0Start);
//
//	if( fft.numGridLocal != static_cast<Int>(localN0) * numGrid[1] * numGrid[0] ){
//		std::ostringstream msg;
//		msg 
//			<< "The dimension of space does not match." << std::endl 
//			<< "numGridLocal obtained from fftw_mpi_local_size_3d is " << fft.numGridLocal << std::endl
//			<< "localN0 = " << localN0 << std::endl
//			<< "localN0 * numGrid[1] * numGrid[2] = " << localN0 * numGrid[1] * numGrid[2] << std::endl;
//		throw std::logic_error( msg.str().c_str() );
//	}

	fft.numGridTotal = fft.domain.NumGridTotal();

	fft.inputComplexVec.Resize( fft.numGridTotal );
	fft.outputComplexVec.Resize( fft.numGridTotal );

	// IMPORTANT: the order of numGrid. This is because the FFTW arrays
	// are row-major ordered.
//	fft.forwardPlan = fftw_mpi_plan_dft_3d( 
//			numGrid[2], numGrid[1], numGrid[0], 
//			reinterpret_cast<fftw_complex*>( &fft.inputComplexVec[0] ), 
//			reinterpret_cast<fftw_complex*>( &fft.outputComplexVec[0] ),
//			dm.comm, FFTW_FORWARD, fft.plannerFlag );
//
//	fft.backwardPlan = fftw_mpi_plan_dft_3d(
//			numGrid[2], numGrid[1], numGrid[0],
//			reinterpret_cast<fftw_complex*>( &fft.outputComplexVec[0] ),
//			reinterpret_cast<fftw_complex*>( &fft.inputComplexVec[0] ),
//			dm.comm, FFTW_BACKWARD, fft.plannerFlag);
	fft.forwardPlan = fftw_plan_dft_3d( 
			numGrid[2], numGrid[1], numGrid[0], 
			reinterpret_cast<fftw_complex*>( &fft.inputComplexVec[0] ), 
			reinterpret_cast<fftw_complex*>( &fft.outputComplexVec[0] ),
			FFTW_FORWARD, fft.plannerFlag );

	fft.backwardPlan = fftw_plan_dft_3d(
			numGrid[2], numGrid[1], numGrid[0],
			reinterpret_cast<fftw_complex*>( &fft.outputComplexVec[0] ),
			reinterpret_cast<fftw_complex*>( &fft.inputComplexVec[0] ),
			FFTW_BACKWARD, fft.plannerFlag);

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

	fft.gkk.Resize( dm.NumGridTotal() );
	fft.TeterPrecond.Resize( dm.NumGridTotal() );
	fft.ik.resize(DIM);
	fft.ik[0].Resize( dm.NumGridTotal() );
	fft.ik[1].Resize( dm.NumGridTotal() );
	fft.ik[2].Resize( dm.NumGridTotal() );
	
	Real*     gkkPtr = fft.gkk.Data();
	Complex*  ikXPtr = fft.ik[0].Data();
	Complex*  ikYPtr = fft.ik[1].Data();
	Complex*  ikZPtr = fft.ik[2].Data();

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
	for( Int i = 0; i < fft.domain.NumGridTotal(); i++ ){
		a = fft.gkk[i] * 2.0;
		b = 27.0 + a * (18.0 + a * (12.0 + a * 8.0) );
		fft.TeterPrecond[i] = b / ( b + 16.0 * pow(a, 4.0) );
	}
	// TODO Real to Complex
	
	fft.isPrepared = true;

#ifndef _RELEASE_
	PopCallStack();
#endif  // ifndef _RELEASE_

	return ;
}		// -----  end of function PrepareFourier  ----- 

} // namespace dgdft
