/// @file ex20.cpp
/// @brief Test the DistFourier interface for solving a Poisson's
/// equation.
///
///   -Delta V = 4*pi*rho
///
/// on a domain of size [1,1,1].
///
/// The result can be compared with the output of utilities/poisson3d.m
///
/// @author Lin Lin
/// @date 2013-01-09
#include "dgdft.hpp"
#include "utility.hpp"
#include "fourier.hpp"
#include "mpi_interf.hpp"

using namespace dgdft;
using namespace std;

void Usage(){
	cout << "Test for the parallel FFTW" << endl;
}

int main(int argc, char **argv) 
{
	MPI_Init(&argc, &argv);
	int mpirank, mpisize;
	MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
	MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

	if( mpirank == 0 )
		Usage();

	try
	{
#ifdef  _RELEASE_
		throw std::runtime_error("Test should be run under debug mode");
#endif
		if( mpisize <= 1 ){
			throw std::runtime_error("At least 2 processors should be used.");
		}
		
		fftw_mpi_init();

		Domain  dm;
		dm.length  = Point3( 1.0, 1.0, 1.0 );
		dm.numGrid = Index3( 32, 32, 32 );

		DistFourier fft;
		Int numProc = mpisize-1;
		
		if( mpirank == 0 ){
			cout << mpisize << " processors used in total. Rank 0 -- " << 
				numProc-1 << " are actually used for computation." << endl;
		}

		fft.Initialize( dm, numProc );
		Point3 gridSize;
		for(Int i = 0; i < 3; i++){
			gridSize[i] = dm.length[i] / dm.numGrid[i];
		}

		Int Nx = dm.numGrid[0], Ny = dm.numGrid[1], Nz = dm.numGrid[2];
		Index3&    numGrid = dm.numGrid;

		DblNumVec  rho(fft.numGridTotal), vhart( fft.numGridTotal );

		Real x, y, z;
		Real sigma = 0.2;
		for( Int k = 0; k < Nz; k++ ){
			for(Int j = 0; j < Ny; j++ ){
				for(Int i = 0; i < Nx; i++ ){
					x = i * gridSize[0] - 0.5;
					y = j * gridSize[1] - 0.5;
					z = k * gridSize[2] - 0.5;
					rho(i + j*Nx + k * Nx * Ny) = exp(-(x*x + y*y + z*z) / (2.0 * sigma*sigma));
				}
			}
		}

		cerr << "mpirank = " << mpirank << ", isInGrid = " << fft.isInGrid << endl;

		if( fft.isInGrid ){
			CpxNumVec& in  = fft.inputComplexVecLocal;
			CpxNumVec& out = fft.outputComplexVecLocal;

			for( Int k = 0; k < fft.localNz; k++ ){
				for(Int j = 0; j < Ny; j++ ){
					for(Int i = 0; i < Nx; i++ ){
						in(i + j*Nx + k * Nx * Ny) = 
							Complex( rho(i + j*Nx + (k+fft.localNzStart) * Nx * Ny), 0.0 );
					}
				}
			}

			fftw_execute( fft.forwardPlan );

			for(Int i = 0; i < fft.numGridLocal; i++){
				if( fft.gkkLocal(i) == 0 ){
					out(i) = Z_ZERO;
				}
				else{
					// NOTE: gkk already contains the factor 1/2.
					out(i) *= 2.0 * PI / fft.gkkLocal(i);
				}
			}

			fftw_execute( fft.backwardPlan );

			DblNumVec  vhartLocal( fft.numGridTotal );
			SetValue( vhartLocal, 0.0 );

			for( Int i = 0; i < fft.numGridLocal; i++ ){
				in(i) /= fft.numGridTotal;
			}

			for( Int k = 0; k < fft.localNz; k++ ){
				for(Int j = 0; j < Ny; j++ ){
					for(Int i = 0; i < Nx; i++ ){
						vhartLocal(i + j*Nx + (k+fft.localNzStart) * Nx * Ny) = 
							in(i + j*Nx + k * Nx * Ny).real();
					}
				}
			}

			mpi::Reduce( vhartLocal.Data(), vhart.Data(), fft.numGridTotal,
					MPI_SUM, 0, fft.comm );

			if( mpirank == 0 ){
				ofstream ofs("vhart");
				if( !ofs.good() ){
					throw std::runtime_error( "Cannot open file." );
				}
				serialize( vhart, ofs, NO_MASK );
				ofs.close();
				cout << "Finished. Output to file vhart." << endl;
			}
		} // if (fft.isInGrid)

		
		fftw_mpi_cleanup();

	}
	catch( std::exception& e )
	{
		std::cerr << " caught exception with message: "
			<< e.what() << std::endl;
		DumpCallStack();
	}

	MPI_Finalize();

	return 0;
}
