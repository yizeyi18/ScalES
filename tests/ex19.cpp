/// @file ex19.cpp
/// @brief Test the parallel FFTW
/// @author Lin Lin
/// @date 2013-01-08
#include "dgdft.hpp"
#include "utility.hpp"

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

	Usage();

	try
	{
#ifdef  _RELEASE_
		throw std::runtime_error("Test should be run under debug mode");
#endif

		CpxNumVec  in, out;

		fftw_mpi_init();

		Int Nx = 2, Ny = 8, Nz = 4;
		
		Int numGrid = Nx * Ny * Nz;

		ptrdiff_t localNz, localNzStart;  // Use ptrdiff_t (long*) is important here.

		Int numGridLocal = fftw_mpi_local_size_3d(Nz, Ny, Nx, 
				MPI_COMM_WORLD, &localNz, &localNzStart );

		cout << "localNz = " << localNz << ", localNzStart = " << localNzStart
			<< endl;

		cout << "numGridLocal = " << numGridLocal << endl;

		in.Resize( numGridLocal );
		out.Resize( numGridLocal );

		fftw_plan plan;

		plan = fftw_mpi_plan_dft_3d( Nz, Ny, Nx, 
				reinterpret_cast<fftw_complex*>(in.Data()), 
				reinterpret_cast<fftw_complex*>(out.Data()), 
				MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE );

		for( Int k = 0; k < localNz; k++ ){
			for(Int j = 0; j < Ny; j++ ){
				for(Int i = 0; i < Nx; i++ ){
					in(i + j*Nx + k * Nx * Ny) = 
						Complex( i + j*Nx + (k+localNzStart) * Nx * Ny, 0.0 );
				}
			}
		}

		fftw_execute( plan );

		cout << "Input  = " << in << endl;
		cout << "Output = " << out << endl;
		
		fftw_destroy_plan( plan );

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
