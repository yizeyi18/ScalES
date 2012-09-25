// *********************************************************************
// Test for the periodic table class without using Petsc.
//
// *********************************************************************
#include "dgdft.hpp"
#include "utility.hpp"
#include "eigensolver.hpp"
#include "periodtable.hpp"

using namespace dgdft;
using namespace std;

void Usage(){
	cout << "Test for the periodic table" << endl;
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

		// *********************************************************************
		// Preparation
		// *********************************************************************

		PushCallStack( "Preparation" );

		Domain  dm;
		dm.length  = Point3( 1.0, 1.0, 1.0 );
		dm.numGrid = Index3( 4, 4, 4 );
		std::vector<Atom> atomList;
		atomList.clear();

		string pseudoType = "HGH";
		Int  XCId = XC_LDA_XC_TETER93;
		
		Int nev = 15;
		KohnSham ks(dm, atomList, pseudoType, XCId, nev, 1);
		Fourier fft;
		PrepareFourier( fft, dm );

#ifdef _USE_COMPLEX_
		Spinor  spn( dm, 1, nev, Complex(1.0, 1.0) );
#else
		Spinor  spn( dm, 1, nev, 1.0 );
#endif

		SetRandomSeed(1);

		UniformRandom( spn.Wavefun() );

		EigenSolver eigSol( ks, spn, fft, 20, 1e-6, 1e-6 );

		PopCallStack();

		// *********************************************************************
		// Solve
		// *********************************************************************
		PushCallStack( "Solving" );
		eigSol.Solve();

		cout << "Eigenvalues " << eigSol.EigVal() << endl;

		PopCallStack();


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
