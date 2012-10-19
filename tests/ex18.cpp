// *********************************************************************
// Test for the SCF procedure for KohnSham class
//
// *********************************************************************
#include "dgdft.hpp"
#include "utility.hpp"
#include "eigensolver.hpp"
#include "periodtable.hpp"
#include "esdf.hpp"
#include "scf.hpp"

using namespace dgdft;
using namespace dgdft::esdf;
using namespace std;

void Usage(){
	cout << "Test for the SCF procedure for KohnSham class" << endl;
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
		// Reading files
		// *********************************************************************
		

		stringstream  ss;
		ss << "statfile." << mpirank;
		cout << "The filename for the statfile is " << ss.str() << endl;
		statusOFS.open( ss.str().c_str() );

		ESDFInputParam  esdfParam;

		ESDFReadInput( esdfParam, "dgdft.in" );

		PrintInitialState( esdfParam );


		// *********************************************************************
		// Preparation
		// *********************************************************************
		PushCallStack( "Preparation" );

		Domain&  dm = esdfParam.domain;

		PeriodTable ptable;
		ptable.Setup( esdfParam.periodTableFile );

		PrintBlock( statusOFS, "Preparing the Hamiltonian" );
		Print( statusOFS, "Periodic table setup finished." );

		KohnSham hamKS( esdfParam, 1 );

		hamKS.CalculatePseudoCharge( ptable );
		hamKS.CalculateNonlocalPP  ( ptable );

		Print( statusOFS, "Pseudopotential setup finished." );

		Fourier fft;
		SetupFourier( fft, dm );

#ifdef _USE_COMPLEX_
		Spinor  spn( dm, 2, hamKS.NumStateTotal(), Complex(1.0, 1.0) );
#else
		Spinor  spn( dm, 2, hamKS.NumStateTotal(), 1.0 );
#endif

		SetRandomSeed(1);

		UniformRandom( spn.Wavefun() );

		EigenSolver eigSol;
		
		eigSol.Setup( esdfParam, hamKS, spn, fft );

		Print( statusOFS, "Eigensolver setup finished." );

		SCF   scf;
		scf.Setup( esdfParam, eigSol, ptable );


		Print( statusOFS, "SCF setup finished." );

		PopCallStack();

		// *********************************************************************
		// Solve
		// *********************************************************************
		scf.Iterate();

		scf.OutputState();

		statusOFS.close();

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
