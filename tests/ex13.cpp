// *********************************************************************
// Test for solving a least square problem, which is going to be used
// for Anderson mixing.  This program is sequential.  The matrix-matrix
// multiplication is performed using BLAS1 vector-vector multiplication
// and then is solved using lapack.
//
// A = [2  3
//      1  1
//      1  1];
//
// b = [0; 2; 0];
//
// A\b = [3; -2];
// *********************************************************************
#include "dgdft.hpp"
#include "utility.hpp"


static char help[] = "Test the least square solver for dense matrices sequentially.";
using namespace dgdft;
using namespace std;

int main(int argc, char **argv) 
{
	PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help); CHKERRQ(ierr);
  
	int mpirank, mpisize;
	MPI_Comm_rank( PETSC_COMM_WORLD, &mpirank );
	MPI_Comm_size( PETSC_COMM_WORLD, &mpisize );

	try
	{
#ifdef  _RELEASE_
		throw std::runtime_error("Test should be run under debug mode");
#endif
		const Int   M = 3, N = 2;

		NumMat<Scalar>   AData(M, N);
		AData(0,0) = 2; AData(0,1) = 3;
		AData(1,0) = 1; AData(1,1) = 1;
		AData(2,0) = 1; AData(2,1) = 1;

		std::vector<Vec>  A(N);
		for(Int j = 0; j < N; j++){
			VecCreateSeqWithArray( PETSC_COMM_WORLD, 1, M, AData.ColData(j), &A[j] );
			VecView( A[j], PETSC_VIEWER_STDOUT_WORLD ); 
		}

		NumVec<Scalar>   bData(M);
		bData(0) = 0; bData(1) = 2; bData(2) = 0;

		Vec b;
		VecCreateSeqWithArray( PETSC_COMM_WORLD, 1, M, bData.Data(), &b );

		// Construct the projected problem
		NumMat<Scalar>  ATA(N, N);
		NumVec<Scalar>  ATb(N);

		SetValue( ATA, static_cast<Complex>(0.0) );
		SetValue( ATb, static_cast<Complex>(0.0) );
		
		for( Int i = 0; i < N; i++ ){
			for( Int j = 0; j < N; j++ ){
				VecDot( A[i], A[j], &ATA(i, j) );
			}
			VecDot( A[i], b, &ATb(i) );
		}

		// Solve the projected problem using LAPACK
		NumVec<Scalar>  x(N);
		DblNumVec  S(N);
		Int rank;
		lapack::SVDLeastSquare( N, N, 1, ATA.Data(), N, ATb.Data(), N, 
				S.Data(), 5e-1, &rank ); 

		cout << "S = " << S << endl;

		x = NumVec<Scalar>( N, true, ATb.Data() );

		// Multiply the solution back into Ax
		NumVec<Scalar>  AxData(M);
		SetValue( AxData, static_cast<Complex>(0.0) );

		Vec   Ax;
		VecCreateSeqWithArray( PETSC_COMM_WORLD, 1, M, AxData.Data(), &Ax );

		for( Int j = 0; j < N; j++ ){
			VecAXPY( Ax, x(j), A[j] );
		}

		// View Ax
		VecView( Ax, PETSC_VIEWER_STDOUT_WORLD );
		 
		for(Int j = 0; j < N; j++){
			VecDestroy( &A[j] );
		}
		VecDestroy(&b);
		VecDestroy(&Ax);


	}
	catch( std::exception& e )
	{
		std::cerr << " caught exception with message: "
			<< e.what() << std::endl;
		DumpCallStack();
	}
	catch ( PetscErrorCode ierr ){
		std::cerr << "Caught PETSc error message " << std::endl;
		DumpCallStack();
		CHKERRQ(ierr);
	}
	
	ierr = SlepcFinalize();CHKERRQ(ierr);
	
  return 0;
}

