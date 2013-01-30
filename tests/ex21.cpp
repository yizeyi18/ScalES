/// @file ex21.cpp
/// @brief Test for HamiltonianDG class.
///
/// @author Lin Lin
/// @date 2013-01-10
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;

void Usage(){
	cout << "Test for the HamiltonianDG" << endl;
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

		fftw_mpi_init();
		
		stringstream  ss;
		ss << "statfile." << mpirank;
		cout << "The filename for the statfile is " << ss.str() << endl;
		statusOFS.open( ss.str().c_str() );

		ESDFInputParam  esdfParam;

		ESDFReadInput( esdfParam, "dgdft.in" );

		// Adjust the input parameters
		{
			bool isGridAdjusted = false;
			Index3& numGrid = esdfParam.domain.numGrid;
			Index3& numElem = esdfParam.numElem;
			for( Int d = 0; d < DIM; d++ ){
				if( numGrid[d] % numElem[d] != 0 ){
					numGrid[d] = IRound( (Real)numGrid[d] / numElem[d] ) * numElem[d];
					isGridAdjusted = true;
				}
			}
			if( isGridAdjusted ){
				statusOFS << std::endl 
					<< "Grid size is adjusted to be a multiple of the number of elements." 
					<< std::endl;
			}
		}

		// Print the initial state
		{
			PrintBlock(statusOFS, "Basic information");

			Print(statusOFS, "Super cell        = ",  esdfParam.domain.length );
			Print(statusOFS, "Grid size         = ",  esdfParam.domain.numGrid ); 
			Print(statusOFS, "Mixing dimension  = ",  esdfParam.mixMaxDim );
			Print(statusOFS, "Mixing type       = ",  esdfParam.mixType );
			Print(statusOFS, "Mixing Steplength = ",  esdfParam.mixStepLength);
			Print(statusOFS, "SCF Tolerence     = ",  esdfParam.scfTolerance);
			Print(statusOFS, "SCF MaxIter       = ",  esdfParam.scfMaxIter);
			Print(statusOFS, "Eig Tolerence     = ",  esdfParam.eigTolerance);
			Print(statusOFS, "Eig MaxIter       = ",  esdfParam.eigMaxIter);

			Print(statusOFS, "RestartDensity    = ",  esdfParam.isRestartDensity);
			Print(statusOFS, "RestartWfn        = ",  esdfParam.isRestartWfn);
			Print(statusOFS, "OutputDensity     = ",  esdfParam.isOutputDensity);
			Print(statusOFS, "OutputWfn         = ",  esdfParam.isOutputWfn);

			Print(statusOFS, "Temperature       = ",  au2K / esdfParam.Tbeta, "[K]");
			Print(statusOFS, "Extra states      = ",  esdfParam.numExtraState );
			Print(statusOFS, "PeriodTable File  = ",  esdfParam.periodTableFile );
			Print(statusOFS, "Pseudo Type       = ",  esdfParam.pseudoType );
			Print(statusOFS, "PW Solver         = ",  esdfParam.PWSolver );
			Print(statusOFS, "XC Type           = ",  esdfParam.XCType );

			Print(statusOFS, "Penalty Alpha     = ",  esdfParam.penaltyAlpha );
			Print(statusOFS, "Element size      = ",  esdfParam.numElem ); 
			Print(statusOFS, "LGL Grid size     = ",  esdfParam.numGridLGL ); 


			PrintBlock(statusOFS, "Atom Type and Coordinates");

			const std::vector<Atom>&  atomList = esdfParam.atomList;
			for(Int i=0; i < atomList.size(); i++) {
				Print(statusOFS, "Type = ", atomList[i].type, "Position  = ", atomList[i].pos);
			}

			statusOFS << std::endl;
		}


		// *********************************************************************
		// Preparation
		// *********************************************************************
		Domain&  dm = esdfParam.domain;

		PeriodTable ptable;
		ptable.Setup( esdfParam.periodTableFile );

		DistFourier fft;
		fft.Initialize( dm, mpisize );

		HamiltonianDG hamDG( esdfParam );

		// FIXME for the time being, calculate PP twice
		hamDG.CalculatePseudoPotential( ptable );

		hamDG.CalculatePseudoPotentialAA( ptable );
		
		// Compute the Hartree potential
		hamDG.CalculateHartree( fft );

		// Output the potential by master processor
		if(0)
		{
			MPI_Comm commMaster;
			bool isInGrid = (mpirank == 0) ? true : false;
			MPI_Comm_split( MPI_COMM_WORLD, isInGrid, mpirank, &commMaster );
			
			Int localNzStart, localNz;
			if( mpirank == 0 ){
				localNzStart = 0;
				localNz      = dm.numGrid[2];
			}
			
			DblNumVec localVec;
			DistNumVecToDistRowVec( 
					hamDG.Vhart(),
					localVec,
					dm.numGrid,
					esdfParam.numElem,
					localNzStart,
					localNz,
					isInGrid,
					MPI_COMM_WORLD);

			// Compare with Kohn-Sham
			if( mpirank == 0 ){
				Fourier seqfft;
				seqfft.Initialize( dm );
				KohnSham hamKS( esdfParam, 1 );
				hamKS.CalculatePseudoPotential( ptable ); 
				hamKS.CalculateHartree( seqfft );

				Real diff = 0.0;
				DblNumVec& val1 = hamKS.Vhart();
				DblNumVec& val2 = localVec;
				for( Int l = 0; l < val1.m(); l++ ){
					diff += pow(val1(l) - val2(l), 2.0);
				}
				diff = std::sqrt( diff );
				std::cout << "diff of vhart = " << diff << std::endl;

				{
					ofstream ofs("vhartDG");
					if( !ofs.good() )
						throw runtime_error("File cannot be opened.");
					serialize( localVec, ofs, NO_MASK );
					ofs.close();
				}
				{
					ofstream ofs("vhartKS");
					if( !ofs.good() )
						throw runtime_error("File cannot be opened.");
					serialize( hamKS.Vhart(), ofs, NO_MASK );
					ofs.close();
				}
			
			}
			
			MPI_Comm_free( &commMaster );
			MPI_Barrier( MPI_COMM_WORLD );

		}

		
		// Test the manipulation of basis functions
		if(0)
		{
			DblNumVec psi(esdfParam.numGridLGL.prod());
			SetValue( psi, 0.0 );
			psi[0] = 1.0;
			DblNumVec Dpsi(psi.m());
			SetValue( Dpsi, 0.0 );
			hamDG.DiffPsi( esdfParam.numGridLGL, 
					psi.Data(), Dpsi.Data(), 0);
			statusOFS << std::endl << "D[0] = " << Dpsi << std::endl;
			hamDG.DiffPsi( esdfParam.numGridLGL, 
					psi.Data(), Dpsi.Data(), 1);
			statusOFS << std::endl << "D[1] = " << Dpsi << std::endl;
			hamDG.DiffPsi( esdfParam.numGridLGL, 
					psi.Data(), Dpsi.Data(), 2);
			statusOFS << std::endl << "D[2] = " << Dpsi << std::endl;
		}

		// Test loading the basis functions from the old code
		if(0)
		{
			istringstream iss;
			SeparateRead("ALB", iss);
			std::vector<DblNumTns>  basis;
			{
				Index3  key;
				deserialize(key, iss, NO_MASK);
				deserialize(basis, iss, NO_MASK);
				if( basis.size() > 0 ){
					if( basis[0].Size() != esdfParam.numGridLGL.prod() ){
						throw std::runtime_error("Grid size does not match.");
					}
					if( hamDG.BasisLGL().Prtn().Owner(key) != mpirank ){
						throw std::runtime_error("Basis owner does not match.");
					}
				}
				Int numBasis = basis.size();
				DblNumMat psi(esdfParam.numGridLGL.prod(), numBasis);
				for( Int p = 0; p < numBasis; p++ ){
					blas::Copy( psi.m(), basis[p].Data(), 1, 
							psi.VecData(p), 1 );
				}
				hamDG.BasisLGL().LocalMap()[key] = psi;
			}
			// Dump out the basis for the first element
			if(0){
				Index3 key(0,0,0);
				if( hamDG.BasisLGL().Prtn().Owner(key) == mpirank ){
					ofstream ofs("ALB_NEW");
					if( !ofs.good() )
						throw runtime_error("File cannot be opened.");
					serialize( hamDG.BasisLGL().LocalMap()[key], ofs, NO_MASK );
					ofs.close();
				}
			}
			
			// Differentiation along three directions
			if(1){
				Index3 key(0,0,0);
				if( hamDG.BasisLGL().Prtn().Owner(key) == mpirank ){
					DblNumMat& psi = hamDG.BasisLGL().LocalMap()[key];
					DblNumMat  Dpsi(psi.m(), DIM);
					hamDG.DiffPsi( esdfParam.numGridLGL, 
							psi.VecData(0), Dpsi.VecData(0), 0 );
					hamDG.DiffPsi( esdfParam.numGridLGL, 
							psi.VecData(0), Dpsi.VecData(1), 1 );
					hamDG.DiffPsi( esdfParam.numGridLGL, 
							psi.VecData(0), Dpsi.VecData(2), 2 );

					ofstream ofs("ALB_DIFF");
					if( !ofs.good() )
						throw runtime_error("File cannot be opened.");
					serialize( Dpsi, ofs, NO_MASK );
					ofs.close();
				}
			}
		}

		// Testing the construction of the Hamiltonian matrix (dry run)
		if(1)
		{
			// Load basis
			{
				istringstream iss;
				SeparateRead("ALB", iss);
				std::vector<DblNumTns>  basis;
				Index3  key;
				deserialize(key, iss, NO_MASK);
				deserialize(basis, iss, NO_MASK);
				if( basis.size() > 0 ){
					if( basis[0].Size() != esdfParam.numGridLGL.prod() ){
						throw std::runtime_error("Grid size does not match.");
					}
					if( hamDG.BasisLGL().Prtn().Owner(key) != mpirank ){
						throw std::runtime_error("Basis owner does not match.");
					}
				}
				Int numBasis = basis.size();
				DblNumMat psi(esdfParam.numGridLGL.prod(), numBasis);
				for( Int p = 0; p < numBasis; p++ ){
					blas::Copy( psi.m(), basis[p].Data(), 1, 
							psi.VecData(p), 1 );
				}
				hamDG.BasisLGL().LocalMap()[key] = psi;
			}

			// Load potential
			{
				istringstream iss;
				SeparateRead("VTOT", iss);
				DblNumVec    vtot;

				Index3  key;
				deserialize(key, iss, NO_MASK);
				deserialize(vtot, iss, NO_MASK);
				if( vtot.Size() > 0 ){
					if( vtot.Size() != esdfParam.numGridLGL.prod() ){
						throw std::runtime_error("Grid size does not match.");
					}
					if( hamDG.VtotLGL().Prtn().Owner(key) != mpirank ){
						throw std::runtime_error("Vtot owner does not match.");
					}
				}
				hamDG.VtotLGL().LocalMap()[key] = vtot;
			}

			// Construct matrix

			hamDG.CalculateDGMatrix();
		}

		// Read the matrix generated by the old code and compare with HMat_
		if(1)
		{
			// Load matrix
			{
				istringstream iss;
				SeparateRead("DGMAT", iss);
				while(1){
					ElemMatKey  key;
					DblNumMat   matRead;
					deserialize(key, iss, NO_MASK);
					
					if( key.first == Index3(-1,-1,-1) )
					  break;

					deserialize( matRead, iss, NO_MASK );
					std::map<ElemMatKey, DblNumMat>::iterator 
						mi = hamDG.HMat().LocalMap().find( key );
					if( mi != hamDG.HMat().LocalMap().end() ){
						// Found the matrix block
						statusOFS << "Comparing block ( " << 
							key.first << " ), ( "<< key.second 
							<< " )" << std::endl;
							
						DblNumMat& mat = (*mi).second;
						if( mat.Size() != matRead.Size() ){
							std::ostringstream msg;
							msg << "The matrix size does not match." << std::endl
								<< "mat.m     = " << mat.m() << ", mat.n     = " << mat.n() << std::endl
								<< "matRead.m = " << matRead.m() << ", matRead.n = " <<
								matRead.n() << std::endl;
							
							throw std::runtime_error( msg.str().c_str() );
						}

//						statusOFS << "Mat     = " << mat << std::endl;
//						statusOFS << "MatRead = " << matRead << std::endl;


						DblNumMat error(mat);
						blas::Axpy( error.Size(), -1.0, matRead.Data(), 1,
								error.Data(), 1 ); 

						statusOFS << "L2 error = " << Energy(error) << std::endl << std::endl;
					}

				}
			}
		}

		// Output the pseudoCharge by master processor
//		{
//			MPI_Comm commMaster;
//			Int color = 0;
//			if( mpirank != 0 && mpirank != 1 )
//				color = 1;
//			MPI_Comm_split( MPI_COMM_WORLD, color, mpirank, &commMaster );
//			if( color != 0 )
//				commMaster = MPI_COMM_NULL;
//			
//			Int localNzStart, localNz;
//			if( mpirank == 0 ){
//				localNzStart = 0;
//				localNz      = dm.numGrid[2]/2;
//			}
//			if( mpirank == 1 ){
//				localNzStart = dm.numGrid[2] / 2;
//				localNz      = dm.numGrid[2] / 2;
//			}
//			
//			DblNumVec localVec;
//			DistNumVecToDistRowVec( 
//					hamDG.PseudoCharge(),
//					localVec,
//					dm.numGrid,
//					esdfParam.numElem,
//					localNzStart,
//					localNz,
//					MPI_COMM_WORLD,
//					commMaster);
//
//			DistDblNumVec  tt;
//			tt.Prtn() = hamDG.PseudoCharge().Prtn();
//
//			Index3 numElem = esdfParam.numElem;
//
//			DistRowVecToDistNumVec(
//					localVec,
//					tt,
//					dm.numGrid,
//					numElem,
//					localNzStart,
//					localNz,
//					MPI_COMM_WORLD,
//					commMaster);
//
//			for( Int k = 0; k < numElem[2]; k++ ){
//				for( Int j = 0; j < numElem[1]; j++ ){
//					for( Int i = 0; i < numElem[0]; i++ ){
//						Index3 key( i, j, k );
//						if( tt.Prtn().Owner( key ) == mpirank ){
//							DblNumVec& val1 = hamDG.PseudoCharge().LocalMap()[key];
//							DblNumVec& val2 = tt.LocalMap()[key];
//							Real diff = 0.0;
//							for( Int l = 0; l < val1.m(); l++ ){
//								cout << val1(l) - val2(l) << endl;
//								diff += pow(val1(l) - val2(l), 2.0);
//							}
//							diff = sqrt( diff );
//							std::cout << "Element " << key << ", diff = " << 
//								diff << std::endl;
//
//						}
//					}
//				}
//			} // for (k)
//
//
//			if( mpirank == 1 ){
//				ofstream ofs("pseudo");
//				if( !ofs.good() )
//					throw runtime_error("File cannot be opened.");
//				serialize( localVec, ofs, NO_MASK );
//				ofs.close();
//			}
//		}
		

		fftw_mpi_cleanup();

	}
	catch( std::exception& e )
	{
		std::cerr << " caught exception with message: "
			<< e.what() << std::endl;
#ifndef _RELEASE_
		DumpCallStack();
#endif
	}

	MPI_Finalize();

	return 0;
}
