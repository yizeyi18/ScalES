/// @file dgdft.cpp
/// @brief Main driver for DGDFT for self-consistent field iteration.
///
/// @author Lin Lin
/// @date 2013-02-11
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;


void Usage(){
  std::cout 
		<< "dgdft -in [inFile] -fftsize [fftsize]" << std::endl
		<< "in:             Input file (default: dgdft.in)" << std::endl
		<< "fftsize:        Number of procs used for distributed memory fft (default:mpisize)" << std::endl;
}


int main(int argc, char **argv) 
{
	MPI_Init(&argc, &argv);
	int mpirank, mpisize;
	MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
	MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
	Real timeSta, timeEnd;

	if( mpirank == 0 )
		Usage();

	try
	{
		// *********************************************************************
		// Input parameter
		// *********************************************************************
		
		// Initialize log file
		stringstream  ss;
		ss << "statfile." << mpirank;
		statusOFS.open( ss.str().c_str() );

		// Initialize FFTW
		fftw_mpi_init();

		// Initialize BLACS
		Int nprow, npcol;
		Int contxt;
		Cblacs_get(0, 0, &contxt);
		for( Int i = IRound(sqrt(double(mpisize))); i <= mpisize; i++){
			nprow = i; npcol = mpisize / nprow;
			if( nprow * npcol == mpisize ) break;
		} 
		Print( statusOFS, "nprow = ", nprow );
		Print( statusOFS, "npcol = ", npcol ); 
		Cblacs_gridinit(&contxt, "C", nprow, npcol);
	
		// Initialize input parameters
		std::map<std::string,std::string> options;
		OptionsCreate(argc, argv, options);

		Int distfftSize;
    if( options.find("-fftsize") != options.end() ){
			distfftSize = std::atoi(options["-fftsize"].c_str());
		}
		else{
			distfftSize = mpisize;
		}

		std::string inFile;                   
		if( options.find("-in") != options.end() ){ 
			inFile = options["-in"];
		}
		else{
			inFile = "dgdft.in";
		}


		// Read ESDF input file
		ESDFInputParam  esdfParam;

		ESDFReadInput( esdfParam, inFile.c_str() );

		// The number of grid points have already been adjusted in ESDFReadInput.
//		{
//			bool isGridAdjusted = false;
//			Index3& numGrid = esdfParam.domain.numGrid;
//			Index3& numElem = esdfParam.numElem;
//			for( Int d = 0; d < DIM; d++ ){
//				if( numGrid[d] % numElem[d] != 0 ){
//					numGrid[d] = IRound( (Real)numGrid[d] / numElem[d] ) * numElem[d];
//					isGridAdjusted = true;
//				}
//			}
//			if( isGridAdjusted ){
//				statusOFS << std::endl 
//					<< "Grid size is adjusted to be a multiple of the number of elements." 
//					<< std::endl;
//			}
//		}

		// Print the initial state
		{
			PrintBlock(statusOFS, "Basic information");

			Print(statusOFS, "Super cell        = ",  esdfParam.domain.length );
			Print(statusOFS, "Grid size         = ",  esdfParam.domain.numGrid ); 
			Print(statusOFS, "Mixing dimension  = ",  esdfParam.mixMaxDim );
			Print(statusOFS, "Mixing type       = ",  esdfParam.mixType );
			Print(statusOFS, "Mixing Steplength = ",  esdfParam.mixStepLength);
			Print(statusOFS, "SCF Outer Tol     = ",  esdfParam.scfOuterTolerance);
			Print(statusOFS, "SCF Outer MaxIter = ",  esdfParam.scfOuterMaxIter);
			Print(statusOFS, "SCF Inner Tol     = ",  esdfParam.scfInnerTolerance);
			Print(statusOFS, "SCF Inner MaxIter = ",  esdfParam.scfInnerMaxIter);
			Print(statusOFS, "Eig Tolerence     = ",  esdfParam.eigTolerance);
			Print(statusOFS, "Eig MaxIter       = ",  esdfParam.eigMaxIter);
			Print(statusOFS, "SVD Basis Tol     = ",  esdfParam.SVDBasisTolerance);

			Print(statusOFS, "RestartDensity    = ",  esdfParam.isRestartDensity);
			Print(statusOFS, "RestartWfn        = ",  esdfParam.isRestartWfn);
			Print(statusOFS, "OutputDensity     = ",  esdfParam.isOutputDensity);
			Print(statusOFS, "OutputWfn         = ",  esdfParam.isOutputWfn);
			Print(statusOFS, "Calculate A Posteriori error estimator at each step = ",  
					esdfParam.isCalculateAPosterioriEachSCF);
			Print(statusOFS, "OutputHMatrix     = ",  esdfParam.isOutputHMatrix );
			Print(statusOFS, "Barrier W         = ",  esdfParam.potentialBarrierW);
			Print(statusOFS, "Barrier S         = ",  esdfParam.potentialBarrierS);
			Print(statusOFS, "Barrier R         = ",  esdfParam.potentialBarrierR);
			Print(statusOFS, "EcutWavefunction  = ",  esdfParam.ecutWavefunction);
			Print(statusOFS, "Density GridFactor= ",  esdfParam.densityGridFactor);
			Print(statusOFS, "LGL GridFactor    = ",  esdfParam.LGLGridFactor);

			Print(statusOFS, "Temperature       = ",  au2K / esdfParam.Tbeta, "[K]");
			Print(statusOFS, "Extra states      = ",  esdfParam.numExtraState );
			Print(statusOFS, "PeriodTable File  = ",  esdfParam.periodTableFile );
			Print(statusOFS, "Pseudo Type       = ",  esdfParam.pseudoType );
			Print(statusOFS, "PW Solver         = ",  esdfParam.PWSolver );
			Print(statusOFS, "XC Type           = ",  esdfParam.XCType );

			Print(statusOFS, "Penalty Alpha     = ",  esdfParam.penaltyAlpha );
			Print(statusOFS, "Element size      = ",  esdfParam.numElem ); 
			Print(statusOFS, "Wfn Elem GridSize = ",  esdfParam.numGridWavefunctionElem ); 
			Print(statusOFS, "LGL Grid size     = ",  esdfParam.numGridLGL ); 
			Print(statusOFS, "ScaLAPACK block   = ",  esdfParam.scaBlockSize); 
			statusOFS << "Number of ALB for each element: " << std::endl 
				<< esdfParam.numALBElem << std::endl;


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
		SetRandomSeed(1);

		Domain&  dm = esdfParam.domain;
		PeriodTable ptable;
		ptable.Setup( esdfParam.periodTableFile );


		// Setup the element and extended element information
		DistVec<Index3, EigenSolver, ElemPrtn>  distEigSol; 
		DistVec<Index3, KohnSham, ElemPrtn>     distHamKS;
		DistVec<Index3, Spinor, ElemPrtn>       distPsi;
		// All extended elements share the same Fourier structure.
		Fourier fftExtElem;

		// Setup the eigenvalue solvers in each extended element
		{
			// Element partition information
			Index3  numElem = esdfParam.numElem;

			IntNumTns& elemPrtnInfo = distEigSol.Prtn().ownerInfo;
			elemPrtnInfo.Resize( numElem[0], numElem[1], numElem[2] );

			if( mpisize != numElem.prod() ){
				std::ostringstream msg;
				msg << "The number of processors is not equal to the total number of elements." << std::endl;
				throw std::runtime_error( msg.str().c_str() );
			}

			Int cnt = 0;
			for( Int k=0; k< numElem[2]; k++ )
				for( Int j=0; j< numElem[1]; j++ )
					for( Int i=0; i< numElem[0]; i++ ) {
						elemPrtnInfo(i,j,k) = cnt++;
					}

			distHamKS.Prtn() = distEigSol.Prtn();
			distPsi.Prtn()   = distEigSol.Prtn(); 


			for( Int k=0; k< numElem[2]; k++ )
				for( Int j=0; j< numElem[1]; j++ )
					for( Int i=0; i< numElem[0]; i++ ) {
						Index3 key (i,j,k);
						if( distEigSol.Prtn().Owner(key) == mpirank ){
							// Setup the domain in the extended element
							Domain dmExtElem;
							for( Int d = 0; d < DIM; d++ ){
								// Assume the global domain starts from 0.0
								if( numElem[d] == 1 ){
									dmExtElem.length[d]     = dm.length[d];
									dmExtElem.numGrid[d]    = esdfParam.numGridWavefunctionElem[d];
									dmExtElem.posStart[d]   = 0.0;
								}
								else if ( numElem[d] >= 3 ){
									dmExtElem.length[d]     = dm.length[d]  / numElem[d] * 3;
									dmExtElem.numGrid[d]    = esdfParam.numGridWavefunctionElem[d] * 3;
									dmExtElem.posStart[d]   = dm.length[d]  / numElem[d] * ( key[d] - 1 );
								}
								else{
									throw std::runtime_error( "numElem[d] is either 1 or >=3." );
								}

								// Do not specify the communicator for the domain yet
								// since it is not used for parallelization
							}
							
							// Atoms	
							std::vector<Atom>&  atomList = esdfParam.atomList;
							std::vector<Atom>   atomListExtElem;
							
							Int numAtom = atomList.size();
							for( Int a = 0; a < numAtom; a++ ){
								Point3 pos = atomList[a].pos;
								if( IsInSubdomain( pos, dmExtElem, dm.length ) ){
									// Update the coordinate relative to the extended
									// element
									for( Int d = 0; d < DIM; d++ ){
										pos[d] -= floor( ( pos[d] - dmExtElem.posStart[d] ) / 
											 dm.length[d] )* dm.length[d]; 
									}
									atomListExtElem.push_back( Atom( atomList[a].type,
												pos, atomList[a].vel, atomList[a].force ) );
								} // Atom is in the extended element
							}

							// Fourier
							fftExtElem.Initialize( dmExtElem );

							// Wavefunction
							Spinor& spn = distPsi.LocalMap()[key];
							spn.Setup( dmExtElem, 1, esdfParam.numALBElem(i,j,k), 0.0 );
							
							UniformRandom( spn.Wavefun() );

							// Hamiltonian
							// The exchange-correlation type and numExtraState is not
							// used in the extended element calculation
							statusOFS << "Hamiltonian begin." << std::endl;
							KohnSham& hamKS = distHamKS.LocalMap()[key];
							
							hamKS.Setup( dmExtElem, atomListExtElem, 
								 esdfParam.pseudoType, esdfParam.XCId );

							// Setup the external barrier potential in the extended element
							Real barrierR = esdfParam.potentialBarrierR;
							Real barrierW = esdfParam.potentialBarrierW;
							Real barrierS = esdfParam.potentialBarrierS;
							std::vector<DblNumVec> gridpos(DIM);
							UniformMesh ( dmExtElem, gridpos );
							// Barrier potential along each dimension
							std::vector<DblNumVec> vBarrier(DIM);

							for( Int d = 0; d < DIM; d++ ){
								Real length   = dmExtElem.length[d];
								Int numGrid   = dmExtElem.numGrid[d];
								Real posStart = dmExtElem.posStart[d]; 
								Real center   = posStart + length / 2.0;
								Real EPS      = 1.0;           // For stability reason
								Real dist;
								statusOFS << "center = " << center << std::endl;
								vBarrier[d].Resize( numGrid );
								SetValue( vBarrier[d], 0.0 );
								for( Int p = 0; p < numGrid; p++ ){
									dist = std::abs( gridpos[d][p] - center );
									// Only apply the barrier for region outside barrierR
									if( dist > barrierR ){
										vBarrier[d][p] = barrierS * std::exp( - barrierW / 
												( dist - barrierR ) ) / std::pow( dist - length / 2.0 - EPS, 2.0 );
									}
								}
							} // for (d)

#if 0
							statusOFS << "gridpos[0] = " << std::endl << gridpos[0] << std::endl;
							statusOFS << "gridpos[1] = " << std::endl << gridpos[1] << std::endl;
							statusOFS << "gridpos[2] = " << std::endl << gridpos[2] << std::endl;
							statusOFS << "vBarrier[0] = " << std::endl << vBarrier[0] << std::endl;
							statusOFS << "vBarrier[1] = " << std::endl << vBarrier[1] << std::endl;
							statusOFS << "vBarrier[2] = " << std::endl << vBarrier[2] << std::endl;
#endif

							DblNumVec& vext = hamKS.Vext();
							SetValue( vext, 0.0 );
#if 0
							for( Int gk = 0; gk < dmExtElem.numGrid[2]; gk++)
								for( Int gj = 0; gj < dmExtElem.numGrid[1]; gj++ )
									for( Int gi = 0; gi < dmExtElem.numGrid[0]; gi++ ){
										Int idx = gi + gj * dmExtElem.numGrid[0] + 
											gk * dmExtElem.numGrid[0] * dmExtElem.numGrid[1];
										vext[idx] = vBarrier[0][gi] + vBarrier[1][gj] + vBarrier[2][gk];
									} // for (gi)
#endif
							
							hamKS.CalculatePseudoPotential( ptable );

							statusOFS << "Hamiltonian constructed." << std::endl;
							

							// Eigensolver class
							EigenSolver& eigSol = distEigSol.LocalMap()[key];
							eigSol.Setup( esdfParam, 
									hamKS, 
									spn, 
									fftExtElem );

						} // own this element
					} // for(i)
		}


		DistFourier distfft;

		distfft.Initialize( dm, distfftSize );

		// Setup HamDG
		HamiltonianDG hamDG( esdfParam );

		hamDG.CalculatePseudoPotential( ptable );

		// Setup SCFDG
		SCFDG  scfDG;
		

		GetTime( timeSta );
		scfDG.Setup( esdfParam, hamDG, distEigSol, distfft, ptable, contxt );
		GetTime( timeEnd );
		statusOFS << "Time for setting up SCFDG is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;

		// *********************************************************************
		// Solve
		// *********************************************************************

		// Main SCF iteration
		GetTime( timeSta );
		scfDG.Iterate();
		GetTime( timeEnd );
		statusOFS << "Time for SCF iteration is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;

		// Compute force
		GetTime( timeSta );
		hamDG.CalculateForce( distfft );
		GetTime( timeEnd );
		statusOFS << "Time for computing the force is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;

		

		// Print out the force
		PrintBlock( statusOFS, "Force" );
		{
			std::vector<Atom>& atomList = hamDG.AtomList();
			Int numAtom = atomList.size();
			for( Int a = 0; a < numAtom; a++ ){
				Print( statusOFS, "Atom", a, "Force", atomList[a].force );
			}
			statusOFS << std::endl;
		}

		// Compute the a posteriori error estimator
		GetTime( timeSta );
		DblNumTns  eta2Total, eta2Residual, eta2GradJump, eta2Jump;
		hamDG.CalculateAPosterioriError( 
				eta2Total, eta2Residual, eta2GradJump, eta2Jump );
		GetTime( timeEnd );
		statusOFS << "Time for computing the a posteriori error is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;

		PrintBlock( statusOFS, "A Posteriori error" );
		{
			statusOFS << std::endl << "Total a posteriori error:" << std::endl;
			statusOFS << eta2Total << std::endl;
			statusOFS << std::endl << "Residual term:" << std::endl;
			statusOFS << eta2Residual << std::endl;
			statusOFS << std::endl << "Face term:" << std::endl;
			statusOFS << eta2GradJump << std::endl;
			statusOFS << std::endl << "Jump term:" << std::endl;
			statusOFS << eta2Jump << std::endl;
		}

		// *********************************************************************
		// Clean up
		// *********************************************************************

		// Finish Cblacs
		Cblacs_gridexit( contxt );

		// Finish fftw
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
