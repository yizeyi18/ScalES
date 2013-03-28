/// @file scf_dg.hpp
/// @brief Self consistent iteration using the DF method.
/// @author Lin Lin
/// @date 2013-02-05
#include  "scf_dg.hpp"
#include	"blas.hpp"
#include	"lapack.hpp"
#include  "utility.hpp"

#define _DEBUGlevel_ 0

namespace  dgdft{

using namespace dgdft::DensityComponent;


SCFDG::SCFDG	(  )
{
#ifndef _RELEASE_
	PushCallStack("SCFDG::SCFDG");
#endif

#ifndef _RELEASE_
	PopCallStack();
#endif
} 		// -----  end of method SCFDG::SCFDG  ----- 


SCFDG::~SCFDG	(  )
{
#ifndef _RELEASE_
	PushCallStack("SCFDG::~SCFDG");
#endif

#ifndef _RELEASE_
	PopCallStack();
#endif
} 		// -----  end of method SCFDG::~SCFDG  ----- 

void
SCFDG::Setup	( 
		const esdf::ESDFInputParam& esdfParam, 
		HamiltonianDG&              hamDG,
	  DistVec<Index3, EigenSolver, ElemPrtn>&  distEigSol,
		DistFourier&                distfft,
		PeriodTable&                ptable,
	  Int                         contxt	)
{
#ifndef _RELEASE_
	PushCallStack("SCFDG::Setup");
#endif
	Int mpirank, mpisize;
	MPI_Comm_rank( domain_.comm, &mpirank );
	MPI_Comm_size( domain_.comm, &mpisize );
	
	// esdf parameters
	{
		domain_        = esdfParam.domain;
    mixMaxDim_     = esdfParam.mixMaxDim;
    mixType_       = esdfParam.mixType;
		mixStepLength_ = esdfParam.mixStepLength;
		scfTolerance_  = esdfParam.scfTolerance;
		scfMaxIter_    = esdfParam.scfMaxIter;
		isRestartDensity_ = esdfParam.isRestartDensity;
		isRestartWfn_     = esdfParam.isRestartWfn;
		isOutputDensity_  = esdfParam.isOutputDensity;
		isOutputWfn_      = esdfParam.isOutputWfn;
    Tbeta_            = esdfParam.Tbeta;
		scaBlockSize_     = esdfParam.scaBlockSize;
		numALB_           = esdfParam.numALB;
		numElem_          = esdfParam.numElem;
	}

	// other SCFDG parameters
	{
		hamDGPtr_      = &hamDG;
		distEigSolPtr_ = &distEigSol;
		distfftPtr_    = &distfft;
    ptablePtr_     = &ptable;
		elemPrtn_      = distEigSol.Prtn();
		contxt_        = contxt;
		
		vtotNew_.Prtn() = elemPrtn_;
		dfMat_.Prtn()   = elemPrtn_;
		dvMat_.Prtn()   = elemPrtn_;

		// FIXME fixed ratio between the size of the extended element and
		// the element
		for( Int d = 0; d < DIM; d++ ){
			extElemRatio_[d] = ( numElem_[d]>1 ) ? 3 : 1;
		}

		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner( key ) == mpirank ){
						DblNumVec  emptyVec( hamDG.NumUniformGridElem().prod() );
						SetValue( emptyVec, 0.0 );
						vtotNew_.LocalMap()[key] = emptyVec;
						DblNumMat  emptyMat( hamDG.NumUniformGridElem().prod(), mixMaxDim_ );
						SetValue( emptyMat, 0.0 );
						dfMat_.LocalMap()[key]   = emptyMat;
						dvMat_.LocalMap()[key]   = emptyMat;
					} // own this element
				}  // for (i)
		
	
//		restartDensityFileName_ = "DEN";
//		restartWfnFileName_     = "WFN";
	}

	// Density
	{
		DistDblNumVec&  density = hamDGPtr_->Density();
		if( isRestartDensity_ ) {
			throw std::runtime_error("Cannot restart density now.");
//			std::istringstream rhoStream;      
//			SharedRead( restartDensityFileName_, rhoStream);
//			// TODO Error checking
//			deserialize( density, rhoStream, NO_MASK );    
		} // else using the zero initial guess
		else {
			// Initialize the electron density using the pseudocharge
			// make sure the pseudocharge is initialized
			DistDblNumVec& pseudoCharge = hamDGPtr_->PseudoCharge();

			ElemPrtn&  elemPrtn = pseudoCharge.Prtn();
			
      
			Real sumDensityLocal = 0.0, sumPseudoChargeLocal = 0.0;
			Real sumDensity, sumPseudoCharge;
			Real EPS = 1e-6;

			// make sure that the electron density is positive
			for( Int k = 0; k < numElem_[2]; k++ )
				for( Int j = 0; j < numElem_[1]; j++ )
					for( Int i = 0; i < numElem_[0]; i++ ){
						Index3 key( i, j, k );
						if( elemPrtn.Owner( key ) == mpirank ){
							DblNumVec&  denVec = density.LocalMap()[key];
							DblNumVec&  ppVec  = pseudoCharge.LocalMap()[key];
							for( Int p = 0; p < denVec.Size(); p++ ){
								denVec(p) = ( ppVec(p) > EPS ) ? ppVec(p) : EPS;
								sumDensityLocal += denVec(p);
								sumPseudoChargeLocal += ppVec(p);
							}
						}
					} // for (i)

			// Rescale the density
			mpi::Allreduce( &sumDensityLocal, &sumDensity, 1, MPI_SUM,
					domain_.comm );
			mpi::Allreduce( &sumPseudoChargeLocal, &sumPseudoCharge, 
					1, MPI_SUM, domain_.comm );

#if ( _DEBUGlevel_ >= 1 )
			Print( statusOFS, "Sum of initial density      = ", 
					sumDensity * domain_.Volume() / domain_.NumGridTotal() );
			Print( statusOFS, "Sum of pseudo charge        = ", 
					sumPseudoCharge * domain_.Volume() / domain_.NumGridTotal() );
#endif

			for( Int k = 0; k < numElem_[2]; k++ )
				for( Int j = 0; j < numElem_[1]; j++ )
					for( Int i = 0; i < numElem_[0]; i++ ){
						Index3 key( i, j, k );
						if( elemPrtn.Owner( key ) == mpirank ){
							DblNumVec&  denVec = density.LocalMap()[key];
							blas::Scal( denVec.Size(), sumPseudoCharge / sumDensity, 
									denVec.Data(), 1 );
						}
					} // for (i)
		} // Restart the density
	}


	// Generate the transfer matrix from the periodic uniform grid on each
	// extended element to LGL grid.  
	{
		PeriodicUniformToLGLMat_.resize(DIM);
		// FIXME
		EigenSolver& eigSol = (*distEigSol.LocalMap().begin()).second;
		Domain dmExtElem = eigSol.FFT().domain;
		Domain dmElem;
		for( Int d = 0; d < DIM; d++ ){
			dmElem.length[d]   = domain_.length[d] / numElem_[d];
			dmElem.numGrid[d]  = domain_.numGrid[d] / numElem_[d];
			// PosStart relative to the extended element
			dmElem.posStart[d] = ( numElem_[d] > 1 ) ? dmElem.length[d] : 0;
		}

		Index3 numLGL        = hamDG.NumLGLGridElem();
		Index3 numUniform    = dmExtElem.numGrid;
		Point3 lengthUniform = dmExtElem.length;

		std::vector<DblNumVec>  LGLGrid(DIM);
		LGLMesh( dmElem, numLGL, LGLGrid ); 

		for( Int d = 0; d < DIM; d++ ){
			DblNumMat&  localMat = PeriodicUniformToLGLMat_[d];
			localMat.Resize( numLGL[d], numUniform[d] );
			SetValue( localMat, 0.0 );
			Int maxK;
			if (numUniform[d] % 2 == 0)
				maxK = numUniform[d] / 2 - 1;
			else
				maxK = ( numUniform[d] - 1 ) / 2;
			for( Int j = 0; j < numUniform[d]; j++ )
				for( Int i = 0; i < numLGL[d]; i++ ){
					// 1.0 accounts for the k=0 mode
					localMat(i,j) = 1.0;
					for( Int k = 1; k < maxK; k++ ){
						localMat(i,j) += 2.0 * std::cos( 
								2 * PI * k / lengthUniform[d] * 
								( LGLGrid[d](i) - j * lengthUniform[d] / numUniform[d] ) );
					} // for (k)
					localMat(i,j) /= numUniform[d];
				} // for (i)
		} // for (d)

#if ( _DEBUGlevel_ >= 1 )
		statusOFS << "PeriodicUniformToLGLMat[0] = "
			<< PeriodicUniformToLGLMat_[0] << std::endl;
		statusOFS << "PeriodicUniformToLGLMat[1] = " 
			<< PeriodicUniformToLGLMat_[1] << std::endl;
		statusOFS << "PeriodicUniformToLGLMat[2] = "
			<< PeriodicUniformToLGLMat_[2] << std::endl;
#endif


	}

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCFDG::Setup  ----- 


void
SCFDG::Iterate	(  )
{
#ifndef _RELEASE_
	PushCallStack("SCFDG::Iterate");
#endif
	Int mpirank, mpisize;
	MPI_Comm_rank( domain_.comm, &mpirank );
	MPI_Comm_size( domain_.comm, &mpisize );

	Real timeSta, timeEnd;

  HamiltonianDG&  hamDG = *hamDGPtr_;

	// Compute the exchange-correlation potential and energy
	hamDG.CalculateXC( Exc_ );

	// Compute the Hartree energy
	hamDG.CalculateHartree( *distfftPtr_ );

	// No external potential

	// Compute the total potential
	hamDG.CalculateVtot( hamDG.Vtot() );

//	CalculateEnergy();
//
//	PrintState(0);


  Real timeIterStart(0), timeIterEnd(0);
  
	bool isSCFConverged = false;

  for (Int iter=1; iter <= scfMaxIter_; iter++) {
    if ( isSCFConverged ) break;
		
		// Performing each iteartion
		{
			std::ostringstream msg;
			msg << "SCF iteration # " << iter;
			PrintBlock( statusOFS, msg.str() );
		}

    GetTime( timeIterStart );
		
		// *********************************************************************
		// Update the potential in the extended element and the element
		// *********************************************************************

		{
			GetTime(timeSta);
			// vtot gather the neighborhood
			DistDblNumVec&  vtot = hamDG.Vtot();
			std::set<Index3> neighborSet;
			for( Int k = 0; k < numElem_[2]; k++ )
				for( Int j = 0; j < numElem_[1]; j++ )
					for( Int i = 0; i < numElem_[0]; i++ ){
						Index3 key( i, j, k );
						if( elemPrtn_.Owner(key) == mpirank ){
							std::vector<Index3>   idx(3);

							for( Int d = 0; d < DIM; d++ ){
								// Previous
								if( key[d] == 0 ) 
									idx[0][d] = numElem_[d]-1; 
								else 
									idx[0][d] = key[d]-1;

								// Current
								idx[1][d] = key[d];

								// Next
								if( key[d] == numElem_[d]-1) 
									idx[2][d] = 0;
								else
									idx[2][d] = key[d] + 1;
							} // for (d)

							// Tensor product 
							for( Int c = 0; c < 3; c++ )
								for( Int b = 0; b < 3; b++ )
									for( Int a = 0; a < 3; a++ ){
										// Not the element key itself
										if( idx[a][0] != i || idx[b][1] != j || idx[c][2] != k ){
											neighborSet.insert( Index3( idx[a][0], idx[b][1], idx[c][2] ) );
										}
									} // for (a)
						} // own this element
					} // for (i)
			std::vector<Index3>  neighborIdx;
			neighborIdx.insert( neighborIdx.begin(), neighborSet.begin(), neighborSet.end() );

#if ( _DEBUGlevel_ >= 1 )
			statusOFS << "neighborIdx = " << neighborIdx << std::endl;
#endif

			// communicate
			vtot.GetBegin( neighborIdx, NO_MASK );
			vtot.GetEnd( NO_MASK );


			// Update of the local potential in each extended element locally.
			// The nonlocal potential does not need to be updated
			//
			// Also update the local potential on the LGL grid in hamDG.
			//
			// NOTE:
			//
			// 1. It is hard coded that the extended element is 1 or 3
			// times the size of the element
			//
			// 2. The local potential on the LGL grid is done by using Fourier
			// interpolation from the extended element to the element. Gibbs
			// phenomena MAY be there but at least this is better than
			// Lagrange interpolation on a uniform grid.
			//  
			//
			for( Int k = 0; k < numElem_[2]; k++ )
				for( Int j = 0; j < numElem_[1]; j++ )
					for( Int i = 0; i < numElem_[0]; i++ ){
						Index3 key( i, j, k );
						if( elemPrtn_.Owner( key ) == mpirank ){
              EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
							Hamiltonian&  hamExtElem  = eigSol.Ham();
							DblNumVec&    vtotExtElem = hamExtElem.Vtot();
							SetValue( vtotExtElem, 0.0 );

							Index3 numGridElem = hamDG.NumUniformGridElem();
							Index3 numGridExtElem = eigSol.FFT().domain.numGrid;
							
							// Update the potential in the extended element
							for(std::map<Index3, DblNumVec>::iterator 
									mi = vtot.LocalMap().begin();
									mi != vtot.LocalMap().end(); mi++ ){
								Index3      keyElem = (*mi).first;
								DblNumVec&  vtotElem = (*mi).second;

								// Determine the shiftIdx which maps the position of vtotElem to 
								// vtotExtElem
								Index3 shiftIdx;
								for( Int d = 0; d < DIM; d++ ){
									shiftIdx[d] = keyElem[d] - key[d];
									shiftIdx[d] = shiftIdx[d] - IRound( Real(shiftIdx[d]) / 
											numElem_[d] ) * numElem_[d];
									// FIXME Adjustment  
									if( numElem_[d] > 1 ) shiftIdx[d] ++;

									shiftIdx[d] *= numGridElem[d];
								}

#if ( _DEBUGlevel_ >= 1 )
								statusOFS << "keyExtElem     = " << key << std::endl;
								statusOFS << "numGridExtElem = " << numGridExtElem << std::endl;
								statusOFS << "numGridElem    = " << numGridElem << std::endl;
								statusOFS << "keyElem        = " << keyElem << ", shiftIdx = " << shiftIdx << std::endl;
#endif

								Int ptrExtElem, ptrElem;
								for( Int k = 0; k < numGridElem[2]; k++ )
									for( Int j = 0; j < numGridElem[1]; j++ )
										for( Int i = 0; i < numGridElem[0]; i++ ){
											ptrExtElem = (shiftIdx[0] + i) + 
												( shiftIdx[1] + j ) * numGridExtElem[0] +
												( shiftIdx[2] + k ) * numGridExtElem[0] * numGridExtElem[1];
											ptrElem    = i + j * numGridElem[0] + 
												k * numGridElem[0] * numGridElem[1];
											vtotExtElem( ptrExtElem ) = vtotElem( ptrElem );
										} // for (i)
							} // for (mi)

							// Update the potential in the element on LGL grid
							DblNumVec&  vtotLGLElem = hamDG.VtotLGL().LocalMap()[key];
							Index3 numLGLGrid       = hamDG.NumLGLGridElem();

							InterpPeriodicUniformToLGL( 
									numGridExtElem,
									numLGLGrid,
									vtotExtElem.Data(),
									vtotLGLElem.Data() );

							// Loop over the neighborhood

						} // own this element
					} // for (i)

			// Clean up vtot not owned by this element
			std::vector<Index3>  eraseKey;
			for( std::map<Index3, DblNumVec>::iterator 
					mi  = vtot.LocalMap().begin();
					mi != vtot.LocalMap().end(); mi++ ){
				Index3 key = (*mi).first;
				if( vtot.Prtn().Owner(key) != mpirank ){
					eraseKey.push_back( key );
				}
			}
			for( std::vector<Index3>::iterator vi = eraseKey.begin();
					vi != eraseKey.end(); vi++ ){
				vtot.LocalMap().erase( *vi );
			}

			MPI_Barrier( domain_.comm );
			GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
			statusOFS << "Time for updating the potential is " <<
				timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
		}

		// *********************************************************************
		// Solve the basis functions in the extended element
		// *********************************************************************

		Real timeBasisSta, timeBasisEnd;
		GetTime(timeBasisSta);
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner( key ) == mpirank ){
						EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
						DblNumVec&    vtotExtElem = eigSol.Ham().Vtot();
						Index3 numGridExtElem = eigSol.FFT().domain.numGrid;
						Index3 numLGLGrid     = hamDG.NumLGLGridElem();

						// Solve the basis functions in the extended element
						GetTime( timeSta );
						eigSol.Solve();
						GetTime( timeEnd );
						statusOFS << "Eigensolver time = " 	<< timeEnd - timeSta
							<< " [sec]" << std::endl;

						// Print out the information
						statusOFS << std::endl 
							<< "ALB calculation in extended element " << key << std::endl;
						for(Int i = 0; i < eigSol.EigVal().m(); i++){
							Print(statusOFS, 
									"basis#   = ", i, 
									"eigval   = ", eigSol.EigVal()(i),
									"resval   = ", eigSol.ResVal()(i));
						}
						statusOFS << std::endl;


						GetTime( timeSta );
						Spinor& psi = eigSol.Psi();

						// Assuming that wavefun has only 1 component
						DblNumTns& wavefun = psi.Wavefun();

						DblNumMat localBasis( 
								numLGLGrid.prod(), 
								psi.NumState() );

						SetValue( localBasis, 0.0 );

//#pragma omp parallel for
						for( Int l = 0; l < psi.NumState(); l++ ){
							InterpPeriodicUniformToLGL( 
									numGridExtElem,
									numLGLGrid,
									wavefun.VecData(0, l), 
									localBasis.VecData(l) );
						}

						GetTime( timeEnd );
						statusOFS << "Time for interpolating basis = " 	<< timeEnd - timeSta
							<< " [sec]" << std::endl;

						// FIXME
//						if( mpirank == 1 ){
//							std::ofstream ofs("psi");
//							serialize( DblNumVec(localBasis.m(), false, localBasis.VecData(5)),
//									ofs, NO_MASK );
//							ofs.close();
//						}
						

						// Perform SVD for the basis functions
						GetTime( timeSta );
						{
							// Compute the LGL weights
							std::vector<DblNumVec>  LGLWeight1D(DIM);
							Point3                  lengthLGL;
							for( Int d = 0; d < DIM; d++ ){
								lengthLGL[d] = domain_.length[d] / numElem_[d];
								DblNumVec  dummyX;
								DblNumMat  dummyP, dummpD;
								GenerateLGL( dummyX, LGLWeight1D[d], dummyP, dummpD, 
										numLGLGrid[d] );
								blas::Scal( numLGLGrid[d], 0.5 * lengthLGL[d], 
										LGLWeight1D[d].Data(), 1 );
							}
							DblNumTns    sqrtLGLWeight3D( numLGLGrid[0], numLGLGrid[1], numLGLGrid[2] );
							for( Int k1 = 0; k1 < numLGLGrid[2]; k1++ )
								for( Int j1 = 0; j1 < numLGLGrid[1]; j1++ )
									for( Int i1 = 0; i1 < numLGLGrid[0]; i1++ ){
										sqrtLGLWeight3D(i1, j1, k1) = 
											std::sqrt ( LGLWeight1D[0](i1) *
													LGLWeight1D[1](j1) * LGLWeight1D[2](k1) ); }
							// for (i1)
							
							// Scale the basis functions by sqrt of integration weight
//#pragma omp parallel for 
							for( Int g = 0; g < localBasis.n(); g++ ){
								Real *ptr1 = localBasis.VecData(g);
								Real *ptr2 = sqrtLGLWeight3D.Data();
								for( Int l = 0; l < localBasis.m(); l++ ){
									*(ptr1++)  *= *(ptr2++);
								}
							}

							DblNumMat    U( localBasis.m(), localBasis.n() );
							DblNumMat   VT( localBasis.n(), localBasis.n() );
							DblNumVec    S( localBasis.n() );


							lapack::QRSVD( localBasis.m(), localBasis.n(), 
									localBasis.Data(), localBasis.m(),
									S.Data(), U.Data(), U.m(), VT.Data(), VT.m() );

							statusOFS << "Singular values of the basis = " 
								<< S << std::endl;

							// Unscale the orthogonal basis functions by sqrt of
							// integration weight
#pragma omp parallel for schedule(dynamic,1) 
							for( Int g = 0; g < localBasis.n(); g++ ){
								Real *ptr1 = U.VecData(g);
								Real *ptr2 = sqrtLGLWeight3D.Data();
								for( Int l = 0; l < localBasis.m(); l++ ){
									*(ptr1++)  /= *(ptr2++);
								}
							}


							// FIXME No SVD truncation
							hamDG.BasisLGL().LocalMap()[key] = U;  
						}
						GetTime( timeEnd );
						statusOFS << "Time for SVD of basis = " 	<< timeEnd - timeSta
							<< " [sec]" << std::endl;

					} // own this element
				} // for (i)
		MPI_Barrier( domain_.comm );
		GetTime( timeBasisEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Total time for generating the adaptive local basis function is " <<
			timeBasisEnd - timeBasisSta << " [s]" << std::endl << std::endl;
#endif


		// *********************************************************************
		// Assemble the DG matrix
		// *********************************************************************

		GetTime(timeSta);
		hamDG.CalculateDGMatrix( );
		MPI_Barrier( domain_.comm );
		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for constructing the DG matrix is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


		// *********************************************************************
		// Diagonalize the DG matrix
		// *********************************************************************

		{
			GetTime(timeSta);
			Int sizeH = hamDG.NumBasisTotal();

			scalapack::Descriptor descH( sizeH, sizeH, scaBlockSize_, scaBlockSize_, 
					0, 0, contxt_ );

			scalapack::ScaLAPACKMatrix<Real>  scaH, scaZ;
			
			std::vector<Real> eigs;

			DistElemMatToScaMat( hamDG.HMat(), 	descH,
					scaH, hamDG.ElemBasisIdx(), domain_.comm );

			scalapack::Syevd('U', scaH, eigs, scaZ);

			DblNumVec& eigval = hamDG.EigVal(); 
      eigval.Resize( hamDG.NumStateTotal() );		
			for( Int i = 0; i < hamDG.NumStateTotal(); i++ )
				eigval[i] = eigs[i];
			
			ScaMatToDistNumMat( scaZ, hamDG.Density().Prtn(), 
					hamDG.EigvecCoef(), hamDG.ElemBasisIdx(), domain_.comm, 
					hamDG.NumStateTotal() );
		
			MPI_Barrier( domain_.comm );
			GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
			statusOFS << "Time for diagonalizing the DG matrix using ScaLAPACK is " <<
				timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
		}


		// *********************************************************************
		// Post processing
		// *********************************************************************
		
		GetTime(timeSta);

		// Compute the occupation rate
		CalculateOccupationRate( hamDG.EigVal(), hamDG.OccupationRate() );

		// Compute the electron density
		hamDG.CalculateDensity( hamDG.OccupationRate() );

		MPI_Barrier( domain_.comm );
		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for computing density in the global domain is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

		GetTime(timeSta);

		// Compute the exchange-correlation potential and energy
		hamDG.CalculateXC( Exc_ );

		// Compute the Hartree energy
		hamDG.CalculateHartree( *distfftPtr_ );

		// No external potential

		// Compute the new total potential
		hamDG.CalculateVtot( vtotNew_ );

		// Compute the error of the potential
		{
			Real normVtotDifLocal = 0.0, normVtotOldLocal = 0.0;
			Real normVtotDif, normVtotOld;
			for( Int k = 0; k < numElem_[2]; k++ )
				for( Int j = 0; j < numElem_[1]; j++ )
					for( Int i = 0; i < numElem_[0]; i++ ){
						Index3 key( i, j, k );
						if( elemPrtn_.Owner( key ) == mpirank ){
							DblNumVec& oldVec = hamDG.Vtot().LocalMap()[key];
							DblNumVec& newVec = vtotNew_.LocalMap()[key];

							for( Int p = 0; p < oldVec.m(); p++ ){
								normVtotDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
								normVtotOldLocal += pow( oldVec(p), 2.0 );
							}
						} // own this element
					} // for (i)


			mpi::Allreduce( &normVtotDifLocal, &normVtotDif, 1, MPI_SUM, 
					domain_.comm );
			mpi::Allreduce( &normVtotOldLocal, &normVtotOld, 1, MPI_SUM,
					domain_.comm );

			normVtotDif = std::sqrt( normVtotDif );
			normVtotOld = std::sqrt( normVtotOld );

			scfNorm_    = normVtotDif / normVtotOld;
		}
		
		// Compute the energies
    CalculateEnergy();

		// Print out the state variables of the current iteration
    PrintState( iter );


    if( scfNorm_ < scfTolerance_ ){
      /* converged */
      Print( statusOFS, "SCF is converged!\n" );
      isSCFConverged = true;
    }

		MPI_Barrier( domain_.comm );
		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for post processing is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


		// Potential mixing
		GetTime( timeSta );
    if( mixType_ == "anderson" ){
      AndersonMix(iter);
    }
    if( mixType_ == "kerker" ){
      KerkerMix();  
      AndersonMix(iter);
    }

		MPI_Barrier( domain_.comm );
		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for potential mixing is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


		GetTime( timeIterEnd );
   
		statusOFS << "Total wall clock time for this SCF iteration = " << timeIterEnd - timeIterStart
			<< " [sec]" << std::endl;
  }

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCFDG::Iterate  ----- 

void
SCFDG::CalculateOccupationRate	( DblNumVec& eigVal, DblNumVec& occupationRate )
{
#ifndef _RELEASE_
	PushCallStack("SCFDG::CalculateOccupationRate");
#endif
	// For a given finite temperature, update the occupation number */
	// FIXME Magic number here
	Real tol = 1e-10; 
	Int maxiter = 100;  

	Real lb, ub, flb, fub, occsum;
	Int ilb, iub, iter;

	Int npsi       = hamDGPtr_->NumStateTotal();
	Int nOccStates = hamDGPtr_->NumOccupiedState();

	if( eigVal.m() != npsi ){
		std::ostringstream msg;
		msg 
			<< "The number of eigenstates do not match."  << std::endl
			<< "eigVal         ~ " << eigVal.m() << std::endl
			<< "numStateTotal  ~ " << npsi << std::endl;
		throw std::logic_error( msg.str().c_str() );
	}


	if( occupationRate.m() != npsi ) occupationRate.Resize( npsi );

	if( npsi > nOccStates )  {
		/* use bisection to find efermi such that 
		 * sum_i fermidirac(ev(i)) = nocc
		 */
		ilb = nOccStates-1;
		iub = nOccStates+1;

		lb = eigVal(ilb-1);
		ub = eigVal(iub-1);

		/* Calculate Fermi-Dirac function and make sure that
		 * flb < nocc and fub > nocc
		 */

		flb = 0.0;
		fub = 0.0;
		for(Int j = 0; j < npsi; j++) {
			flb += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-lb)));
			fub += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-ub))); 
		}

		while( (nOccStates-flb)*(fub-nOccStates) < 0 ) {
			if( flb > nOccStates ) {
				if(ilb > 0){
					ilb--;
					lb = eigVal(ilb-1);
					flb = 0.0;
					for(Int j = 0; j < npsi; j++) flb += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-lb)));
				}
				else {
					throw std::logic_error( "Cannot find a lower bound for efermi" );
				}
			}

			if( fub < nOccStates ) {
				if( iub < npsi ) {
					iub++;
					ub = eigVal(iub-1);
					fub = 0.0;
					for(Int j = 0; j < npsi; j++) fub += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-ub)));
				}
				else {
					throw std::logic_error( "Cannot find a lower bound for efermi, try to increase the number of wavefunctions" );
				}
			}
		}  /* end while */

		fermi_ = (lb+ub)*0.5;
		occsum = 0.0;
		for(Int j = 0; j < npsi; j++) {
			occupationRate(j) = 1.0 / (1.0 + exp(Tbeta_*(eigVal(j) - fermi_)));
			occsum += occupationRate(j);
		}

		/* Start bisection iteration */
		iter = 1;
		while( (fabs(occsum - nOccStates) > tol) && (iter < maxiter) ) {
			if( occsum < nOccStates ) {lb = fermi_;}
			else {ub = fermi_;}

			fermi_ = (lb+ub)*0.5;
			occsum = 0.0;
			for(Int j = 0; j < npsi; j++) {
				occupationRate(j) = 1.0 / (1.0 + exp(Tbeta_*(eigVal(j) - fermi_)));
				occsum += occupationRate(j);
			}
			iter++;
		}
	}
	else {
		if (npsi == nOccStates ) {
			for(Int j = 0; j < npsi; j++) 
				occupationRate(j) = 1.0;
			fermi_ = eigVal(npsi-1);
		}
		else {
			throw std::logic_error( "The number of eigenvalues in ev should be larger than nocc" );
		}
	}

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCFDG::CalculateOccupationRate  ----- 



void
SCFDG::InterpPeriodicUniformToLGL	( 
		const Index3& numUniformGrid, 
		const Index3& numLGLGrid, 
		const Real*   psiUniform, 
		Real*         psiLGL )
{
#ifndef _RELEASE_
	PushCallStack("SCFDG::InterpPeriodicUniformToLGL");
#endif

	Index3 Ns1 = numUniformGrid;
	Index3 Ns2 = numLGLGrid;
	
	DblNumVec  tmp1( Ns2[0] * Ns1[1] * Ns1[2] );
	DblNumVec  tmp2( Ns2[0] * Ns2[1] * Ns1[2] );
	SetValue( tmp1, 0.0 );
	SetValue( tmp2, 0.0 );

	// x-direction, use Gemm
	{
		Int m = Ns2[0], n = Ns1[1] * Ns1[2], k = Ns1[0];
		blas::Gemm( 'N', 'N', m, n, k, 1.0, PeriodicUniformToLGLMat_[0].Data(),
				m, psiUniform, k, 0.0, tmp1.Data(), m );
	}
	
	// y-direction, use Gemv
	{
		Int   m = Ns2[1], n = Ns1[1];
		Int   ptrShift1, ptrShift2;
		Int   inc = Ns2[0];
		for( Int k = 0; k < Ns1[2]; k++ ){
			for( Int i = 0; i < Ns2[0]; i++ ){
				ptrShift1 = i + k * Ns2[0] * Ns1[1];
				ptrShift2 = i + k * Ns2[0] * Ns2[1];
				blas::Gemv( 'N', m, n, 1.0, 
						PeriodicUniformToLGLMat_[1].Data(), m, 
						tmp1.Data() + ptrShift1, inc, 0.0, 
						tmp2.Data() + ptrShift2, inc );
			} // for (i)
		} // for (k)
	}

	
	// z-direction, use Gemm
	{
		Int m = Ns2[0] * Ns2[1], n = Ns2[2], k = Ns1[2]; 
		blas::Gemm( 'N', 'T', m, n, k, 1.0, 
				tmp2.Data(), m, 
				PeriodicUniformToLGLMat_[2].Data(), n, 0.0, psiLGL, m );
	}

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCFDG::InterpPeriodicUniformToLGL  ----- 

void
SCFDG::CalculateEnergy	(  )
{
#ifndef _RELEASE_
	PushCallStack("SCFDG::CalculateEnergy");
#endif
	Int mpirank, mpisize;
	MPI_Comm_rank( domain_.comm, &mpirank );
	MPI_Comm_size( domain_.comm, &mpisize );

  HamiltonianDG&  hamDG = *hamDGPtr_;

	DblNumVec&  eigVal         = hamDG.EigVal();
	DblNumVec&  occupationRate = hamDG.OccupationRate();

	// Kinetic energy
	Int numSpin = hamDG.NumSpin();
	Ekin_ = 0.0;
	for (Int i=0; i < eigVal.m(); i++) {
		Ekin_  += numSpin * eigVal(i) * occupationRate(i);
	}

	// Hartree and xc part
	Ehart_ = 0.0;
	EVxc_  = 0.0;

	Real EhartLocal = 0.0, EVxcLocal = 0.0;
	
	for( Int k = 0; k < numElem_[2]; k++ )
		for( Int j = 0; j < numElem_[1]; j++ )
			for( Int i = 0; i < numElem_[0]; i++ ){
				Index3 key( i, j, k );
				if( elemPrtn_.Owner( key ) == mpirank ){
					DblNumVec&  density      = hamDG.Density().LocalMap()[key];
					DblNumVec&  vxc          = hamDG.Vxc().LocalMap()[key];
					DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
					DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

					for (Int p=0; p < density.Size(); p++) {
						EVxcLocal  += vxc(p) * density(p);
						EhartLocal += 0.5 * vhart(p) * ( density(p) + pseudoCharge(p) );
					}

				} // own this element
			} // for (i)

	mpi::Allreduce( &EVxcLocal, &EVxc_, 1, MPI_SUM, domain_.comm );
	mpi::Allreduce( &EhartLocal, &Ehart_, 1, MPI_SUM, domain_.comm );

	Ehart_ *= domain_.Volume() / domain_.NumGridTotal();
	EVxc_  *= domain_.Volume() / domain_.NumGridTotal();

	// Self energy part
	Eself_ = 0.0;
	std::vector<Atom>&  atomList = hamDG.AtomList();
	for(Int a=0; a< atomList.size() ; a++) {
		Int type = atomList[a].type;
		Eself_ +=  ptablePtr_->ptemap()[type].params(PTParam::ESELF);
	}

	// Correction energy
	Ecor_   = (Exc_ - EVxc_) - Ehart_ - Eself_;

	// Total energy
	Etot_ = Ekin_ + Ecor_;

	// Helmholtz fre energy
	if( hamDG.NumOccupiedState() == 
			hamDG.NumStateTotal() ){
		// Zero temperature
		Efree_ = Etot_;
	}
	else{
		// Finite temperature
		Efree_ = 0.0;
		Real fermi = fermi_;
		Real Tbeta = Tbeta_;
		for(Int l=0; l< eigVal.m(); l++) {
			Real eig = eigVal(l);
			if( eig - fermi >= 0){
				Efree_ += -numSpin /Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
			}
			else{
				Efree_ += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
			}
		}
		Efree_ += Ecor_ + fermi * hamDG.NumOccupiedState() * numSpin; 
	}


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCFDG::CalculateEnergy  ----- 

void
SCFDG::AndersonMix	( const Int iter )
{
#ifndef _RELEASE_
	PushCallStack("SCFDG::AndersonMix");
#endif
	Int mpirank, mpisize;
	MPI_Comm_rank( domain_.comm, &mpirank );
	MPI_Comm_size( domain_.comm, &mpisize );
	
	DistDblNumVec distvin, distvout, distvinsave, distvoutsave;

	Int ntot  = hamDGPtr_->NumUniformGridElem().prod();
	
	DistDblNumVec&  distvtot = hamDGPtr_->Vtot();
	DistDblNumVec&  distvtotnew = vtotNew_;

	Int iterused = std::min( iter-1, mixMaxDim_ ); // iter should start from 1
	Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;


	for( Int k = 0; k < numElem_[2]; k++ )
		for( Int j = 0; j < numElem_[1]; j++ )
			for( Int i = 0; i < numElem_[0]; i++ ){
				Index3 key( i, j, k );
				if( elemPrtn_.Owner( key ) == mpirank ){
					DblNumVec& vin        = distvin.LocalMap()[key];
					DblNumVec& vout       = distvout.LocalMap()[key];
					DblNumVec& vinsave    = distvinsave.LocalMap()[key];
					DblNumVec& voutsave   = distvoutsave.LocalMap()[key];
					DblNumMat& df         = dfMat_.LocalMap()[key];
					DblNumMat& dv         = dvMat_.LocalMap()[key];

					// vin(:)  = vtot(:)
					// vout(:) = vtotnew(:) - vtot(:)
					vin  = distvtot.LocalMap()[key];
					vout = distvtotnew.LocalMap()[key];
					blas::Axpy( ntot, -1.0, vin.Data(), 1, vout.Data(), 1 );

					// save vin and vout
					vinsave  = vin;
					voutsave = vout;

				  // dfMat_(:, ipos-1) -= vout(:);
				  // dvMat_(:, ipos-1) -= vin(:);
					if( iter > 1 ){
						blas::Axpy( ntot, -1.0, vout.Data(), 1, df.VecData(ipos-1), 1);
						blas::Axpy( ntot, -1.0, vin.Data(),  1, dv.VecData(ipos-1), 1);
					}
				} // own this element
			} // for (i)



	if( iter > 1 ){

		Int nrow = iterused;

		// Normal matrix FTF = F^T * F
		DblNumMat FTFLocal( nrow, nrow ), FTF( nrow, nrow );
		SetValue( FTFLocal, 0.0 );
		SetValue( FTF, 0.0 );

		// Right hand side FTv = F^T * vout
		DblNumVec FTvLocal( nrow ), FTv( nrow );
		SetValue( FTvLocal, 0.0 );
		SetValue( FTv, 0.0 );

		// Local construction of FTF and FTv
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner( key ) == mpirank ){
						DblNumMat& df     = dfMat_.LocalMap()[key];
						DblNumVec& vout   = distvout.LocalMap()[key];
						for( Int q = 0; q < nrow; q++ ){
							FTvLocal(q) += blas::Dot( ntot, df.VecData(q), 1,
									vout.Data(), 1 );

							for( Int p = q; p < nrow; p++ ){
								FTFLocal(p, q) += blas::Dot( ntot, df.VecData(p), 1, 
										df.VecData(q), 1 );
								if( p > q )
									FTFLocal(q,p) = FTFLocal(p,q);
							} // for (p)
						} // for (q)

					} // own this element
				} // for (i)
		
		// Reduce the data
		mpi::Allreduce( FTFLocal.Data(), FTF.Data(), nrow * nrow, MPI_SUM, domain_.comm );
		mpi::Allreduce( FTvLocal.Data(), FTv.Data(), nrow, MPI_SUM, domain_.comm );

		// All processors solve the least square problem

		// FIXME Magic number
		Real rcond = 1e-6;
		Int rank;

		DblNumVec  S( nrow );

		// FTv = pinv( FTF ) * vout
		lapack::SVDLeastSquare( nrow, nrow, 1, 
				FTF.Data(), nrow, FTv.Data(), nrow,
        S.Data(), rcond, &rank );

		Print( statusOFS, "  Rank of dfmat = ", rank );
			

		// Update vin, vout
		// vin  = vin  - dv * FTv
		// vout = vout - df * FTv
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner( key ) == mpirank ){
						DblNumVec& vin    = distvin.LocalMap()[key];
						DblNumVec& vout   = distvout.LocalMap()[key];
						DblNumMat& df     = dfMat_.LocalMap()[key];
						DblNumMat& dv     = dvMat_.LocalMap()[key];
						
						blas::Gemv('N', ntot, nrow, -1.0, dv.Data(),
								ntot, FTv.Data(), 1, 1.0, vin.Data(), 1 );
						blas::Gemv('N', ntot, nrow, -1.0, df.Data(),
								ntot, FTv.Data(), 1, 1.0, vout.Data(), 1 );
					} // own this element
				} // for (i)
	} // (iter > 1)

	Int inext = iter - ((iter-1)/ mixMaxDim_) * mixMaxDim_;

	// Update df, dv, vtot
	for( Int k = 0; k < numElem_[2]; k++ )
		for( Int j = 0; j < numElem_[1]; j++ )
			for( Int i = 0; i < numElem_[0]; i++ ){
				Index3 key( i, j, k );
				if( elemPrtn_.Owner( key ) == mpirank ){
					DblNumVec& vtot       = distvtot.LocalMap()[key];
					DblNumVec& vin        = distvin.LocalMap()[key];
					DblNumVec& vout       = distvout.LocalMap()[key];
					DblNumVec& vinsave    = distvinsave.LocalMap()[key];
					DblNumVec& voutsave   = distvoutsave.LocalMap()[key];
					DblNumMat& df         = dfMat_.LocalMap()[key];
					DblNumMat& dv         = dvMat_.LocalMap()[key];

					// dfMat_(:, inext-1) = voutsave(:)
					// dvMat_(:, inext-1) = vinsave(:)
					blas::Copy( ntot, voutsave.Data(), 1, df.VecData(inext-1), 1 );
					blas::Copy( ntot, vinsave.Data(),  1, dv.VecData(inext-1), 1 );

					// vtot(:) = vin(:) + alpha * vout(:)
					blas::Copy( ntot, vin.Data(), 1, vtot.Data(), 1 );
					blas::Axpy( ntot, mixStepLength_, vout.Data(), 1, vtot.Data(), 1);
				} // own this element
			} // for (i)

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCFDG::AndersonMix  ----- 

void
SCFDG::KerkerMix	(  )
{
#ifndef _RELEASE_
	PushCallStack("SCFDG::KerkerMix");
#endif
	Int mpirank, mpisize;
	MPI_Comm_rank( domain_.comm, &mpirank );
	MPI_Comm_size( domain_.comm, &mpisize );

	DistFourier& fft = *distfftPtr_;

  Int ntot      = fft.numGridTotal;
	Int ntotLocal = fft.numGridLocal;

	DistDblNumVec   tempVec;
	tempVec.Prtn() = elemPrtn_;

	DistDblNumVec&  distvtot    = hamDGPtr_->Vtot();
	DistDblNumVec&  distvtotnew = vtotNew_;

	Index3 numUniformGridElem = hamDGPtr_->NumUniformGridElem();

	// FIXME Magic number here
	Real mixStepLengthKerker = 0.8; 

	// tempVec(:) = vtotnew(:) - vtot(:)
	// FIXME Why add the residue here?
	// vtot(:) += mixStepLengthKerker * (vtotnew(:) - vtot(:))
	for( Int k = 0; k < numElem_[2]; k++ )
		for( Int j = 0; j < numElem_[1]; j++ )
			for( Int i = 0; i < numElem_[0]; i++ ){
				Index3 key = Index3( i, j, k );
				if( elemPrtn_.Owner( key ) == mpirank ){

					tempVec.LocalMap()[key] = 
						distvtotnew.LocalMap()[key];

					blas::Axpy( numUniformGridElem.prod(), -1.0, 
							distvtot.LocalMap()[key].Data(), 1,
							tempVec.LocalMap()[key].Data(), 1 );

					blas::Axpy( numUniformGridElem.prod(), mixStepLengthKerker, 
							tempVec.LocalMap()[key].Data(), 1,
							distvtot.LocalMap()[key].Data(), 1 );
				} // own this element
			} // for (i)

	// Convert tempVec to tempVecLocal in distributed row vector format
	DblNumVec  tempVecLocal;

  DistNumVecToDistRowVec(
			tempVec,
			tempVecLocal,
			domain_.numGrid,
			numElem_,
			fft.localNzStart,
			fft.localNz,
			fft.isInGrid,
			domain_.comm );

	// Only part of the processors participate in the FFTW calculation

	if( fft.isInGrid ){

		for( Int i = 0; i < ntotLocal; i++ ){
			fft.inputComplexVecLocal(i) = Complex( 
					tempVecLocal(i), 0.0 );
		}
		fftw_execute( fft.forwardPlan );

		for( Int i = 0; i < ntotLocal; i++ ){
			if( fft.gkkLocal(i) == 0 ){
				fft.outputComplexVecLocal(i) = Z_ZERO;
			}
			else{
				// FIXME Magic number
				fft.outputComplexVecLocal(i) *= 
					mixStepLengthKerker * ( fft.gkkLocal(i) / 
							( fft.gkkLocal(i) + 0.5 ) - 1.0 );
			}
		}
		fftw_execute( fft.backwardPlan );

		// tempVecLocal saves the update of vtot

		for( Int i = 0; i < ntotLocal; i++ ){
			tempVecLocal(i) = fft.inputComplexVecLocal(i).real() / ntot;
		}
	} // if (fft.isInGrid)

	// Convert tempVecLocal to tempVec in the DistNumVec format 

  DistRowVecToDistNumVec(
			tempVecLocal,
			tempVec,
			domain_.numGrid,
			numElem_,
			fft.localNzStart,
			fft.localNz,
			fft.isInGrid,
			domain_.comm );
  
	// Update vtot
	// vtot(:) += tempVec(:)
	for( Int k = 0; k < numElem_[2]; k++ )
		for( Int j = 0; j < numElem_[1]; j++ )
			for( Int i = 0; i < numElem_[0]; i++ ){
				Index3 key = Index3( i, j, k );
				if( elemPrtn_.Owner( key ) == mpirank ){
					blas::Axpy( numUniformGridElem.prod(), 1.0, 
							tempVec.LocalMap()[key].Data(), 1,
							distvtot.LocalMap()[key].Data(), 1 );
				} // own this element
			} // for (i)


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCFDG::KerkerMix  ----- 


void
SCFDG::PrintState	( const Int iter  )
{
#ifndef _RELEASE_
	PushCallStack("SCFDG::PrintState");
#endif
  
	HamiltonianDG&  hamDG = *hamDGPtr_;

	statusOFS << std::endl << "Eigenvalues in the global domain." << std::endl;
	for(Int i = 0; i < hamDG.EigVal().m(); i++){
    Print(statusOFS, 
				"band#    = ", i, 
	      "eigval   = ", hamDG.EigVal()(i),
	      "occrate  = ", hamDG.OccupationRate()(i));
	}
	statusOFS << std::endl;
	statusOFS 
		<< "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself" << std::endl
	  << "       Etot  = Ekin + Ecor" << std::endl
	  << "       Efree = Etot	+ Entropy" << std::endl << std::endl;
	Print(statusOFS, "Etot              = ",  Etot_, "[au]");
	Print(statusOFS, "Efree             = ",  Efree_, "[au]");
	Print(statusOFS, "Ekin              = ",  Ekin_, "[au]");
	Print(statusOFS, "Ehart             = ",  Ehart_, "[au]");
	Print(statusOFS, "EVxc              = ",  EVxc_, "[au]");
	Print(statusOFS, "Exc               = ",  Exc_, "[au]"); 
	Print(statusOFS, "Eself             = ",  Eself_, "[au]");
	Print(statusOFS, "Ecor              = ",  Ecor_, "[au]");
	Print(statusOFS, "Fermi             = ",  fermi_, "[au]");
	Print(statusOFS, "norm(vout-vin)/norm(vin) = ", scfNorm_ );

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCFDG::PrintState  ----- 



} // namespace dgdft
