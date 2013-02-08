/// @file scf_dg.hpp
/// @brief Self consistent iteration using the DF method.
/// @author Lin Lin
/// @date 2013-02-05
#include  "scf_dg.hpp"
#include	"blas.hpp"
#include	"lapack.hpp"
#include  "utility.hpp"

#define _DEBUGlevel_ 1

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
		PeriodTable&                ptable )
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
		
		vtotNew_.Prtn() = elemPrtn_;

//		vtotNew_.Resize(ntot); SetValue(vtotNew_, 0.0);
//		dfMat_.Resize( ntot, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
//		dvMat_.Resize( ntot, mixMaxDim_ ); SetValue( dvMat_, 0.0 );
	
//		restartDensityFileName_ = "DEN";
//		restartWfnFileName_     = "WFN";
	}

	// Density
	{
		DistDblNumVec&  density = hamDGPtr_->Density();
		if( isRestartDensity_ ) {
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

  HamiltonianDG&  hamDG = *hamDGPtr_;

	// Compute the exchange-correlation potential and energy
	hamDG.CalculateXC( Exc_ );

	statusOFS << "Exc = " << Exc_ << std::endl;

	// Compute the Hartree energy
	hamDG.CalculateHartree( *distfftPtr_ );

	// No external potential

	// Compute the total potential
	hamDG.CalculateVtot( hamDG.Vtot() );

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
		// Update the potential in the extended element
		// *********************************************************************

		{
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


			// Update of the local potential in each element locally.  The
			// nonlocal potential does not need to be updated
			// NOTE: It is hard coded that the extended element is 1 or 3
			// times the size of the element
			for( Int k = 0; k < numElem_[2]; k++ )
				for( Int j = 0; j < numElem_[1]; j++ )
					for( Int i = 0; i < numElem_[0]; i++ ){
						Index3 key( i, j, k );
						if( elemPrtn_.Owner( key ) == mpirank ){
              EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
							Hamiltonian&  hamExtElem  = eigSol.Ham();
							DblNumVec&    vtotExtElem = hamExtElem.Vtot();
							SetValue( vtotExtElem, 0.0 );

							Index3 numGridElem;
							for( Int d = 0; d < DIM; d++ ){
								numGridElem[d] = domain_.numGrid[d] / numElem_[d];
							}

							Index3 numGridExtElem = eigSol.FFT().domain.numGrid;
							
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
									// Adjustment
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


							// Loop over the neighborhood

						} // own this element
					} // for (i)

			// Clean up
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

		}

		// *********************************************************************
		// Solve the basis functions in the extended element
		// *********************************************************************

		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner( key ) == mpirank ){
						EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
						DblNumVec&    vtotExtElem = eigSol.Ham().Vtot();

						// Solve the basis functions in the extended element
						eigSol.Solve();

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


//						Spinor& psi = eigSol.Psi();
//
//						// Assuming that wavefun has only 1 component
//						DblNumTns& wavefun = psi.Wavefun();
//
//						DblNumMat localBasis( 
//								psi.NumGridTotal(), 
//								psi.NumState() );
//
//						SetValue( localBasis, 0.0 );
//
//						for( Int l = 0; l < psi.NumState(); l++ ){
//							InterpPeriodicUniformToLGL( 
//									wavefun.VecData(0, l), 
//									localBasis.VecData(l) );
//						}
//
//						// Perform SVD for the basis functions
//						DblNumMat localOrthoBasis;
//
//						
//						// Write localOrthoBasis to basisLGL_
//						hamDG.BasisLGL().LocalMap()[key] = localOrthoBasis;
						
					} // own this element
				} // for (i)
//
//
//		// *********************************************************************
//		// Assemble the DG matrix
//		// *********************************************************************
//
//		hamDG.CalculateDGMatrix( );
//
//
//		// *********************************************************************
//		// Diagonalize the DG matrix
//		// *********************************************************************
//
//		{
//			Int sizeH = hamDG.NumBasisTotal();
//
//			Descriptor descH( sizeH, sizeH, MB_, MB_, 0, 0, contxt_ );
//
//			ScaLAPACKMatrix<Real>  scaH, scaZ;
//			
//			std::vector<Real> eigs;
//
//			DistElemMatToScaMat( hamDG.HMat(), 	descH,
//					scaH, hamDG.ElemBasisIdx(), domain_.comm );
//
//			scalapack::Syevd('U', scaH, eigs, scaZ);
//
//			DblNumVec& eigval = hamDG.EigVal(); 
//      eigval.Resize( hamDG.NumStateTotal() );		
//			for( Int i = 0; i < hamDG.NumStateTotal(); i++ )
//				eigval[i] = eigs[i];
//			
//			ScaMatToDistNumMat( scaZ, hamDG.Density().Prtn(), 
//					hamDG.EigvecCoef(), hamDG.ElemBasisIdx(), domain_.comm, 
//					hamDG.NumStateTotal() );
//
//		}
//
//
//		// *********************************************************************
//		// Post processing
//		// *********************************************************************
//
//		// Compute the occupation rate
//		CalculateOccupationRate( hamDG.EigVal(), hamDG.OccupationRate() );
//
//		// Compute the electron density
//		hamDG.CalculateDensity( hamDG.OccupationRate() );
//
//		// Compute the exchange-correlation potential and energy
//		hamDG.CalculateXC( Exc_ );
//
//		// Compute the Hartree energy
//		hamDG.CalculateHartree( *distfftPtr_ );
//
//		// No external potential
//
//		// Compute the new total potential
//		hamDG.CalculateVtot( vtotNew_ );
//
//		// Compute the error of the potential
//		{
//			Real normVtotDifLocal = 0.0, normVtotOldLocal = 0.0;
//			Real normVtotDif, normVtotOld;
//			for( Int k = 0; k < numElem_[2]; k++ )
//				for( Int j = 0; j < numElem_[1]; j++ )
//					for( Int i = 0; i < numElem_[0]; i++ ){
//						Index3 key( i, j, k );
//						if( elemPrtn_.Owner( key ) == mpirank ){
//							DblNumVec& oldVec = hamDG.Vtot().LocalMap()[key];
//							DblNumVec& newVec = vtotNew_.LocalMap()[key];
//
//							for( Int p = 0; p < oldVec.m(); p++ ){
//								normVtotDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
//								normVtotOldLocal += pow( oldVec(p), 2.0 );
//							}
//						} // own this element
//					} // for (i)
//
//
//			mpi::Allreduce( &normVtotDifLocal, &normVtotDif, 1, MPI_SUM, 
//					domain_.comm );
//			mpi::Allreduce( &normVtotOldLocal, &normVtotOld, 1, MPI_SUM,
//					domain_.comm );
//
//			normVtotDif = std::sqrt( normVtotDif );
//			normVtotOld = std::sqrt( normVtotOld );
//
//			scfNorm_    = normVtotDif / normVtotOld;
//		}
		
//		// Compute the energies
    CalculateEnergy();
//
//		// Print out the state variables of the current iteration
    PrintState( iter );
//
//
//    if( scfNorm_ < scfTolerance_ ){
//      /* converged */
//      Print( statusOFS, "SCF is converged!\n" );
//      isSCFConverged = true;
//    }
//
//		// Potential mixing
//    if( mixType_ == "anderson" ){
//      AndersonMix(iter);
//    }
//    if( mixType_ == "kerker" ){
//      KerkerMix();  
//      AndersonMix(iter);
//    }

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
	Eself_ = 0;
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
				Efree_ += (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
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
SCFDG::PrintState	( const Int iter  )
{
#ifndef _RELEASE_
	PushCallStack("SCFDG::PrintState");
#endif
  
	HamiltonianDG&  hamDG = *hamDGPtr_;

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
	Print(statusOFS, "Total charge      = ",  totalCharge_, "[au]");
	Print(statusOFS, "norm(vout-vin)/norm(vin) = ", scfNorm_ );

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCFDG::PrintState  ----- 


} // namespace dgdft
