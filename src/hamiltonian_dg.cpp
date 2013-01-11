#include  "hamiltonian_dg.hpp"
#include  "mpi_interf.hpp"

namespace dgdft{


// *********************************************************************
// Hamiltonian class for DG
// *********************************************************************


HamiltonianDG::HamiltonianDG	( const esdf::ESDFInputParam& esdfParam )
{
#ifndef _RELEASE_
	PushCallStack("HamiltonianDG::HamiltonianDG");
#endif
	domain_            = esdfParam.domain;
	atomList_          = esdfParam.atomList;
	pseudoType_        = esdfParam.pseudoType;
	XCId_              = esdfParam.XCId;
	numExtraState_     = esdfParam.numExtraState;
	numElem_           = esdfParam.numElem;
	penaltyAlpha_      = esdfParam.penaltyAlpha;
	numLGLGridElem_    = esdfParam.numGridLGL;
  
	Int ntot = domain_.NumGridTotal();

	// Only consider numSpin == 2 in the DG calculation.
	numSpin_ = 2;

	for( Int d = 0; d < DIM; d++ ){
		if( domain_.numGrid[d] % numElem_[d] != 0 ){
			throw std::runtime_error( 
					"The number of global grid points is not divisible by the number of elements" );
		}
		numUniformGridElem_[d] = domain_.numGrid[d] / numElem_[d];
	}

	// Setup the element domains
	domainElem_.Resize( numElem_[0], numElem_[1], numElem_[2] );
	for( Int k=0; k< numElem_[2]; k++ )
		for( Int j=0; j< numElem_[1]; j++ )
			for( Int i=0; i< numElem_[0]; i++ ) {
				Index3 key( i, j, k );
				Domain& dm = domainElem_(i, j, k);
				for( Int d = 0; d < DIM; d++ ){
					dm.length[d]     = domain_.length[d] / numElem_[d];
					dm.numGrid[d]    = numUniformGridElem_[d];
					dm.posStart[d]   = dm.length[d] * key[d];
				}
				dm.comm = domain_.comm;
			}
	
	// Partition by element

	IntNumTns& elemPrtnInfo = elemPrtn_.ownerInfo;
	elemPrtnInfo.Resize( numElem_[0], numElem_[1], numElem_[2] );

	Int cnt = 0;
	for( Int k=0; k< numElem_[2]; k++ )
		for( Int j=0; j< numElem_[1]; j++ )
			for( Int i=0; i< numElem_[0]; i++ ) {
				elemPrtnInfo(i,j,k) = cnt++;
			}

#if ( _DEBUGlevel_ >= 1 )
	for( Int k=0; k< numElem_[2]; k++ )
		for( Int j=0; j< numElem_[1]; j++ )
			for( Int i=0; i< numElem_[0]; i++ ) {
				statusOFS << "Element # " << Index3(i,j,k) << " belongs to proc " << 
					elemPrtn_.Owner(Index3(i,j,k)) << std::endl;
			}
#endif

	pseudoCharge_.Prtn()  = elemPrtn_;
	density_.Prtn()       = elemPrtn_;
	vext_.Prtn()          = elemPrtn_;
	vhart_.Prtn()         = elemPrtn_;
	vxc_.Prtn()           = elemPrtn_;
	epsxc_.Prtn()         = elemPrtn_;
	vtot_.Prtn()          = elemPrtn_;
	vtotLGL_.Prtn()       = elemPrtn_;
	basisLGL_.Prtn()      = elemPrtn_;

	// Partition by atom
	// The ownership of an atom is determined by whether the element
	// containing the atom is owned
	
	Int numAtom = atomList_.size();

	std::vector<Int>& atomPrtnInfo = atomPrtn_.ownerInfo;
	atomPrtnInfo.resize( numAtom );
	
	for( Int a = 0; a < numAtom; a++ ){
		Point3 pos = atomList_[a].pos;
		bool isAtomIn = false;
		Index3 idx(-1, -1, -1);
		for( Int k = 0; k < numElem_[2]; k++ ){
			for( Int j = 0; j < numElem_[1]; j++ ){
				for( Int i = 0; i < numElem_[0]; i++ ){
					isAtomIn = IsInSubdomain( pos, domainElem_(i, j, k), domain_.length );
					if( isAtomIn == true ){
						idx = Index3(i, j, k);
						break;
					}
				}
			}
		}
		if( idx[0] < 0 || idx[1] < 0 || idx[2] < 0 ){
			std::ostringstream msg;
			msg << "Cannot find element for atom #" << a << std::endl;
			throw std::runtime_error( msg.str().c_str() );
		}
		// Determine the ownership by the ownership of the corresponding element
		atomPrtnInfo[a] = elemPrtn_.Owner( idx );
	} // for (a)

	pseudo_.Prtn()     = atomPrtn_;

#if ( _DEBUGlevel_ >= 1 )
	for( Int a = 0; a < numAtom; a++ ){
		statusOFS << "Atom # " << a << " belongs to proc " << 
			atomPrtn_.Owner(a) << std::endl;
	}
#endif

	// Generate the grids

	UniformMesh( domain_, uniformGrid_ );

	uniformGridElem_.Resize( numElem_[0], numElem_[1], numElem_[2] );

	LGLGridElem_.Resize( numElem_[0], numElem_[1], numElem_[2] );

	for( Int k = 0; k < numElem_[2]; k++ ){
		for( Int j = 0; j < numElem_[1]; j++ ){
			for( Int i = 0; i < numElem_[0]; i++ ){
				UniformMesh( domainElem_(i, j, k), 
						uniformGridElem_(i, j, k) );
				LGLMesh( domainElem_(i, j, k),
						numLGLGridElem_,
						LGLGridElem_(i, j, k) );
			}
		}
	}

#if ( _DEBUGlevel_ >= 1 )
		for( Int k = 0; k < numElem_[2]; k++ ){
			for( Int j = 0; j < numElem_[1]; j++ ){
				for( Int i = 0; i < numElem_[0]; i++ ){
					statusOFS << "Uniform grid for element " << Index3(i,j,k) << std::endl;
					for( Int d = 0; d < DIM; d++ ){
						statusOFS << uniformGridElem_(i,j,k)[d] << std::endl;
					}
					statusOFS << "LGL grid for element " << Index3(i,j,k) << std::endl;
					for( Int d = 0; d < DIM; d++ ){
						statusOFS << LGLGridElem_(i,j,k)[d] << std::endl;
					}
				}
			}
		}
#endif


	// Initialize the XC functional.  
	// Spin-unpolarized functional is used here
	if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
    throw std::runtime_error( "XC functional initialization error." );
	} 


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method HamiltonianDG::HamiltonianDG  ----- 

void
HamiltonianDG::CalculatePseudoPotential	( PeriodTable &ptable ){
#ifndef _RELEASE_
	PushCallStack("HamiltonianDG::CalculatePseudoPotential");
#endif
	Int ntot = domain_.NumGridTotal();
	Int numAtom = atomList_.size();
	Int mpirank, mpisize;
	MPI_Comm_rank( domain_.comm, &mpirank );
	MPI_Comm_size( domain_.comm, &mpisize );

	Real vol = domain_.Volume();

  // Calculate the number of occupied states
  Int nelec = 0;
  for (Int a=0; a<numAtom; a++) {
		Int atype  = atomList_[a].type;
		if( ptable.ptemap().find(atype) == ptable.ptemap().end() ){
			std::ostringstream msg;
			msg << "Cannot find the atom type for atom #" << a << std::endl;
			throw std::runtime_error( msg.str().c_str() );
		}
		nelec = nelec + ptable.ptemap()[atype].params(PTParam::ZION);
  }

	if( nelec % 2 != 0 ){
		throw std::runtime_error( "This is a spin-restricted calculation. nelec should be even." );
	}
	
	numOccupiedState_ = nelec / numSpin_;

	Print( statusOFS, "Number of Occupied States                    = ", numOccupiedState_ );

	// Generate the atomic pseudopotentials

	Index3 elemNumGridFirst;
	for( Int d = 0; d < DIM; d++ ){
		elemNumGridFirst[d] = domain_.numGrid[d] / numElem_[d];
	}

  for (Int a=0; a<numAtom; a++) {
		if( pseudo_.Prtn().Owner(a) == mpirank ){
			PseudoPotElem pp;
			// Pseudocharge
			ptable.CalculatePseudoCharge( atomList_[a], domain_, uniformGridElem_, pp.pseudoCharge );
			// Nonlocal pseudopotential
//			ptable.CalculateNonlocalPP( atomList_[a], domain_, uniformGridElem_, pp.vnlList );

			pseudo_.LocalMap()[a] = pp;
		}
  }

#if ( _DEBUGlevel_ >= 1 )
	statusOFS << std::endl << "Atomic pseudocharge computed." << std::endl;
#endif

	// Assembly the pseudocharge 

	// First step: local assemly of values for pseudocharge
	for( Int a = 0; a < numAtom; a++ ){
		if( pseudo_.Prtn().Owner(a) == mpirank ){
			for( Int k = 0; k < numElem_[2]; k++ )
				for( Int j = 0; j < numElem_[1]; j++ )
					for( Int i = 0; i < numElem_[0]; i++ ){
						SparseVec& sp  = pseudo_.LocalMap()[a].pseudoCharge(i,j,k);
						IntNumVec& idx = sp.first;
						DblNumMat& val = sp.second;
						if( idx.m() > 0 ){
							Index3 key( i, j, k );
							if( pseudoCharge_.LocalMap().find(key) == pseudoCharge_.LocalMap().end() ){
								// Start a new element
								Int numTotal = 
									numUniformGridElem_[0] *
									numUniformGridElem_[1] * 
									numUniformGridElem_[2];
								DblNumVec empty( numTotal );
							  SetValue( empty, 0.0 );
								pseudoCharge_.LocalMap()[key] = empty;
							}
							DblNumVec& vec = pseudoCharge_.LocalMap()[key];
							for( Int l = 0; l < idx.m(); l++ ){
								vec[idx(l)] += val(l, PseudoComponent::VAL);
							}
						}
					} // for (i)
		}
	} // for (a)

#if ( _DEBUGlevel_ >= 1 )
	statusOFS << std::endl << "Assembly of pseudocharge: first step passed." << std::endl;
#endif

	// Second step: communication of the pseudoCharge among all processors
	{
		std::vector<Index3>  putKey;
		for( std::map<Index3, DblNumVec>::iterator mi = pseudoCharge_.LocalMap().begin();
				 mi != pseudoCharge_.LocalMap().end(); mi++ ){
			Index3 key = (*mi).first;
			if( pseudoCharge_.Prtn().Owner( key ) != mpirank ){
				putKey.push_back( key );
			}
		}
		pseudoCharge_.PutBegin( putKey, NO_MASK );
		pseudoCharge_.PutEnd( NO_MASK, PutMode::COMBINE );
	}

#if ( _DEBUGlevel_ >= 1 )
	statusOFS << std::endl << "Assembly of pseudocharge: second step passed." << std::endl;
#endif

	// Third step: erase the vectors the current processor does not own
	{
		std::vector<Index3>  eraseKey;
		for( std::map<Index3, DblNumVec>::iterator mi = pseudoCharge_.LocalMap().begin();
				 mi != pseudoCharge_.LocalMap().end(); mi++ ){
			Index3 key = (*mi).first;
			if( pseudoCharge_.Prtn().Owner( key ) != mpirank ){
				eraseKey.push_back( key );
			}
		}
		for( std::vector<Index3>::iterator vi = eraseKey.begin();
			   vi != eraseKey.end(); vi++ ){
			pseudoCharge_.LocalMap().erase( *vi );
		}
	}

#if ( _DEBUGlevel_ >= 1 )
	statusOFS << std::endl << "Assembly of pseudocharge: third step passed." << std::endl;
#endif

	// Compute the sum of pseudocharge
	{
		Real localSum = 0.0, sumRho = 0.0;
		for( std::map<Index3, DblNumVec>::iterator mi = pseudoCharge_.LocalMap().begin();
				mi != pseudoCharge_.LocalMap().end(); mi++ ){
			if( pseudoCharge_.Prtn().Owner((*mi).first) != mpirank ){
				throw std::runtime_error("The current processor should not own an element in the pseudocharge.");
			}	
			DblNumVec& vec = (*mi).second;
			for( Int i = 0; i < vec.m(); i++ ){
				localSum += vec[i];
			}
			localSum *= domain_.Volume() / domain_.NumGridTotal();
		}

		mpi::Allreduce( &localSum, &sumRho, 1, MPI_SUM, domain_.comm );

		Print( statusOFS, "Sum of Pseudocharge                          = ", sumRho );
		
		// Make adjustments to the pseudocharge
		Real diff = ( numSpin_ * numOccupiedState_ - sumRho ) / domain_.Volume();
		
		for( std::map<Index3, DblNumVec>::iterator mi = pseudoCharge_.LocalMap().begin();
				mi != pseudoCharge_.LocalMap().end(); mi++ ){
			DblNumVec& vec = (*mi).second;
			for( Int i = 0; i < vec.m(); i++ ){
				vec[i] += diff;
			}
		}
	
		Print( statusOFS, "After adjustment, sum of Pseudocharge        = ", 
				numSpin_ * numOccupiedState_);
	}


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method HamiltonianDG::CalculatePseudoPotential  ----- 


void HamiltonianDG::CalculateHartree( DistFourier& fft ) {
#ifndef _RELEASE_ 
	PushCallStack("HamiltonianDG::CalculateHartree");
#endif
	if( !fft.isInitialized ){
		throw std::runtime_error("Fourier is not prepared.");
	}
 	
  Int ntot      = fft.numGridTotal;
	Int ntotLocal = fft.numGridLocal;

	// TODO Convert density to densityLocal;

	// The contribution of the pseudoCharge is subtracted. So the Poisson
	// equation is well defined for neutral system.

	for( Int i = 0; i < ntotLocal; i++ ){
		fft.inputComplexVecLocal(i) = Complex( 
				densityLocal_(i) - pseudoChargeLocal_(i), 0.0 );
	}
	fftw_execute( fft.forwardPlan );

	for( Int i = 0; i < ntotLocal; i++ ){
		if( fft.gkkLocal(i) == 0 ){
			fft.outputComplexVecLocal(i) = Z_ZERO;
		}
		else{
			// NOTE: gkk already contains the factor 1/2.
			fft.outputComplexVecLocal(i) *= 2.0 * PI / fft.gkkLocal(i);
		}
	}
	fftw_execute( fft.backwardPlan );

	for( Int i = 0; i < ntotLocal; i++ ){
		vhartLocal_(i) = fft.inputComplexVecLocal(i).real() / ntot;
	}

#ifndef _RELEASE_
	PopCallStack();
#endif
	return; 
}  // -----  end of method HamiltonianDG::CalculateHartree ----- 


} // namespace dgdft
