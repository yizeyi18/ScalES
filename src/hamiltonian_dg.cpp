/// @file hamiltonian_dg.cpp
/// @brief Implementation of the Hamiltonian class for DG calculation.
/// @author Lin Lin
/// @date 2013-01-09
#include  "hamiltonian_dg.hpp"
#include  "mpi_interf.hpp"
#include  "blas.hpp"

#define _DEBUGlevel_ 1

namespace dgdft{

using namespace PseudoComponent;

// *********************************************************************
// Hamiltonian class for DG
// *********************************************************************

HamiltonianDG::HamiltonianDG() {
	XCInitialized_ = false;
}

HamiltonianDG::HamiltonianDG	( const esdf::ESDFInputParam& esdfParam )
{
#ifndef _RELEASE_
	PushCallStack("HamiltonianDG::HamiltonianDG");
#endif
	Int mpirank, mpisize;
	MPI_Comm_rank( domain_.comm, &mpirank );
	MPI_Comm_size( domain_.comm, &mpisize );

	domain_            = esdfParam.domain;
	atomList_          = esdfParam.atomList;
	pseudoType_        = esdfParam.pseudoType;
	XCId_              = esdfParam.XCId;
	numExtraState_     = esdfParam.numExtraState;
	numElem_           = esdfParam.numElem;
	penaltyAlpha_      = esdfParam.penaltyAlpha;
	numLGLGridElem_    = esdfParam.numGridLGL;
  
	Int ntot = domain_.NumGridTotal();

	XCInitialized_ = false;

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

	// FIXME Assign one element to one processor. (Hopefully) this is the
	// only place where such assumption is made explicitly, and the rest
	// of the code should be adapted to the general partition plan: 
	// both the case of one processor owns more than one element, and also
	// several processors own the same element.

	if( mpisize != numElem_.prod() ){
			std::ostringstream msg;
			msg << "The number of processors is not equal to the total number of elements." << std::endl;
			throw std::runtime_error( msg.str().c_str() );
	}

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

	// Initialize the DistNumVecs.
	pseudoCharge_.Prtn()  = elemPrtn_;
	density_.Prtn()       = elemPrtn_;
	densityLGL_.Prtn()    = elemPrtn_;
	vext_.Prtn()          = elemPrtn_;
	vhart_.Prtn()         = elemPrtn_;
	vxc_.Prtn()           = elemPrtn_;
	epsxc_.Prtn()         = elemPrtn_;
	vtot_.Prtn()          = elemPrtn_;

	for( Int k=0; k< numElem_[2]; k++ )
		for( Int j=0; j< numElem_[1]; j++ )
			for( Int i=0; i< numElem_[0]; i++ ) {
				Index3 key = Index3(i,j,k);
				if( elemPrtn_.Owner(key) == mpirank ){
					DblNumVec  empty( numUniformGridElem_.prod() );
					SetValue( empty, 0.0 );
					DblNumVec emptyLGL( numLGLGridElem_.prod() );
					density_.LocalMap()[key]     = empty;
					densityLGL_.LocalMap()[key]  = emptyLGL;
					vext_.LocalMap()[key]        = empty;
					vhart_.LocalMap()[key]       = empty;
					vxc_.LocalMap()[key]         = empty;
					epsxc_.LocalMap()[key]       = empty;
					vtot_.LocalMap()[key]        = empty;
				} // own this element
			}  // for (i)
  
	vtotLGL_.Prtn()       = elemPrtn_;
	basisLGL_.Prtn()      = elemPrtn_;

	for( Int k=0; k< numElem_[2]; k++ )
		for( Int j=0; j< numElem_[1]; j++ )
			for( Int i=0; i< numElem_[0]; i++ ) {
				Index3 key = Index3(i,j,k);
				if( elemPrtn_.Owner(key) == mpirank ){
					DblNumVec  empty( numLGLGridElem_.prod() );
					SetValue( empty, 0.0 );
					vtotLGL_.LocalMap()[key]        = empty;
				}
			}


	// Pseudopotential
	pseudo_.Prtn()      = elemPrtn_;
	vnlCoef_.Prtn()       = elemPrtn_;
	vnlDrvCoef_.resize(DIM);
	for( Int d = 0; d < DIM; d++ ){
		vnlDrvCoef_[d].Prtn() = elemPrtn_;
	}

	// Partition of the DG matrix
	elemMatPrtn_.ownerInfo = elemPrtn_.ownerInfo;
	// Initialize HMat_
	HMat_.LocalMap().clear();
	HMat_.Prtn() = elemMatPrtn_;


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
	} // for (k)
#endif

	// Generate the differentiation matrix on the LGL grid
	// NOTE: This assumes uniform mesh used for each element.
	DMat_.resize( DIM );
	for( Int d = 0; d < DIM; d++ ){
		DblNumVec  dummyX, dummyW;
		DblNumMat  dummyP;
		GenerateLGL( dummyX, dummyW, dummyP, DMat_[d], 
				numLGLGridElem_[d] );

		// Scale the differentiation matrix
		blas::Scal( numLGLGridElem_[d] * numLGLGridElem_[d],
				2.0 / (domain_.length[d] / numElem_[d]), 
				DMat_[d].Data(), 1 );
	}

	// Generate the transfer matrix from LGL grid to uniform grid on each
	// element. The Lagrange polynomials involved in the transfer matrix
	// is computed using the Barycentric method.  For more information see
	//
	// [J.P. Berrut and L.N. Trefethen, Barycentric Lagrange Interpolation,
	// SIAM Rev. 2004]
	//
	// NOTE: This assumes uniform mesh used for each element.
	{
		LGLToUniformMat_.resize(DIM);
		Index3& numLGL                      = numLGLGridElem_;
		Index3& numUniform                  = numUniformGridElem_;
		// Small stablization parameter 
		const Real EPS                      = 1e-13; 
		for( Int d = 0; d < DIM; d++ ){
			DblNumVec& LGLGrid     = LGLGridElem_(0,0,0)[d];
			DblNumVec& uniformGrid = uniformGridElem_(0,0,0)[d];
			// Stablization constant factor, according to Berrut and Trefethen
			Real    stableFac = 0.25 * domainElem_(0,0,0).length[d];
			DblNumMat& localMat = LGLToUniformMat_[d];
			localMat.Resize( numUniform[d], numLGL[d] );
			DblNumVec lambda( numLGL[d] );
			DblNumVec denom( numUniform[d] );
			SetValue( lambda, 0.0 );
			SetValue( denom, 0.0 );
			for( Int i = 0; i < numLGL[d]; i++ ){
				lambda[i] = 1.0;
				for( Int j = 0; j < numLGL[d]; j++ ){
					if( j != i ) 
						lambda[i] *= (LGLGrid[i] - LGLGrid[j]) / stableFac; 
				} // for (j)
				lambda[i] = 1.0 / lambda[i];
				for( Int j = 0; j < numUniform[d]; j++ ){
					denom[j] += lambda[i] / ( uniformGrid[j] - LGLGrid[i] + EPS );
				}
			} // for (i)

			for( Int i = 0; i < numLGL[d]; i++ ){
				for( Int j = 0; j < numUniform[d]; j++ ){
					localMat( j, i ) = (lambda[i] / ( uniformGrid[j] - LGLGrid[i]
								+ EPS )) / denom[j]; 
				} // for (j)
			} // for (i)
		} // for (d)
	}

#if ( _DEBUGlevel_ >= 1 )
	statusOFS << "LGLToUniformMat[0] = " << LGLToUniformMat_[0] << std::endl;
	statusOFS << "LGLToUniformMat[1] = " << LGLToUniformMat_[1] << std::endl; 
	statusOFS << "LGLToUniformMat[2] = " << LGLToUniformMat_[2] << std::endl; 
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

HamiltonianDG::~HamiltonianDG	( )
{
#ifndef _RELEASE_
	PushCallStack("HamiltonianDG::~HamiltonianDG");
#endif

	if( XCInitialized_ )
		xc_func_end(&XCFuncType_);

#ifndef _RELEASE_
	PopCallStack();
#endif
} 		// -----  end of method HamiltonianDG::HamiltonianDG  ----- 

void
HamiltonianDG::DiffPsi	(const Index3& numGrid, const Real* psi, Real* Dpsi, Int d)
{
#ifndef _RELEASE_
	PushCallStack("HamiltonianDG::DiffPsi");
#endif
	if( d == 0 ){
		// Use Gemm
		Int m = numGrid[0], n = numGrid[1]*numGrid[2];
		blas::Gemm( 'N', 'N', m, n, m, 1.0, DMat_[0].Data(),
				m, psi, m, 0.0, Dpsi, m );
	}
	else if( d == 1 ){
		// Middle dimension, use Gemv
		Int   m = numGrid[1], n = numGrid[0]*numGrid[2];
		Int   ptrShift;
		Int   inc = numGrid[0];
		for( Int k = 0; k < numGrid[2]; k++ ){
			for( Int i = 0; i < numGrid[0]; i++ ){
				ptrShift = i + k * numGrid[0] * numGrid[1];
				blas::Gemv( 'N', m, m, 1.0, DMat_[1].Data(), m, 
						psi + ptrShift, inc, 0.0, Dpsi + ptrShift, inc );
			}
		} // for (k)
	}
	else if ( d == 2 ){
		// Use Gemm
		Int m = numGrid[0]*numGrid[1], n = numGrid[2];
		blas::Gemm( 'N', 'T', m, n, n, 1.0, psi, m,
				DMat_[2].Data(), n, 0.0, Dpsi, m );
	}
	else{
		throw std::logic_error("Wrong dimension.");
	}

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method HamiltonianDG::DiffPsi  ----- 


void
HamiltonianDG::InterpLGLToUniform	( const Index3& numLGLGrid, const
		Index3& numUniformGrid, const Real* psiLGL, Real* psiUniform )
{
#ifndef _RELEASE_
	PushCallStack("HamiltonianDG::InterpLGLToUniform");
#endif
	Index3 Ns1 = numLGLGrid;
	Index3 Ns2 = numUniformGrid;
	
	DblNumVec  tmp1( Ns2[0] * Ns1[1] * Ns1[2] );
	DblNumVec  tmp2( Ns2[0] * Ns2[1] * Ns1[2] );
	SetValue( tmp1, 0.0 );
	SetValue( tmp2, 0.0 );

	// x-direction, use Gemm
	{
		Int m = Ns2[0], n = Ns1[1] * Ns1[2], k = Ns1[0];
		blas::Gemm( 'N', 'N', m, n, k, 1.0, LGLToUniformMat_[0].Data(),
				m, psiLGL, k, 0.0, tmp1.Data(), m );
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
						LGLToUniformMat_[1].Data(), m, 
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
				LGLToUniformMat_[2].Data(), n, 0.0, psiUniform, m );
	}


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method HamiltonianDG::InterpLGLToUniform  ----- 


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

	// *********************************************************************
	// Atomic information
	// *********************************************************************

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

#if ( _DEBUGlevel_ >= 0 )
	Print( statusOFS, "Number of Occupied States                    = ", numOccupiedState_ );
#endif

	// Generate the atomic pseudopotentials

	// Also prepare the integration weights for constructing the DG matrix later.

	vnlWeightMap_.clear();
	for( Int k = 0; k < numElem_[2]; k++ )
		for( Int j = 0; j < numElem_[1]; j++ )
			for( Int i = 0; i < numElem_[0]; i++ ){
				Index3 key( i, j, k );
				if( elemPrtn_.Owner( key ) == mpirank ){
					std::map<Int, PseudoPot>   ppMap;
					std::vector<DblNumVec>&    gridpos = uniformGridElem_( i, j, k );
					for( Int a = 0; a < numAtom; a++ ){
						PTEntry& ptentry = ptable.ptemap()[atomList_[a].type];
						// Cutoff radius: Take the largest one
						Real Rzero = ptentry.cutoffs( PTSample::PSEUDO_CHARGE );
						if(ptentry.cutoffs.m()>PTSample::NONLOCAL)      
							Rzero = std::max( Rzero, ptentry.cutoffs(PTSample::NONLOCAL) );

						// Compute the minimum distance of this atom to all grid points
						Point3 minDist;
						Point3& length = domain_.length;
						Point3& pos    = atomList_[a].pos;
						for( Int d = 0; d < DIM; d++ ){
							minDist[d] = Rzero;
							Real dist;
							for( Int i = 0; i < gridpos[d].m(); i++ ){
								dist = gridpos[d](i) - pos[d];
								dist = dist - IRound( dist / length[d] ) * length[d];
								if( std::abs( dist ) < minDist[d] )
									minDist[d] = std::abs( dist );
							}
						}
						// If this atom overlaps with this element, compute the pseudopotential
						if( minDist.l2() <= Rzero ){
							PseudoPot   pp;
							ptable.CalculatePseudoCharge( atomList_[a], 
									domain_, uniformGridElem_(i, j, k), pp.pseudoCharge );
							ptable.CalculateNonlocalPP( atomList_[a], 
									domain_, LGLGridElem_(i, j, k), pp.vnlList );
							ppMap[a] = pp;
							DblNumVec   weight( pp.vnlList.size() );
							for( Int l = 0; l < weight.Size(); l++ ){
								weight(l) = pp.vnlList[l].second;
							}
							vnlWeightMap_[a] = weight;
						}
					} // for (a)
					pseudo_.LocalMap()[key] = ppMap;
				} // own this element
			} // for (i)

#if ( _DEBUGlevel_ >= 1 )
	statusOFS << std::endl << "Atomic pseudocharge computed." << std::endl;
#endif

	// *********************************************************************
	// Local pseudopotential: computed by pseudocharge
	// *********************************************************************
	
	// Compute the pseudocharge by summing over contributions from all atoms
	{
		Real localSum = 0.0, sumRho = 0.0;

		pseudoCharge_.LocalMap().clear();

		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner( key ) == mpirank ){
						std::map<Int, PseudoPot>&  ppMap = pseudo_.LocalMap()[key];
						DblNumVec  localVec( numUniformGridElem_.prod() );
						SetValue( localVec, 0.0 );
						for( std::map<Int, PseudoPot>::iterator mi = ppMap.begin();
								 mi != ppMap.end(); mi++ ){
							Int atomIdx = (*mi).first;
							PseudoPot& pp = (*mi).second;
							SparseVec& sp = pp.pseudoCharge;
							IntNumVec& idx = sp.first;
							DblNumMat& val = sp.second;
							for( Int l = 0; l < idx.m(); l++ ){
								localVec[idx(l)] += val(l, VAL);
							}
						} // for (mi)
						pseudoCharge_.LocalMap()[key] = localVec;
					} // own this element
				} // for (i)

		// Compute the sum of the pseudocharge and make adjustment.
	
		for( std::map<Index3, DblNumVec>::iterator mi = pseudoCharge_.LocalMap().begin();
				mi != pseudoCharge_.LocalMap().end(); mi++ ){
			DblNumVec& vec = (*mi).second;
			for( Int i = 0; i < vec.m(); i++ ){
				localSum += vec[i];
			}
		}

		localSum *= domain_.Volume() / domain_.NumGridTotal();

		mpi::Allreduce( &localSum, &sumRho, 1, MPI_SUM, domain_.comm );

#if ( _DEBUGlevel_ >= 0 )
		Print( statusOFS, "Sum of Pseudocharge                          = ", sumRho );
		Print( statusOFS, "numOccupiedState                             = ", 
				numOccupiedState_ );
#endif
		
		// Make adjustments to the pseudocharge
		Real diff = ( numSpin_ * numOccupiedState_ - sumRho ) / domain_.Volume();
		
		for( std::map<Index3, DblNumVec>::iterator mi = pseudoCharge_.LocalMap().begin();
				mi != pseudoCharge_.LocalMap().end(); mi++ ){
			DblNumVec& vec = (*mi).second;
			for( Int i = 0; i < vec.m(); i++ ){
				vec[i] += diff;
			}
		}
	
#if ( _DEBUGlevel_ >= 0 )
		Print( statusOFS, "After adjustment, sum of Pseudocharge        = ", 
				(Real) numSpin_ * numOccupiedState_ );
#endif
	}

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method HamiltonianDG::CalculatePseudoPotential  ----- 

void
HamiltonianDG::CalculateDensity	( const DblNumVec& occrate  )
{
#ifndef _RELEASE_
	PushCallStack("HamiltonianDG::CalculateDensity");
#endif
	Int mpirank, mpisize;
	MPI_Comm_rank( domain_.comm, &mpirank );
	MPI_Comm_size( domain_.comm, &mpisize );

	Int numEig = occrate.m();

	DistDblNumVec  psiUniform;
	psiUniform.Prtn() = elemPrtn_;

	// Clear the density
	for( Int k = 0; k < numElem_[2]; k++ )
		for( Int j = 0; j < numElem_[1]; j++ )
			for( Int i = 0; i < numElem_[0]; i++ ){
				Index3 key( i, j, k );
				if( elemPrtn_.Owner( key ) == mpirank ){
					DblNumVec& localRho = density_.LocalMap()[key];
					SetValue( localRho, 0.0 );
				} // own this element
			} // for (i)

	// Method 1: Normalize each eigenfunctions.  This may take many
	// interpolation steps, and the communication cost may be large
	if(0)
	{
		// Loop over all the eigenfunctions
		for( Int g = 0; g < numEig; g++ ){
			// Normalization constants
			Real normPsiLocal  = 0.0;
			Real normPsi       = 0.0;
			for( Int k = 0; k < numElem_[2]; k++ )
				for( Int j = 0; j < numElem_[1]; j++ )
					for( Int i = 0; i < numElem_[0]; i++ ){
						Index3 key( i, j, k );
						if( elemPrtn_.Owner( key ) == mpirank ){
							DblNumMat& localBasis = basisLGL_.LocalMap()[key];
							Int numGrid  = localBasis.m();
							Int numBasis = localBasis.n();

							DblNumMat& localCoef  = eigvecCoef_.LocalMap()[key];
							if( localCoef.n() != numEig ){
								throw std::runtime_error( 
										"Numbers of eigenfunction coefficients do not match.");
							}
							if( localCoef.m() != numBasis ){
								throw std::runtime_error(
										"Number of LGL grids do not match.");
							}
							DblNumVec  localPsiLGL( numGrid );
							DblNumVec  localPsiUniform( numUniformGridElem_.prod() );
							SetValue( localPsiLGL, 0.0 );

							// Compute local wavefunction on the LGL grid
							blas::Gemv( 'N', numGrid, numBasis, 1.0, 
									localBasis.Data(), numGrid, 
									localCoef.VecData(g), 1, 0.0,
									localPsiLGL.Data(), 1 );

							// Interpolate local wavefunction from LGL grid to uniform grid
							InterpLGLToUniform( 
									numLGLGridElem_, 
									numUniformGridElem_, 
									localPsiLGL.Data(), 
									localPsiUniform.Data() );

							// Compute the local norm
							normPsiLocal += Energy( localPsiUniform );

							psiUniform.LocalMap()[key] = localPsiUniform;

						} // own this element
					} // for (i)

			// All processors get the normalization factor
			mpi::Allreduce( &normPsiLocal, &normPsi, 1, MPI_SUM, domain_.comm );

			// pre-constant in front of psi^2 for density
			Real rhofac = (numSpin_ * domain_.NumGridTotal() / domain_.Volume() ) 
				* occrate[g] / normPsi;

			// Add the normalized wavefunction to density
			for( Int k = 0; k < numElem_[2]; k++ )
				for( Int j = 0; j < numElem_[1]; j++ )
					for( Int i = 0; i < numElem_[0]; i++ ){
						Index3 key( i, j, k );
						if( elemPrtn_.Owner( key ) == mpirank ){
							DblNumVec& localRho = density_.LocalMap()[key];
							DblNumVec& localPsi = psiUniform.LocalMap()[key];
							for( Int p = 0; p < localRho.Size(); p++ ){
								localRho[p] += localPsi[p] * localPsi[p] * rhofac;
							}	
						} // own this element
					} // for (i)
		} // for (g)
		// Check the sum of the electron density
#if ( _DEBUGlevel_ >= 0 )
		Real sumRhoLocal = 0.0;
		Real sumRho      = 0.0;
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner( key ) == mpirank ){
						DblNumVec& localRho = density_.LocalMap()[key];
						for( Int p = 0; p < localRho.Size(); p++ ){
							sumRhoLocal += localRho[p];
						}	
					} // own this element
				} // for (i)
		mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.comm );

		sumRho *= domain_.Volume() / domain_.NumGridTotal();

		Print( statusOFS, "Sum rho = ", sumRho );
#endif
	} // Method 1


	// Method 2: Compute the electron density locally, and then normalize
	// only in the global domain. The result should be almost the same as
	// that in Method 1, but should be much faster.
	if(0)
	{
		Real sumRhoLocal = 0.0, sumRho = 0.0;
		// Compute the local density in each element
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner( key ) == mpirank ){
						DblNumMat& localBasis = basisLGL_.LocalMap()[key];
						Int numGrid  = localBasis.m();
						Int numBasis = localBasis.n();

						// Skip the element if there is no basis functions.
						if( numBasis == 0 )
							continue;

						DblNumMat& localCoef  = eigvecCoef_.LocalMap()[key];
						if( localCoef.n() != numEig ){
							throw std::runtime_error( 
									"Numbers of eigenfunction coefficients do not match.");
						}
						if( localCoef.m() != numBasis ){
							throw std::runtime_error(
									"Number of LGL grids do not match.");
						}
						
						DblNumVec& localRho = density_.LocalMap()[key];

						DblNumVec  localRhoLGL( numGrid );
						DblNumVec  localPsiLGL( numGrid );
						SetValue( localRhoLGL, 0.0 );
						SetValue( localPsiLGL, 0.0 );

						// Loop over all the eigenfunctions
						// 
						// NOTE: Gemm is not a feasible choice when a large number of
						// eigenfunctions are there.
						for( Int g = 0; g < numEig; g++ ){
							// Compute local wavefunction on the LGL grid
							blas::Gemv( 'N', numGrid, numBasis, 1.0, 
									localBasis.Data(), numGrid, 
									localCoef.VecData(g), 1, 0.0,
									localPsiLGL.Data(), 1 );
							// Update the local density
							Real  occ    = occrate[g];
							for( Int p = 0; p < numGrid; p++ ){
								localRhoLGL(p) += pow( localPsiLGL(p), 2.0 ) * occ;
							}
						}

						// Interpolate the local density from LGL grid to uniform
						// grid
						InterpLGLToUniform( 
								numLGLGridElem_, 
								numUniformGridElem_, 
								localRhoLGL.Data(), 
								localRho.Data() );

						Real* ptrRho = localRho.Data();
						for( Int p = 0; p < localRho.Size(); p++ ){
							sumRhoLocal += (*ptrRho);
							ptrRho++;
						}

					} // own this element
				} // for (i)

		// All processors get the normalization factor
		mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.comm );

		Real rhofac = numSpin_ * numOccupiedState_ 
			* (domain_.NumGridTotal() / domain_.Volume()) / sumRho;

		// Normalize the electron density in the global domain
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner( key ) == mpirank ){
						DblNumVec& localRho = density_.LocalMap()[key];
						blas::Scal( localRho.Size(), rhofac, localRho.Data(), 1 );
					} // own this element
				} // for (i)
	} // Method 2

	// Method 3: Method 3 is the same as the Method 2, but to output the
	// eigenfunctions locally. TODO
	if(1)
	{
		Real sumRhoLocal = 0.0, sumRho = 0.0;
		Real sumRhoLGLLocal = 0.0, sumRhoLGL = 0.0;
		// Generate the LGL weight. FIXME. Put it to hamiltonian_dg
		// Compute the integration weights
		DblNumTns               LGLWeight3D;
		{
			std::vector<DblNumVec>  LGLWeight1D(DIM);
			Point3 length       = domainElem_(0,0,0).length;
			Index3 numGrid      = numLGLGridElem_;             
			Int    numGridTotal = numGrid.prod();

			// Compute the integration weights
			// 1D
			for( Int d = 0; d < DIM; d++ ){
				DblNumVec  dummyX;
				DblNumMat  dummyP, dummpD;
				GenerateLGL( dummyX, LGLWeight1D[d], dummyP, dummpD, 
						numGrid[d] );
				blas::Scal( numGrid[d], 0.5 * length[d], 
						LGLWeight1D[d].Data(), 1 );
			}

			// 3D
			LGLWeight3D.Resize( numGrid[0], numGrid[1], numGrid[2] );
			for( Int k = 0; k < numGrid[2]; k++ )
				for( Int j = 0; j < numGrid[1]; j++ )
					for( Int i = 0; i < numGrid[0]; i++ ){
						LGLWeight3D(i, j, k) = LGLWeight1D[0](i) * LGLWeight1D[1](j) *
							LGLWeight1D[2](k);
					} // for (i)
		}

		// Clear the density FIXME. Combine with above
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner( key ) == mpirank ){
						DblNumVec& localRho    = density_.LocalMap()[key];
						DblNumVec& localRhoLGL = densityLGL_.LocalMap()[key];

						SetValue( localRho, 0.0 );
						SetValue( localRhoLGL, 0.0 );
						
					}
				} // for (i)

		// Compute the local density in each element
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner( key ) == mpirank ){
						DblNumMat& localBasis = basisLGL_.LocalMap()[key];
						Int numGrid  = localBasis.m();
						Int numBasis = localBasis.n();

						// Skip the element if there is no basis functions.
						if( numBasis == 0 )
							continue;

						DblNumMat& localCoef  = eigvecCoef_.LocalMap()[key];
						if( localCoef.n() != numEig ){
							throw std::runtime_error( 
									"Numbers of eigenfunction coefficients do not match.");
						}
						if( localCoef.m() != numBasis ){
							throw std::runtime_error(
									"Number of LGL grids do not match.");
						}
						
						DblNumVec& localRho    = density_.LocalMap()[key];
						DblNumVec& localRhoLGL = densityLGL_.LocalMap()[key];

						DblNumVec  localPsiLGL( numGrid );
						SetValue( localPsiLGL, 0.0 );


						// Loop over all the eigenfunctions
						// 
						// NOTE: Gemm is not a feasible choice when a large number of
						// eigenfunctions are there.
						for( Int g = 0; g < numEig; g++ ){
							// Compute local wavefunction on the LGL grid
							blas::Gemv( 'N', numGrid, numBasis, 1.0, 
									localBasis.Data(), numGrid, 
									localCoef.VecData(g), 1, 0.0,
									localPsiLGL.Data(), 1 );
							// Update the local density
							Real  occ    = occrate[g];
							for( Int p = 0; p < numGrid; p++ ){
								localRhoLGL(p) += pow( localPsiLGL(p), 2.0 ) * occ * numSpin_;
							}
						}

						statusOFS << "Before interpolation" << std::endl;

						// Interpolate the local density from LGL grid to uniform
						// grid
						InterpLGLToUniform( 
								numLGLGridElem_, 
								numUniformGridElem_, 
								localRhoLGL.Data(), 
								localRho.Data() );
						statusOFS << "After interpolation" << std::endl;

						sumRhoLGLLocal += blas::Dot( localRhoLGL.Size(),
								localRhoLGL.Data(), 1, 
								LGLWeight3D.Data(), 1 );

						Real* ptrRho = localRho.Data();
						for( Int p = 0; p < localRho.Size(); p++ ){
							sumRhoLocal += (*ptrRho);
							ptrRho++;
						}

					} // own this element
				} // for (i)

		sumRhoLocal *= domain_.Volume() / domain_.NumGridTotal(); 

		// All processors get the normalization factor
		mpi::Allreduce( &sumRhoLGLLocal, &sumRhoLGL, 1, MPI_SUM, domain_.comm );
		mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.comm );

#if ( _DEBUGlevel_ >= 0 )
		statusOFS << std::endl;
		Print( statusOFS, "Sum Rho on LGL grid (raw data) = ", sumRhoLGL );
		Print( statusOFS, "Sum Rho on uniform grid (interpolated) = ", sumRho );
		statusOFS << std::endl;
#endif
		

		Real rhofac = numSpin_ * numOccupiedState_ / sumRho;
	  
		// FIXME No normalizatoin of the electron density!

//		// Normalize the electron density in the global domain
//		for( Int k = 0; k < numElem_[2]; k++ )
//			for( Int j = 0; j < numElem_[1]; j++ )
//				for( Int i = 0; i < numElem_[0]; i++ ){
//					Index3 key( i, j, k );
//					if( elemPrtn_.Owner( key ) == mpirank ){
//						DblNumVec& localRho = density_.LocalMap()[key];
//						blas::Scal( localRho.Size(), rhofac, localRho.Data(), 1 );
//					} // own this element
//				} // for (i)
	}
#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method HamiltonianDG::CalculateDensity  ----- 



void
HamiltonianDG::CalculateXC	( Real &Exc )
{
#ifndef _RELEASE_
	PushCallStack("HamiltonianDG::CalculateXC");
#endif
	Int mpirank, mpisize;
	MPI_Comm_rank( domain_.comm, &mpirank );
	MPI_Comm_size( domain_.comm, &mpisize );
	
	Real ExcLocal = 0.0;

	for( Int k = 0; k < numElem_[2]; k++ )
		for( Int j = 0; j < numElem_[1]; j++ )
			for( Int i = 0; i < numElem_[0]; i++ ){
				Index3 key( i, j, k );
				if( elemPrtn_.Owner( key ) == mpirank ){
					DblNumVec& localRho   = density_.LocalMap()[key];
					DblNumVec& localEpsxc = epsxc_.LocalMap()[key];
					DblNumVec& localVxc   = vxc_.LocalMap()[key];

					switch( XCFuncType_.info->family ){
						case XC_FAMILY_LDA:
							xc_lda_exc_vxc( &XCFuncType_, localRho.Size(), 
									localRho.Data(),
									localEpsxc.Data(), 
									localVxc.Data() );
							break;
						default:
							throw std::logic_error( "Unsupported XC family!" );
							break;
					}
					ExcLocal += blas::Dot( localRho.Size(), 
							localRho.Data(), 1, localEpsxc.Data(), 1 );
				} // own this element
			} // for (i)

	ExcLocal *= domain_.Volume() / domain_.NumGridTotal();

	mpi::Allreduce( &ExcLocal, &Exc, 1, MPI_SUM, domain_.comm );

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method HamiltonianDG::CalculateXC  ----- 

void HamiltonianDG::CalculateHartree( DistFourier& fft ) {
#ifndef _RELEASE_ 
	PushCallStack("HamiltonianDG::CalculateHartree");
#endif
	if( !fft.isInitialized ){
		throw std::runtime_error("Fourier is not prepared.");
	}
 
	Int mpirank, mpisize;
	MPI_Comm_rank( domain_.comm, &mpirank );
	MPI_Comm_size( domain_.comm, &mpisize );

  Int ntot      = fft.numGridTotal;
	Int ntotLocal = fft.numGridLocal;

	DistDblNumVec   tempVec;
	tempVec.Prtn() = elemPrtn_;

	// tempVec = density_ - pseudoCharge_
	for( Int k = 0; k < numElem_[2]; k++ )
		for( Int j = 0; j < numElem_[1]; j++ )
			for( Int i = 0; i < numElem_[0]; i++ ){
				Index3 key = Index3( i, j, k );
				if( elemPrtn_.Owner( key ) == mpirank ){
					tempVec.LocalMap()[key] = density_.LocalMap()[key];
					blas::Axpy( numUniformGridElem_.prod(), -1.0, 
							pseudoCharge_.LocalMap()[key].Data(), 1,
							tempVec.LocalMap()[key].Data(), 1 );
				}
			}

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

	// The contribution of the pseudoCharge is subtracted. So the Poisson
	// equation is well defined for neutral system.
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
				// NOTE: gkk already contains the factor 1/2.
				fft.outputComplexVecLocal(i) *= 2.0 * PI / fft.gkkLocal(i);
			}
		}
		fftw_execute( fft.backwardPlan );

		// tempVecLocal saves the Hartree potential

		for( Int i = 0; i < ntotLocal; i++ ){
			tempVecLocal(i) = fft.inputComplexVecLocal(i).real() / ntot;
		}
	} // if (fft.isInGrid)

	// Convert tempVecLocal to vhart_ in the DistNumVec format

  DistRowVecToDistNumVec(
			tempVecLocal,
			vhart_,
			domain_.numGrid,
			numElem_,
			fft.localNzStart,
			fft.localNz,
			fft.isInGrid,
			domain_.comm );


#ifndef _RELEASE_
	PopCallStack();
#endif
	return; 
}  // -----  end of method HamiltonianDG::CalculateHartree ----- 

void
HamiltonianDG::CalculateVtot	( DistDblNumVec& vtot  )
{
#ifndef _RELEASE_
	PushCallStack("HamiltonianDG::CalculateVtot");
#endif
	Int mpirank, mpisize;
	MPI_Comm_rank( domain_.comm, &mpirank );
	MPI_Comm_size( domain_.comm, &mpisize );

	for( Int k = 0; k < numElem_[2]; k++ )
		for( Int j = 0; j < numElem_[1]; j++ )
			for( Int i = 0; i < numElem_[0]; i++ ){
				Index3 key = Index3( i, j, k );
				if( elemPrtn_.Owner( key ) == mpirank ){
					DblNumVec&   localVtot  = vtot.LocalMap()[key];
					DblNumVec&   localVext  = vext_.LocalMap()[key];
					DblNumVec&   localVhart = vhart_.LocalMap()[key];
					DblNumVec&   localVxc   = vxc_.LocalMap()[key];

					localVtot.Resize( localVxc.Size() );
					for( Int p = 0; p < localVtot.Size(); p++){
						localVtot[p] = localVext[p] + localVhart[p] +
							localVxc[p];
					}
				}
			} // for (i)

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method HamiltonianDG::CalculateVtot  ----- 


void
HamiltonianDG::CalculateForce	( DistFourier& fft )
{
#ifndef _RELEASE_
	PushCallStack("HamiltonianDG::CalculateForce");
#endif
	if( !fft.isInitialized ){
		throw std::runtime_error("Fourier is not prepared.");
	}
 
	Int mpirank, mpisize;
	MPI_Comm_rank( domain_.comm, &mpirank );
	MPI_Comm_size( domain_.comm, &mpisize );


	// *********************************************************************
	// Initialize the force computation
	// *********************************************************************
  Int ntot      = fft.numGridTotal;
	Int ntotLocal = fft.numGridLocal;
	Int numAtom   = atomList_.size();


	DblNumMat   forceLocal( numAtom, DIM );
	DblNumMat   force( numAtom, DIM );
	SetValue( forceLocal, 0.0 );
	SetValue( force, 0.0 );
	


	// Compute the integration weights
	DblNumTns               LGLWeight3D;
	{
		std::vector<DblNumVec>  LGLWeight1D(DIM);
		Point3 length       = domainElem_(0,0,0).length;
		Index3 numGrid      = numLGLGridElem_;             
		Int    numGridTotal = numGrid.prod();

		// Compute the integration weights
		// 1D
		for( Int d = 0; d < DIM; d++ ){
			DblNumVec  dummyX;
			DblNumMat  dummyP, dummpD;
			GenerateLGL( dummyX, LGLWeight1D[d], dummyP, dummpD, 
					numGrid[d] );
			blas::Scal( numGrid[d], 0.5 * length[d], 
					LGLWeight1D[d].Data(), 1 );
		}

		// 3D
		LGLWeight3D.Resize( numGrid[0], numGrid[1], numGrid[2] );
		for( Int k = 0; k < numGrid[2]; k++ )
			for( Int j = 0; j < numGrid[1]; j++ )
				for( Int i = 0; i < numGrid[0]; i++ ){
					LGLWeight3D(i, j, k) = LGLWeight1D[0](i) * LGLWeight1D[1](j) *
						LGLWeight1D[2](k);
				} // for (i)
	}


	// *********************************************************************
	// Compute the derivative of the Hartree potential for computing the 
	// local pseudopotential contribution to the Hellmann-Feynman force
	// *********************************************************************
	std::vector<DistDblNumVec>  vhartDrv(DIM);
	std::vector<DblNumVec>      vhartDrvLocal(DIM);
	DistDblNumVec   tempVec;

	tempVec.Prtn() = elemPrtn_;
	for( Int d = 0; d < DIM; d++ )
		vhartDrv[d].Prtn() = elemPrtn_;

	// tempVec = density_ - pseudoCharge_
	for( Int k = 0; k < numElem_[2]; k++ )
		for( Int j = 0; j < numElem_[1]; j++ )
			for( Int i = 0; i < numElem_[0]; i++ ){
				Index3 key = Index3( i, j, k );
				if( elemPrtn_.Owner( key ) == mpirank ){
					tempVec.LocalMap()[key] = density_.LocalMap()[key];
					blas::Axpy( numUniformGridElem_.prod(), -1.0, 
							pseudoCharge_.LocalMap()[key].Data(), 1,
							tempVec.LocalMap()[key].Data(), 1 );
				}
			}

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

	// The contribution of the pseudoCharge is subtracted. So the Poisson
	// equation is well defined for neutral system.
	// Only part of the processors participate in the FFTW calculation

	if( fft.isInGrid ){

		// cpxVecLocal saves the Fourier transform of 
		// density_ - pseudoCharge_ 
		CpxNumVec  cpxVecLocal( tempVecLocal.Size() );

		for( Int i = 0; i < ntotLocal; i++ ){
			fft.inputComplexVecLocal(i) = Complex( 
					tempVecLocal(i), 0.0 );
		}

		fftw_execute( fft.forwardPlan );

		blas::Copy( ntotLocal, fft.outputComplexVecLocal.Data(), 1,
				cpxVecLocal.Data(), 1 );

		// Compute the derivative of the Hartree potential via Fourier
		// transform 
		for( Int d = 0; d < DIM; d++ ){
			CpxNumVec& ikLocal  = fft.ikLocal[d];
			for( Int i = 0; i < ntotLocal; i++ ){
				if( fft.gkkLocal(i) == 0 ){
					fft.outputComplexVecLocal(i) = Z_ZERO;
				}
				else{
					// NOTE: gkk already contains the factor 1/2.
					fft.outputComplexVecLocal(i) = cpxVecLocal(i) *
						2.0 * PI / fft.gkkLocal(i) * ikLocal(i);
				}
			}

			fftw_execute( fft.backwardPlan );

			// vhartDrvLocal saves the derivative of the Hartree potential in
			// the distributed row format
			vhartDrvLocal[d].Resize( tempVecLocal.Size() );

			for( Int i = 0; i < ntotLocal; i++ ){
				vhartDrvLocal[d](i) = fft.inputComplexVecLocal(i).real() / ntot;
			}

		} // for (d)


	} // if (fft.isInGrid)

	// Convert vhartDrvLocal to vhartDrv in the DistNumVec format

	for( Int d = 0; d < DIM; d++ ){
		DistRowVecToDistNumVec( 
				vhartDrvLocal[d],
				vhartDrv[d],
				domain_.numGrid,
				numElem_,
				fft.localNzStart,
				fft.localNz,
				fft.isInGrid,
				domain_.comm );
	}


	
		
	// *********************************************************************
	// Compute the force from local pseudopotential
	// *********************************************************************
	{
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner( key ) == mpirank ){
						std::map<Int, PseudoPot>&  ppMap = pseudo_.LocalMap()[key];
						for( std::map<Int, PseudoPot>::iterator mi = ppMap.begin();
								 mi != ppMap.end(); mi++ ){
							Int atomIdx = (*mi).first;
							PseudoPot& pp = (*mi).second;
							SparseVec& sp = pp.pseudoCharge;
							IntNumVec& idx = sp.first;
							DblNumMat& val = sp.second;
							Real    wgt = domain_.Volume() / domain_.NumGridTotal();
							for( Int d = 0; d < DIM; d++ ){
								DblNumVec&  drv = vhartDrv[d].LocalMap()[key];
								Real res = 0.0;
								for( Int l = 0; l < idx.m(); l++ ){
									res += val(l, VAL) * drv[idx(l)] * wgt;
								}
								forceLocal( atomIdx, d ) += res;
							}
						} // for (mi)
					} // own this element
				} // for (i)
	}


	// *********************************************************************
	// Compute the force from nonlocal pseudopotential
	// *********************************************************************
	{
		// Step 1. Collect the eigenvectors from the neighboring elements
		// according to the support of the pseudopotential
		// Note: the following part is the same as that in 
		//
		// HamiltonianDG::CalculateDGMatrix
		//
		// Each element owns all the coefficient matrices in its neighbors
		// and then perform data processing later. It can be as many as 
		// 3^3-1 = 26 elements. 
		//
		// Note that it is assumed that the size of the element size cannot
		// be smaller than the pseudopotential (local or nonlocal) cutoff radius.
		//
		// Use std::set to avoid repetitive entries
		std::set<Index3>  pseudoSet;
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner(key) == mpirank ){
						IntNumVec  idxX(3);
						IntNumVec  idxY(3);
						IntNumVec  idxZ(3); 

						// Previous
						if( i == 0 )  idxX(0) = numElem_[0]-1; else   idxX(0) = i-1;
						if( j == 0 )  idxY(0) = numElem_[1]-1; else   idxY(0) = j-1;
						if( k == 0 )  idxZ(0) = numElem_[2]-1; else   idxZ(0) = k-1;

						// Current
						idxX(1) = i;
						idxY(1) = j;
						idxZ(1) = k;

						// Next
						if( i == numElem_[0]-1 )  idxX(2) = 0; else   idxX(2) = i+1;
						if( j == numElem_[1]-1 )  idxY(2) = 0; else   idxY(2) = j+1;
						if( k == numElem_[2]-1 )  idxZ(2) = 0; else   idxZ(2) = k+1;

						// Tensor product 
						for( Int c = 0; c < 3; c++ )
							for( Int b = 0; b < 3; b++ )
								for( Int a = 0; a < 3; a++ ){
									// Not the element key itself
									if( idxX[a] != i || idxY[b] != j || idxZ[c] != k ){
										pseudoSet.insert( Index3( idxX(a), idxY(b), idxZ(c) ) );
									}
								} // for (a)
					}
				} // for (i)
		std::vector<Index3>  pseudoIdx;
		pseudoIdx.insert( pseudoIdx.begin(), pseudoSet.begin(), pseudoSet.end() );
		
		eigvecCoef_.GetBegin( pseudoIdx, NO_MASK );
		eigvecCoef_.GetEnd( NO_MASK );

		// Step 2. Loop through the atoms and eigenvecs for the contribution
		// to the force
		//
		// Note: this procedure shall be substituted with the density matrix
		// formulation when PEXSI is used. TODO

		// Loop over atoms and pseudopotentials
		Int numEig = occupationRate_.m();
		for( Int atomIdx = 0; atomIdx < numAtom; atomIdx++ ){
			if( atomPrtn_.Owner(atomIdx) == mpirank ){
			  DblNumVec&  vnlWeight = vnlWeightMap_[atomIdx];	
				Int numVnl = vnlWeight.Size();
				DblNumMat resVal ( numEig, numVnl );
				DblNumMat resDrvX( numEig, numVnl );
				DblNumMat resDrvY( numEig, numVnl );
				DblNumMat resDrvZ( numEig, numVnl );
				SetValue( resVal,  0.0 );
				SetValue( resDrvX, 0.0 );
				SetValue( resDrvY, 0.0 );
				SetValue( resDrvZ, 0.0 );

				// Loop over the elements overlapping with the nonlocal
				// pseudopotential
				for( std::map<Index3, std::map<Int, DblNumMat> >::iterator 
						ei  = vnlCoef_.LocalMap().begin();
						ei != vnlCoef_.LocalMap().end(); ei++ ){
					Index3 key = (*ei).first;
					std::map<Int, DblNumMat>& coefMap = (*ei).second; 
					std::map<Int, DblNumMat>& coefDrvXMap = vnlDrvCoef_[0].LocalMap()[key];
					std::map<Int, DblNumMat>& coefDrvYMap = vnlDrvCoef_[1].LocalMap()[key];
					std::map<Int, DblNumMat>& coefDrvZMap = vnlDrvCoef_[2].LocalMap()[key];
					
					if( eigvecCoef_.LocalMap().find( key ) == eigvecCoef_.LocalMap().end() ){
						throw std::runtime_error( "Eigenfunction coefficient matrix cannot be located." );
					}

					DblNumMat&  localCoef = eigvecCoef_.LocalMap()[key];

					Int numBasis = localCoef.m();

					if( coefMap.find( atomIdx ) != coefMap.end() ){

						DblNumMat&  coef      = coefMap[atomIdx];
						DblNumMat&  coefDrvX  = coefDrvXMap[atomIdx];
						DblNumMat&  coefDrvY  = coefDrvYMap[atomIdx];
						DblNumMat&  coefDrvZ  = coefDrvZMap[atomIdx];
						
						// Skip the calculation if there is no adaptive local
						// basis function.  
						if( coef.m() == 0 ){
							continue;
						}

						// Value
						blas::Gemm( 'T', 'N', numEig, numVnl, numBasis,
								1.0, localCoef.Data(), numBasis, 
								coef.Data(), numBasis,
								1.0, resVal.Data(), numEig );
						
						// Derivative
						blas::Gemm( 'T', 'N', numEig, numVnl, numBasis,
								1.0, localCoef.Data(), numBasis, 
								coefDrvX.Data(), numBasis,
								1.0, resDrvX.Data(), numEig );

						blas::Gemm( 'T', 'N', numEig, numVnl, numBasis,
								1.0, localCoef.Data(), numBasis, 
								coefDrvY.Data(), numBasis,
								1.0, resDrvY.Data(), numEig );

						blas::Gemm( 'T', 'N', numEig, numVnl, numBasis,
								1.0, localCoef.Data(), numBasis, 
								coefDrvZ.Data(), numBasis,
								1.0, resDrvZ.Data(), numEig );

					} // found the atom
				} // for (ei)

				// Add the contribution to the local force
				// The minus sign comes from integration by parts
				// The 4.0 comes from spin (2.0) and that |l> appears twice (2.0)
				for( Int g = 0; g < numEig; g++ ){
					for( Int l = 0; l < numVnl; l++ ){
						forceLocal(atomIdx, 0) += -4.0 * occupationRate_[g] * vnlWeight[l] *
							resVal(g, l) * resDrvX(g, l);
						forceLocal(atomIdx, 1) += -4.0 * occupationRate_[g] * vnlWeight[l] *
							resVal(g, l) * resDrvY(g, l);
						forceLocal(atomIdx, 2) += -4.0 * occupationRate_[g] * vnlWeight[l] *
							resVal(g, l) * resDrvZ(g, l);
					}
				}
			} // own this atom
		} // for (atomIdx)

	}


	// *********************************************************************
	// Compute the total force and give the value to atomList
	// *********************************************************************
	mpi::Allreduce( forceLocal.Data(), force.Data(), numAtom * DIM,
			MPI_SUM, domain_.comm );

	for( Int a = 0; a < numAtom; a++ ){
		atomList_[a].force = Point3( force(a,0), force(a,1), force(a,2) );
	} 

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method HamiltonianDG::CalculateForce  ----- 



void
HamiltonianDG::CalculateAPosterioriError	( 
		DblNumTns&       eta2Total,
		DblNumTns&       eta2Residual,
		DblNumTns&       eta2GradJump,
		DblNumTns&       eta2Jump	)
{
#ifndef _RELEASE_
	PushCallStack("HamiltonianDG::CalculateAPosterioriError");
#endif
  
	Int mpirank, mpisize;
	Int numAtom = atomList_.size();
	MPI_Comm_rank( domain_.comm, &mpirank );
	MPI_Comm_size( domain_.comm, &mpisize );
	Real timeSta, timeEnd;
  
	// Here numGrid is the LGL grid
	Point3 length       = domainElem_(0,0,0).length;
	Index3 numGrid      = numLGLGridElem_;             
	Int    numGridTotal = numGrid.prod();
	Int    numSpin      = this->NumSpin();

	// Hamiltonian acting on the basis functions
	DistDblNumMat   HbasisLGL;

	// The derivative of basisLGL along x,y,z directions
	std::vector<DistDblNumMat>   Dbasis(DIM);

	// The inner product of <l|psi> for each pair of nonlocal projector
	// saved on the local processor and all eigenfunctions psi.
	//
	// The data attribute of this structure is 
	//
	// atom index (map to) Matrix value of the corresponding <l|psi>
	DistVec<Index3, std::map<Int, DblNumMat>, ElemPrtn>  vnlPsi;

	// Jump of the value of the basis, and average of the
	// derivative of the basis function, each of size 6 describing
	// the different faces along the X/Y/Z directions. L/R: left/right.
	enum{
		XL = 0,
		XR = 1,
		YL = 2,
		YR = 3,
		ZL = 4,
		ZR = 5,
		NUM_FACE = 6,
	};
	std::vector<DistDblNumMat>   basisJump(NUM_FACE);
	std::vector<DistDblNumMat>   DbasisJump(NUM_FACE);


	// Define the local estimators on each processor.
	DblNumTns eta2TotalLocal( numElem_[0], numElem_[1], numElem_[2] );
	DblNumTns eta2ResidualLocal( numElem_[0], numElem_[1], numElem_[2] );
	DblNumTns eta2GradJumpLocal( numElem_[0], numElem_[1], numElem_[2] );
	DblNumTns eta2JumpLocal( numElem_[0], numElem_[1], numElem_[2] );



	// Integration weights
	std::vector<DblNumVec>  LGLWeight1D(DIM);
	std::vector<DblNumMat>  LGLWeight2D(DIM);
	DblNumTns               LGLWeight3D;


	// *********************************************************************
	// Initial setup
	// *********************************************************************

	{
		HbasisLGL.Prtn()     = elemPrtn_;

		for( Int i = 0; i < NUM_FACE; i++ ){
			basisJump[i].Prtn()     = elemPrtn_;
			DbasisJump[i].Prtn()    = elemPrtn_;
		}

		for( Int i = 0; i < DIM; i++ ){
			Dbasis[i].Prtn() = elemPrtn_;
		}

		vnlPsi.Prtn()        = elemPrtn_;

		eta2Total.Resize( numElem_[0], numElem_[1], numElem_[2] );
		eta2Residual.Resize( numElem_[0], numElem_[1], numElem_[2] );
		eta2GradJump.Resize( numElem_[0], numElem_[1], numElem_[2] );
		eta2Jump.Resize( numElem_[0], numElem_[1], numElem_[2] );
		
		SetValue( eta2Total, 0.0 );
		SetValue( eta2Residual, 0.0 );
		SetValue( eta2GradJump, 0.0 );
		SetValue( eta2Jump, 0.0 );

		SetValue( eta2TotalLocal, 0.0 );
		SetValue( eta2ResidualLocal, 0.0 );
		SetValue( eta2GradJumpLocal, 0.0 );
		SetValue( eta2JumpLocal, 0.0 );


		// Compute the integration weights
		// 1D
		for( Int d = 0; d < DIM; d++ ){
			DblNumVec  dummyX;
			DblNumMat  dummyP, dummpD;
			GenerateLGL( dummyX, LGLWeight1D[d], dummyP, dummpD, 
					numGrid[d] );
			blas::Scal( numGrid[d], 0.5 * length[d], 
					LGLWeight1D[d].Data(), 1 );
		}

		// 2D: faces labeled by normal vectors, i.e. 
		// yz face : 0
		// xz face : 1
		// xy face : 2

		// yz face
		LGLWeight2D[0].Resize( numGrid[1], numGrid[2] );
		for( Int k = 0; k < numGrid[2]; k++ )
			for( Int j = 0; j < numGrid[1]; j++ ){
				LGLWeight2D[0](j, k) = LGLWeight1D[1](j) * LGLWeight1D[2](k);
			} // for (j)

		// xz face
		LGLWeight2D[1].Resize( numGrid[0], numGrid[2] );
		for( Int k = 0; k < numGrid[2]; k++ )
			for( Int i = 0; i < numGrid[0]; i++ ){
				LGLWeight2D[1](i, k) = LGLWeight1D[0](i) * LGLWeight1D[2](k);
			} // for (i)

		// xy face
		LGLWeight2D[2].Resize( numGrid[0], numGrid[1] );
		for( Int j = 0; j < numGrid[1]; j++ )
			for( Int i = 0; i < numGrid[0]; i++ ){
				LGLWeight2D[2](i, j) = LGLWeight1D[0](i) * LGLWeight1D[1](j);
			}


		// 3D
		LGLWeight3D.Resize( numGrid[0], numGrid[1],
				numGrid[2] );
		for( Int k = 0; k < numGrid[2]; k++ )
			for( Int j = 0; j < numGrid[1]; j++ )
				for( Int i = 0; i < numGrid[0]; i++ ){
					LGLWeight3D(i, j, k) = LGLWeight1D[0](i) * LGLWeight1D[1](j) *
						LGLWeight1D[2](k);
				} // for (i)
	}


	// *********************************************************************
	// Collect the eigenvectors from the neighboring elements according to
	// the support of the pseudopotential
	// *********************************************************************
	{
		// Each element owns all the coefficient matrices in its neighbors
		// and then perform data processing later. It can be as many as 
		// 3^3-1 = 26 elements. 
		//
		// Note that it is assumed that the size of the element size cannot
		// be smaller than the pseudopotential (local or nonlocal) cutoff radius.
		//
		// Use std::set to avoid repetitive entries
		GetTime( timeSta );
		std::set<Index3>  pseudoSet;
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner(key) == mpirank ){
						IntNumVec  idxX(3);
						IntNumVec  idxY(3);
						IntNumVec  idxZ(3); 

						// Previous
						if( i == 0 )  idxX(0) = numElem_[0]-1; else   idxX(0) = i-1;
						if( j == 0 )  idxY(0) = numElem_[1]-1; else   idxY(0) = j-1;
						if( k == 0 )  idxZ(0) = numElem_[2]-1; else   idxZ(0) = k-1;

						// Current
						idxX(1) = i;
						idxY(1) = j;
						idxZ(1) = k;

						// Next
						if( i == numElem_[0]-1 )  idxX(2) = 0; else   idxX(2) = i+1;
						if( j == numElem_[1]-1 )  idxY(2) = 0; else   idxY(2) = j+1;
						if( k == numElem_[2]-1 )  idxZ(2) = 0; else   idxZ(2) = k+1;

						// Tensor product 
						for( Int c = 0; c < 3; c++ )
							for( Int b = 0; b < 3; b++ )
								for( Int a = 0; a < 3; a++ ){
									// Not the element key itself
									if( idxX[a] != i || idxY[b] != j || idxZ[c] != k ){
										pseudoSet.insert( Index3( idxX(a), idxY(b), idxZ(c) ) );
									}
								} // for (a)
					}
				} // for (i)
		std::vector<Index3>  pseudoIdx;
		pseudoIdx.insert( pseudoIdx.begin(), pseudoSet.begin(), pseudoSet.end() );
		
		eigvecCoef_.GetBegin( pseudoIdx, NO_MASK );
		eigvecCoef_.GetEnd( NO_MASK );
		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for getting the coefficients is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	}

	// *********************************************************************
	// Compute the residual term.
	// *********************************************************************

	// Prepare the H*basis but without the nonlocal contribution
	{
		GetTime(timeSta);

		// Compute H * basis
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner(key) == mpirank ){
						DblNumMat&  basis = basisLGL_.LocalMap()[key];
						Int numBasis = basis.n();

						DblNumMat empty( basis.m(), basis.n() );
						SetValue( empty, 0.0 );
						HbasisLGL.LocalMap()[key] = empty; 

						// Skip the calculation if there is no basis functions in
						// the element.
						if( numBasis == 0 )
							continue;

						DblNumMat& Hbasis = HbasisLGL.LocalMap()[key];

						// Laplacian part
						for( Int d = 0; d < DIM; d++ ){
							DblNumMat D(basis.m(), basis.n());
							DblNumMat D2(basis.m(), basis.n());
							SetValue( D, 0.0 );
							SetValue( D2, 0.0 );
							for( Int g = 0; g < numBasis; g++ ){
								DiffPsi( numGrid, basis.VecData(g), D.VecData(g), d );
								DiffPsi( numGrid, D.VecData(g), D2.VecData(g), d );
							}
							Dbasis[d].LocalMap()[key] = D;
							blas::Axpy( D2.Size(), -0.5, D2.Data(), 1,
									Hbasis.Data(), 1 );
						}

						// Local pseudopotential part
						{
							DblNumVec&  vtot  = vtotLGL_.LocalMap()[key];
							for( Int g = 0; g < numBasis; g++ ){
								Real*   ptrVtot   = vtot.Data();
								Real*   ptrBasis  = basis.VecData(g);
								Real*   ptrHbasis = Hbasis.VecData(g);
								for( Int p = 0; p < vtot.Size(); p++ ){
									*(ptrHbasis++) += (*(ptrVtot++)) * (*(ptrBasis++));
								}
							}
						}

					} // if (own this element)
				} // for (i)
		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for H * basis is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	}

	// Prepare <l|psi> for each nonlocal projector l and eigenfunction
	// psi.  This is done through the structure vnlPsi
	{
		GetTime(timeSta);
		
		Int numEig = occupationRate_.m();

		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner(key) == mpirank ){
						std::map<Int, PseudoPot>& pseudoMap =
							pseudo_.LocalMap()[key];
						std::map<Int, DblNumMat>  vnlPsiMap;

						// Loop over atoms, regardless of whether this atom belongs
						// to this element or not.
						for(std::map<Int, PseudoPot>::iterator 
								mi  = pseudoMap.begin();
								mi != pseudoMap.end(); mi++ ){
							Int atomIdx = (*mi).first;
							std::vector<NonlocalPP>&  vnlList = (*mi).second.vnlList;
							Int numVnl = vnlList.size();
							DblNumMat   vnlPsiMat( numVnl, numEig );
							SetValue( vnlPsiMat, 0.0 );

							// Loop over all neighboring elements to compute the
							// inner product. 
							// vnlCoef saves the information of <l|phi> where phi are
							// the basis functions.  
							//
							// <l|psi_i> = \sum_{jk} <l|phi_{jk}> C_{jk;i}
							//
							// where C_{jk;i} is saved in eigvecCoef
							for(std::map<Index3, std::map<Int, DblNumMat> >::iterator 
									ei  = vnlCoef_.LocalMap().begin();
									ei != vnlCoef_.LocalMap().end(); ei++ ){
								Index3 keyNB                      = (*ei).first;
								std::map<Int, DblNumMat>& coefMap = (*ei).second; 
								DblNumMat&  eigCoef               = eigvecCoef_.LocalMap()[keyNB];

								if( coefMap.find( atomIdx ) != coefMap.end() ){

									DblNumMat&  coef      = coefMap[atomIdx];

									Int numBasis = coef.m();

									// Skip the calculation if there is no adaptive local
									// basis function.  
									if( numBasis == 0 ){
										continue;
									}

									// Inner product (NOTE The conjugate is done in a
									// very crude way here, may need to be revised
									// when complex arithmetic is considered)
									blas::Gemm( 'T', 'N', numVnl, numEig, numBasis,
											1.0, coef.Data(), numBasis, 
											eigCoef.Data(), numBasis,
											1.0, vnlPsiMat.Data(), numVnl );

								} // found the atom
							} // for (ei)
							vnlPsiMap[atomIdx] = vnlPsiMat;
						} // for (mi)

						vnlPsi.LocalMap()[key] = vnlPsiMap;

					} // if (own this element)
				} // for (i)
		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for computing <l|psi> is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	}

	{
		GetTime(timeSta);


		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner(key) == mpirank ){
						DblNumMat&  basis =  basisLGL_.LocalMap()[key];
						Int numLGL   = basis.m();
						Int numBasis = basis.n();

						// Do not directly skip the calculatoin even if numBasis ==
						// 0, due to the contribution from the nonlocal
						// pseudopotential. 


						DblNumMat& Hbasis     = HbasisLGL.LocalMap()[key];
						DblNumMat& localCoef  = eigvecCoef_.LocalMap()[key];
						DblNumVec& eig        = eigVal_;
						DblNumVec& occrate    = occupationRate_;

						Int numEig = localCoef.n();

						// Prefactor for the residual term.
						Real hK = length.l2(); // diameter of the domain
						Real pK = std::max( 1, numBasis );
						Real facR = ( hK * hK ) / ( pK * pK );

						DblNumVec  residual( numLGL );
						for( Int g = 0; g < numEig; g++ ){
							SetValue( residual, 0.0 );
							// Contribution from local part
							if( numBasis > 0 ){
								// r = Hbasis * V(g) 
								blas::Gemv( 'N', numLGL, numBasis, 1.0, Hbasis.Data(),
										numLGL, localCoef.VecData(g), 1, 1.0, 
										residual.Data(), 1 );
								// r = r - e(g) * basis * V(g)
								blas::Gemv( 'N', numLGL, numBasis, -eig(g), basis.Data(),
										numLGL, localCoef.VecData(g), 1, 1.0,
										residual.Data(), 1 );
							} // local part

							// Contribution from the nonlocal pseudopotential
							{
								std::map<Int, PseudoPot>& pseudoMap  = pseudo_.LocalMap()[key];
								std::map<Int, DblNumMat>&  vnlPsiMap = vnlPsi.LocalMap()[key];
								for(std::map<Int, PseudoPot>::iterator 
										mi  = pseudoMap.begin();
										mi != pseudoMap.end(); mi++ ){
									Int atomIdx = (*mi).first;
									std::vector<NonlocalPP>&  vnlList = (*mi).second.vnlList;
									Int numVnl = vnlList.size();
									DblNumMat&  vnlPsiMat = vnlPsiMap[atomIdx];
									DblNumVec&  vnlWeight = vnlWeightMap_[atomIdx];	

									// Loop over projector
									for( Int l = 0; l < vnlList.size(); l++ ){
										SparseVec&  vnl = vnlList[l].first;
										IntNumVec&  idx = vnl.first;
										DblNumMat&  val = vnl.second;


										Real fac = vnlWeight[l] * vnlPsiMat( l, g );

										for( Int p = 0; p < idx.Size(); p++ ){
											residual[idx(p)] += fac * val(p, VAL);
										}
									}

								} // for (mi)
							}

							Real* ptrR = residual.Data();
							Real* ptrW = LGLWeight3D.Data();
							Real  tmpR = 0.0;
							for( Int p = 0; p < numLGL; p++ ){
								tmpR += (*ptrR) * (*ptrR) * (*ptrW);
								ptrR++; ptrW++;
							}
							eta2ResidualLocal( i, j, k ) += 
								tmpR * occrate(g) * numSpin * facR;
						} // for (eigenfunction)


					} // if (own this element)
				} // for (i)

		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for computing the local residual is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	}

	// *********************************************************************
	// Compute the jump term
	// *********************************************************************
	{
		GetTime(timeSta);
		// Compute average of derivatives and jump of values
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner(key) == mpirank ){
						DblNumMat&  basis = basisLGL_.LocalMap()[key];
						Int numBasis = basis.n();

						// NOTE Still construct empty matrices when numBasis = 0


						// x-direction
						{
							Int  numGridFace = numGrid[1] * numGrid[2];
							DblNumMat emptyX( numGridFace, numBasis );
							SetValue( emptyX, 0.0 );
							basisJump[XL].LocalMap()[key] = emptyX;
							basisJump[XR].LocalMap()[key] = emptyX;
							DbasisJump[XL].LocalMap()[key] = emptyX;
							DbasisJump[XR].LocalMap()[key] = emptyX;

							DblNumMat&  valL = basisJump[XL].LocalMap()[key];
							DblNumMat&  valR = basisJump[XR].LocalMap()[key];
							DblNumMat&  drvL = DbasisJump[XL].LocalMap()[key];
							DblNumMat&  drvR = DbasisJump[XR].LocalMap()[key];
							DblNumMat&  DbasisX = Dbasis[0].LocalMap()[key];

							// Form jumps of the values and derivatives of the basis
							// functions from volume to face.

							// basis(0,:,:)             -> valL
							// basis(numGrid[0]-1,:,:)  -> valR
							// Dbasis(0,:,:)            -> drvL
							// Dbasis(numGrid[0]-1,:,:) -> drvR
							for( Int g = 0; g < numBasis; g++ ){
								Int idx, idxL, idxR;
								for( Int gk = 0; gk < numGrid[2]; gk++ )
									for( Int gj = 0; gj < numGrid[1]; gj++ ){
										idx  = gj + gk*numGrid[1];
										idxL = 0 + gj*numGrid[0] + gk * (numGrid[0] *
													 numGrid[1]);
										idxR = (numGrid[0]-1) + gj*numGrid[0] + gk * (numGrid[0] *
													numGrid[1]);

										// 1.0, -1.0 comes from jump with different normal vectors
										// [[a]] = -(1.0) a_L + (1.0) a_R
										drvL(idx, g) = -1.0 * DbasisX( idxL, g );
										drvR(idx, g) = +1.0 * DbasisX( idxR, g );

										valL(idx, g) = -1.0 * basis( idxL, g );
										valR(idx, g) = +1.0 * basis( idxR, g );
									} // for (gj)
							} // for (g)

						} // x-direction


						// y-direction
						{
							Int  numGridFace = numGrid[0] * numGrid[2];
							DblNumMat emptyY( numGridFace, numBasis );
							SetValue( emptyY, 0.0 );
							basisJump[YL].LocalMap()[key] = emptyY;
							basisJump[YR].LocalMap()[key] = emptyY;
							DbasisJump[YL].LocalMap()[key] = emptyY;
							DbasisJump[YR].LocalMap()[key] = emptyY;

							DblNumMat&  valL = basisJump[YL].LocalMap()[key];
							DblNumMat&  valR = basisJump[YR].LocalMap()[key];
							DblNumMat&  drvL = DbasisJump[YL].LocalMap()[key];
							DblNumMat&  drvR = DbasisJump[YR].LocalMap()[key];
							DblNumMat&  DbasisY = Dbasis[1].LocalMap()[key];

							// Form jumps of the values and derivatives of the basis
							// functions from volume to face.
							
							// basis(0,:,:)             -> valL
							// basis(numGrid[0]-1,:,:)  -> valR
							// Dbasis(0,:,:)            -> drvL
							// Dbasis(numGrid[0]-1,:,:) -> drvR
							for( Int g = 0; g < numBasis; g++ ){
								Int idx, idxL, idxR;
								for( Int gk = 0; gk < numGrid[2]; gk++ )
									for( Int gi = 0; gi < numGrid[0]; gi++ ){
										idx  = gi + gk*numGrid[0];
										idxL = gi + 0 *numGrid[0] +
											gk * (numGrid[0] * numGrid[1]);
										idxR = gi + (numGrid[1]-1)*numGrid[0] + 
											gk * (numGrid[0] * numGrid[1]);

										// 1.0, -1.0 comes from jump with different normal vectors
										// [[a]] = -(1.0) a_L + (1.0) a_R
										drvL(idx, g) = -1.0 * DbasisY( idxL, g );
										drvR(idx, g) = +1.0 * DbasisY( idxR, g );

										valL(idx, g) = -1.0 * basis( idxL, g );
										valR(idx, g) = +1.0 * basis( idxR, g );
									} // for (gj)
							} // for (g)

						} // y-direction

						// z-direction
						{
							Int  numGridFace = numGrid[0] * numGrid[1];
							DblNumMat emptyZ( numGridFace, numBasis );
							SetValue( emptyZ, 0.0 );
							basisJump[ZL].LocalMap()[key] = emptyZ;
							basisJump[ZR].LocalMap()[key] = emptyZ;
							DbasisJump[ZL].LocalMap()[key] = emptyZ;
							DbasisJump[ZR].LocalMap()[key] = emptyZ;

							DblNumMat&  valL = basisJump[ZL].LocalMap()[key];
							DblNumMat&  valR = basisJump[ZR].LocalMap()[key];
							DblNumMat&  drvL = DbasisJump[ZL].LocalMap()[key];
							DblNumMat&  drvR = DbasisJump[ZR].LocalMap()[key];
							DblNumMat&  DbasisZ = Dbasis[2].LocalMap()[key];

							// Form jumps of the values and derivatives of the basis
							// functions from volume to face.
							
							// basis(0,:,:)             -> valL
							// basis(numGrid[0]-1,:,:)  -> valR
							// Dbasis(0,:,:)            -> drvL
							// Dbasis(numGrid[0]-1,:,:) -> drvR
							for( Int g = 0; g < numBasis; g++ ){
								Int idx, idxL, idxR;
								for( Int gj = 0; gj < numGrid[1]; gj++ )
									for( Int gi = 0; gi < numGrid[0]; gi++ ){
										idx  = gi + gj*numGrid[0];
										idxL = gi + gj*numGrid[0] +
											0 * (numGrid[0] * numGrid[1]);
										idxR = gi + gj*numGrid[0] +
											(numGrid[2]-1) * (numGrid[0] * numGrid[1]);

										// 1.0, -1.0 comes from jump with different normal vectors
										// [[a]] = -(1.0) a_L + (1.0) a_R
										drvL(idx, g) = -1.0 * DbasisZ( idxL, g );
										drvR(idx, g) = +1.0 * DbasisZ( idxR, g );

										valL(idx, g) = -1.0 * basis( idxL, g );
										valR(idx, g) = +1.0 * basis( idxR, g );
									} // for (gj)
							} // for (g)

						} // z-direction

					}
				} // for (i)

		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for constructing the boundary terms is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	}

	{
		GetTime(timeSta);
		std::set<Index3>   boundaryXset;
		std::set<Index3>   boundaryYset;
		std::set<Index3>   boundaryZset;

		std::vector<Index3>   boundaryXIdx;
		std::vector<Index3>   boundaryYIdx; 
		std::vector<Index3>   boundaryZIdx;
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key(i, j, k);
					if( elemPrtn_.Owner(key) == mpirank ){
						// Periodic boundary condition. Everyone only considers the
					  // previous(left/back/down) directions and compute.
						Int p1; if( i == 0 )  p1 = numElem_[0]-1; else   p1 = i-1;
						Int p2; if( j == 0 )  p2 = numElem_[1]-1; else   p2 = j-1;
						Int p3; if( k == 0 )  p3 = numElem_[2]-1; else   p3 = k-1;
						boundaryXset.insert( Index3( p1, j,  k) );
						boundaryYset.insert( Index3( i, p2,  k ) );
						boundaryZset.insert( Index3( i,  j, p3 ) ); 
					}
				} // for (i)
		
		// The left element passes the values on the right face.

		boundaryXIdx.insert( boundaryXIdx.begin(), boundaryXset.begin(), boundaryXset.end() );
		boundaryYIdx.insert( boundaryYIdx.begin(), boundaryYset.begin(), boundaryYset.end() );
		boundaryZIdx.insert( boundaryZIdx.begin(), boundaryZset.begin(), boundaryZset.end() );

		DbasisJump[XR].GetBegin( boundaryXIdx, NO_MASK );
		DbasisJump[YR].GetBegin( boundaryYIdx, NO_MASK );
		DbasisJump[ZR].GetBegin( boundaryZIdx, NO_MASK );

		basisJump[XR].GetBegin( boundaryXIdx, NO_MASK );
		basisJump[YR].GetBegin( boundaryYIdx, NO_MASK );
		basisJump[ZR].GetBegin( boundaryZIdx, NO_MASK );
		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "After the GetBegin part of communication." << std::endl;
		statusOFS << "Time for GetBegin is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	}
	
	{
		GetTime( timeSta );
		DbasisJump[XR].GetEnd( NO_MASK );
		DbasisJump[YR].GetEnd( NO_MASK );
		DbasisJump[ZR].GetEnd( NO_MASK );

		basisJump[XR].GetEnd( NO_MASK );
		basisJump[YR].GetEnd( NO_MASK );
		basisJump[ZR].GetEnd( NO_MASK );

		MPI_Barrier( domain_.comm );
		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for the remaining communication cost is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	}

	{
		GetTime( timeSta );
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner(key) == mpirank ){

						// x-direction
						{
							// keyL is the previous element received from GetBegin/GetEnd.
							// keyR is the current element
							Int p1; if( i == 0 )  p1 = numElem_[0]-1; else   p1 = i-1;
							Index3 keyL( p1, j, k );
							Index3 keyR = key;

							Int  numGridFace = numGrid[1] * numGrid[2];

							// Note that the notation can be very confusing here:
							// The left element (keyL) contributes to the right face
							// (XR), and the right element (keyR) contributes to the
							// left face (XL)
							DblNumMat&  valL = basisJump[XR].LocalMap()[keyL];
							DblNumMat&  valR = basisJump[XL].LocalMap()[keyR];
							DblNumMat&  drvL = DbasisJump[XR].LocalMap()[keyL];
							DblNumMat&  drvR = DbasisJump[XL].LocalMap()[keyR];

							Int numBasisL = valL.n();
							Int numBasisR = valR.n();

							DblNumMat& localCoefL  = eigvecCoef_.LocalMap()[keyL];
							DblNumMat& localCoefR  = eigvecCoef_.LocalMap()[keyR];
							DblNumVec& eig         = eigVal_;
							DblNumVec& occrate     = occupationRate_;

							// Diameter of the face
							Real hF = std::sqrt( length[1] * length[1] + length[2] * length[2] );
							Real pF = std::max( numBasisL, numBasisR );

							// Skip the computation if there is no basis function on
							// either side.  
							if( pF > 0 ){
								// factor in the estimator for jump of the derivative and
								// the jump of the values
								// NOTE:
								// Originally in [Giani 2012] the penalty term in the a
								// posteriori error estimator should be
								//
								// facJ = gamma^2 * p_F^3 / h_F.  However, there the jump term
								// should be
								//
								// gamma * p_F^2 / h_F = penaltyAlpha_
								//
								// used in this code.  So 
								//
								// facJ = penaltyAlpha_^2 * h_F / p_F


								// One 0.5 comes from double counting on the faces F,
								// and the other comes from the 1/2 in front of the
								// Laplacian operator.
								Real facGJ = 0.5 * 0.5 * hF / pF;
								Real facJ  = 0.5 * 0.5 * penaltyAlpha_ * penaltyAlpha_ * hF / pF;

								if( localCoefL.n() != localCoefR.n() ){
									throw std::runtime_error( 
											"The number of eigenfunctions do not match." );
								}

								Int numEig = localCoefL.n();

								DblNumVec  jump( numGridFace );
								DblNumVec  gradJump( numGridFace );

								for( Int g = 0; g < numEig; g++ ){
									SetValue( jump, 0.0 );
									SetValue( gradJump, 0.0 );

									// Left side: gradjump and jump
									if( numBasisL > 0 ){
										blas::Gemv( 'N', numGridFace, numBasisL, 1.0,
												drvL.Data(), numGridFace, localCoefL.VecData(g), 1,
												1.0, gradJump.Data(), 1 );
										blas::Gemv( 'N', numGridFace, numBasisL, 1.0, 
												valL.Data(), numGridFace, localCoefL.VecData(g), 1,
												1.0, jump.Data(), 1 );
									}

									// Right side: gradjump and jump
									if( numBasisR > 0 ){
										blas::Gemv( 'N', numGridFace, numBasisR, 1.0,
												drvR.Data(), numGridFace, localCoefR.VecData(g), 1,
												1.0, gradJump.Data(), 1 );
										blas::Gemv( 'N', numGridFace, numBasisR, 1.0, 
												valR.Data(), numGridFace, localCoefR.VecData(g), 1,
												1.0, jump.Data(), 1 );
									}

									Real* ptrGJ = gradJump.Data();
									Real* ptrJ  = jump.Data();
									Real* ptrW  = LGLWeight2D[0].Data();
									Real  tmpGJ = 0.0;
									Real  tmpJ  = 0.0;
									for( Int p = 0; p < numGridFace; p++ ){
										tmpGJ += (*ptrGJ) * (*ptrGJ) * (*ptrW);
										tmpJ  += (*ptrJ)  * (*ptrJ)  * (*ptrW);
										ptrGJ++; ptrJ++; ptrW++;
									}
									// Previous element
									eta2GradJumpLocal( p1, j, k ) += tmpGJ * occrate(g) * numSpin * facGJ;
									eta2JumpLocal( p1, j, k )     += tmpJ  * occrate(g) * numSpin * facJ;
									// Current element
									eta2GradJumpLocal( i, j, k ) += tmpGJ * occrate(g) * numSpin * facGJ;
									eta2JumpLocal( i, j, k )     += tmpJ  * occrate(g) * numSpin * facJ;
								} // for (eigenfunction)
							} // if (pF>0)
						} // x-direction


						// y-direction
						{
							// keyL is the previous element received from GetBegin/GetEnd.
							// keyR is the current element
							Int p2; if( j == 0 )  p2 = numElem_[1]-1; else   p2 = j-1;
							Index3 keyL( i, p2, k );
							Index3 keyR = key;

							Int  numGridFace = numGrid[0] * numGrid[2];

							// Note that the notation can be very confusing here:
							// The left element (keyL) contributes to the right face
							// (YR), and the right element (keyR) contributes to the
							// left face (YL)
							DblNumMat&  valL = basisJump[YR].LocalMap()[keyL];
							DblNumMat&  valR = basisJump[YL].LocalMap()[keyR];
							DblNumMat&  drvL = DbasisJump[YR].LocalMap()[keyL];
							DblNumMat&  drvR = DbasisJump[YL].LocalMap()[keyR];

							Int numBasisL = valL.n();
							Int numBasisR = valR.n();

							DblNumMat& localCoefL  = eigvecCoef_.LocalMap()[keyL];
							DblNumMat& localCoefR  = eigvecCoef_.LocalMap()[keyR];
							DblNumVec& eig         = eigVal_;
							DblNumVec& occrate     = occupationRate_;

							Real hF = std::sqrt( length[0] * length[0] + length[2] * length[2] );
							Real pF = std::max( numBasisL, numBasisR );

							// Skip the computation if there is no basis function on
							// either side.  
							if( pF > 0 ){

								// factor in the estimator for jump of the derivative and
								// the jump of the values
								// NOTE:
								// Originally in [Giani 2012] the penalty term in the a
								// posteriori error estimator should be
								//
								// facJ = gamma^2 * p_F^3 / h_F.  However, there the jump term
								// should be
								//
								// gamma * p_F^2 / h_F = penaltyAlpha_
								//
								// used in this code.  So 
								//
								// facJ = penaltyAlpha_^2 * h_F / p_F
								//
								
								// One 0.5 comes from double counting on the faces F,
								// and the other comes from the 1/2 in front of the
								// Laplacian operator.
								Real facGJ = 0.5 * 0.5 * hF / pF;
								Real facJ  = 0.5 * 0.5 * penaltyAlpha_ * penaltyAlpha_ * hF / pF;

								if( localCoefL.n() != localCoefR.n() ){
									throw std::runtime_error( 
											"The number of eigenfunctions do not match." );
								}

								Int numEig = localCoefL.n();

								DblNumVec  jump( numGridFace );
								DblNumVec  gradJump( numGridFace );

								for( Int g = 0; g < numEig; g++ ){
									SetValue( jump, 0.0 );
									SetValue( gradJump, 0.0 );

									// Left side: gradjump and jump
									if( numBasisL > 0 ){
										blas::Gemv( 'N', numGridFace, numBasisL, 1.0,
												drvL.Data(), numGridFace, localCoefL.VecData(g), 1,
												1.0, gradJump.Data(), 1 );
										blas::Gemv( 'N', numGridFace, numBasisL, 1.0, 
												valL.Data(), numGridFace, localCoefL.VecData(g), 1,
												1.0, jump.Data(), 1 );
									}

									// Right side: gradjump and jump
									if( numBasisR > 0 ){
										blas::Gemv( 'N', numGridFace, numBasisR, 1.0,
												drvR.Data(), numGridFace, localCoefR.VecData(g), 1,
												1.0, gradJump.Data(), 1 );
										blas::Gemv( 'N', numGridFace, numBasisR, 1.0, 
												valR.Data(), numGridFace, localCoefR.VecData(g), 1,
												1.0, jump.Data(), 1 );
									}

									Real* ptrGJ = gradJump.Data();
									Real* ptrJ  = jump.Data();
									Real* ptrW  = LGLWeight2D[1].Data();
									Real  tmpGJ = 0.0;
									Real  tmpJ  = 0.0;
									for( Int p = 0; p < numGridFace; p++ ){
										tmpGJ += (*ptrGJ) * (*ptrGJ) * (*ptrW);
										tmpJ  += (*ptrJ)  * (*ptrJ)  * (*ptrW);
										ptrGJ++; ptrJ++; ptrW++;
									}
									// Previous element
									eta2GradJumpLocal( i, p2, k ) += tmpGJ * occrate(g) * numSpin * facGJ;
									eta2JumpLocal( i, p2, k )     += tmpJ  * occrate(g) * numSpin * facJ;
									// Current element
									eta2GradJumpLocal( i, j, k ) += tmpGJ * occrate(g) * numSpin * facGJ;
									eta2JumpLocal( i, j, k )     += tmpJ  * occrate(g) * numSpin * facJ;
								} // for (eigenfunction)
							} // if (pF>0)
						} // y-direction
				

						// z-direction
						{
							// keyL is the previous element received from GetBegin/GetEnd.
							// keyR is the current element
							Int p3; if( k == 0 )  p3 = numElem_[2]-1; else   p3 = k-1;
							Index3 keyL( i, j, p3 );
							Index3 keyR = key;

							Int  numGridFace = numGrid[0] * numGrid[1];

							// Note that the notation can be very confusing here:
							// The left element (keyL) contributes to the right face
							// (ZR), and the right element (keyR) contributes to the
							// left face (ZL)
							DblNumMat&  valL = basisJump[ZR].LocalMap()[keyL];
							DblNumMat&  valR = basisJump[ZL].LocalMap()[keyR];
							DblNumMat&  drvL = DbasisJump[ZR].LocalMap()[keyL];
							DblNumMat&  drvR = DbasisJump[ZL].LocalMap()[keyR];

							Int numBasisL = valL.n();
							Int numBasisR = valR.n();

							DblNumMat& localCoefL  = eigvecCoef_.LocalMap()[keyL];
							DblNumMat& localCoefR  = eigvecCoef_.LocalMap()[keyR];
							DblNumVec& eig         = eigVal_;
							DblNumVec& occrate     = occupationRate_;


							Real hF = std::sqrt( length[0] * length[0] + length[1] * length[1] );
							Real pF = std::max( numBasisL, numBasisR );

							// Skip the computation if there is no basis function on
							// either side.  
							if( pF > 0 ){

								// factor in the estimator for jump of the derivative and
								// the jump of the values
								// NOTE:
								// Originally in [Giani 2012] the penalty term in the a
								// posteriori error estimator should be
								//
								// facJ = gamma^2 * p_F^3 / h_F.  However, there the jump term
								// should be
								//
								// gamma * p_F^2 / h_F = penaltyAlpha_
								//
								// used in this code.  So 
								//
								// facJ = penaltyAlpha_^2 * h_F / p_F
								//
								
								// One 0.5 comes from double counting on the faces F,
								// and the other comes from the 1/2 in front of the
								// Laplacian operator.
								Real facGJ = 0.5 * 0.5 * hF / pF;
								Real facJ  = 0.5 * 0.5 * penaltyAlpha_ * penaltyAlpha_ * hF / pF;

								if( localCoefL.n() != localCoefR.n() ){
									throw std::runtime_error( 
											"The number of eigenfunctions do not match." );
								}

								Int numEig = localCoefL.n();

								DblNumVec  jump( numGridFace );
								DblNumVec  gradJump( numGridFace );

								for( Int g = 0; g < numEig; g++ ){
									SetValue( jump, 0.0 );
									SetValue( gradJump, 0.0 );

									// Left side: gradjump and jump
									if( numBasisL > 0 ){
										blas::Gemv( 'N', numGridFace, numBasisL, 1.0,
												drvL.Data(), numGridFace, localCoefL.VecData(g), 1,
												1.0, gradJump.Data(), 1 );
										blas::Gemv( 'N', numGridFace, numBasisL, 1.0, 
												valL.Data(), numGridFace, localCoefL.VecData(g), 1,
												1.0, jump.Data(), 1 );
									}

									// Right side: gradjump and jump
									if( numBasisR > 0 ){
										blas::Gemv( 'N', numGridFace, numBasisR, 1.0,
												drvR.Data(), numGridFace, localCoefR.VecData(g), 1,
												1.0, gradJump.Data(), 1 );
										blas::Gemv( 'N', numGridFace, numBasisR, 1.0, 
												valR.Data(), numGridFace, localCoefR.VecData(g), 1,
												1.0, jump.Data(), 1 );
									}

									Real* ptrGJ = gradJump.Data();
									Real* ptrJ  = jump.Data();
									Real* ptrW  = LGLWeight2D[2].Data();
									Real  tmpGJ = 0.0;
									Real  tmpJ  = 0.0;
									for( Int p = 0; p < numGridFace; p++ ){
										tmpGJ += (*ptrGJ) * (*ptrGJ) * (*ptrW);
										tmpJ  += (*ptrJ)  * (*ptrJ)  * (*ptrW);
										ptrGJ++; ptrJ++; ptrW++;
									}
									// Previous element
									eta2GradJumpLocal( i, j, p3 ) += tmpGJ * occrate(g) * numSpin * facGJ;
									eta2JumpLocal( i, j, p3 )     += tmpJ  * occrate(g) * numSpin * facJ;
									// Current element
									eta2GradJumpLocal( i, j, k ) += tmpGJ * occrate(g) * numSpin * facGJ;
									eta2JumpLocal( i, j, k )     += tmpJ  * occrate(g) * numSpin * facJ;
								} // for (eigenfunction)
							} // if (pF>0)
						} // z-direction


					} // if (own this element)
				} // for (i)

		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for the face and jump term is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	}

	// *********************************************************************
	// Reduce the computed error estimator among all elements.
	// *********************************************************************
	
	for( Int k = 0; k < numElem_[2]; k++ )
		for( Int j = 0; j < numElem_[1]; j++ )
			for( Int i = 0; i < numElem_[0]; i++ ){
				Index3 key( i, j, k );
				if( elemPrtn_.Owner(key) == mpirank ){
					eta2TotalLocal(i,j,k) = 
						eta2ResidualLocal(i,j,k) +
						eta2GradJumpLocal(i,j,k) +
						eta2JumpLocal(i,j,k);
				} // if (own this element)
			} // for (i)


	mpi::Allreduce( eta2TotalLocal.Data(), eta2Total.Data(),
			numElem_.prod(), MPI_SUM, domain_.comm );

	mpi::Allreduce( eta2ResidualLocal.Data(), eta2Residual.Data(),
			numElem_.prod(), MPI_SUM, domain_.comm );

	mpi::Allreduce( eta2GradJumpLocal.Data(), eta2GradJump.Data(),
			numElem_.prod(), MPI_SUM, domain_.comm );

	mpi::Allreduce( eta2JumpLocal.Data(), eta2Jump.Data(),
			numElem_.prod(), MPI_SUM, domain_.comm );

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method HamiltonianDG::CalculateAPosterioriError  ----- 

} // namespace dgdft
