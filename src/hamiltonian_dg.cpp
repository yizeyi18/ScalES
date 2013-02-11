/// @file hamiltonian_dg.cpp
/// @brief Implementation of the Hamiltonian class for DG calculation.
/// @author Lin Lin
/// @date 2013-01-09
#include  "hamiltonian_dg.hpp"
#include  "mpi_interf.hpp"
#include  "blas.hpp"

#define _DEBUGlevel_ 0

namespace dgdft{

using namespace PseudoComponent;

// *********************************************************************
// Utility functions used in this subroutine
// *********************************************************************

inline Real ThreeDotProduct(Real* x, Real* y, Real* z, Int ntot) {
  Real sum =0;
  for(Int i=0; i<ntot; i++) {
    sum += (*x++)*(*y++)*(*z++);
  }
  return sum;
}

inline Real FourDotProduct(Real* w, Real* x, Real* y, Real* z, Int ntot) {
  Real sum =0;
  for(Int i=0; i<ntot; i++) {
    sum += (*w++)*(*x++)*(*y++)*(*z++);
  }
  return sum;
}



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
					density_.LocalMap()[key]     = empty;
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
HamiltonianDG::CalculateDGMatrix	(  )
{
#ifndef _RELEASE_
	PushCallStack("HamiltonianDG::CalculateDGMatrix");
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
	std::vector<DistDblNumMat>   DbasisAverage(NUM_FACE);

	// The derivative of basisLGL along x,y,z directions
	std::vector<DistDblNumMat>   Dbasis(DIM);

	// Integration weights
	std::vector<DblNumVec>  LGLWeight1D(DIM);
	std::vector<DblNumMat>  LGLWeight2D(DIM);
	DblNumTns               LGLWeight3D;

	// Clear the DG Matrix
	HMat_.LocalMap().clear();

	// *********************************************************************
	// Initial setup
	// *********************************************************************
	

	{
		GetTime(timeSta);
		// Compute the global index set
		IntNumTns  numBasisLocal(numElem_[0], numElem_[1], numElem_[2]);
		SetValue( numBasisLocal, 0 );
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key(i, j, k);
					if( elemPrtn_.Owner(key) == mpirank ){
						numBasisLocal(i, j, k) = basisLGL_.LocalMap()[key].n();
					}
				} // for (i)
		IntNumTns numBasis(numElem_[0], numElem_[1], numElem_[2]);
		mpi::Allreduce( numBasisLocal.Data(), numBasis.Data(),
				numElem_.prod(), MPI_SUM, domain_.comm );
		// Every processor compute all index sets
		elemBasisIdx_.Resize(numElem_[0], numElem_[1], numElem_[2]);

		Int cnt = 0;
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					std::vector<Int> idxVec;
					for(Int g = 0; g < numBasis(i, j, k); g++){
						idxVec.push_back( cnt++ );
					}
					elemBasisIdx_(i, j, k) = idxVec;
				} // for (i)

		sizeHMat_ = cnt;

		for( Int i = 0; i < NUM_FACE; i++ ){
			basisJump[i].Prtn()     = elemPrtn_;
			DbasisAverage[i].Prtn() = elemPrtn_;
		}

		for( Int i = 0; i < DIM; i++ ){
			Dbasis[i].Prtn() = elemPrtn_;
		}

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
		
		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for initial setup is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	}
	
	
	// *********************************************************************
	// Local gradient calculation: Overlap communication with computation
	// *********************************************************************
	{
		GetTime(timeSta);

		// Compute derivatives
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner(key) == mpirank ){
						DblNumMat&  basis = basisLGL_.LocalMap()[key];
						Int numBasis = basis.n();
						
						for( Int d = 0; d < DIM; d++ ){
							DblNumMat D(basis.m(), basis.n());
							SetValue( D, 0.0 );
							for( Int g = 0; g < numBasis; g++ ){
								DiffPsi( numGrid, basis.VecData(g), D.VecData(g), d );
							}
							Dbasis[d].LocalMap()[key] = D;
						}
					}
				} // for (i)
		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for the local gradient calculation is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

		GetTime(timeSta);
		// Compute average of derivatives and jump of values
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner(key) == mpirank ){
						DblNumMat&  basis = basisLGL_.LocalMap()[key];
						Int numBasis = basis.n();

						// x-direction
						{
							Int  numGridFace = numGrid[1] * numGrid[2];
							DblNumMat emptyX( numGridFace, numBasis );
							SetValue( emptyX, 0.0 );
							basisJump[XL].LocalMap()[key] = emptyX;
							basisJump[XR].LocalMap()[key] = emptyX;
							DbasisAverage[XL].LocalMap()[key] = emptyX;
							DbasisAverage[XR].LocalMap()[key] = emptyX;

							DblNumMat&  valL = basisJump[XL].LocalMap()[key];
							DblNumMat&  valR = basisJump[XR].LocalMap()[key];
							DblNumMat&  drvL = DbasisAverage[XL].LocalMap()[key];
							DblNumMat&  drvR = DbasisAverage[XR].LocalMap()[key];
							DblNumMat&  DbasisX = Dbasis[0].LocalMap()[key];

							// Form jumps and averages from volume to face.
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

										// 0.5 comes from average
										// {{a}} = 1/2 (a_L + a_R)
										drvL(idx, g) = +0.5 * DbasisX( idxL, g );
										drvR(idx, g) = +0.5 * DbasisX( idxR, g );
										// 1.0, -1.0 comes from jump with different normal vectors
										// [[a]] = -(1.0) a_L + (1.0) a_R
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
							DbasisAverage[YL].LocalMap()[key] = emptyY;
							DbasisAverage[YR].LocalMap()[key] = emptyY;

							DblNumMat&  valL = basisJump[YL].LocalMap()[key];
							DblNumMat&  valR = basisJump[YR].LocalMap()[key];
							DblNumMat&  drvL = DbasisAverage[YL].LocalMap()[key];
							DblNumMat&  drvR = DbasisAverage[YR].LocalMap()[key];
							DblNumMat&  DbasisY = Dbasis[1].LocalMap()[key];

							// Form jumps and averages from volume to face.
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

										// 0.5 comes from average
										// {{a}} = 1/2 (a_L + a_R)
										drvL(idx, g) = +0.5 * DbasisY( idxL, g );
										drvR(idx, g) = +0.5 * DbasisY( idxR, g );
										// 1.0, -1.0 comes from jump with different normal vectors
										// [[a]] = -(1.0) a_L + (1.0) a_R
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
							DbasisAverage[ZL].LocalMap()[key] = emptyZ;
							DbasisAverage[ZR].LocalMap()[key] = emptyZ;

							DblNumMat&  valL = basisJump[ZL].LocalMap()[key];
							DblNumMat&  valR = basisJump[ZR].LocalMap()[key];
							DblNumMat&  drvL = DbasisAverage[ZL].LocalMap()[key];
							DblNumMat&  drvR = DbasisAverage[ZR].LocalMap()[key];
							DblNumMat&  DbasisZ = Dbasis[2].LocalMap()[key];

							// Form jumps and averages from volume to face.
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

										// 0.5 comes from average
										// {{a}} = 1/2 (a_L + a_R)
										drvL(idx, g) = +0.5 * DbasisZ( idxL, g );
										drvR(idx, g) = +0.5 * DbasisZ( idxR, g );
										// 1.0, -1.0 comes from jump with different normal vectors
										// [[a]] = -(1.0) a_L + (1.0) a_R
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

	// *********************************************************************
	// Start the communication of boundary terms
	// *********************************************************************
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

		DbasisAverage[XR].GetBegin( boundaryXIdx, NO_MASK );
		DbasisAverage[YR].GetBegin( boundaryYIdx, NO_MASK );
		DbasisAverage[ZR].GetBegin( boundaryZIdx, NO_MASK );

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
	
	
	// *********************************************************************
	// Diagonal part: Overlap communication with computation
	// *********************************************************************

	// Diagonal part:
  // 1) Laplacian 
	// 2) Local potential
	// 3) Intra-element part of boundary terms
	{
		GetTime(timeSta);


		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner(key) == mpirank ){
						DblNumMat&  basis = basisLGL_.LocalMap()[key];
						Int numBasis = basis.n();
						DblNumMat   localMat( numBasis, numBasis );
						SetValue( localMat, 0.0 );
						// In all matrix assembly process, Only compute the upper
						// triangular matrix use symmetry later

						// Laplacian part
						{
							DblNumMat&  DbasisX = Dbasis[0].LocalMap()[key];
							DblNumMat&  DbasisY = Dbasis[1].LocalMap()[key];
							DblNumMat&  DbasisZ = Dbasis[2].LocalMap()[key];

							for( Int a = 0; a < numBasis; a++ )
								for( Int b = a; b < numBasis; b++ ){
									localMat(a,b) += 
										+ 0.5 * ThreeDotProduct( 
												DbasisX.VecData(a), DbasisX.VecData(b), 
												LGLWeight3D.Data(), numGridTotal )
										+ 0.5 * ThreeDotProduct( 
												DbasisY.VecData(a), DbasisY.VecData(b), 
												LGLWeight3D.Data(), numGridTotal )
										+ 0.5 * ThreeDotProduct( 
												DbasisZ.VecData(a), DbasisZ.VecData(b), 
												LGLWeight3D.Data(), numGridTotal );
								} // for (b)

							// Release the gradient as volume data to save memory
							for( Int d = 0; d < DIM; d++ ){
								Dbasis[d].LocalMap().erase(key);
							}
						}
#if ( _DEBUGlevel_ >= 0 )
						statusOFS << "After the Laplacian part." << std::endl;
#endif

						// Local potential part
						{
							DblNumVec&  vtot  = vtotLGL_.LocalMap()[key];
							for( Int a = 0; a < numBasis; a++ )
								for( Int b = a; b < numBasis; b++ ){
									localMat(a,b) += FourDotProduct( 
											basis.VecData(a), basis.VecData(b), 
											vtot.Data(), LGLWeight3D.Data(), numGridTotal );
								} // for (b)
						}
						
#if ( _DEBUGlevel_ >= 0 )
						statusOFS << "After the local potential part." << std::endl;
#endif

						// x-direction: intra-element part of the boundary term
						{
							Int  numGridFace = numGrid[1] * numGrid[2];

							DblNumMat&  valL = basisJump[XL].LocalMap()[key];
							DblNumMat&  valR = basisJump[XR].LocalMap()[key];
							DblNumMat&  drvL = DbasisAverage[XL].LocalMap()[key];
							DblNumMat&  drvR = DbasisAverage[XR].LocalMap()[key];

							// intra-element part of the boundary term
							Real intByPartTerm, penaltyTerm;
							for( Int a = 0; a < numBasis; a++ )
								for( Int b = a; b < numBasis; b++ ){
									intByPartTerm = 
										-0.5 * ThreeDotProduct( 
												drvL.VecData(a), 
												valL.VecData(b),
												LGLWeight2D[0].Data(),
												numGridFace )
										-0.5 * ThreeDotProduct(
												valL.VecData(a),
												drvL.VecData(b), 
												LGLWeight2D[0].Data(),
												numGridFace )
										-0.5 * ThreeDotProduct( 
												drvR.VecData(a), 
												valR.VecData(b),
												LGLWeight2D[0].Data(),
												numGridFace )
										-0.5 * ThreeDotProduct(
												valR.VecData(a),
												drvR.VecData(b), 
												LGLWeight2D[0].Data(),
												numGridFace );
									penaltyTerm = 
										penaltyAlpha_ * ThreeDotProduct(
												valL.VecData(a),
												valL.VecData(b),
												LGLWeight2D[0].Data(),
												numGridFace )
										+ penaltyAlpha_ * ThreeDotProduct(
												valR.VecData(a),
												valR.VecData(b),
												LGLWeight2D[0].Data(),
												numGridFace );

									localMat(a,b) += 
										intByPartTerm + penaltyTerm;
								} // for (b)
						} // x-direction

						// y-direction: intra-element part of the boundary term
						{
							Int  numGridFace = numGrid[0] * numGrid[2];

							DblNumMat&  valL = basisJump[YL].LocalMap()[key];
							DblNumMat&  valR = basisJump[YR].LocalMap()[key];
							DblNumMat&  drvL = DbasisAverage[YL].LocalMap()[key];
							DblNumMat&  drvR = DbasisAverage[YR].LocalMap()[key];

							// intra-element part of the boundary term
							Real intByPartTerm, penaltyTerm;
							for( Int a = 0; a < numBasis; a++ )
								for( Int b = a; b < numBasis; b++ ){
									intByPartTerm = 
										-0.5 * ThreeDotProduct( 
												drvL.VecData(a), 
												valL.VecData(b),
												LGLWeight2D[1].Data(),
												numGridFace )
										-0.5 * ThreeDotProduct(
												valL.VecData(a),
												drvL.VecData(b), 
												LGLWeight2D[1].Data(),
												numGridFace )
										-0.5 * ThreeDotProduct( 
												drvR.VecData(a), 
												valR.VecData(b),
												LGLWeight2D[1].Data(),
												numGridFace )
										-0.5 * ThreeDotProduct(
												valR.VecData(a),
												drvR.VecData(b), 
												LGLWeight2D[1].Data(),
												numGridFace );
									penaltyTerm = 
										penaltyAlpha_ * ThreeDotProduct(
												valL.VecData(a),
												valL.VecData(b),
												LGLWeight2D[1].Data(),
												numGridFace )
										+ penaltyAlpha_ * ThreeDotProduct(
												valR.VecData(a),
												valR.VecData(b),
												LGLWeight2D[1].Data(),
												numGridFace );

									localMat(a,b) += 
										intByPartTerm + penaltyTerm;
								} // for (b)
						} // y-direction

						// z-direction: intra-element part of the boundary term
						{
							Int  numGridFace = numGrid[0] * numGrid[1];

							DblNumMat&  valL = basisJump[ZL].LocalMap()[key];
							DblNumMat&  valR = basisJump[ZR].LocalMap()[key];
							DblNumMat&  drvL = DbasisAverage[ZL].LocalMap()[key];
							DblNumMat&  drvR = DbasisAverage[ZR].LocalMap()[key];

							// intra-element part of the boundary term
							Real intByPartTerm, penaltyTerm;
							for( Int a = 0; a < numBasis; a++ )
								for( Int b = a; b < numBasis; b++ ){
									intByPartTerm = 
										-0.5 * ThreeDotProduct( 
												drvL.VecData(a), 
												valL.VecData(b),
												LGLWeight2D[2].Data(),
												numGridFace )
										-0.5 * ThreeDotProduct(
												valL.VecData(a),
												drvL.VecData(b), 
												LGLWeight2D[2].Data(),
												numGridFace )
										-0.5 * ThreeDotProduct( 
												drvR.VecData(a), 
												valR.VecData(b),
												LGLWeight2D[2].Data(),
												numGridFace )
										-0.5 * ThreeDotProduct(
												valR.VecData(a),
												drvR.VecData(b), 
												LGLWeight2D[2].Data(),
												numGridFace );
									penaltyTerm = 
										penaltyAlpha_ * ThreeDotProduct(
												valL.VecData(a),
												valL.VecData(b),
												LGLWeight2D[2].Data(),
												numGridFace )
										+ penaltyAlpha_ * ThreeDotProduct(
												valR.VecData(a),
												valR.VecData(b),
												LGLWeight2D[2].Data(),
												numGridFace );

									localMat(a,b) += 
										intByPartTerm + penaltyTerm;
								} // for (b)
						} // z-direction

#if ( _DEBUGlevel_ >= 0 )
						statusOFS << "After the boundary part." << std::endl;
#endif

						// Symmetrize
						for( Int a = 0; a < numBasis; a++ )
							for( Int b = 0; b < a; b++ ){
								localMat(a,b) = localMat(b,a);
							}


						// Add to HMat_
						ElemMatKey matKey( key, key );
						std::map<ElemMatKey, DblNumMat>::iterator mi = 
							HMat_.LocalMap().find( matKey );
						if( mi == HMat_.LocalMap().end() ){
							HMat_.LocalMap()[matKey] = localMat;
						}
						else{
							DblNumMat&  mat = (*mi).second;
							blas::Axpy( mat.Size(), 1.0, localMat.Data(), 1,
									mat.Data(), 1);
						}
					}
				} // for (i)

		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for the diagonal part of the DG matrix is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	} 


	// *********************************************************************
	// Nonlocal pseudopotential term
	// *********************************************************************
	if(1)
	{
		GetTime( timeSta );
		// Compute the coefficient (i.e. the inner product of the nonlocal
		// pseudopotential and basis functions in the form of <phi|l>) for
		// nonlocal pseudopotential projectors locally

		vnlCoef_.LocalMap().clear();
		for( Int k = 0; k < numElem_[2]; k++ )
			for( Int j = 0; j < numElem_[1]; j++ )
				for( Int i = 0; i < numElem_[0]; i++ ){
					Index3 key( i, j, k );
					if( elemPrtn_.Owner(key) == mpirank ){
						std::map<Int, DblNumMat>  coefMap;
						std::map<Int, PseudoPot>& pseudoMap =
							pseudo_.LocalMap()[key];
						DblNumMat&   basis = basisLGL_.LocalMap()[key];
						Int numBasis = basis.n();

						// Loop over atoms, regardless of whether this atom belongs
						// to this element or not.
						for( std::map<Int, PseudoPot>::iterator 
							 	 mi  = pseudoMap.begin();
							   mi != pseudoMap.end(); mi++ ){
							Int atomIdx = (*mi).first;
							std::vector<NonlocalPP>&  vnlList = (*mi).second.vnlList;
							DblNumMat coef( numBasis, vnlList.size() );
							SetValue( coef, 0.0 );
							// Loop over projector
							for( Int g = 0; g < vnlList.size(); g++ ){
								SparseVec&  vnl = vnlList[g].first;
								IntNumVec&  idx = vnl.first;
								DblNumMat&  val = vnl.second;
								Real*       ptrWeight = LGLWeight3D.Data();
								if( idx.Size() > 0 ) {
									// Loop over basis function
									for( Int a = 0; a < numBasis; a++ ){
										// Loop over grid point
										for( Int l = 0; l < idx.Size(); l++ ){
											coef(a, g) += basis( idx(l), a ) * val(l, VAL) * 
												ptrWeight[idx(l)];
										}
									}
								} // non-empty
							} // for (g)

							coefMap[atomIdx] = coef;
						}
						vnlCoef_.LocalMap()[key] = coefMap;
					} // own this element
				} // for (i)


		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << 
			"Time for computing the coefficient for nonlocal projector is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

		GetTime( timeSta );
		
		// Communication of the coefficient matrices.
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
		
		vnlCoef_.GetBegin( pseudoIdx, NO_MASK );
		vnlCoef_.GetEnd( NO_MASK );

		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << 
			"Time for the communication of pseudopotential coefficent is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

		GetTime( timeSta );

		// Loop over atoms
		for( Int atomIdx = 0; atomIdx < numAtom; atomIdx++ ){
			if( atomPrtn_.Owner(atomIdx) == mpirank ){
			  DblNumVec&  vnlWeight = vnlWeightMap_[atomIdx];	
				// Loop over element 1
				for( std::map<Index3, std::map<Int, DblNumMat> >::iterator 
						ei  = vnlCoef_.LocalMap().begin();
						ei != vnlCoef_.LocalMap().end(); ei++ ){
					Index3 key1 = (*ei).first;
					std::map<Int, DblNumMat>& coefMap1 = (*ei).second; 
					std::map<Int, PseudoPot>& pseudoMap = pseudo_.LocalMap()[key1];

					std::map<Int, DblNumMat>::iterator mi = 
						coefMap1.find( atomIdx );
					if( mi != coefMap1.end() ){
						DblNumMat&  coef1 = (*mi).second;
						// Loop over element j
						for( std::map<Index3, std::map<Int, DblNumMat> >::iterator 
								ej  = vnlCoef_.LocalMap().begin();
								ej != vnlCoef_.LocalMap().end(); ej++ ){
							Index3 key2 = (*ej).first;
							std::map<Int, DblNumMat>& coefMap2 = (*ej).second;

							std::map<Int, DblNumMat>::iterator ni = 
								coefMap2.find( atomIdx );
							// Compute the contribution to HMat_(key1, key2)
							if( ni != coefMap2.end() ){
								DblNumMat& coef2 = (*ni).second;
								DblNumMat localMat( coef1.m(), coef2.m() );
								SetValue( localMat, 0.0 );
								// Check size consistency
								if( coef1.n() != coef2.n() ||
										coef1.n() != vnlWeight.Size() ){
									std::ostringstream msg;
									msg 
										<< "Error in assembling the nonlocal pseudopotential part of the DG matrix." << std::endl
										<< "Atom number " << atomIdx << std::endl
										<< "Element 1: " << key1 << ", Element 2: " << key2 << std::endl
										<< "Coef matrix 1 size : " << coef1.m() << " x " << coef1.n() << std::endl
										<< "Coef matrix 2 size : " << coef2.m() << " x " << coef2.n() << std::endl
										<< "vnlWeight     size : " << vnlWeight.Size() << std::endl;

									throw std::runtime_error( msg.str().c_str() );
								}
								// Outer product with the weight of the nonlocal
								// pseudopotential to form local matrix
								//
								// localMat = coef1 * diag(weight) * coef2^T.
								for( Int l = 0; l < vnlWeight.Size(); l++ ){
									Real weight = vnlWeight(l);
									blas::Ger( coef1.m(), coef2.m(), weight, 
											coef1.VecData(l), 1, coef2.VecData(l), 1,
											localMat.Data(), localMat.m() );
								}
								// Add to HMat_
								ElemMatKey matKey( key1, key2 );
								std::map<ElemMatKey, DblNumMat>::iterator mati = 
									HMat_.LocalMap().find( matKey );
								if( mati == HMat_.LocalMap().end() ){
									HMat_.LocalMap()[matKey] = localMat;
								}
								else{
									DblNumMat&  mat = (*mati).second;
									blas::Axpy( mat.Size(), 1.0, localMat.Data(), 1,
											mat.Data(), 1 );
								}
							} // found atomIdx in element 2
						} // for (ej)
					} // found atomIdx in element 1
				} // for (ei)
			} // own this atom
		}

		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << 
			"Time for updating the nonlocal potential part of the matrix is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	}



	// *********************************************************************
	// Finish the communication of boundary terms
	// *********************************************************************

	// New code for communicating boundary elements
	{
		GetTime( timeSta );
		DbasisAverage[XR].GetEnd( NO_MASK );
		DbasisAverage[YR].GetEnd( NO_MASK );
		DbasisAverage[ZR].GetEnd( NO_MASK );

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

	// *********************************************************************
	// The inter-element boundary term
	// *********************************************************************
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
							DblNumMat&  drvL = DbasisAverage[XR].LocalMap()[keyL];
							DblNumMat&  drvR = DbasisAverage[XL].LocalMap()[keyR];

							Int numBasisL = valL.n();
							Int numBasisR = valR.n();
							DblNumMat   localMat( numBasisL, numBasisR );
							SetValue( localMat, 0.0 );

							// inter-element part of the boundary term
							Real intByPartTerm, penaltyTerm;
							for( Int a = 0; a < numBasisL; a++ )
								for( Int b = 0; b < numBasisR; b++ ){
									intByPartTerm = 
										-0.5 * ThreeDotProduct( 
												drvL.VecData(a), 
												valR.VecData(b),
												LGLWeight2D[0].Data(),
												numGridFace )
										-0.5 * ThreeDotProduct(
												valL.VecData(a),
												drvR.VecData(b), 
												LGLWeight2D[0].Data(),
												numGridFace );
									penaltyTerm = 
										penaltyAlpha_ * ThreeDotProduct(
												valL.VecData(a),
												valR.VecData(b),
												LGLWeight2D[0].Data(),
												numGridFace );

									localMat(a,b) += 
										intByPartTerm + penaltyTerm;
								} // for (b)
						
							// Add (keyL, keyR) to HMat_
							{
								ElemMatKey matKey( keyL, keyR );
								std::map<ElemMatKey, DblNumMat>::iterator mi = 
									HMat_.LocalMap().find( matKey );
								if( mi == HMat_.LocalMap().end() ){
									HMat_.LocalMap()[matKey] = localMat;
								}
								else{
									DblNumMat&  mat = (*mi).second;
									blas::Axpy( mat.Size(), 1.0, localMat.Data(), 1,
											mat.Data(), 1);
								}
							}

							// Add (keyR, keyL) to HMat_
							{
								DblNumMat localMatTran;
								Transpose( localMat, localMatTran );
								ElemMatKey matKey( keyR, keyL );
								std::map<ElemMatKey, DblNumMat>::iterator mi = 
									HMat_.LocalMap().find( matKey );
								if( mi == HMat_.LocalMap().end() ){
									HMat_.LocalMap()[matKey] = localMatTran;
								}
								else{
									DblNumMat&  mat = (*mi).second;
									blas::Axpy( mat.Size(), 1.0, localMatTran.Data(), 1,
											mat.Data(), 1);
								}
							}
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
							DblNumMat&  drvL = DbasisAverage[YR].LocalMap()[keyL];
							DblNumMat&  drvR = DbasisAverage[YL].LocalMap()[keyR];

							Int numBasisL = valL.n();
							Int numBasisR = valR.n();
							DblNumMat   localMat( numBasisL, numBasisR );
							SetValue( localMat, 0.0 );

							// inter-element part of the boundary term
							Real intByPartTerm, penaltyTerm;
							for( Int a = 0; a < numBasisL; a++ )
								for( Int b = 0; b < numBasisR; b++ ){
									intByPartTerm = 
										-0.5 * ThreeDotProduct( 
												drvL.VecData(a), 
												valR.VecData(b),
												LGLWeight2D[1].Data(),
												numGridFace )
										-0.5 * ThreeDotProduct(
												valL.VecData(a),
												drvR.VecData(b), 
												LGLWeight2D[1].Data(),
												numGridFace );
									penaltyTerm = 
										penaltyAlpha_ * ThreeDotProduct(
												valL.VecData(a),
												valR.VecData(b),
												LGLWeight2D[1].Data(),
												numGridFace );

									localMat(a,b) += 
										intByPartTerm + penaltyTerm;
								} // for (b)
						
							// Add (keyL, keyR) to HMat_
							{
								ElemMatKey matKey( keyL, keyR );
								std::map<ElemMatKey, DblNumMat>::iterator mi = 
									HMat_.LocalMap().find( matKey );
								if( mi == HMat_.LocalMap().end() ){
									HMat_.LocalMap()[matKey] = localMat;
								}
								else{
									DblNumMat&  mat = (*mi).second;
									blas::Axpy( mat.Size(), 1.0, localMat.Data(), 1,
											mat.Data(), 1);
								}
							}

							// Add (keyR, keyL) to HMat_
							{
								DblNumMat localMatTran;
								Transpose( localMat, localMatTran );
								ElemMatKey matKey( keyR, keyL );
								std::map<ElemMatKey, DblNumMat>::iterator mi = 
									HMat_.LocalMap().find( matKey );
								if( mi == HMat_.LocalMap().end() ){
									HMat_.LocalMap()[matKey] = localMatTran;
								}
								else{
									DblNumMat&  mat = (*mi).second;
									blas::Axpy( mat.Size(), 1.0, localMatTran.Data(), 1,
											mat.Data(), 1);
								}
							}
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
							DblNumMat&  drvL = DbasisAverage[ZR].LocalMap()[keyL];
							DblNumMat&  drvR = DbasisAverage[ZL].LocalMap()[keyR];

							Int numBasisL = valL.n();
							Int numBasisR = valR.n();
							DblNumMat   localMat( numBasisL, numBasisR );
							SetValue( localMat, 0.0 );

							// inter-element part of the boundary term
							Real intByPartTerm, penaltyTerm;
							for( Int a = 0; a < numBasisL; a++ )
								for( Int b = 0; b < numBasisR; b++ ){
									intByPartTerm = 
										-0.5 * ThreeDotProduct( 
												drvL.VecData(a), 
												valR.VecData(b),
												LGLWeight2D[2].Data(),
												numGridFace )
										-0.5 * ThreeDotProduct(
												valL.VecData(a),
												drvR.VecData(b), 
												LGLWeight2D[2].Data(),
												numGridFace );
									penaltyTerm = 
										penaltyAlpha_ * ThreeDotProduct(
												valL.VecData(a),
												valR.VecData(b),
												LGLWeight2D[2].Data(),
												numGridFace );

									localMat(a,b) += 
										intByPartTerm + penaltyTerm;
								} // for (b)
						
							// Add (keyL, keyR) to HMat_
							{
								ElemMatKey matKey( keyL, keyR );
								std::map<ElemMatKey, DblNumMat>::iterator mi = 
									HMat_.LocalMap().find( matKey );
								if( mi == HMat_.LocalMap().end() ){
									HMat_.LocalMap()[matKey] = localMat;
								}
								else{
									DblNumMat&  mat = (*mi).second;
									blas::Axpy( mat.Size(), 1.0, localMat.Data(), 1,
											mat.Data(), 1);
								}
							}

							// Add (keyR, keyL) to HMat_
							{
								DblNumMat localMatTran;
								Transpose( localMat, localMatTran );
								ElemMatKey matKey( keyR, keyL );
								std::map<ElemMatKey, DblNumMat>::iterator mi = 
									HMat_.LocalMap().find( matKey );
								if( mi == HMat_.LocalMap().end() ){
									HMat_.LocalMap()[matKey] = localMatTran;
								}
								else{
									DblNumMat&  mat = (*mi).second;
									blas::Axpy( mat.Size(), 1.0, localMatTran.Data(), 1,
											mat.Data(), 1);
								}
							}
						} // z-direction
					
					}
				} // for (i)

		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for the inter-element boundary calculation is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	}


	// *********************************************************************
	// Collect information and combine HMat_
	// *********************************************************************
	{
		GetTime( timeSta );
		std::vector<ElemMatKey>  keyIdx;
		for( std::map<ElemMatKey, DblNumMat>::iterator 
				 mi  = HMat_.LocalMap().begin();
				 mi != HMat_.LocalMap().end(); mi++ ){
			ElemMatKey key = (*mi).first;
			if( HMat_.Prtn().Owner(key) != mpirank ){
				keyIdx.push_back( key );
			}
		}

		// Communication
		HMat_.PutBegin( keyIdx, NO_MASK );
		HMat_.PutEnd( NO_MASK, PutMode::COMBINE );

		// Clean up
    std::vector<ElemMatKey>  eraseKey;
		for( std::map<ElemMatKey, DblNumMat>::iterator 
				 mi  = HMat_.LocalMap().begin();
				 mi != HMat_.LocalMap().end(); mi++ ){
			ElemMatKey key = (*mi).first;
			if( HMat_.Prtn().Owner(key) != mpirank ){
				eraseKey.push_back( key );
			}
		}
		for( std::vector<ElemMatKey>::iterator vi = eraseKey.begin();
			   vi != eraseKey.end(); vi++ ){
			HMat_.LocalMap().erase( *vi );
		}
				 

		GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Time for combining the matrix is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	}

	// *********************************************************************
	// Clean up
	// *********************************************************************


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method HamiltonianDG::CalculateDGMatrix  ----- 
} // namespace dgdft
