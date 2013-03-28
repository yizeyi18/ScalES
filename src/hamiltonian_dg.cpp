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
HamiltonianDG::CalculateAPosterioriError	(  )
{
#ifndef _RELEASE_
	PushCallStack("HamiltonianDG::CalculateAPosterioriError");
#endif

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method HamiltonianDG::CalculateAPosterioriError  ----- 

} // namespace dgdft
