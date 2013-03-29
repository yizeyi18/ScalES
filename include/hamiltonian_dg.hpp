/// @file hamiltonian_dg.hpp
/// @brief The Hamiltonian class for DG calculation.
/// @author Lin Lin
/// @date 2013-01-09
#ifndef _HAMILTONIAN_DG_HPP_
#define _HAMILTONIAN_DG_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "numtns_impl.hpp"
#include  "distvec_impl.hpp"
#include  "domain.hpp"
#include  "periodtable.hpp"
#include  "utility.hpp"
#include  "esdf.hpp"
#include  "fourier.hpp"
#include  <xc.h>
#include  "mpi_interf.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"


namespace dgdft{

// *********************************************************************
// Partitions
// *********************************************************************

/// @struct ElemPrtn
/// @brief Partition class (used by DistVec) according to the element
/// index.
struct ElemPrtn
{
	IntNumTns                 ownerInfo;

	Int Owner (const Index3& key) const {
		return ownerInfo(key(0), key(1), key(2));
	}
};

typedef std::pair<Index3, Index3>  ElemMatKey;

/// @struct ElemMatPrtn
/// @brief Partition class of a matrix according to the element index
/// (row index).  This is used to represent the stiffness matrix and
/// mass matrix.
struct ElemMatPrtn
{
	IntNumTns                 ownerInfo;

	Int Owner (const ElemMatKey& key) const {
		Index3 keyRow = key.first;
		return ownerInfo(keyRow(0), keyRow(1), keyRow(2));
	}
};

/// @struct AtomPrtn
/// @brief Partition class (used by DistVec) according to the atom
/// index.
struct AtomPrtn 
{
	std::vector<Int> ownerInfo; 

	Int Owner(Int key) const {
		return ownerInfo[key];
	}
};

/// @struct BlockMatPrtn
/// @brief Partition class of a matrix according to the 2D block cyclic
/// distribution.  This is used for ScaLAPACK calculation.
struct BlockMatPrtn
{
	IntNumMat                 ownerInfo;

	Int Owner (const Index2& key) const {
		return ownerInfo(key(0), key(1));
	}
};



// *********************************************************************
// Typedefs
// *********************************************************************

typedef DistVec<Index3, DblNumVec, ElemPrtn>   DistDblNumVec;

typedef DistVec<Index3, CpxNumVec, ElemPrtn>   DistCpxNumVec;

typedef DistVec<Index3, DblNumMat, ElemPrtn>   DistDblNumMat;

typedef DistVec<Index3, CpxNumMat, ElemPrtn>   DistCpxNumMat;

typedef DistVec<Index3, DblNumTns, ElemPrtn>   DistDblNumTns;

typedef DistVec<Index3, CpxNumTns, ElemPrtn>   DistCpxNumTns;


// *********************************************************************
// Main class
// *********************************************************************

/// @class HamiltonianDG 
/// @brief Main class of DG for storing and assembling the DG matrix.
class HamiltonianDG {
private:
	
	// *********************************************************************
	// Physical variables
	// *********************************************************************
	/// @brief Global domain.
	Domain                      domain_;

	/// @brief Element subdomains.
	NumTns<Domain>              domainElem_;

	/// @brief Uniform grid in the global domain
	std::vector<DblNumVec>      uniformGrid_;

	/// @brief Number of uniform grids in each element.  
	///
	/// Note: It must be satisifed that
	///
	/// domain_.numGrid[d] = numUniformGridElem_[d] * numElem_[d]
	Index3                      numUniformGridElem_;

	/// @brief Number of LGL grids in each element.
	Index3                      numLGLGridElem_;

	/// @brief Uniform grid in the elements, each has size 
	/// numUniformGridElem_
	NumTns<std::vector<DblNumVec> >   uniformGridElem_;

	/// @brief Legendre-Gauss-Lobatto grid in the elements, each has size
	/// numLGLGridElem_
	NumTns<std::vector<DblNumVec> >   LGLGridElem_;

	/// @brief List of atoms.
	std::vector<Atom>           atomList_;
	/// @brief Number of spin-degeneracy, can be 1 or 2.
	Int                         numSpin_;
	/// @brief Number of extra states for fractional occupation number.
	Int                         numExtraState_;
	/// @brief Number of occupied states.
	Int                         numOccupiedState_;
	/// @brief Type of pseudopotential, default is HGH
	std::string                 pseudoType_;
	/// @brief Id of the exchange-correlation potential
	Int                         XCId_;
	/// @brief Exchange-correlation potential using libxc package.
	xc_func_type                XCFuncType_; 
	/// @brief Whether libXC has been initialized.
	bool                        XCInitialized_;




	// *********************************************************************
	// Computational variables
	// *********************************************************************

	/// @brief The number of elements.
	Index3                      numElem_;

	/// @brief Partition of element.
	ElemPrtn                    elemPrtn_;

	/// @brief Partition of a matrix defined through elements.
	ElemMatPrtn                 elemMatPrtn_;

	/// @brief Partition of atom.
	AtomPrtn                    atomPrtn_;

	/// @brief Interior penalty parameter.
	Real                        penaltyAlpha_;

	/// @brief Pseudocharge in the global domain. 
	DistDblNumVec    pseudoCharge_;

	/// @brief Electron density in the global domain. No magnitization for
	/// DG calculation.
	DistDblNumVec    density_;

	/// @brief External potential in the global domain. (not implemented)
	DistDblNumVec    vext_;

	/// @brief Hartree potential in the global domain.
	DistDblNumVec    vhart_;

	/// @brief Exchange-correlation potential in the global domain. No
	/// magnization calculation in the DG code.
	DistDblNumVec    vxc_;

	/// @brief Exchange-correlation energy density in the global domain.
	DistDblNumVec    epsxc_;

	/// @brief Total potential in the global domain.
	DistDblNumVec    vtot_;


	/// @brief Total potential on the local LGL grid.
	DistDblNumVec    vtotLGL_;

	/// @brief Basis functions on the local LGL grid.
	DistDblNumMat    basisLGL_;

	/// @brief Eigenvalues
	DblNumVec        eigVal_;

	/// @brief Occupation number
	DblNumVec        occupationRate_;

	/// @brief Coefficients of the eigenfunctions
	DistDblNumMat    eigvecCoef_;

	/// @brief Pseudopotential and nonlocal projectors in each element for
	/// each atom.
	DistVec<Index3, std::map<Int, PseudoPot>, ElemPrtn>  pseudo_;

	// FIXME
	DistVec<Index3, std::map<Int, DblNumMat>, ElemPrtn>  vnlCoef_;
	
	// FIXME
	std::vector<DistVec<Index3, std::map<Int, DblNumMat>, ElemPrtn> >  vnlDrvCoef_;

	std::map<Int, DblNumVec>  vnlWeightMap_;

	/// @brief Differentiation matrix on the LGL grid.
	std::vector<DblNumMat>    DMat_;

	/// @brief Interpolation matrix from LGL to uniform grid in each
	/// element (assuming all the elements are the same).
	std::vector<DblNumMat>    LGLToUniformMat_;

	/// @brief DG Hamiltonian matrix.
	DistVec<ElemMatKey, DblNumMat, ElemMatPrtn>  HMat_;


	/// @brief The size of the H matrix.
	Int    sizeHMat_;

	/// @brief Indices of all the basis functions.
  NumTns< std::vector<Int> >  elemBasisIdx_;

public:

	// *********************************************************************
	// Lifecycle
	// *********************************************************************
	HamiltonianDG();

	~HamiltonianDG();

	HamiltonianDG( const esdf::ESDFInputParam& esdfParam );


	// *********************************************************************
	// Operations
	// *********************************************************************

	/// @brief Differentiate the basis functions on a certain element
	/// along the dimension d.
	void DiffPsi(const Index3& numGrid, const Real* psi, Real* Dpsi, Int d);

	/// @brief Differentiate the basis functions on a certain element
	/// along the dimension d.
	void InterpLGLToUniform( const Index3& numLGLGrid, const Index3& numUniformGrid, 
			const Real* psiLGL, Real* psiUniform );

	void CalculatePseudoPotential( PeriodTable &ptable );
	
	/// @brief Compute the electron density after the diagonalization
	/// of the DG Hamiltonian matrix.
	void CalculateDensity( const DblNumVec &occrate );
	

	/// @brief Compute the exchange-correlation potential and energy.
	void CalculateXC ( Real &Exc );

	/// @brief Compute the Hartree potential.
	void CalculateHartree( DistFourier& fft );
	
	/// @brief Compute the total potential
	void CalculateVtot( DistDblNumVec& vtot );

	/// @brief Assemble the DG Hamiltonian matrix. The mass matrix is
	/// identity in the framework of adaptive local basis functions.
	void CalculateDGMatrix( ); 

	/// @brief Calculate the Hellmann-Feynman force for each atom.
	void CalculateForce ( DistFourier& fft );

	/// @brief Calculate the residual type a posteriori error estimator
	/// for the solution. 
	///
	/// Currently only the residual term is computed, and it is assumed
	/// that the eigenvalues and eigenfunctions have been computed and
	/// saved in eigVal_ and eigvecCoef_.
	///
	/// Currently the nonlocal pseudopotential is not implemented in this
	/// subroutine.
	void CalculateAPosterioriError( DblNumTns&  eta2Residual );

	// *********************************************************************
	// Access
	// *********************************************************************

	/// @brief Total potential in the global domain.
	DistDblNumVec&  Vtot( ) { return vtot_; }

	/// @brief Exchange-correlation potential in the global domain. No
	/// magnization calculation in the DG code.
	DistDblNumVec&  Vxc()  { return vxc_; }

	/// @brief Hartree potential in the global domain.
	DistDblNumVec&  Vhart() { return vhart_; }

	/// @brief Electron density in the global domain. No magnitization for
	/// DG calculation.
	DistDblNumVec&  Density() { return density_; }

	DistDblNumVec&  PseudoCharge() { return pseudoCharge_; }
	
	std::vector<Atom>&  AtomList() { return atomList_; }

	Int NumSpin () { return numSpin_; }

	DblNumVec&  EigVal() { return eigVal_; }
	
	DblNumVec&  OccupationRate() { return occupationRate_; }


	DistDblNumVec&  VtotLGL() { return vtotLGL_; }

	DistDblNumMat&  BasisLGL() { return basisLGL_; }

	DistDblNumMat&  EigvecCoef() { return eigvecCoef_; }

	/// @brief DG Hamiltonian matrix.
	DistVec<ElemMatKey, DblNumMat, ElemMatPrtn>&  
		HMat() { return HMat_; } 

	Int NumBasisTotal() const { return sizeHMat_; }

  NumTns< std::vector<Int> >&  ElemBasisIdx() { return elemBasisIdx_; }
	
	/// domain_.numGrid[d] = numUniformGridElem_[d] * numElem_[d]
	Index3 NumUniformGridElem() const { return numUniformGridElem_; }

	/// @brief Number of LGL grids in each element.
	Index3 NumLGLGridElem() const { return numLGLGridElem_; }


	// *********************************************************************
	// Inquiry
	// *********************************************************************

	Int NumStateTotal() const { return numExtraState_ + numOccupiedState_; }

	Int NumOccupiedState() const { return numOccupiedState_; }

	Int NumExtraState() const { return numExtraState_; }

};


// *********************************************************************
// Utility subroutines
// *********************************************************************

/// @brief Convert a DistNumVec structure to the row partitioned vector,
/// which can be used for instance by DistFourier.
template<typename F>
void DistNumVecToDistRowVec(
		const DistVec<Index3, NumVec<F>, ElemPrtn>&   distVec,
		NumVec<F>&                                    distRowVec,
		const Index3&                                 numGrid,
		const Index3&                                 numElem,
    Int                                           localNzStart,
		Int                                           localNz,
		bool                                          isInGrid,
	  MPI_Comm                                      commDistVec )
{
#ifndef _RELEASE_
	PushCallStack("DistNumVecToDistRowVec");
#endif

	Int mpirank, mpisize;
	MPI_Comm_rank( commDistVec, &mpirank );
	MPI_Comm_size( commDistVec, &mpisize );
	
  Index3  numGridElem;

	for( Int d = 0; d < DIM; d++ ){
		numGridElem[d] = numGrid[d] / numElem[d];
	}

	// Prepare distVecRecv

	DistVec<Index3, NumVec<F>, ElemPrtn>  distVecRecv;
	distVecRecv.Prtn() = distVec.Prtn();

	for(typename std::map<Index3, NumVec<F> >::const_iterator 
			mi = distVec.LocalMap().begin();
			mi != distVec.LocalMap().end(); mi++ ){
		Index3 key = (*mi).first;  
		if( distVec.Prtn().Owner( key ) != mpirank ){
			throw std::runtime_error( "DistNumVec owns a wrong key." );
		}
		distVecRecv.LocalMap()[key] = (*mi).second;
	}


	// Specify the elements of data to receive according to the index of
	// the z-dimension. 
	std::vector<Index3>  getKey;
	if( isInGrid ){
		for( Int k = 0; k < numElem[2]; k++ ){
			if( k     * numGridElem[2] < localNzStart + localNz &&
					(k+1) * numGridElem[2] > localNzStart ){
				// The processor needs these elements
				for( Int j = 0; j < numElem[1]; j++ ){
					for( Int i = 0; i < numElem[0]; i++ ){
						getKey.push_back( Index3(i,j,k) );
					}
				} // for (j)
			}
		} // for (k)
	} // if (isInGrid)

	// Data communication.
	distVecRecv.GetBegin( getKey, NO_MASK );
	distVecRecv.GetEnd( NO_MASK );

	// Unpack the data locally into distRowVec
	if( isInGrid ){
		Int numGridLocal = numGrid[0] * numGrid[1] * localNz;
		distRowVec.Resize( numGridLocal );
		for(typename std::map<Index3, NumVec<F> >::const_iterator 
				mi = distVecRecv.LocalMap().begin();
				mi != distVecRecv.LocalMap().end(); mi++ ){
			Index3 key = (*mi).first;  
			const NumVec<F> & val = (*mi).second;
			Int kStart = std::max( 0, localNzStart - key[2] * numGridElem[2] );
			Int kEnd   = std::min( numGridElem[2], 
					localNzStart + localNz - key[2] * numGridElem[2] );
			for( Int k = kStart; k < kEnd; k++ )
				for( Int j = 0; j < numGridElem[1]; j++ )
					for( Int i = 0; i < numGridElem[0]; i++ ){
						Index3 glbIdx( 
								key[0] * numGridElem[0] + i,
								key[1] * numGridElem[1] + j,
								key[2] * numGridElem[2] + k - localNzStart );

						distRowVec( glbIdx[0] + glbIdx[1] * numGrid[0] + glbIdx[2] *
								numGrid[0] * numGrid[1] ) = 
							val( i + j * numGridElem[0] + k * numGridElem[0] *
									numGridElem[1] );
					} // for (i)
		}
	} // if (isInGrid)
#ifndef _RELEASE_
	PopCallStack();
#endif
}  // -----  end of function DistNumVecToDistRowVec  ----- 


/// @brief Convert a distributed row partitioned vector to a DistNumVec,
/// which can be used for instance after DistFourier.
template<typename F>
void DistRowVecToDistNumVec(
		const NumVec<F>&                              distRowVec,
		DistVec<Index3, NumVec<F>, ElemPrtn>&         distVec,
		const Index3&                                 numGrid,
		const Index3&                                 numElem,
    Int                                           localNzStart,
		Int                                           localNz,
		bool                                          isInGrid,
	  MPI_Comm                                      commDistVec )
{
#ifndef _RELEASE_
	PushCallStack("DistRowVecToDistNumVec");
#endif

	Int mpirank, mpisize;
	MPI_Comm_rank( commDistVec, &mpirank );
	MPI_Comm_size( commDistVec, &mpisize );
	
  Index3  numGridElem;

	distVec.LocalMap().clear();

	for( Int d = 0; d < DIM; d++ ){
		numGridElem[d] = numGrid[d] / numElem[d];
	}

	// Collect the element indicies
	std::vector<Index3> putKey;
	if( isInGrid ){
		for( Int k = 0; k < numElem[2]; k++ ){
			if( k     * numGridElem[2] < localNzStart + localNz &&
					(k+1) * numGridElem[2] > localNzStart ){
				// The current processor contains these elements
				for( Int j = 0; j < numElem[1]; j++ ){
					for( Int i = 0; i < numElem[0]; i++ ){
						putKey.push_back( Index3(i,j,k) );
					}
				} 
			}
		} // for (k)
	} // if (isInGrid)

	// Prepare the local data
	for( std::vector<Index3>::iterator vi = putKey.begin();
			 vi != putKey.end(); vi++ ){
		Index3 key = (*vi);
		typename std::map<Index3, NumVec<F> >::iterator mi = 
			distVec.LocalMap().find(key);
		if( mi == distVec.LocalMap().end() ){
			NumVec<F> empty( numGridElem[0] * numGridElem[1] * numGridElem[2] );
			SetValue( empty, (F)(0) ); 
			distVec.LocalMap()[key] = empty;
		}
		NumVec<F>& val = distVec.LocalMap()[key];

		Int kStart = std::max( 0, localNzStart - key[2] * numGridElem[2] );
		Int kEnd   = std::min( numGridElem[2], 
				localNzStart + localNz - key[2] * numGridElem[2] );

		for( Int k = kStart; k < kEnd; k++ )
			for( Int j = 0; j < numGridElem[1]; j++ )
				for( Int i = 0; i < numGridElem[0]; i++ ){
					Index3 glbIdx( 
							key[0] * numGridElem[0] + i,
							key[1] * numGridElem[1] + j,
							key[2] * numGridElem[2] + k - localNzStart );

					val( i + j * numGridElem[0] + k * numGridElem[0] *
							 numGridElem[1] ) +=
						distRowVec( glbIdx[0] + glbIdx[1] * numGrid[0] +
								glbIdx[2] * numGrid[0] * numGrid[1] );
				}
	}


	// Data communication.
	distVec.PutBegin( putKey, NO_MASK );
	distVec.PutEnd( NO_MASK, PutMode::COMBINE );

	// Erase the elements distVec does not own
	std::vector<Index3>  eraseKey;
	for(typename std::map<Index3, NumVec<F> >::iterator
			mi = distVec.LocalMap().begin();
			mi != distVec.LocalMap().end(); mi++ ){
		Index3 key = (*mi).first;
		if( distVec.Prtn().Owner( key ) != mpirank ){
			eraseKey.push_back( key );
		}
	}
	for( std::vector<Index3>::iterator vi = eraseKey.begin();
			 vi != eraseKey.end(); vi++ ){
		distVec.LocalMap().erase( *vi );
	}


#ifndef _RELEASE_
	PopCallStack();
#endif
}  // -----  end of function DistNumVecToDistRowVec  ----- 


/// @brief Convert a matrix distributed according to 2D element indices
/// to a matrix distributed block-cyclicly as in ScaLAPACK.
///
/// NOTE: 
///
/// 1. This subroutine can be considered as one implementation of Gemr2d
/// type process.
///
/// 2. This subroutine assumes that proper descriptor has been given
/// by desc, which agrees with distMat.
///
/// 3. This subroutine is mainly used for converting the DG Hamiltonian
/// matrix to ScaLAPACK format for diagonalization purpose.
///
template<typename F>
void DistElemMatToScaMat(
		const DistVec<ElemMatKey, NumMat<F>, ElemMatPrtn>&   distMat,
		const scalapack::Descriptor&                         desc, 
		scalapack::ScaLAPACKMatrix<F>&                       scaMat,
		const NumTns<std::vector<Int> >&                     basisIdx,
	  MPI_Comm  	                                         comm ){
#ifndef _RELEASE_
	PushCallStack("DistElemMatToScaMat");
#endif
	using namespace dgdft::scalapack;

	Int mpirank, mpisize;
	MPI_Comm_rank( comm, &mpirank );
	MPI_Comm_size( comm, &mpisize );

	Int nprow  = desc.NpRow();
	Int npcol  = desc.NpCol();
	Int myprow = desc.MypRow();
	Int mypcol = desc.MypCol();

	scaMat.SetDescriptor( desc );

	{
    std::vector<F>& vec = scaMat.LocalMatrix();
		for( Int i = 0; i < vec.size(); i++ ){
			vec[i] = (F) 0.0;
		}
	}

	Int MB = scaMat.MB(); 
	Int NB = scaMat.NB();

	if( MB != NB ){
		throw std::runtime_error("MB must be equal to NB.");
	}

	Int numRowBlock = scaMat.NumRowBlocks();
	Int numColBlock = scaMat.NumColBlocks();

	// Get the processor map
	IntNumMat  procGrid( nprow, npcol );
	SetValue( procGrid, 0 );
	{
		IntNumMat  procTmp( nprow, npcol );
		SetValue( procTmp, 0 );
		procTmp( myprow, mypcol ) = mpirank;
		mpi::Allreduce( procTmp.Data(), procGrid.Data(), nprow * npcol,
				MPI_SUM, comm );
	}


	BlockMatPrtn  blockPrtn;
	blockPrtn.ownerInfo.Resize( numRowBlock, numColBlock );
	IntNumMat&    blockOwner = blockPrtn.ownerInfo;
	for( Int jb = 0; jb < numColBlock; jb++ ){
		for( Int ib = 0; ib < numRowBlock; ib++ ){
			blockOwner( ib, jb ) = procGrid( ib % nprow, jb % npcol );
		}
	}

	// Intermediate variable for constructing the distributed matrix in
	// ScaLAPACK format.  
  DistVec<Index2, NumMat<F>, BlockMatPrtn> distScaMat;
	distScaMat.Prtn() = blockPrtn;

	// Initialize
	DblNumMat empty( MB, MB );
	SetValue( empty, 0.0 );
	for( Int jb = 0; jb < numColBlock; jb++ )
		for( Int ib = 0; ib < numRowBlock; ib++ ){
			Index2 key( ib, jb );
			if( distScaMat.Prtn().Owner( key ) == mpirank ){
				distScaMat.LocalMap()[key] = empty;
			}
		} // for (ib)

	// Convert matrix distributed according to elements to distScaMat
	for( typename std::map<ElemMatKey, NumMat<F> >::const_iterator 
			 mi  = distMat.LocalMap().begin();
			 mi != distMat.LocalMap().end(); mi++ ){
		ElemMatKey elemKey = (*mi).first;
		if( distMat.Prtn().Owner( elemKey ) == mpirank ){
			Index3 key1 = elemKey.first;
			Index3 key2 = elemKey.second;
			const std::vector<Int>& idx1 = basisIdx( key1(0), key1(1), key1(2) );
			const std::vector<Int>& idx2 = basisIdx( key2(0), key2(1), key2(2) );
			const NumMat<F>& localMat = (*mi).second;
			if( localMat.m() != idx1.size() ||
					localMat.n() != idx2.size() ){
				std::ostringstream msg;
				msg 
					<< "Local matrix size is not consistent." << std::endl
					<< "localMat   size : " << localMat.m() << " x " << localMat.n() << std::endl
					<< "idx1       size : " << idx1.size() << std::endl
					<< "idx2       size : " << idx2.size() << std::endl;
				throw std::runtime_error( msg.str().c_str() );
			}

			// Reshape the matrix element by element
			Int ib, jb, io, jo;
			for( Int b = 0; b < localMat.n(); b++ ){
				for( Int a = 0; a < localMat.m(); a++ ){
					ib = idx1[a] / MB;
					jb = idx2[b] / MB;
					io = idx1[a] % MB;
					jo = idx2[b] % MB;
					typename std::map<Index2, NumMat<F> >::iterator 
						ni = distScaMat.LocalMap().find( Index2(ib, jb) );
					// Contributes to blocks not owned by the current processor
					if( ni == distScaMat.LocalMap().end() ){
						distScaMat.LocalMap()[Index2(ib, jb)] = empty;
						ni = distScaMat.LocalMap().find( Index2(ib, jb) );
					}
					DblNumMat&  scaMat = (*ni).second;
					scaMat(io, jo) += localMat(a, b);
				} // for (a)
			} // for (b)


		} // own this matrix block
	} // for (mi)

	// Communication of the matrix
	{
		// Prepare
		std::vector<Index2>  keyIdx;
		for( typename std::map<Index2, NumMat<F> >::iterator 
				 mi  = distScaMat.LocalMap().begin();
				 mi != distScaMat.LocalMap().end(); mi++ ){
			Index2 key = (*mi).first;
			if( distScaMat.Prtn().Owner( key ) != mpirank ){
				keyIdx.push_back( key );
			}
		} // for (mi)

		// Communication
		distScaMat.PutBegin( keyIdx, NO_MASK );
		distScaMat.PutEnd( NO_MASK, PutMode::COMBINE );

		// Clean to save space
		std::vector<Index2>  eraseKey;
		for( typename std::map<Index2, NumMat<F> >::iterator 
				 mi  = distScaMat.LocalMap().begin();
				 mi != distScaMat.LocalMap().end(); mi++ ){
			Index2 key = (*mi).first;
			if( distScaMat.Prtn().Owner( key ) != mpirank ){
				eraseKey.push_back( key );
			}
		} // for (mi)

		for( std::vector<Index2>::iterator vi = eraseKey.begin();
				 vi != eraseKey.end(); vi++ ){
			distScaMat.LocalMap().erase( *vi );
		}	
	}

	// Copy the matrix values from distScaMat to scaMat
	{
		for( typename std::map<Index2, NumMat<F> >::iterator 
				 mi  = distScaMat.LocalMap().begin();
				 mi != distScaMat.LocalMap().end(); mi++ ){
			Index2 key = (*mi).first;
			if( distScaMat.Prtn().Owner( key ) == mpirank ){
				Int ib = key(0), jb = key(1);
				Int offset = ( jb / npcol ) * MB * scaMat.LocalLDim() + 
					( ib / nprow ) * MB;
				lapack::Lacpy( 'A', MB, MB, (*mi).second.Data(),
						MB, scaMat.Data() + offset, scaMat.LocalLDim() );
			} // own this block
		} // for (mi)
	}

#ifndef _RELEASE_
	PopCallStack();
#endif
	return;
}  // -----  end of function DistElemMatToScaMat  ----- 


/// @brief Convert a matrix distributed block-cyclicly as in ScaLAPACK 
/// back to a matrix distributed according to 1D element indices.
///
/// Note:
///
/// 1.  This routine is mainly used for converting the eigenvector
/// matrix from ScaLAPACK format back to the DG matrix format for
/// constructing density etc.
///
/// @param[in] numColKeep The first numColKeep columns are kept in each
/// matrix of distMat (i.e. distMat.LocalMap()[key].n() ) to save disk
/// space.  However, if numColKeep < 0 or omitted, *all* columns are kept in
/// distMat. 
///
template<typename F>
void ScaMatToDistNumMat(
		const scalapack::ScaLAPACKMatrix<F>&           scaMat,
		const ElemPrtn&                                prtn,
		DistVec<Index3, NumMat<F>, ElemPrtn>&          distMat,
		const NumTns<std::vector<Int> >&               basisIdx,
	  MPI_Comm  	                                   comm,
	  Int                                            numColKeep = -1 ){
#ifndef _RELEASE_
	PushCallStack("ScaMatToDistNumMat");
#endif
	using namespace dgdft::scalapack;

	Int mpirank, mpisize;
	MPI_Comm_rank( comm, &mpirank );
	MPI_Comm_size( comm, &mpisize );

	Descriptor desc = scaMat.Desc();
	distMat.Prtn()  = prtn;

	Index3 numElem;
	numElem[0] = prtn.ownerInfo.m();
	numElem[1] = prtn.ownerInfo.n(); 
	numElem[2] = prtn.ownerInfo.p(); 

	if( numColKeep < 0 )
		numColKeep = scaMat.Width();

	if( numColKeep > scaMat.Width() )
		throw std::runtime_error("NumColKeep cannot be bigger than the matrix width.");

	Int nprow  = desc.NpRow();
	Int npcol  = desc.NpCol();
	Int myprow = desc.MypRow();
	Int mypcol = desc.MypCol();

	Int MB = scaMat.MB(); 
	Int NB = scaMat.NB();

	if( MB != NB ){
		throw std::runtime_error("MB must be equal to NB.");
	}

	Int numRowBlock = scaMat.NumRowBlocks();
	Int numColBlock = scaMat.NumColBlocks();

	// Get the processor map
	IntNumMat  procGrid( nprow, npcol );
	SetValue( procGrid, 0 );
	{
		IntNumMat  procTmp( nprow, npcol );
		SetValue( procTmp, 0 );
		procTmp( myprow, mypcol ) = mpirank;
		mpi::Allreduce( procTmp.Data(), procGrid.Data(), nprow * npcol,
				MPI_SUM, comm );
	}


	BlockMatPrtn  blockPrtn;
	blockPrtn.ownerInfo.Resize( numRowBlock, numColBlock );
	IntNumMat&    blockOwner = blockPrtn.ownerInfo;
	for( Int jb = 0; jb < numColBlock; jb++ ){
		for( Int ib = 0; ib < numRowBlock; ib++ ){
			blockOwner( ib, jb ) = procGrid( ib % nprow, jb % npcol );
		}
	}

	// Intermediate variable for constructing the distributed matrix in
	// ScaLAPACK format.  
  DistVec<Index2, NumMat<F>, BlockMatPrtn> distScaMat;
	distScaMat.Prtn() = blockPrtn;

	// Initialize
	DblNumMat empty( MB, MB );
	SetValue( empty, 0.0 );
	for( Int jb = 0; jb < numColBlock; jb++ )
		for( Int ib = 0; ib < numRowBlock; ib++ ){
			Index2 key( ib, jb );
			if( distScaMat.Prtn().Owner( key ) == mpirank ){
				distScaMat.LocalMap()[key] = empty;
			}
		} // for (ib)

	// Convert ScaLAPACK matrix to distScaMat
	for( Int jb = 0; jb < numColBlock; jb++ )
		for( Int ib = 0; ib < numRowBlock; ib++ ){
			Index2 key( ib, jb );
			if( distScaMat.Prtn().Owner(key) == mpirank ){
				typename std::map<Index2, NumMat<F> >::iterator
					mi = distScaMat.LocalMap().find(key);
				Int offset = ( jb / npcol ) * MB * scaMat.LocalLDim() + 
					( ib / nprow ) * MB;
				lapack::Lacpy( 'A', MB, MB, scaMat.Data() + offset,
						scaMat.LocalLDim(), (*mi).second.Data(), MB );
			} // own this block
		} // for (ib)


	// Communication of the matrix
	{
		// Prepare
		std::set<Index2>  keySet;
		for( Int k = 0; k < numElem[2]; k++ )
			for( Int j = 0; j < numElem[1]; j++ )
				for( Int i = 0; i < numElem[0]; i++ ){
					Index3 key( i, j, k );
					if( distMat.Prtn().Owner( key ) == mpirank ) {
						const std::vector<Int>&  idx = basisIdx(i, j, k);
						for( Int g = 0; g < idx.size(); g++ ){
							Int ib = idx[g] / MB;
							for( Int jb = 0; jb < numColBlock; jb++ )
								keySet.insert( Index2( ib, jb ) );
						} // for (g)
					}
				} // for (i)

		std::vector<Index2>  keyIdx;
		keyIdx.insert( keyIdx.begin(), keySet.begin(), keySet.end() );

		// Actual communication
		distScaMat.GetBegin( keyIdx, NO_MASK );
		distScaMat.GetEnd( NO_MASK );
	}


	// Write back to distMat

	for( Int k = 0; k < numElem[2]; k++ )
		for( Int j = 0; j < numElem[1]; j++ )
			for( Int i = 0; i < numElem[0]; i++ ){
				Index3 key( i, j, k );
				if( distMat.Prtn().Owner( key ) == mpirank ) {
					const std::vector<Int>&  idx = basisIdx(i, j, k);
					NumMat<F>& localMat = distMat.LocalMap()[key];
					localMat.Resize( idx.size(), numColKeep );
					SetValue( localMat, (F)0 );
					for( Int g = 0; g < idx.size(); g++ ){
						Int ib = idx[g] / MB;
						Int io = idx[g] % MB;
						for( Int jb = 0; jb < numColBlock; jb++ ){
							NumMat<F>& localScaMat = 
								distScaMat.LocalMap()[Index2(ib, jb)];
							for( Int jo = 0; jo < MB; jo++ ){
								Int h = jb * MB + jo;
								if( h < numColKeep )
									localMat(g, h) = localScaMat(io, jo);
							}
						}
					} // for (g)
				} // own this element
			} // for (i)


#ifndef _RELEASE_
	PopCallStack();
#endif
	return;
}  // -----  end of function ScaMatToDistNumMat  ----- 


/// @brief Computes the inner product of three terms.
inline Real ThreeDotProduct(Real* x, Real* y, Real* z, Int ntot) {
  Real sum =0;
  for(Int i=0; i<ntot; i++) {
    sum += (*x++)*(*y++)*(*z++);
  }
  return sum;
}

/// @brief Computes the inner product of four terms.
inline Real FourDotProduct(Real* w, Real* x, Real* y, Real* z, Int ntot) {
  Real sum =0;
  for(Int i=0; i<ntot; i++) {
    sum += (*w++)*(*x++)*(*y++)*(*z++);
  }
  return sum;
}

} // namespace dgdft


#endif // _HAMILTONIAN_DG_HPP_
