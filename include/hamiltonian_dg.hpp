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

	Int Owner (Index3 key) const {
		return ownerInfo(key(0), key(1), key(2));
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

/// @struct PseudoPotElem
/// @brief The pseudocharge and nonlocal projectors for each atom. 
///
/// Each vector is defined element-by-element, in the format of
/// SparseVec.
/// 
struct PseudoPotElem 
{
	/// @brief Pseudocharge of an atom, defined on the uniform grid.
  NumTns<SparseVec>                                    pseudoCharge; 
	/// @brief Nonlocal projectors of an atom, defined on the LGL grid.
	std::vector<std::pair<NumTns<SparseVec>, Real> >     vnlList;
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

typedef DistVec<Int, PseudoPotElem, AtomPrtn>  DistPseudoPotElem;

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




	// *********************************************************************
	// Computational variables
	// *********************************************************************

	/// @brief The number of elements.
	Index3                      numElem_;

	/// @brief Partition of element.
	ElemPrtn                    elemPrtn_;

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

	/// @brief Pseudopotential and nonlocal projectors associated with
	/// each atom.
	DistPseudoPotElem  pseudo_;

	/// @brief Differentiation matrix on the LGL grid.
	std::vector<DblNumMat>    DMat_;

public:

	// *********************************************************************
	// Lifecycle
	// *********************************************************************
	HamiltonianDG() {}
	~HamiltonianDG() {}
	HamiltonianDG( const esdf::ESDFInputParam& esdfParam );


	// *********************************************************************
	// Operations
	// *********************************************************************

	/// @brief Differentiate the basis functions on a certain element
	/// along the dimension d.
	void DiffPsi(const Index3& numGrid, const Real* psi, Real* Dpsi, Int d);



	void CalculatePseudoPotential( PeriodTable &ptable );
	//
	//	virtual void CalculateNonlocalPP( PeriodTable &ptable ) = 0;
	//
	//	virtual void CalculateDensity( const Spinor &psi, const DblNumVec &occrate, Real &val ) = 0;
	//
	//	virtual void CalculateXC (Real &val) = 0;
	//
	void CalculateHartree( DistFourier& fft );
	//
	//	virtual void CalculateVtot( DblNumVec& vtot ) = 0;
	//

	// *********************************************************************
	// Access
	// *********************************************************************

	/// @brief Total potential in the global domain.
	DistDblNumVec&  Vtot() { return vtot_; }

	/// @brief Exchange-correlation potential in the global domain. No
	/// magnization calculation in the DG code.
	DistDblNumVec&  Vxc()  { return vxc_; }

	/// @brief Hartree potential in the global domain.
	DistDblNumVec&  Vhart() { return vhart_; }

	DistDblNumVec&  Density() { return density_; }

	DistDblNumVec&  PseudoCharge() { return pseudoCharge_; }
	
	std::vector<Atom>&  AtomList() { return atomList_; }

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


} // namespace dgdft


#endif // _HAMILTONIAN_HPP_
