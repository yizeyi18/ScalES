#ifndef _DIST_NUMVEC_DECL_HPP_
#define _DIST_NUMVEC_DECL_HPP_

#include "numvec_impl.hpp"
#include "numtns_impl.hpp"
#include "tinyvec_impl.hpp"
#include "distvec_impl.hpp"

namespace dgdft{

/// @class ElemPtn
/// @brief Partition class (used by DistVec) for the elements.
///
/// ElemPtn contains the information of the ownership of an element, as
/// well as the local-to-global / global-to-local index maps. 
///
/// elemPtr[d][e] returns the leading grid point index of the e-th
/// element along the d-th direction.  
///
/// elemIdx[d][i] returns the element index of the i-th grid point along
/// the d-th direction.
class ElemPtn
{
private:
	IntNumTns                 ownerInfo_;

	std::vector<IntNumVec>    elemPtr_;

  std::vector<IntNumVec>    elemIdx_;	

public:
	ElemPtn() {;}

	~ElemPtn() {;}

	IntNumTns& OwnerInfo() { return ownerInfo_; }

	// OWNER is determined by the row index only
	Int Owner(Index3 key) {
		return ownerInfo_(key(0), key(1), key(2));
	}

	Int  NumGrid( Int d ) { return elemPtr_[d][elemPtr_[d].m()-1]; }

	Index3  NumGrid() { return Index3( elemPtr_[0][elemPtr_[0].m()-1],
		                                 elemPtr_[1][elemPtr_[1].m()-1],
		                                 elemPtr_[2][elemPtr_[2].m()-1]	); }

	void GlobalToLocalIndex( 
			const Index3& globalIdx,
			Index3&       elemKey,
			Index3&       localIdx );

	void GlobalToLocalIndex(
			const Int     globalIdx,
			Index3&       elemKey,
			Int&          localIdx );

	void LocalToGlobalIndex( 
			const Index3& elemKey,
			const Index3& localIdx,
			Index3&       globalIdx);

	void LocalToGlobalIndex(
			const Index3& elemKey,
			const Int&    localIdx,
			Int&          globalIdx );

};

template<typename F>
class DistNumVec{
private:

	DistVec<Index3, NumVec<F>, ElemPtn>    data_;

public:

  DistNumVec();

	DistNumVec( const ElemPtn& prtn );

	~DistNumVec();

	DistVec<Index3, NumVec<F>, ElemPtn>& Data() {return data_;}

  std::map<Index3, NumVec<F> >&        LocalMap() {return data_.LocalMap();} 

	void AssemblyBegin();

	void AssemblyEnd();


};


// Commonly used
typedef DistNumVec<Real>        DistDblNumVec;
typedef DistNumVec<Complex>     DistCpxNumVec;

} //  namespace dgdft

#endif // _DIST_NUMVEC_DECL_HPP_

