/// @file distvec.hpp
/// @brief General purpose parallel vectors.
/// @author Lexing Ying and Lin Lin
/// @date 2013-01-09
#ifndef _DISTVEC_DECL_HPP_
#define _DISTVEC_DECL_HPP_

#include "environment.hpp"

namespace dgdft{

/// @namespace PutMode
/// @brief Mode for put operation of DistVec.
namespace PutMode{
enum {
	REPLACE = 0,
	COMBINE = 1,
};
}

/// @class DistVec
/// @brief General purpose parallel vector interfaces.
///
/// DistVec uses a triplet <Key, Data, Partition> to describe a general
/// parallel vectors.
template <class Key, class Data, class Partition>
class DistVec
{
private:
	std::map<Key,Data> lclmap_;
	Partition          prtn_;   //has function owner:Key->pid
	MPI_Comm           comm_;

	// Data communication variables
	std::vector<Int> snbvec_;
	std::vector<Int> rnbvec_;
	std::vector< std::vector<char> > sbufvec_;
	std::vector< std::vector<char> > rbufvec_;
	MPI_Request *reqs_;
	MPI_Status  *stats_;
public:
	DistVec( MPI_Comm comm = MPI_COMM_WORLD ) : comm_(comm) {;}
	~DistVec() {;}
	//
	std::map<Key,Data>& LocalMap()             { return lclmap_; }
	const std::map<Key,Data>& LocalMap() const { return lclmap_; }
	Partition& Prtn()                          { return prtn_; }
	const Partition& Prtn() const              { return prtn_; }
	//
	Int Insert(Key, Data&);
	Data& Access(Key);
	//
	Int GetBegin(Int (*e2ps)(Key, Data& ,std::vector<Int>&), const std::vector<Int>& mask); //gather all entries st pid contains this proc
	Int GetBegin(std::vector<Key>& keyvec, const std::vector<Int>& mask); //gather all entries with key in keyvec
	Int GetEnd(const std::vector<Int>& mask);
	Int PutBegin(std::vector<Key>& keyvec, const std::vector<Int>& mask); //put data for all entries with key in keyvec
	Int PutEnd(const std::vector<Int>& mask, Int putmode=0);
	//
	Int Expand( std::vector<Key>& keyvec); //allocate space for not-owned entries
	Int Discard(std::vector<Key>& keyvec); //remove non-owned entries
	Int mpirank() const { Int rank; MPI_Comm_rank(comm_, &rank); return rank; }
	Int mpisize() const { Int size; MPI_Comm_size(comm_, &size); return size; }
};

} //  namespace dgdft

#endif // _DISTVEC_DECL_HPP_

