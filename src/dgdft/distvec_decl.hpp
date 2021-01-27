//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lexing Ying and Lin Lin

/// @file distvec_decl.hpp
/// @brief General purpose parallel vectors.
/// @date 2013-01-09
#ifndef _DISTVEC_DECL_HPP_
#define _DISTVEC_DECL_HPP_

#include "environment.hpp"

namespace scales{

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
  void SetComm( MPI_Comm comm ) {comm_ = comm;}
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

} //  namespace scales

#endif // _DISTVEC_DECL_HPP_

