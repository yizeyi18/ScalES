//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

/// @file device_numvec_decl.hpp
/// @brief  Numerical vector on device.
/// @date 2020-08-12
#ifdef DEVICE
#ifndef _DEVICE_NUMVEC_DECL_HPP_
#define _DEVICE_NUMVEC_DECL_HPP_

#include "environment.hpp"
//#include "numvec_decl.hpp"
//#include "numvec_impl.hpp"
#include "device_utils.h"

namespace  scales{

// Templated form of numerical vectors
//
// The main advantage of this portable NumVec structure is that it can
// either own (owndata == true) or view (owndata == false) a piece of
// data.

template <class F> class deviceNumVec
{
public:
  Int  m_;                                // The number of elements 
  bool owndata_;                          // Whether it owns the data
  F* data_;                               // The pointer for the actual data
public:
  deviceNumVec(Int m = 0);
  deviceNumVec(Int m, bool owndata, F* data);
  deviceNumVec(const deviceNumVec& C);
  ~deviceNumVec();

  deviceNumVec& operator=(const deviceNumVec& C);

  void Resize ( Int m );

  const F& operator()(Int i) const;  
  F& operator()(Int i);  
  const F& operator[](Int i) const;
  F& operator[](Int i);

  bool IsOwnData() const { return owndata_; }

  F*   Data() const { return data_; }

  Int  m () const { return m_; }

  Int Size() const { return m_; }

  void CopyTo(NumVec<F> & C); 

  void CopyFrom(const NumVec<F> &C);

};

// Commonly used
typedef deviceNumVec<bool>	deviceBolNumVec;
typedef deviceNumVec<Int>	deviceIntNumVec;
typedef deviceNumVec<Real>	deviceDblNumVec;
typedef deviceNumVec<cuDoubleComplex>	deviceCpxNumVec;


// Utilities
#if 0
template <class F> inline void SetValue( deviceNumVec<F>& vec, F val );
template <class F> inline Real Energy( const deviceNumVec<F>& vec );
template <class F> inline void Sort( deviceNumVec<F>& vec );
#endif

} // namespace scales

#endif // _DEVICE_NUMVEC_DECL_HPP_
#endif // DEVICE
