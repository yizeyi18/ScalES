/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Weile Jia

This file is part of DGDFT. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
(2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to Lawrence Berkeley National
Laboratory, without imposing a separate written license agreement for such
Enhancements, then you hereby grant the following license: a non-exclusive,
royalty-free perpetual license to install, use, modify, prepare derivative
works, incorporate into other computer software, distribute, and sublicense
such enhancements or derivative works thereof, in binary and source code form.
 */
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

namespace  dgdft{

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

} // namespace dgdft

#endif // _DEVICE_NUMVEC_DECL_HPP_
#endif // DEVICE