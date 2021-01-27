//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

/// @file device_nummat_decl.hpp
/// @brief Numerical matrix on device
/// @date 2020-08-12
#ifdef DEVICE
#ifndef _DEVICE_NUMMAT_DECL_HPP_
#define _DEVICE_NUMMAT_DECL_HPP_

#include "environment.hpp"
#include "device_nummat_decl.hpp"
#include "device_utils.h"

namespace  scales{

// Templated form of numerical matrix
//
// The main advantage of this portable deviceNumMat structure is that it can
// either own (owndata == true) or view (owndata == false) a piece of
// data.

template <class F>
  class deviceNumMat
  {
  public:
    Int m_, n_;
    bool owndata_;
    F* data_;

  public:

    deviceNumMat(Int m=0, Int n=0);

    deviceNumMat(Int m, Int n, bool owndata, F* data);

    deviceNumMat(const deviceNumMat& C); // the C must be a deviceMat

    ~deviceNumMat();

    deviceNumMat& operator=(const deviceNumMat& C);

    void Resize(Int m, Int n);

    const F& operator()(Int i, Int j) const;   // must be a device function, currently do not implement.

    F& operator()(Int i, Int j);   // must be device function.

    bool IsOwnData() const { return owndata_; }

    F* Data() const { return data_; }

    F* VecData(Int j)  const;  // current do no implement this func.

    Int m() const { return m_; }

    Int n() const { return n_; }

    Int Size() const { return m_ * n_; }

    void CopyTo(NumMat<F> & C); 

    void CopyFrom(const NumMat<F> &C);

    void CopyTo(deviceNumMat<F> & C); 

    void CopyFrom(const deviceNumMat<F> &C);
  };

// Commonly used
typedef deviceNumMat<bool>     deviceBolNumMat;
typedef deviceNumMat<Int>      deviceIntNumMat;
typedef deviceNumMat<Real>     deviceDblNumMat;
typedef deviceNumMat<float>    deviceFltNumMat;
//typedef deviceNumMat<deviceDoubleComplex>  deviceCpxNumMat;

// Utilities
/*
template <class F> inline void SetValue(deviceNumMat<F>& M, F val);
template <class F> inline Real Energy(const deviceNumMat<F>& M);
template <class F> inline void Transpose ( const deviceNumMat<F>& A, deviceNumMat<F>& B );
template <class F> inline void Symmetrize( deviceNumMat<F>& A );
*/
} // namespace scales

#endif // _DEVICE_NUMMAT_DECL_HPP_
#endif // DEVICE

