//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

/// @file device_numtns_decl.hpp
/// @brief Numerical tensor on device.
/// @date 2020-08-12
#ifdef DEVICE
#ifndef _DEVICE_NUMTNS_DECL_HPP_
#define _DEVICE_NUMTNS_DECL_HPP_

#include "environment.h"
#include "device_nummat_impl.h"

namespace  scales{

// Templated form of numerical tensor
//
// The main advantage of this portable NumVec structure is that it can
// either own (owndata == true) or view (owndata == false) a piece of
// data.

template <class F>
  class deviceNumTns
  {
  public:
    Int m_, n_, p_;
    bool owndata_;
    F* data_;
  public:
    deviceNumTns(Int m=0, Int n=0, Int p=0);

    deviceNumTns(Int m, Int n, Int p, bool owndata, F* data);

    deviceNumTns(const deviceNumTns& C);

    ~deviceNumTns();

    deviceNumTns& operator=(const deviceNumTns& C);

    void Resize(Int m, Int n, Int p);

    const F& operator()(Int i, Int j, Int k) const;

    F& operator()(Int i, Int j, Int k);

    bool IsOwnData() const { return owndata_; }

    F* Data() const { return data_; }

    F* MatData (Int k) const; 

    F* VecData (Int j, Int k) const; 

    Int m() const { return m_; }

    Int n() const { return n_; }

    Int p() const { return p_; }

    Int Size() const { return m_ * n_ * p_; }
  };


// Commonly used
typedef deviceNumTns<bool>       deviceBolNumTns;
typedef deviceNumTns<Int>        deviceIntNumTns;
typedef deviceNumTns<Real>       deviceDblNumTns;
typedef deviceNumTns<cuDoubleComplex>    deviceCpxNumTns;

// Utilities
//template <class F> inline void SetValue(deviceNumTns<F>& T, F val);

//template <class F> inline Real Energy(const deviceNumTns<F>& T);



} // namespace scales

#endif // _DEVICE_NUMTNS_DECL_HPP_
#endif // DEVICE
