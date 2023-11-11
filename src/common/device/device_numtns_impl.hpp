//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

/// @file device_numtns_impl.hpp
/// @brief Implementation of numerical tensor on device.
/// @date 2020-08-12
#ifdef DEVICE
#ifndef _DEVICE_NUMTNS_IMPL_HPP_
#define _DEVICE_NUMTNS_IMPL_HPP_

#include  "device_numtns_decl.h"

namespace  scales{

template <class F> 
  inline deviceNumTns<F>::deviceNumTns(Int m, Int n, Int p): m_(m), n_(n), p_(p), owndata_(true) {
    if(m_>0 && n_>0 && p_>0) 
      data_ = (F*)device_malloc( sizeof(F) * m_ * n_ *p_); 
    else 
      data_=NULL;
  }

template <class F> 
  inline deviceNumTns<F>::deviceNumTns(Int m, Int n, Int p, bool owndata, F* data): m_(m), n_(n), p_(p), owndata_(owndata) {
    if(owndata_) {
      if(m_>0 && n_>0 && p_>0) {
        data_ = (F*) device_malloc( sizeof(F) * m_ * n_ * p_);
        device_memcpy_DEVICE2DEVICE(data_, data, sizeof(F) * m_ * n_ * p_ );
      }
      else 
        data_=NULL;
    }
    else{ data_ = data; } 
  } 

template <class F> 
  inline deviceNumTns<F>::deviceNumTns(const deviceNumTns& C): m_(C.m_), n_(C.n_), p_(C.p_), owndata_(C.owndata_) {
    if(owndata_) {
      if(m_>0 && n_>0 && p_>0) {
        data_ = (F*) device_malloc( sizeof(F) * m_ * n_ * p_);
        device_memcpy_DEVICE2DEVICE(data_, C.data_, sizeof(F) * m_ * n_ * p_ );
      }
      else 
        data_=NULL;
    }
    else{ data_ = C.data_; } 
  } 


template <class F> 
  inline deviceNumTns<F>::~deviceNumTns() { 
    if(owndata_) { 
      if(m_>0 && n_>0 && p_>0) { device_free(data_) ; data_ = NULL; } 
    }
  }

template <class F> 
  inline deviceNumTns<F>& deviceNumTns<F>::operator=(const deviceNumTns& C) {
    // Do not copy if it is the same matrix.
    if(C.data_ == data_) return *this;

    if(owndata_) { 
      if(m_>0 && n_>0 && p_>0) { device_free(data_) ; data_ = NULL; } 
    }
    m_ = C.m_; n_=C.n_; p_=C.p_; owndata_=C.owndata_;
    if(owndata_) {
      if(m_>0 && n_>0 && p_>0) {
        data_ = (F*) device_malloc( sizeof(F) * m_ * n_ * p_);
        device_memcpy_DEVICE2DEVICE(data_, C.data_, sizeof(F) * m_ * n_ * p_ );
      }
      else data_ = NULL;
    } 
    else data_ = C.data_;
    return (*this);
  }


template <class F> 
  inline void deviceNumTns<F>::Resize(Int m, Int n, Int p)  {
    if( owndata_ == false ){
      ErrorHandling("Tensor being resized must own data.");
    }
    if(m_!=m || n_!=n || p_!=p) {
      if(m_>0 && n_>0 && p_>0) { device_free(data_); data_ = NULL; } 
      m_ = m; n_ = n; p_=p;
      if(m_>0 && n_>0 && p_>0) { data_ = (F*) device_malloc( sizeof(F) * m_ * n_ * p_); } else data_=NULL;
    }
  }

template <class F> 
  inline const F& deviceNumTns<F>::operator()(Int i, Int j, Int k) const  {
    if( i < 0 || i >= m_ ||
        j < 0 || j >= n_ ||
        k < 0 || k >= p_ ) {
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ", " << n_ << ", " << p_ << ")" << std::endl
        << "This index     ~ (" << i  << ", " << j  << ", " << k  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
    return data_[i+j*m_+k*m_*n_];
  }

template <class F> 
  inline F& deviceNumTns<F>:: operator()(Int i, Int j, Int k)  {
    if( i < 0 || i >= m_ ||
        j < 0 || j >= n_ ||
        k < 0 || k >= p_ ) {
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ", " << n_ << ", " << p_ << ")" << std::endl
        << "This index     ~ (" << i  << ", " << j  << ", " << k  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
    return data_[i+j*m_+k*m_*n_];
  }

template <class F> 
  inline F* deviceNumTns<F>::MatData (Int k) const {
    if( k < 0 || k >= p_ ) {
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << p_ << ")" << std::endl
        << "This index     ~ (" << k  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
    return &(data_[k*m_*n_]);
  }

template <class F> 
  inline F* deviceNumTns<F>::VecData (Int j, Int k) const {
    if( j < 0 || j >= n_ ||
        k < 0 || k >= p_ ) {
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << n_ << ", " << p_ << ")" << std::endl
        << "This index     ~ (" << j  << ", " << k  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
    return &(data_[k*m_*n_+j*m_]);
  }


// *********************************************************************
// Utilities
// *********************************************************************
/*
template <class F> inline void SetValue(deviceNumTns<F>& T, F val)
{
  F *ptr = T.data_;
  for(Int i=0; i < T.m() * T.n() * T.p(); i++) *(ptr++) = val; 

  return;
}

template <class F> inline Real Energy(const deviceNumTns<F>& T)
{
  Real sum = 0;

  F *ptr = T.Data();
  for(Int i=0; i < T.m() * T.n() * T.p(); i++) 
    sum += std::abs(ptr[i]) * std::abs(ptr[i]);

  return sum;
}
*/
} // namespace scales

#endif // _DEVICE_NUMTNS_IMPL_HPP_
#endif // DEVICE
