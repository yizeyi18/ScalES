//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lexing Ying and Lin Lin 

/// @file numtns_impl.hpp
/// @brief Implementation of numerical tensor.
/// @date 2010-09-27
#ifndef _NUMTNS_IMPL_HPP_
#define _NUMTNS_IMPL_HPP_

#include  "numtns_decl.h"

namespace  scales{

template <class F> 
  inline NumTns<F>::NumTns(Int m, Int n, Int p): m_(m), n_(n), p_(p), owndata_(true) {
    if(m_>0 && n_>0 && p_>0) {
      data_ = new F[m_*n_*p_]; 
      if( data_ == nullptr ) ErrorHandling("Cannot allocate memory."); 
    }//m_>0 && n_>0 && p_>0
    else data_=nullptr;
  }

template <class F> 
  inline NumTns<F>::NumTns(Int m, Int n, Int p, bool owndata, F* data): m_(m), n_(n), p_(p), owndata_(owndata) {
    if(owndata_) {
      if(m_>0 && n_>0 && p_>0) { 
        data_ = new F[m_*n_*p_]; 
	if( data_ == nullptr ) ErrorHandling("Cannot allocate memory."); 
      }//m_>0 && n_>0 && p_>0
      else data_=nullptr;
      if(m_>0 && n_>0 && p_>0) 
        for(Int i=0; i<m_*n_*p_; i++) data_[i] = data[i]; 
    }//if owndata_
    else 
      data_ = data;
  }

template <class F> 
  inline NumTns<F>::NumTns(const NumTns& C): m_(C.m_), n_(C.n_), p_(C.p_), owndata_(C.owndata_) {
    if(owndata_) {
      if(m_>0 && n_>0 && p_>0) { 
        data_ = new F[m_*n_*p_]; 
	if( data_ == nullptr ) ErrorHandling("Cannot allocate memory."); 
      }//m_>0 && n_>0 && p_>0
      else data_=nullptr;
      if(m_>0 && n_>0 && p_>0) 
        for(Int i=0; i<m_*n_*p_; i++) data_[i] = C.data_[i]; 
    }//if owndata_
    else 
      data_ = C.data_;
  }

template <class F> 
  inline NumTns<F>::~NumTns() { 
    if(owndata_ && m_>0 && n_>0 && p_>0) { 
        delete[] data_; data_ = nullptr; 
    } 
  }

template <class F> 
  inline NumTns<F>& NumTns<F>::operator=(const NumTns& C) {
    // Do not copy if it is the same matrix.
    if(C.data_ == data_) return *this;

    if(owndata_ && m_>0 && n_>0 && p_>0) { 
      delete[] data_; data_ = nullptr; 
    }//if owndata_
    m_ = C.m_; 
    n_=C.n_; 
    p_=C.p_; 
    owndata_=C.owndata_;
    if(owndata_) {
      if(m_>0 && n_>0 && p_>0) { 
        data_ = new F[m_*n_*p_]; 
	if( data_ == nullptr ) ErrorHandling("Cannot allocate memory."); 
      }//m_>0 && n_>0 && p_>0 
      else data_=nullptr;
      if(m_>0 && n_>0 && p_>0)  
        for(Int i=0; i<m_*n_*p_; i++) data_[i] = C.data_[i]; 
    }//if owndata_ 
    else 
      data_ = C.data_;
    return *this;
  }//NumTns::operator=

template <class F> 
  inline void NumTns<F>::Resize(Int m, Int n, Int p)  {
    if( owndata_ == false ){
      ErrorHandling("Tensor being resized must own data.");
    }
    if(m_!=m || n_!=n || p_!=p) {
      if(m_>0 && n_>0 && p_>0) { 
        delete[] data_; data_ = nullptr; } 
      m_ = m; n_ = n; p_=p;
      if(m_>0 && n_>0 && p_>0) { 
        data_ = new F[m_*n_*p_]; 
	if( data_ == nullptr ) ErrorHandling("Cannot allocate memory."); 
      }//m_>0 && n_>0 && p_>0 
      else data_=nullptr;
    }//if m_!=m || n_!=n || p_!=p
  }//NumTns::Resize

template <class F> 
  inline const F& NumTns<F>::operator()(Int i, Int j, Int k) const  {
#if ( _DEBUGlevel_ >= 1 )
    if( i < 0 || i >= m_ ||
        j < 0 || j >= n_ ||
        k < 0 || k >= p_ ) {//FIXME 检查m_ n_ p_是否小于0
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ", " << n_ << ", " << p_ << ")" << std::endl
        << "This index     ~ (" << i  << ", " << j  << ", " << k  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
#endif
    return data_[i+j*m_+k*m_*n_];
  }

template <class F> 
  inline F& NumTns<F>:: operator()(Int i, Int j, Int k)  {
#if ( _DEBUGlevel_ >= 1 )
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
#endif
    return data_[i+j*m_+k*m_*n_];
  }

template <class F> 
  inline F* NumTns<F>::MatData (Int k) const {                 //返回第三指标是k的首元素地址，不负责矩阵大小
#if ( _DEBUGlevel_ >= 1 )
    if( k < 0 || k >= p_ ) {
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << p_ << ")" << std::endl
        << "This index     ~ (" << k  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
#endif
    return &(data_[k*m_*n_]);
  }

template <class F> 
  inline F* NumTns<F>::VecData (Int j, Int k) const {          //返回第二指标是j，第三指标是k的首元素地址，不负责向量大小
#if ( _DEBUGlevel_ >= 1 )
    if( j < 0 || j >= n_ ||
        k < 0 || k >= p_ ) {
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << n_ << ", " << p_ << ")" << std::endl
        << "This index     ~ (" << j  << ", " << k  << ")" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
#endif
    return &(data_[k*m_*n_+j*m_]);
  }


// *********************************************************************
// Utilities
// *********************************************************************

template <class F> inline void SetValue(NumTns<F>& T, F val) { //FIXME 是否有非逐个遍历的更好写法？
  F *ptr = T.data_;
  Int max= T.m() * T.n() * T.p();
  for(Int i=0; i < max; i++) {
    *ptr = val; 
    ptr++;
  }//for
  return;
}

template <class F> inline Real Energy(const NumTns<F>& T) {    //取模
  Real sum = 0;

  Int max=T.m() * T.n() * T.p();
  F *ptr = T.Data();
  for(Int i=0; i < max; i++) 
    sum += std::abs(ptr[i]) * std::abs(ptr[i]);
  return sum;
}

} // namespace scales

#endif // _NUMTNS_IMPL_HPP_
