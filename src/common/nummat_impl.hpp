//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lexing Ying and Lin Lin 

/// @file nummat_impl.hpp
/// @brief Implementation of numerical matrix.
/// @date 2010-09-27
#ifndef _NUMMAT_IMPL_HPP_
#define _NUMMAT_IMPL_HPP_

#include "nummat_decl.h"

namespace  scales{

template <class F> inline NumMat<F>::NumMat(Int m, Int n): m_(m), n_(n), owndata_(true) {
  if(m_>0 && n_>0) { 
    data_ = new F[m_*n_]; 
    if( data_ == nullptr ) ErrorHandling("Cannot allocate memory."); } 
  else data_=nullptr;
}//NumMat::NumMat()

template <class F> inline NumMat<F>::NumMat(Int m, Int n, bool owndata, F* data): m_(m), n_(n), owndata_(owndata) {
  if(owndata_) {
    if(m_>0 && n_>0) { 
      data_ = new F[m_*n_]; 
      if( data_ == nullptr ) ErrorHandling("Cannot allocate memory."); 
    }//m_ > 0 && n_ > 0 
    else data_=nullptr;
    if(m_>0 && n_>0) //复制
      for(Int i=0; i<m_*n_; i++) data_[i] = data[i];
  }//if owndata_
  else {
    data_ = data;
  }
}//NumMat::NumMat()

template <class F> inline NumMat<F>::NumMat(const NumMat& C): m_(C.m_), n_(C.n_), owndata_(C.owndata_) {
  if(owndata_) {
    if(m_>0 && n_>0) { 
      data_ = new F[m_*n_]; 
      if( data_ == nullptr ) ErrorHandling("Cannot allocate memory."); 
    }//m_ > 0 && n_ > 0 
    else data_=nullptr;
    if(m_>0 && n_>0) 
      for(Int i=0; i<m_*n_; i++) data_[i] = C.data_[i]; 
  }//if owndata_ 
  else {
    data_ = C.data_;
  }
}//NumMat::NumMat()

template <class F> inline NumMat<F>::~NumMat() {
  if(owndata_ && m_>0 && n_>0){ 
    delete[] data_; data_ = nullptr; 
  }
}//NumMat::~NumMat()

template <class F> inline NumMat<F>& NumMat<F>::operator=(const NumMat& C) {
  // Do not copy if it is the same matrix.
  if(C.data_ == data_) return *this;
  if(owndata_ && m_>0 && n_>0) {
    delete[] data_; data_ = nullptr; 
  }//if owndata_
  m_ = C.m_; 
  n_=C.n_; 
  owndata_=C.owndata_;
  if(owndata_) {
    if(m_>0 && n_>0) {
      data_ = new F[m_*n_];
      if( data_ == nullptr ) ErrorHandling("Cannot allocate memory."); 
    }//m_ > 0 && n_ > 0
    else data_=NULL;
    if(m_>0 && n_>0) { for(Int i=0; i<m_*n_; i++) data_[i] = C.data_[i]; }
  }//if owndata_
  else {
    data_ = C.data_;
  }
  return *this;
}//operator =

template <class F> inline void NumMat<F>::Resize(Int m, Int n)  { //成功的Resize会删除存储的数据
  if( owndata_ == false ){
    ErrorHandling("Matrix being resized must own data.");
  }
  if(m_!=m || n_!=n) {
    if(m_>0 && n_>0) { delete[] data_; data_ = nullptr; }
    m_ = m; n_ = n;
    if(m_>0 && n_>0) {
      data_ = new F[m_*n_]; 
      if( data_ == nullptr ) ErrorHandling("Cannot allocate memory."); 
    } //m_ > 0 && n_ > 0
    else data_=nullptr;
  }//if m_!=m || n_!=n
}//NumMat::Resize()

template <class F> 
inline const F& NumMat<F>::operator()(Int i, Int j) const  { 
#if ( _DEBUGlevel_ >= 1 )
  if( i < 0 || i >= m_ ||
      j < 0 || j >= n_ ) {//FIXME m_,n_ < 0时候怎么办？构造、赋值时都不抛错误，等着运行时抛？
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << m_ << ", " << n_ << ")" << std::endl
      << "This index     ~ (" << i  << ", " << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
#endif
  return data_[i+j*m_];
}

template <class F>
inline F& NumMat<F>::operator()(Int i, Int j)  { 
#if ( _DEBUGlevel_ >= 1 )
  if( i < 0 || i >= m_ ||
      j < 0 || j >= n_ ) {
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << m_ << ", " << n_ << ")" << std::endl
      << "This index     ~ (" << i  << ", " << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
#endif
  return data_[i+j*m_];
}

template <class F>
inline F* NumMat<F>::VecData(Int j)  const {                   //返回列j首元素的地址，不负责传出指针的尺寸
#if ( _DEBUGlevel_ >= 1 )
  if( j < 0 || j >= n_ ) {
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << n_ << ")" << std::endl
      << "This index     ~ (" << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
#endif
  return &(data_[j*m_]); 
}


// *********************************************************************
// Utilities
// *********************************************************************

template <class F> inline void SetValue(NumMat<F>& M, F val) { //FIXME 是否有非逐个遍历的更好写法？
  F *ptr = M.data_;
  Int max = M.m()*M.n();
  for (Int i=0; i < max; i++){
    *ptr = val;
    ptr++;
  }
}

template <class F> inline Real Energy(const NumMat<F>& M) {    //取模
  Real sum = 0;
  F *ptr = M.data_;
  Int max = M.m()*M.n();
  for (Int i=0; i < max; i++) 
    sum += std::abs(ptr[i]) * std::abs(ptr[i]);
  return sum;
}


template <class F> inline void
Transpose ( const NumMat<F>& A, NumMat<F>& B ) {               //转置，结果存到B里
  if( A.m() != B.n() || A.n() != B.m() ){
    B.Resize( A.n(), A.m() );//知道Resize为什么不管旧数据了吧？
  }

  F* Adata = A.Data();
  F* Bdata = B.Data();
  Int m = A.m(), n = A.n();

  for( Int i = 0; i < m; i++ )  //FIXME BLAS/LAPACK里有没有相关内容，或者这个就是最快的？
    for( Int j = 0; j < n; j++ )
      Bdata[ j + n*i ] = Adata[ i + j*m ];

  return ;
}        // -----  end of function Transpose  ----- 

template <class F> inline void
Symmetrize( NumMat<F>& A ){                                    //FIXME 将A与其转置相加后除以2，实现之抽象堪比NumVec的Sort......
  if( A.m() != A.n() ){
    ErrorHandling( "The matrix to be symmetrized should be a square matrix." );
  }

  NumMat<F> B;
  Transpose( A, B );

  F* Adata = A.Data();
  F* Bdata = B.Data();

  F  half = (F) 0.5;

  for( Int i = 0; i < A.m() * A.n(); i++ ){
    *Adata = half * (*Adata + *Bdata);
    Adata++; Bdata++;
  }


  return ;
}        // -----  end of function Symmetrize ----- 

} // namespace scales

#endif // _NUMMAT_IMPL_HPP_
