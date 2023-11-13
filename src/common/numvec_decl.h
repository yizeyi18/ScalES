//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lexing Ying and Lin Lin 

/// @file numvec_decl.hpp
/// @brief  Numerical vector.
/// @date 2010-09-27
#ifndef _NUMVEC_DECL_HPP_
#define _NUMVEC_DECL_HPP_

#include "environment.h"

namespace  scales{

// Templated form of numerical vectors
//
// The main advantage of this portable NumVec structure is that it can
// either own (owndata == true) or view (owndata == false) a piece of
// data.

template <class F> class NumVec
{
public:
  Int  m_;                                // The number of elements 
  bool owndata_;                          // Whether it owns the data
  F* data_;                               // The pointer for the actual data
public:
  NumVec(Int m = 0);                      //三种构造函数
  NumVec(Int m, bool owndata, F* data);
  NumVec(const NumVec& C);
  ~NumVec();                              //析构

  NumVec& operator=(const NumVec& C);     //赋值算符

  void Resize ( Int m );                  //改大小用，没闹懂。不保留旧的为什么不析构掉建个新的？

  const F& operator()(Int i) const;       //常量与非常量型的值访问算符
  F& operator()(Int i);  
  const F& operator[](Int i) const;
  F& operator[](Int i);

  bool IsOwnData() const { return owndata_; }//访问成员的方法

  F*   Data() const { return data_; }

  Int  m () const { return m_; }

  Int Size() const { return m_; }
};

// Commonly used
typedef NumVec<bool>       BolNumVec;
typedef NumVec<Int>        IntNumVec;
typedef NumVec<Real>       DblNumVec;
typedef NumVec<Complex>    CpxNumVec;


// Utilities
template <class F> inline void SetValue( NumVec<F>& vec, F val );  //给所有元素设值
template <class F> inline Real Energy( const NumVec<F>& vec );     //给实向量取模
template <class F> inline Real findMin( const NumVec<F>& vec );    //遍历搜极大，假设任意vec存在小于0元素
template <class F> inline Real findMax( const NumVec<F>& vec );    //遍历搜极大，假设任意vec存在大于0元素
template <class F> inline void Sort( NumVec<F>& vec );             //FIXME vec元素排序，多少带点......

} // namespace scales

#endif // _NUMVEC_DECL_HPP_
