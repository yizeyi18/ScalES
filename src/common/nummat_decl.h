//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lexing Ying and Lin Lin 

/// @file nummat_decl.hpp
/// @brief Numerical matrix.
/// @date 2010-09-27
#ifndef _NUMMAT_DECL_HPP_
#define _NUMMAT_DECL_HPP_

#include "environment.h"

namespace  scales{

// Templated form of numerical matrix
//
// The main advantage of this portable NumVec structure is that it can
// either own (owndata == true) or view (owndata == false) a piece of
// data.

template <class F>
  class NumMat
  {
  public:
    Int m_, n_;
    bool owndata_;
    F* data_;
  public:
    NumMat(Int m=0, Int n=0);                                  //三种构造函数
    NumMat(Int m, Int n, bool owndata, F* data);
    NumMat(const NumMat& C);
    ~NumMat();                                                 //析构函数

    NumMat& operator=(const NumMat& C);                        //赋值

    void Resize(Int m, Int n);                                 //没闹懂预备

    const F& operator()(Int i, Int j) const;                   //可变/不可变地访问元素
    F& operator()(Int i, Int j);  
    F* VecData(Int j)  const;                                  //返回列j首元素的地址，不负责传出指针的尺寸

    bool IsOwnData() const { return owndata_; }                //访问类成员
    F* Data() const { return data_; } 
    Int m() const { return m_; }
    Int n() const { return n_; }
    Int Size() const { return m_ * n_; }

  };

// Commonly used
typedef NumMat<bool>     BolNumMat;
typedef NumMat<Int>      IntNumMat;
typedef NumMat<Real>     DblNumMat;
typedef NumMat<Complex>  CpxNumMat;

// Utilities
template <class F> inline void SetValue(NumMat<F>& M, F val);  //将矩阵全部元素值设为val
template <class F> inline Real Energy(const NumMat<F>& M);     //取模，全部元素取std::abs后平方相加
template <class F> inline void Transpose ( const NumMat<F>& A, NumMat<F>& B );//转置
template <class F> inline void Symmetrize( NumMat<F>& A );     //FIXME 将A与其转置相加后除以2，待重写

} // namespace scales

#endif // _NUMMAT_DECL_HPP_
