//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lexing Ying and Lin Lin 

/// @file numtns_decl.hpp
/// @brief Numerical tensor
/// @date 2010-09-27
#ifndef _NUMTNS_DECL_HPP_
#define _NUMTNS_DECL_HPP_

#include "environment.h"
#include "nummat_impl.hpp"

namespace  scales{

// Templated form of numerical 3-dimention tensor
//
// The main advantage of this portable NumVec structure is that it can
// either own (owndata == true) or view (owndata == false) a piece of
// data.

template <class F>
  class NumTns
  {
  public:
    Int m_, n_, p_;
    bool owndata_;
    F* data_;
  public:
    NumTns(Int m=0, Int n=0, Int p=0);                         //三种构造函数
    NumTns(Int m, Int n, Int p, bool owndata, F* data);
    NumTns(const NumTns& C);
    ~NumTns();                                                 //析构函数

    NumTns& operator=(const NumTns& C);                        //赋值

    void Resize(Int m, Int n, Int p);                          //可能只是为与NumMat一致？

    const F& operator()(Int i, Int j, Int k) const;            //可变/不可变地访问元素
    F& operator()(Int i, Int j, Int k);
    F* MatData (Int k) const;
    F* VecData (Int j, Int k) const;

    bool IsOwnData() const { return owndata_; }                //访问类成员，统统不可变
    F* Data() const { return data_; }
    Int m() const { return m_; }
    Int n() const { return n_; }
    Int p() const { return p_; }
    Int Size() const { return m_ * n_ * p_; }
  };

// Commonly used
typedef NumTns<bool>       BolNumTns;
typedef NumTns<Int>        IntNumTns;
typedef NumTns<Real>       DblNumTns;
typedef NumTns<Complex>    CpxNumTns;

// Utilities
template <class F> inline void SetValue(NumTns<F>& T, F val);  //将三阶张量全部元素值设为val
template <class F> inline Real Energy(const NumTns<F>& T);     //取模，全部元素取std::abs后平方相加


} // namespace scales

#endif // _NUMTNS_DECL_HPP_
