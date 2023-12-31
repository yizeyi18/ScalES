//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lexing Ying and Lin Lin

/// @file tinyvec_decl.hpp
/// @brief Tiny vectors of dimension 2 and 3.
/// @date 2010-09-20
#ifndef  _TINYVEC_DECL_HPP_
#define  _TINYVEC_DECL_HPP_

#include "environment.hpp"

namespace scales{

// *********************************************************************
// Tiny vectors of dimension 3.
// *********************************************************************

template <class F> class Vec3T {
private:
  F v_[3];
public:
  enum{ X=0, Y=1, Z=2 };
  //------------CONSTRUCTOR AND DESTRUCTOR 
  Vec3T()              { v_[0]=F(0);    v_[1]=F(0);    v_[2]=F(0); }
  Vec3T(const F* f)    { v_[0]=f[0];    v_[1]=f[1];    v_[2]=f[2]; }
  Vec3T(const F a, const F b, const F c)   { v_[0]=a;       v_[1]=b;       v_[2]=c; }
  Vec3T(const Vec3T& c){ v_[0]=c.v_[0]; v_[1]=c.v_[1]; v_[2]=c.v_[2]; }
  ~Vec3T() {}
  //------------POINTER and ACCESS
  operator F*()             { return &v_[0]; }
  operator const F*() const { return &v_[0]; }
  F* Data()                 { return &v_[0]; }  //access array
  F& operator()(Int i);
  const F& operator()(Int i) const;
  F& operator[](Int i);
  const F& operator[](Int i) const;
  //------------ASSIGN
  Vec3T& operator= ( const Vec3T& c ) { v_[0] =c.v_[0]; v_[1] =c.v_[1]; v_[2] =c.v_[2]; return *this; }
  Vec3T& operator+=( const Vec3T& c ) { v_[0]+=c.v_[0]; v_[1]+=c.v_[1]; v_[2]+=c.v_[2]; return *this; }
  Vec3T& operator-=( const Vec3T& c ) { v_[0]-=c.v_[0]; v_[1]-=c.v_[1]; v_[2]-=c.v_[2]; return *this; }
  Vec3T& operator*=( const F& s )     { v_[0]*=s;       v_[1]*=s;       v_[2]*=s;       return *this; }
  Vec3T& operator/=( const F& s )     { v_[0]/=s;       v_[1]/=s;       v_[2]/=s;       return *this; }
  //-----------LENGTH
  F l1( void )     const  { F sum=F(0); for(Int i=0; i<3; i++) sum=sum+std::abs(v_[i]); return sum; }
  F linfty( void ) const  { F cur=F(0); for(Int i=0; i<3; i++) cur=std::max(cur,std::abs(v_[i])); return cur; }
  F l2( void )     const  { F sum=F(0); for(Int i=0; i<3; i++) sum=sum+std::abs(v_[i]*v_[i]); return std::sqrt(sum); }
  F prod( void )   const  { return v_[0]*v_[1]*v_[2]; }
};

// Commonly used Vec3T types

typedef Vec3T<Real>   Point3;
typedef Vec3T<Int>    Index3;

// *********************************************************************
// Tiny vectors of dimension 3.
// *********************************************************************

template <class F>
class Vec2T {
private:
  F v_[2];
public:
  enum{ X=0, Y=1 };
  //------------CONSTRUCTOR AND DESTRUCTOR 
  Vec2T()              { v_[0]=F(0);    v_[1]=F(0); }  //Vec2T(F f)           { v_[0]=f;       v_[1]=f; }
Vec2T(const F* f)    { v_[0]=f[0];    v_[1]=f[1]; }
Vec2T(F a,F b)       { v_[0]=a;       v_[1]=b; }
Vec2T(const Vec2T& c){ v_[0]=c.v_[0]; v_[1]=c.v_[1]; }
~Vec2T() {}
//------------POINTER and ACCESS
operator F*()             { return &v_[0]; }
operator const F*() const { return &v_[0]; }
F* data()                { return &v_[0]; }  //access array
F& operator()(int i)             { assert(i<2); return v_[i]; }
const F& operator()(int i) const { assert(i<2); return v_[i]; }
F& operator[](int i)             { assert(i<2); return v_[i]; }
const F& operator[](int i) const { assert(i<2); return v_[i]; }
//------------ASSIGN
Vec2T& operator= ( const Vec2T& c ) { v_[0] =c.v_[0]; v_[1] =c.v_[1]; return *this; }
Vec2T& operator+=( const Vec2T& c ) { v_[0]+=c.v_[0]; v_[1]+=c.v_[1]; return *this; }
Vec2T& operator-=( const Vec2T& c ) { v_[0]-=c.v_[0]; v_[1]-=c.v_[1]; return *this; }
Vec2T& operator*=( const F& s )     { v_[0]*=s;       v_[1]*=s;       return *this; }
Vec2T& operator/=( const F& s )     { v_[0]/=s;       v_[1]/=s;       return *this; }
//-----------LENGTH...
F l1( void )     const  { F sum=F(0); for(int i=0; i<2; i++) sum=sum+std::abs(v_[i]); return sum; }
F linfty( void ) const  { F cur=F(0); for(int i=0; i<2; i++) cur=std::max(cur,std::abs(v_[i])); return cur; }
F l2( void )     const  { F sum=F(0); for(int i=0; i<2; i++) sum=sum+std::abs(v_[i]*v_[i]); return std::sqrt(sum); }
F prod( void )   const  { return v_[0]*v_[1]; }
};

// Commonly used Vec2T types
typedef Vec2T<double> Point2;
typedef Vec2T<int>    Index2;

} // namespace scales

#endif // _TINYVEC_DECL_HPP_
