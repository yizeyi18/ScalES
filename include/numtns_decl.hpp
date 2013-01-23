/// @file numtns_decl.hpp
/// @brief Numerical tensor
/// @author Lexing Ying and Lin Lin
/// @date 2010-09-27
#ifndef _NUMTNS_DECL_HPP_
#define _NUMTNS_DECL_HPP_

#include "environment.hpp"
#include "nummat_impl.hpp"

namespace  dgdft{

// Templated form of numerical tensor
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
  NumTns(Int m=0, Int n=0, Int p=0);

  NumTns(Int m, Int n, Int p, bool owndata, F* data);

  NumTns(const NumTns& C);

  ~NumTns();
 	
  NumTns& operator=(const NumTns& C);

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
typedef NumTns<bool>       BolNumTns;
typedef NumTns<Int>        IntNumTns;
typedef NumTns<Real>       DblNumTns;
typedef NumTns<Complex>    CpxNumTns;

// Utilities
template <class F> inline void SetValue(NumTns<F>& T, F val);

template <class F> inline Real Energy(const NumTns<F>& T);



} // namespace dgdft

#endif // _NUMTNS_DECL_HPP_
