/// @file nummat_decl.hpp
/// @brief Numerical matrix.
/// @author Lexing Ying and Lin Lin
/// @date 2010-09-27
#ifndef _NUMMAT_DECL_HPP_
#define _NUMMAT_DECL_HPP_

#include "environment.hpp"

namespace  dgdft{

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
  NumMat(Int m=0, Int n=0);

  NumMat(Int m, Int n, bool owndata, F* data);
	
  NumMat(const NumMat& C);

	~NumMat();

  NumMat& operator=(const NumMat& C);

	void Resize(Int m, Int n);

  const F& operator()(Int i, Int j) const;  

  F& operator()(Int i, Int j);  
	
	bool IsOwnData() const { return owndata_; }
  
  F* Data() const { return data_; }

  F* VecData(Int j)  const; 
	
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
template <class F> inline void SetValue(NumMat<F>& M, F val);
template <class F> inline Real Energy(const NumMat<F>& M);
 
} // namespace dgdft

#endif // _NUMMAT_DECL_HPP_
