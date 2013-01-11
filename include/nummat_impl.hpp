/// @file nummat_impl.hpp
/// @brief Implementation of numerical matrix.
/// @author Lexing Ying and Lin Lin
/// @date 2010-09-27
#ifndef _NUMMAT_IMPL_HPP_
#define _NUMMAT_IMPL_HPP_

#include  "nummat_decl.hpp"

namespace  dgdft{

template <class F> NumMat<F>::NumMat(Int m, Int n): m_(m), n_(n), owndata_(true) {
	if(m_>0 && n_>0) { data_ = new F[m_*n_]; if( data_ == NULL ) throw std::runtime_error("Cannot allocate memory."); } else data_=NULL;
}

template <class F> NumMat<F>::NumMat(Int m, Int n, bool owndata, F* data): m_(m), n_(n), owndata_(owndata) {
	if(owndata_) {
		if(m_>0 && n_>0) { data_ = new F[m_*n_]; if( data_ == NULL ) throw std::runtime_error("Cannot allocate memory."); } else data_=NULL;
		if(m_>0 && n_>0) { for(Int i=0; i<m_*n_; i++) data_[i] = data[i]; }
	} else {
		data_ = data;
	}
}

template <class F> NumMat<F>::NumMat(const NumMat& C): m_(C.m_), n_(C.n_), owndata_(C.owndata_) {
	if(owndata_) {
		if(m_>0 && n_>0) { data_ = new F[m_*n_]; if( data_ == NULL ) throw std::runtime_error("Cannot allocate memory."); } else data_=NULL;
		if(m_>0 && n_>0) { for(Int i=0; i<m_*n_; i++) data_[i] = C.data_[i]; }
	} else {
		data_ = C.data_;
	}
}

template <class F> NumMat<F>::~NumMat() {
	if(owndata_) {
		if(m_>0 && n_>0) { delete[] data_; data_ = NULL; }
	}
}

template <class F> inline NumMat<F>& NumMat<F>::operator=(const NumMat& C) {
	if(owndata_) {
		if(m_>0 && n_>0) { delete[] data_; data_ = NULL; }
	}
	m_ = C.m_; n_=C.n_; owndata_=C.owndata_;
	if(owndata_) {
		if(m_>0 && n_>0) { data_ = new F[m_*n_]; if( data_ == NULL ) throw std::runtime_error("Cannot allocate memory."); } else data_=NULL;
		if(m_>0 && n_>0) { for(Int i=0; i<m_*n_; i++) data_[i] = C.data_[i]; }
	} else {
		data_ = C.data_;
	}
	return *this;
}

template <class F> inline void NumMat<F>::Resize(Int m, Int n)  {
	if( owndata_ == false ){
		throw std::logic_error("Matrix being resized must own data.");
	}
	if(m_!=m || n_!=n) {
		if(m_>0 && n_>0) { delete[] data_; data_ = NULL; }
		m_ = m; n_ = n;
		if(m_>0 && n_>0) { data_ = new F[m_*n_]; if( data_ == NULL ) throw std::runtime_error("Cannot allocate memory."); } else data_=NULL;
	}
}

template <class F> 
inline const F& NumMat<F>::operator()(Int i, Int j) const  { 
#ifndef _RELEASE_
	PushCallStack("NumMat<F>::operator()");
#endif  
	if( i < 0 || i >= m_ ||
			j < 0 || j >= n_ ) {
		std::ostringstream msg;
		msg 
			<< "Index is out of bound."  << std::endl
			<< "Index bound    ~ (" << m_ << ", " << n_ << ")" << std::endl
			<< "This index     ~ (" << i  << ", " << j  << ")" << std::endl;
		throw std::logic_error( msg.str().c_str() ); 
	}
#ifndef _RELEASE_
	PopCallStack();
#endif  
	return data_[i+j*m_];
}

template <class F>
inline F& NumMat<F>::operator()(Int i, Int j)  { 
#ifndef _RELEASE_
	PushCallStack("NumMat<F>::operator()");
#endif  
	if( i < 0 || i >= m_ ||
			j < 0 || j >= n_ ) {
		std::ostringstream msg;
		msg 
			<< "Index is out of bound."  << std::endl
			<< "Index bound    ~ (" << m_ << ", " << n_ << ")" << std::endl
			<< "This index     ~ (" << i  << ", " << j  << ")" << std::endl;
		throw std::logic_error( msg.str().c_str() ); 
	}
#ifndef _RELEASE_
	PopCallStack();
#endif  
	return data_[i+j*m_];
}

template <class F>
inline F* NumMat<F>::VecData(Int j)  const 
{ 
#ifndef _RELEASE_
	PushCallStack("NumMat<F>::VecData");
#endif  
	if( j < 0 || j >= n_ ) {
		std::ostringstream msg;
		msg 
			<< "Index is out of bound."  << std::endl
			<< "Index bound    ~ (" << n_ << ")" << std::endl
			<< "This index     ~ (" << j  << ")" << std::endl;
		throw std::logic_error( msg.str().c_str() ); 
	}
#ifndef _RELEASE_
	PopCallStack();
#endif  
	return &(data_[j*m_]); 
}


// *********************************************************************
// Utilities
// *********************************************************************

template <class F> inline void SetValue(NumMat<F>& M, F val)
{
	F *ptr = M.data_;
	for (Int i=0; i < M.m()*M.n(); i++) *(ptr++) = val;
}

template <class F> inline Real Energy(const NumMat<F>& M)
{
	Real sum = 0;
	F *ptr = M.data_;
	for (Int i=0; i < M.m()*M.n(); i++) 
		sum += abs(ptr[i]) * abs(ptr[i]);
	return sum;
}

} // namespace dgdft

#endif // _NUMMAT_IMPL_HPP_
