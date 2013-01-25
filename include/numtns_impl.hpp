/// @file numtns_impl.hpp
/// @brief Implementation of numerical tensor.
/// @author Lexing Ying and Lin Lin
/// @date 2010-09-27
#ifndef _NUMTNS_IMPL_HPP_
#define _NUMTNS_IMPL_HPP_

#include  "numtns_decl.hpp"

namespace  dgdft{

template <class F> 
inline NumTns<F>::NumTns(Int m, Int n, Int p): m_(m), n_(n), p_(p), owndata_(true) {
	if(m_>0 && n_>0 && p_>0) { data_ = new F[m_*n_*p_]; if( data_ == NULL ) throw std::runtime_error("Cannot allocate memory."); } else data_=NULL;
}

template <class F> 
inline NumTns<F>::NumTns(Int m, Int n, Int p, bool owndata, F* data): m_(m), n_(n), p_(p), owndata_(owndata) {
	if(owndata_) {
		if(m_>0 && n_>0 && p_>0) { data_ = new F[m_*n_*p_]; if( data_ == NULL ) throw std::runtime_error("Cannot allocate memory."); } else data_=NULL;
		if(m_>0 && n_>0 && p_>0) { for(Int i=0; i<m_*n_*p_; i++) data_[i] = data[i]; }
	} else {
		data_ = data;
	}
}

template <class F> 
inline NumTns<F>::NumTns(const NumTns& C): m_(C.m_), n_(C.n_), p_(C.p_), owndata_(C.owndata_) {
	if(owndata_) {
		if(m_>0 && n_>0 && p_>0) { data_ = new F[m_*n_*p_]; if( data_ == NULL ) throw std::runtime_error("Cannot allocate memory."); } else data_=NULL;
		if(m_>0 && n_>0 && p_>0) { for(Int i=0; i<m_*n_*p_; i++) data_[i] = C.data_[i]; }
	} else {
		data_ = C.data_;
	}
}

template <class F> 
inline NumTns<F>::~NumTns() { 
	if(owndata_) { 
		if(m_>0 && n_>0 && p_>0) { delete[] data_; data_ = NULL; } 
	}
}

template <class F> 
inline NumTns<F>& NumTns<F>::operator=(const NumTns& C) {
	if(owndata_) { 
		if(m_>0 && n_>0 && p_>0) { delete[] data_; data_ = NULL; } 
	}
	m_ = C.m_; n_=C.n_; p_=C.p_; owndata_=C.owndata_;
	if(owndata_) {
		if(m_>0 && n_>0 && p_>0) { data_ = new F[m_*n_*p_]; if( data_ == NULL ) throw std::runtime_error("Cannot allocate memory."); } else data_=NULL;
		if(m_>0 && n_>0 && p_>0) { for(Int i=0; i<m_*n_*p_; i++) data_[i] = C.data_[i]; }
	} else {
		data_ = C.data_;
	}
	return *this;
}

template <class F> 
inline void NumTns<F>::Resize(Int m, Int n, Int p)  {
	if( owndata_ == false ){
		throw std::logic_error("Tensor being resized must own data.");
	}
	if(m_!=m || n_!=n || p_!=p) {
		if(m_>0 && n_>0 && p_>0) { delete[] data_; data_ = NULL; } 
		m_ = m; n_ = n; p_=p;
		if(m_>0 && n_>0 && p_>0) { data_ = new F[m_*n_*p_]; if( data_ == NULL ) throw std::runtime_error("Cannot allocate memory."); } else data_=NULL;
	}
}

template <class F> 
inline const F& NumTns<F>::operator()(Int i, Int j, Int k) const  {
#ifndef _RELEASE_
	PushCallStack("NumTns<F>::operator()");
#endif 
	if( i < 0 || i >= m_ ||
			j < 0 || j >= n_ ||
			k < 0 || k >= p_ ) {
		std::ostringstream msg;
		msg 
			<< "Index is out of bound."  << std::endl
			<< "Index bound    ~ (" << m_ << ", " << n_ << ", " << p_ << ")" << std::endl
			<< "This index     ~ (" << i  << ", " << j  << ", " << k  << ")" << std::endl;
		throw std::logic_error( msg.str().c_str() );
	}
#ifndef _RELEASE_
	PopCallStack();
#endif  
	return data_[i+j*m_+k*m_*n_];
}

template <class F> 
inline F& NumTns<F>:: operator()(Int i, Int j, Int k)  {
#ifndef _RELEASE_
	PushCallStack("NumTns<F>::operator()");
#endif 
	if( i < 0 || i >= m_ ||
			j < 0 || j >= n_ ||
			k < 0 || k >= p_ ) {
		std::ostringstream msg;
		msg 
			<< "Index is out of bound."  << std::endl
			<< "Index bound    ~ (" << m_ << ", " << n_ << ", " << p_ << ")" << std::endl
			<< "This index     ~ (" << i  << ", " << j  << ", " << k  << ")" << std::endl;
		throw std::logic_error( msg.str().c_str() );
	}
#ifndef _RELEASE_
	PopCallStack();
#endif  
	return data_[i+j*m_+k*m_*n_];
}

template <class F> 
inline F* NumTns<F>::MatData (Int k) const {
#ifndef _RELEASE_
	PushCallStack("NumTns<F>::MatData");
#endif 
	if( k < 0 || k >= p_ ) {
		std::ostringstream msg;
		msg 
			<< "Index is out of bound."  << std::endl
			<< "Index bound    ~ (" << p_ << ")" << std::endl
			<< "This index     ~ (" << k  << ")" << std::endl;
		throw std::logic_error( msg.str().c_str() );
	}
#ifndef _RELEASE_
	PopCallStack();
#endif  
	return &(data_[k*m_*n_]);
}

template <class F> 
inline F* NumTns<F>::VecData (Int j, Int k) const {
#ifndef _RELEASE_
	PushCallStack("NumTns<F>::VecData");
#endif 
	if( j < 0 || j >= n_ ||
			k < 0 || k >= p_ ) {
		std::ostringstream msg;
		msg 
			<< "Index is out of bound."  << std::endl
			<< "Index bound    ~ (" << n_ << ", " << p_ << ")" << std::endl
			<< "This index     ~ (" << j  << ", " << k  << ")" << std::endl;
		throw std::logic_error( msg.str().c_str() );
	}
#ifndef _RELEASE_
	PopCallStack();
#endif  
	return &(data_[k*m_*n_+j*m_]);
}


// *********************************************************************
// Utilities
// *********************************************************************

template <class F> inline void SetValue(NumTns<F>& T, F val)
{
	F *ptr = T.data_;
  for(Int i=0; i < T.m() * T.n() * T.p(); i++) *(ptr++) = val; 

	return;
}

template <class F> inline Real Energy(const NumTns<F>& T)
{
  Real sum = 0;

	F *ptr = T.Data();
  for(Int i=0; i < T.m() * T.n() * T.p(); i++) 
		sum += std::abs(ptr[i]) * std::abs(ptr[i]);

	return sum;
}

} // namespace dgdft

#endif // _NUMTNS_IMPL_HPP_
