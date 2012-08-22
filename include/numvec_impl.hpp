#ifndef _NUMVEC_IMPL_HPP_
#define _NUMVEC_IMPL_HPP_

#include  "numvec_decl.hpp"

namespace  dgdft{

// Templated form of numerical vectors
//
// The main advantage of this portable NumVec structure is that it can
// either own (owndata == true) or view (owndata == false) a piece of
// data.


template <class F> 
	NumVec<F>::NumVec	( Int m ) : m_(m), owndata_(true)
	{
#ifndef _RELEASE_
		PushCallStack("NumVec<F>::NumVec");
#endif  // ifndef _RELEASE_
		if(m_>0) { 
			data_ = new F[m_]; 
			if( data_ == NULL ){
				throw std::runtime_error("Cannot allocate memory.");
			}
		} 
		else 
			data_=NULL;
#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_
	} 		// -----  end of method NumVec<F>::NumVec  ----- 

template <class F> 
	NumVec<F>::NumVec	( Int m, bool owndata, F* data ) : m_(m), owndata_(owndata)
	{
#ifndef _RELEASE_
		PushCallStack("NumVec<F>::NumVec");
#endif  // ifndef _RELEASE_
		if( owndata_ ){
			if( m_ > 0 ) { 
				data_ = new F[m_]; 
				if( data_ == NULL ){
					throw std::runtime_error("Cannot allocate memory.");
				}
			}
			else
				data_ = NULL;

			if( m_ > 0 ) {
				for( Int i = 0; i < m_; i++ ){
					data_[i] = data[i];
				}
			}
		}
		else{
			data_ = data;
		}
#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_
	} 		// -----  end of method NumVec<F>::NumVec  ----- 

template <class F> 
	NumVec<F>::NumVec	( const NumVec<F>& C ) : m_(C.m_), owndata_(C.owndata_)
	{
#ifndef _RELEASE_
		PushCallStack("NumVec<F>::NumVec");
#endif  // ifndef _RELEASE_
		if( owndata_ ){
			if( m_ > 0 ) { 
				data_ = new F[m_]; 
				if( data_ == NULL ){
					throw std::runtime_error("Cannot allocate memory.");
				}
			}
			else
				data_ = NULL;

			if( m_ > 0 ) {
				for( Int i = 0; i < m_; i++ ){
					data_[i] = C.data_[i];
				}
			}
		}
		else{
			data_ = C.data_;
		}
#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_
	} 		// -----  end of method NumVec<F>::NumVec  ----- 


template < class F >
	NumVec<F>::~NumVec	(  )
	{
#ifndef _RELEASE_
		PushCallStack("NumVec<F>::~NumVec");
#endif  // ifndef _RELEASE_
		if( owndata_ ){
			if( m_ > 0 ){
				delete[] data_;  
				data_ = NULL;
			}
		}
#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_

	} 		// -----  end of method NumVec<F>::~NumVec  ----- 


template < class F >
	inline NumVec<F>& NumVec<F>::operator =	( const NumVec& C  )
	{
#ifndef _RELEASE_
		PushCallStack("NumVec<F>::operator=");
#endif  // ifndef _RELEASE_
		if( owndata_ ){
			if( m_ > 0 ){
				delete[]  data_;
				data_ = NULL;
			}
		}
		m_ = C.m_;
		owndata_ = C.owndata_;

		if( owndata_ ) {
			if( m_ > 0 ){
				data_ = new F[m_];
				if( data_ == NULL ){
					throw std::runtime_error("Cannot allocate memory.");
				}
			}
			else{
				data_ = NULL;
			}

			if( m_ > 0 ){
				for( Int i = 0; i < m_; i++ ){
					data_[i] = C.data_[i];
				}
			}
		}
		else{
			data_ = C.data_;
		}

#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_

		return *this;
	} 		// -----  end of method NumVec<F>::operator=  ----- 


template < class F >
	inline void NumVec<F>::Resize	( const Int m )
	{
#ifndef _RELEASE_
		PushCallStack("NumVec<F>::Resize");
#endif  // ifndef _RELEASE_
		if( owndata_ == false ){
			throw std::logic_error("Vector being resized must own data.");
		}
		if( m != m_ ){
			if( m_ > 0 ){
				delete[] data_;
				data_ = NULL;
			}
			m_ = m;
			if( m_ > 0 ){
				data_ = new F[m_];
				if( data_ == NULL ){
					throw std::runtime_error("Cannot allocate memory.");
				}
			}
		}
		else{
			std::cerr << "NumVec<F>::Resize is doing nothin" << std::endl;
		}

#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_
		return ;
	} 		// -----  end of method NumVec<F>::Resize  ----- 


template <class F>
	inline F&
	NumVec<F>::operator()	( Int i )
	{
#ifndef _RELEASE_
		PushCallStack("NumVec<F>::operator()");
#endif  // ifndef _RELEASE_
		if( i < 0 || i >= m_ ){
			throw std::logic_error( "Index is out of bound." );
		}
#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_
		return data_[i];

	} 		// -----  end of method NumVec<F>::operator()  ----- 


template <class F>
	inline const F&
	NumVec<F>::operator()	( Int i ) const
	{
#ifndef _RELEASE_
		PushCallStack("NumVec<F>::operator()");
#endif  // ifndef _RELEASE_
		if( i < 0 || i >= m_ ){
			throw std::logic_error( "Index is out of bound." );
		}
#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_
		return data_[i];

	} 		// -----  end of method NumVec<F>::operator()  ----- 


template <class F>
	inline F& NumVec<F>::operator[]	( Int i )
	{
#ifndef _RELEASE_
		PushCallStack("NumVec<F>::operator[]");
#endif  // ifndef _RELEASE_
		if( i < 0 || i >= m_ ){
			throw std::logic_error( "Index is out of bound." );
		}
#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_
		return data_[i];

	} 		// -----  end of method NumVec<F>::operator[]  ----- 


template <class F>
	inline const F&
	NumVec<F>::operator[]	( Int i ) const
	{
#ifndef _RELEASE_
		PushCallStack("NumVec<F>::operator[]");
#endif  // ifndef _RELEASE_
		if( i < 0 || i >= m_ ){
			throw std::logic_error( "Index is out of bound." );
		}
#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_
		return data_[i];

	} 		// -----  end of method NumVec<F>::operator[]  ----- 


// *********************************************************************
// Input and output
// *********************************************************************

template <class F> inline std::ostream& operator<<( std::ostream& os, const NumVec<F>& vec)
{
	os<<vec.m()<<std::endl;
	os.setf(std::ios_base::scientific, std::ios_base::floatfield);
	for(Int i=0; i<vec.m(); i++)	 
		os<<" "<<vec(i);
	os<<std::endl;
	return os;
}

template <class F> inline std::istream& operator>>( std::istream& is, NumVec<F>& vec)
{
	Int m;  is>>m;  vec.resize(m);
	for(Int i=0; i<vec.m(); i++)	 
		is >> vec(i);
	return is;
}


template <class F> inline void SetValue( NumVec<F>& vec, F val )
{
	for(Int i=0; i<vec.m(); i++)
		vec(i) = val;
}

template <class F> inline Real Energy( const NumVec<F>& vec )
{
	Real sum = 0;
	for(Int i=0; i<vec.m(); i++)
		sum += abs(vec(i)*vec(i));
	return sum;
}  

template <class F> inline void Sort( NumVec<F>& vec ){
	std::vector<F>  tvec(vec.m());
	std::copy( vec.Data(), vec.Data() + vec.m(), tvec.begin() );
	std::sort( tvec.begin(), tvec.end() );
	for(Int i = 0; i < vec.m(); i++){
		vec(i) = tvec[i];
	}
	return;
}


} // namespace dgdft

#endif


