//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

/// @file device_numvec_impl.hpp
/// @brief Implementation of Numerical Vector on device.
/// @date 2020-08-12
#ifdef DEVICE
#ifndef _DEVICE_NUMVEC_IMPL_HPP_
#define _DEVICE_NUMVEC_IMPL_HPP_

#include  "device_numvec_decl.h"

namespace  scales{

// Templated form of numerical vectors
//
// The main advantage of this portable deviceNumVec structure is that it can
// either own (owndata == true) or view (owndata == false) a piece of
// data.


template <class F> 
  inline deviceNumVec<F>::deviceNumVec( Int m ) 
  {
    owndata_ = true;
    if(m>0) { 
      m_ = m;
      data_ = (F*) device_malloc( sizeof(F) * m_ );
      /*
      if( data_ == NULL ){
        ErrorHandling("Cannot allocate memory.");
      }
      */
    } 
    else 
      data_=NULL;
  }         // -----  end of method deviceNumVec<F>::deviceNumVec  ----- 
template <class F> 
  inline deviceNumVec<F>::deviceNumVec    ( Int m, bool owndata, F* data ) : m_(m), owndata_(owndata)
  {
    if( owndata_ ){
      if( m_ > 0 ) { 
        data_ = (F*) device_malloc( sizeof(F) * m_ );
        /*
        if( data_ == NULL ){
          ErrorHandling("Cannot allocate memory.");
        }
        */
      }
      else
        data_ = NULL;

      if( m_ > 0 ) {
        device_memcpy_DEVICE2DEVICE(data_, data, sizeof(F)*m_);
      }
    }
    else{
      data_ = data;
    }
  }         // -----  end of method deviceNumVec<F>::deviceNumVec  ----- 

template <class F> 
  inline deviceNumVec<F>::deviceNumVec    ( const deviceNumVec<F>& C ) : m_(C.m_), owndata_(C.owndata_)
  {
    if( owndata_ ){
      if( m_ > 0 ) { 
        data_ = (F*) device_malloc( sizeof(F) * m_ );
        /*
        if( data_ == NULL ){
          ErrorHandling("Cannot allocate memory.");
        }
        */
      }
      else
        data_ = NULL;

      if( m_ > 0 ) {
        device_memcpy_DEVICE2DEVICE(data_, C.data_, sizeof(F)*m_);
      }
    }
    else{
      data_ = C.data_;
    }
  }         // -----  end of method deviceNumVec<F>::deviceNumVec  ----- 

template < class F > 
  inline deviceNumVec<F>::~deviceNumVec    (  )
  {
    if( owndata_ ){
      if( m_ > 0 ){
        device_free(data_);
        data_ = NULL;
      }
    }

  }         // -----  end of method deviceNumVec<F>::~deviceNumVec  ----- 


template < class F > 
  inline deviceNumVec<F>& deviceNumVec<F>::operator =    ( const deviceNumVec& C  )
  {
    // Do not copy if it is the same matrix.
    if(C.data_ != data_){
      if( owndata_ ){
        if( m_ > 0 ){
          device_free(data_);
          data_ = NULL;
        }
      }
      m_ = C.m_;
      owndata_ = C.owndata_;

      if( owndata_ ) {
        if( m_ > 0 ){
          data_ = (F*) device_malloc( sizeof(F) * m_ );
          /*
          if( data_ == NULL ){
            ErrorHandling("Cannot allocate memory.");
          }
          */
        }
        else{
          data_ = NULL;
        }

        if( m_ > 0 ){
          device_memcpy_DEVICE2DEVICE(data_, C.data_, sizeof(F)*m_);
        }
      }
      else{
        data_ = C.data_;
      }
    }


    return *this;
  }         // -----  end of method deviceNumVec<F>::operator=  ----- 


template < class F > 
  inline void deviceNumVec<F>::Resize    ( const Int m )
  {
    if( owndata_ == false ){
      ErrorHandling("Vector being resized must own data.");
    }
    if( m != m_ ){
      if( m_ > 0 ){
        device_free(data_);
        data_ = NULL;
      }
      m_ = m;
      if( m_ > 0 ){
        data_ = (F*) device_malloc( sizeof(F) * m_ );
        /*
        if( data_ == NULL ){
          ErrorHandling("Cannot allocate memory.");
        }
        */
      }
    }

    return ;
  }         // -----  end of method deviceNumVec<F>::Resize  ----- 


template <class F> 
  inline F& deviceNumVec<F>::operator()    ( Int i )
  {
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
    return data_[i];

  }         // -----  end of method deviceNumVec<F>::operator()  ----- 


template <class F>
  inline const F& deviceNumVec<F>::operator()    ( Int i ) const
  {
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
    return data_[i];

  }         // -----  end of method deviceNumVec<F>::operator()  ----- 


template <class F> 
  inline F& deviceNumVec<F>::operator[]    ( Int i )
  {
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
    return data_[i];

  }         // -----  end of method deviceNumVec<F>::operator[]  ----- 


template <class F> 
  inline const F& deviceNumVec<F>::operator[]    ( Int i ) const
  {
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
    return data_[i];

  }         // -----  end of method deviceNumVec<F>::operator[]  ----- 

template <class F> inline void deviceNumVec<F>::CopyTo(NumVec<F> &C) {
  // copy from the DEVICE NumVec to the HOST NumVec.
  if( C.m_ < m_ ) 
  { 
    C.Resize(m_);
  }
  if(C.m_ >= m_) {
    if(m_>0 ) { device_memcpy_DEVICE2HOST(C.data_, data_, sizeof(F)*m_);}
  }
}
template <class F> inline void deviceNumVec<F>::CopyFrom(const NumVec<F> &C) {
  // copy from the DEVICE NumVec to the HOST NumVec.
  if( C.m_ > m_ ) 
  { 
    std:: cout << " DEVICE memory not big enough. " << m_<<" "<< C.m_ << std:: endl;
    device_free(data_);
    m_ = C.m_; 
    if(m_>0 ) { data_ = (F*)device_malloc( sizeof(F) * m_ ); } else data_=NULL;
   }
  if(C.m_ <= m_) {
    if(m_>0 ) { device_memcpy_HOST2DEVICE(data_, C.data_, sizeof(F)*C.m_);}
  }
}


// *********************************************************************
// Utilities
// *********************************************************************
template <class F> inline void SetValue( deviceNumVec<F>& vec, F val )
{
  // note, device setValue only works for float and double.
  device_setValue(vec.data_, val, vec.m_);
}

/*
template <class F> inline Real Energy( const deviceNumVec<F>& vec )
{
  Real sum = 0;
  for(Int i=0; i<vec.m(); i++)
    sum += std::abs(vec(i)*vec(i));
  return sum;
}  

template <class F> inline void Sort( deviceNumVec<F>& vec ){
  std::vector<F>  tvec(vec.m());
  std::copy( vec.Data(), vec.Data() + vec.m(), tvec.begin() );
  std::sort( tvec.begin(), tvec.end() );
  for(Int i = 0; i < vec.m(); i++){
    vec(i) = tvec[i];
  }
  return;
}
*/

} // namespace scales

#endif // _DEVICE_NUMVEC_IMPL_HPP_
#endif // DEVICE
