/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Weile Jia

This file is part of DGDFT. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
(2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to Lawrence Berkeley National
Laboratory, without imposing a separate written license agreement for such
Enhancements, then you hereby grant the following license: a non-exclusive,
royalty-free perpetual license to install, use, modify, prepare derivative
works, incorporate into other computer software, distribute, and sublicense
such enhancements or derivative works thereof, in binary and source code form.
 */
/// @file device_numvec_impl.hpp
/// @brief Implementation of Numerical Vector on device.
/// @date 2020-08-12
#ifdef DEVICE
#ifndef _DEVICE_NUMVEC_IMPL_HPP_
#define _DEVICE_NUMVEC_IMPL_HPP_

#include  "device_numvec_decl.hpp"

namespace  dgdft{

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

} // namespace dgdft

#endif // _DEVICE_NUMVEC_IMPL_HPP_
#endif // DEVICE
