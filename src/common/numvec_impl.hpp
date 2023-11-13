//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lexing Ying and Lin Lin 

/// @file numvec_impl.hpp
/// @brief Implementation of Numerical Vector.
/// @date 2010-09-27
#ifndef _NUMVEC_IMPL_HPP_
#define _NUMVEC_IMPL_HPP_

#include  "numvec_decl.h"

namespace  scales{

// Templated form of numerical vectors
//
// The main advantage of this portable NumVec structure is that it can
// either own (owndata == true) or view (owndata == false) a piece of
// data.

//三种构造函数的实现
template <class F> 
  inline NumVec<F>::NumVec    ( Int m ) : m_(m), owndata_(true)
  {
    if(m_>0) { 
      data_ = new F[m_]; 
      if( data_ == nullptr ){//FIXME new在申请失败时会抛异常。需要再判断一遍吗？
        ErrorHandling("Cannot allocate memory.");
      }
    } 
    else 
      data_=nullptr;         //FIXME m<0，但不抛异常？
  }         // -----  end of method NumVec<F>::NumVec  ----- 

template <class F> 
  inline NumVec<F>::NumVec    ( Int m, bool owndata, F* data ) : m_(m), owndata_(owndata)
  {
    if( owndata_ ){
      if( m_ > 0 ) { 
        data_ = new F[m_]; 
        if( data_ == nullptr ){
          ErrorHandling("Cannot allocate memory.");
        }
      }
      else
        data_ = nullptr;

      if( m_ > 0 ) {         //FIXME 从data向this->data_复制数据。是否有循环外的写法？
        for( Int i = 0; i < m_; i++ ){
          data_[i] = data[i];
        }
      }
    }
    else{
      data_ = data;
    }
  }         // -----  end of method NumVec<F>::NumVec  ----- 

template <class F> 
  inline NumVec<F>::NumVec    ( const NumVec<F>& C ) : m_(C.m_), owndata_(C.owndata_)
  {
    if( owndata_ ){
      if( m_ > 0 ) { 
        data_ = new F[m_]; 
        if( data_ == nullptr ){
          ErrorHandling("Cannot allocate memory.");
        }
      }
      else
        data_ = nullptr;

      if( m_ > 0 ) {//同第51行
        for( Int i = 0; i < m_; i++ ){
          data_[i] = C.data_[i];
        }
      }
    }//if owndata_
    else{
      data_ = C.data_;
    }
  }         // -----  end of method NumVec<F>::NumVec  ----- 


template < class F > 
  inline NumVec<F>::~NumVec    (  )
  {
    if( owndata_ && m_ > 0 ){
        delete[] data_;  
        data_ = nullptr;
    }
  }         // -----  end of method NumVec<F>::~NumVec  ----- 


template < class F > 
  inline NumVec<F>& NumVec<F>::operator =    ( const NumVec& C  )
  {
    // Do not copy if it is the same matrix.
    if(C.data_ != data_){//除旧
      if( owndata_ && m_ > 0 ){
        delete[]  data_;
        data_ = nullptr;
      }//if owndata_
      m_ = C.m_;
      owndata_ = C.owndata_;

      if( owndata_ ) {//迎新
        if( m_ > 0 ){
          data_ = new F[m_];
          if( data_ == nullptr ){
            ErrorHandling("Cannot allocate memory.");
          }
        }//if m_ > 0
        else
          data_ = nullptr;

        if( m_ > 0 ){//同51行
          for( Int i = 0; i < m_; i++ ){
            data_[i] = C.data_[i];
          }
        }
      }//if owndata_
      else{
        data_ = C.data_;
      }
    }//if C.data_!=data_

    return *this;
  }         // -----  end of method NumVec<F>::operator=  ----- 


template < class F > 
  inline void NumVec<F>::Resize    ( const Int m )//成功的Resize会把存的数据直接删光
  {
    if( owndata_ == false ){
      ErrorHandling("Vector being resized must own data.");
    }
    if( m != m_ ){
      if( m_ > 0 ){
        delete[] data_;
        data_ = nullptr;
      }
      m_ = m;      
      if( m_ > 0 ){
        data_ = new F[m_];
        if( data_ == nullptr ){
          ErrorHandling("Cannot allocate memory.");
        }
      }
    }//if m != m_

    return ;
  }         // -----  end of method NumVec<F>::Resize  ----- 


template <class F> 
  inline F& NumVec<F>::operator()    ( Int i )
  {
#if ( _DEBUGlevel_ >= 1 )
    if( i < 0 || i >= m_ ){//FIXME m_ < 0时候怎么办？构造、赋值时都不抛错误，等着运行时抛？
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
#endif
    return data_[i];
  }         // -----  end of method NumVec<F>::operator()  ----- 


template <class F>
  inline const F& NumVec<F>::operator()    ( Int i ) const
  {
#if ( _DEBUGlevel_ >= 1 )
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
#endif
    return data_[i];
  }         // -----  end of method NumVec<F>::operator()  ----- 


template <class F> 
  inline F& NumVec<F>::operator[]    ( Int i )
  {
#if ( _DEBUGlevel_ >= 1 )
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
#endif
    return data_[i];

  }         // -----  end of method NumVec<F>::operator[]  ----- 


template <class F> 
  inline const F& NumVec<F>::operator[]    ( Int i ) const
  {
#if ( _DEBUGlevel_ >= 1 )
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
#endif
    return data_[i];

  }         // -----  end of method NumVec<F>::operator[]  ----- 

// *********************************************************************
// Utilities
// *********************************************************************

template <class F> inline void SetValue( NumVec<F>& vec, F val )//赋值
{
  for(Int i=0; i<vec.m(); i++)
    vec(i) = val;
}

template <class F> inline Real Energy( const NumVec<F>& vec )   //取模
{
  Real sum = 0;
  for(Int i=0; i<vec.m(); i++)
    sum += std::abs(vec(i)*vec(i));
  return sum;
}

template <class F> inline Real findMin( const NumVec<F>& vec )  //遍历搜极小，假设任意vec存在小于0元素？
{
  Real min = 0.0;
  for(Int i=0; i<vec.m(); i++)
    if(vec(i) < min)
	    min = vec(i);
  return min;
}  

template <class F> inline Real findMax( const NumVec<F>& vec )  //遍历搜极大，假设任意vec存在大于0元素？
{
  Real max = 0.0;
  for(Int i=0; i<vec.m(); i++)
    if(vec(i) > max)
	    max = vec(i);
  return max;
}  



template <class F> inline void Sort( NumVec<F>& vec ){          //FIXME vec元素排序，多少带点......
  std::vector<F>  tvec(vec.m());
  std::copy( vec.Data(), vec.Data() + vec.m(), tvec.begin() );
  std::sort( tvec.begin(), tvec.end() );
  for(Int i = 0; i < vec.m(); i++){
    vec(i) = tvec[i];
  }
  return;
}


} // namespace scales

#endif // _NUMVEC_IMPL_HPP_
