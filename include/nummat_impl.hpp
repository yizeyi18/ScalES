/*
	 Copyright (c) 2012 The Regents of the University of California,
	 through Lawrence Berkeley National Laboratory.  

   Authors: Lexing Ying and Lin Lin
	 
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
/// @file nummat_impl.hpp
/// @brief Implementation of numerical matrix.
/// @date 2010-09-27
#ifndef _NUMMAT_IMPL_HPP_
#define _NUMMAT_IMPL_HPP_

#include "nummat_decl.hpp"

namespace  dgdft{

template <class F> inline NumMat<F>::NumMat(Int m, Int n): m_(m), n_(n), owndata_(true) {
	if(m_>0 && n_>0) { data_ = new F[m_*n_]; if( data_ == NULL ) throw std::runtime_error("Cannot allocate memory."); } else data_=NULL;
}

template <class F> inline NumMat<F>::NumMat(Int m, Int n, bool owndata, F* data): m_(m), n_(n), owndata_(owndata) {
	if(owndata_) {
		if(m_>0 && n_>0) { data_ = new F[m_*n_]; if( data_ == NULL ) throw std::runtime_error("Cannot allocate memory."); } else data_=NULL;
		if(m_>0 && n_>0) { for(Int i=0; i<m_*n_; i++) data_[i] = data[i]; }
	} else {
		data_ = data;
	}
}

template <class F> inline NumMat<F>::NumMat(const NumMat& C): m_(C.m_), n_(C.n_), owndata_(C.owndata_) {
	if(owndata_) {
		if(m_>0 && n_>0) { data_ = new F[m_*n_]; if( data_ == NULL ) throw std::runtime_error("Cannot allocate memory."); } else data_=NULL;
		if(m_>0 && n_>0) { for(Int i=0; i<m_*n_; i++) data_[i] = C.data_[i]; }
	} else {
		data_ = C.data_;
	}
}

template <class F> inline NumMat<F>::~NumMat() {
	if(owndata_) {
		if(m_>0 && n_>0) { delete[] data_; data_ = NULL; }
	}
}

template <class F> inline NumMat<F>& NumMat<F>::operator=(const NumMat& C) {
	// Do not copy if it is the same matrix.
	if(C.data_ == data_) return *this;
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
		sum += std::abs(ptr[i]) * std::abs(ptr[i]);
	return sum;
}


template <class F> inline void
Transpose ( const NumMat<F>& A, NumMat<F>& B )
{
#ifndef _RELEASE_
	PushCallStack("Transpose");
#endif
	if( A.m() != B.n() || A.n() != B.m() ){
		B.Resize( A.n(), A.m() );
	}

	F* Adata = A.Data();
	F* Bdata = B.Data();
	Int m = A.m(), n = A.n();

	for( Int i = 0; i < m; i++ ){
		for( Int j = 0; j < n; j++ ){
			Bdata[ j + n*i ] = Adata[ i + j*m ];
		}
	}

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
}		// -----  end of function Transpose  ----- 

template <class F> inline void
Symmetrize( NumMat<F>& A )
{
#ifndef _RELEASE_
	PushCallStack("Symmetrize");
#endif
	if( A.m() != A.n() ){
		throw std::logic_error( "The matrix to be symmetrized should be a square matrix." );
	}

	NumMat<F> B;
	Transpose( A, B );

	F* Adata = A.Data();
	F* Bdata = B.Data();

	F  half = (F) 0.5;

	for( Int i = 0; i < A.m() * A.n(); i++ ){
		*Adata = half * (*Adata + *Bdata);
		Adata++; Bdata++;
	}

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
}		// -----  end of function Symmetrize ----- 


inline void AlltoallForward( DblNumMat& A, DblNumMat& B, MPI_Comm comm )
{
#ifndef _RELEASE_
	PushCallStack("AlltoallForward");
#endif

	int mpirank, mpisize;
	MPI_Comm_rank( comm, &mpirank );
	MPI_Comm_size( comm, &mpisize );
	
  Int height = A.m();
  Int widthTemp = A.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }

  if(mpirank == (mpisize - 1)){
    heightLocal = heightBlocksize + height % mpisize;
  }

  double sendbuf[height*widthLocal]; 
  double recvbuf[heightLocal*width];
  int sendcounts[mpisize];
  int recvcounts[mpisize];
  int senddispls[mpisize];
  int recvdispls[mpisize];
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    if( k < (mpisize - 1)){
      sendcounts[k] = heightBlocksize * widthLocal;
    }
    else {
      sendcounts[mpisize - 1] = (heightBlocksize + (height % mpisize)) * widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  if((height % heightBlocksize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if((i / heightBlocksize) < (mpisize - 1)){
          sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
        }
        else {
          sendk(i, j) = senddispls[mpisize -1] + j * (heightBlocksize + height % heightBlocksize) 
            + (i - (mpisize - 1) * heightBlocksize) % (heightBlocksize + height % heightBlocksize);
        }
      }
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvk(i, j) = recvdispls[j % mpisize] + (j / mpisize) * heightLocal + i;
    }
  }

  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      sendbuf[sendk(i, j)] = A(i, j); 
    }
  }
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, comm );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      B(i, j) = recvbuf[recvk(i, j)];
    }
  }

#ifndef _RELEASE_
  PopCallStack();
#endif

  return ;
}		// -----  end of function AlltoallForward ----- 

inline void AlltoallBackward( DblNumMat& A, DblNumMat& B, MPI_Comm comm )
{
#ifndef _RELEASE_
	PushCallStack("AlltoallBackward");
#endif

	int mpirank, mpisize;
	MPI_Comm_rank( comm, &mpirank );
	MPI_Comm_size( comm, &mpisize );
	
  Int height = B.m();
  Int widthTemp = B.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );
 
  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }

  if(mpirank == (mpisize - 1)){
    heightLocal = heightBlocksize + height % mpisize;
  }

  double sendbuf[height*widthLocal]; 
  double recvbuf[heightLocal*width];
  int sendcounts[mpisize];
  int recvcounts[mpisize];
  int senddispls[mpisize];
  int recvdispls[mpisize];
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    if( k < (mpisize - 1)){
      sendcounts[k] = heightBlocksize * widthLocal;
    }
    else {
      sendcounts[mpisize - 1] = (heightBlocksize + (height % mpisize)) * widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  if((height % heightBlocksize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if((i / heightBlocksize) < (mpisize - 1)){
          sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
        }
        else {
          sendk(i, j) = senddispls[mpisize -1] + j * (heightBlocksize + height % heightBlocksize) 
            + (i - (mpisize - 1) * heightBlocksize) % (heightBlocksize + height % heightBlocksize);
        }
      }
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvk(i, j) = recvdispls[j % mpisize] + (j / mpisize) * heightLocal + i;
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvbuf[recvk(i, j)] = A(i, j);
    }
  }
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, comm );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      B(i, j) = sendbuf[sendk(i, j)]; 
    }
  }

#ifndef _RELEASE_
  PopCallStack();
#endif

  return ;
}		// -----  end of function AlltoallBackward ----- 

} // namespace dgdft

#endif // _NUMMAT_IMPL_HPP_
