/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

   Author: Lin Lin

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
/// @file scalapack.cpp
/// @brief Thin interface to ScaLAPACK
/// @date 2012-06-05
#include "scalapack.hpp"

namespace dgdft {
namespace scalapack {


// *********************************************************************
// ScaLAPACK routines in C interface
// *********************************************************************
extern "C"{

  void SCALAPACK(descinit)(Int* desc, const Int* m, const Int * n, const Int* mb, 
      const Int* nb, const Int* irsrc, const Int* icsrc, 
      const Int* contxt, const Int* lld, Int* info);

  void SCALAPACK(pdsyev)(const char *jobz, const char *uplo, const Int *n, double *a, 
      const Int *ia, const Int *ja, const Int *desca, double *w, 
      double *z, const Int *iz, const Int *jz, const Int *descz, 
      double *work, const Int *lwork, Int *info);

  void SCALAPACK(pdsyevd)(const char *jobz, const char *uplo, const Int *n, double *a, 
      const Int *ia, const Int *ja, const Int *desca, double *w, 
      const double *z, const Int *iz, const Int *jz, const Int *descz, 
      double *work, const Int *lwork, Int* iwork, const Int* liwork, 
      Int *info);

  void SCALAPACK(pdsyevr)(const char *jobz, const char *range, const char *uplo,
      const Int *n, double* a, const Int *ia, const Int *ja,
      const Int *desca, const double* vl, const double *vu,
      const Int *il, const Int* iu, Int *m, Int *nz, 
      double *w, double *z, const Int *iz, const Int *jz, 
      const Int *descz, double *work, const Int *lwork, 
      Int *iwork, const Int *liwork, Int *info);

// huwei
  void SCALAPACK(pdlacpy)(const char* uplo,
      const Int* m, const Int* n,
      const double* A, const Int* ia, const Int* ja, const Int* desca, 
      const double* B, const Int* ib, const Int* jb, const Int* descb );

  // FIXME huwei
  void SCALAPACK(pdgemm)(const char* transA, const char* transB,
      const Int* m, const Int* n, const Int* k,
      const double* alpha,
      const double* A, const Int* ia, const Int* ja, const Int* desca, 
      const double* B, const Int* ib, const Int* jb, const Int* descb,
      const double* beta,
      double* C, const Int* ic, const Int* jc, const Int* descc,
      const Int* contxt);

  void SCALAPACK(pdgemr2d)(const Int* m, const Int* n, const double* A, const Int* ia, 
      const Int* ja, const Int* desca, double* B,
      const Int* ib, const Int* jb, const Int* descb,
      const Int* contxt);

  void SCALAPACK(pdpotrf)( const char* uplo, const Int* n, 
      double* A, const Int* ia, const Int* ja, const Int* desca, 
      Int* info );

  void SCALAPACK(pdsygst)( const Int* ibtype, const char* uplo, 
      const Int* n, double* A, const Int* ia, const Int* ja, 
      const Int* desca, const double* b, const Int* ib, const Int* jb,
      const Int* descb, double* scale, Int* info );

  void SCALAPACK(pdtrsm)( const char* side, const char* uplo, 
      const char* trans, const char* diag,
      const Int* m, const Int* n, const double* alpha,
      const double* a, const Int* ia, const Int* ja, const Int* desca, 
      double* b, const Int* ib, const Int* jb, const Int* descb );

  // Factorization and triangular solve
  void SCALAPACK(pzgetrf)( const Int* m, const Int* n, dcomplex* A,
      const Int* ia, const Int* ja, const Int* desca, Int* ipiv,
      Int* info );

  void SCALAPACK(pzgetri)( const Int* n, dcomplex* A, const Int* ia,
      const Int* ja, const Int* desca, const Int* ipiv, 
      dcomplex *work, const Int* lwork, Int *iwork, const Int *liwork, 
      Int* info );
}

// *********************************************************************
// Descriptor
// *********************************************************************

void
Descriptor::Init(Int m, Int n, Int mb,
    Int nb, Int irsrc, Int icsrc,
    Int contxt )
{
#ifndef _RELEASE_
  PushCallStack("Descriptor::Init");
#endif
  values_.resize(DLEN);
  Cblacs_gridinfo(contxt, &nprow_, &npcol_, &myprow_, &mypcol_);

  // Compute the leading dimension.  Use the upper bound directly,
  // which costs a bit memory but saves the coding effort for book-keeping.
  Int lld = ( ( ( (m + mb - 1 ) / mb ) + nprow_ - 1 ) / nprow_ ) * mb;

  Int info;
  SCALAPACK(descinit)(&values_[0], &m, &n, &mb, &nb, &irsrc, &icsrc,
      &contxt, &lld, &info);
  if( info )
  {
    std::ostringstream msg;
    msg << "Descriptor:: descinit returned with info = " << info;
    throw std::logic_error( msg.str().c_str() );
  }

#ifndef _RELEASE_
  PopCallStack();
#endif

  return ;
} 		// -----  end of method Descriptor::Init  ----- 


void
Descriptor::Init(Int m, Int n, Int mb,
    Int nb, Int irsrc, Int icsrc,
    Int contxt, Int lld )
{
#ifndef _RELEASE_
  PushCallStack("Descriptor::Init");
#endif
  values_.resize(DLEN);
  Cblacs_gridinfo(contxt, &nprow_, &npcol_, &myprow_, &mypcol_);

  Int info;
  SCALAPACK(descinit)(&values_[0], &m, &n, &mb, &nb, &irsrc, &icsrc,
      &contxt, &lld, &info);
  if( info )
  {
    std::ostringstream msg;
    msg << "Descriptor:: descinit returned with info = " << info;
    throw std::logic_error( msg.str().c_str() );
  }

#ifndef _RELEASE_
  PopCallStack();
#endif

  return ;
} 		// -----  end of method Descriptor::Init  ----- 


Descriptor& Descriptor::operator =	( const Descriptor& desc  )
{
#ifndef _RELEASE_
  PushCallStack("Descriptor::operator=");
#endif
  if( this == &desc ) return *this;
  values_ = desc.values_;
  Cblacs_gridinfo(values_[CTXT], &nprow_, &npcol_, &myprow_, &mypcol_);
  if( nprow_ != desc.nprow_ ||
      npcol_ != desc.npcol_ ||
      myprow_ != desc.myprow_ ||
      mypcol_ != desc.mypcol_ ){
    std::ostringstream msg;
    msg << "Descriptor:: the context information does not match" << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }
#ifndef _RELEASE_
  PopCallStack();
#endif

  return *this;
} 		// -----  end of method Descriptor::operator=  ----- 

// *********************************************************************
// ScaLAPACK routines
// *********************************************************************


template<class F>
inline F ScaLAPACKMatrix<F>::GetLocal	( Int iLocal, Int jLocal ) const
{
#ifndef _RELEASE_
  PushCallStack("ScaLAPACKMatrix::GetLocal");
#endif
  if( iLocal < 0 || iLocal > this->LocalHeight() ||
      jLocal < 0 || jLocal > this->LocalWidth() ){
    std::ostringstream msg;
    msg << "ScaLAPACK::GetLocal index is out of range" << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }
#ifndef _RELEASE_
  PopCallStack();
#endif

  return localMatrix_[iLocal + jLocal * (this->LocalHeight())];
} 		// -----  end of method ScaLAPACKMatrix::GetLocal  ----- 


template<class F>
inline void ScaLAPACKMatrix<F>::SetLocal	( Int iLocal, Int jLocal, F val )
{
#ifndef _RELEASE_
  PushCallStack("ScaLAPACKMatrix::SetLocal");
#endif
  if( iLocal < 0 || iLocal > this->LocalHeight() ||
      jLocal < 0 || jLocal > this->LocalWidth() ){
    std::ostringstream msg;
    msg << "ScaLAPACK::SetLocal index is out of range" << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }
  localMatrix_[iLocal+jLocal*this->LocalHeight()] = val;
#ifndef _RELEASE_
  PopCallStack();
#endif

  return ;
} 		// -----  end of method ScaLAPACKMatrix::SetLocal  ----- 


template<class F>
inline ScaLAPACKMatrix<F>& 
ScaLAPACKMatrix<F>::operator=	( const ScaLAPACKMatrix<F>& A )
{
#ifndef _RELEASE_
  PushCallStack("ScaLAPACKMatrix::operator=");
#endif
  if( this == &A ) return *this;
  desc_ = A.desc_;
  localMatrix_ = A.localMatrix_;
#ifndef _RELEASE_
  PopCallStack();
#endif
  return *this;
} 		// -----  end of method ScaLAPACKMatrix::operator=  ----- 


// huwei
//void
//Gemm(char transA, char transB, const double alpha,
//    const ScaLAPACKMatrix<double>& A, ScaLAPACKMatrix<double>& B, 
//    const double beta, ScaLAPACKMatrix<double>& C){
//#ifndef _RELEASE_
//  PushCallStack("scalapack::Gemm");
//#endif
//  if( A.Height() != B.Height() ){
//    std::ostringstream msg;
//    msg 
//      << "Gemm:: Global matrix dimension does not match\n" 
//      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl
//      << "The dimension of B is " << B.Height() << " x " << B.Width() << std::endl;
//    throw std::logic_error( msg.str().c_str() );
//  }

//  if( A.Context() != B.Context() ){
//    std::ostringstream msg;
//    msg << "Gemm:: A and B are not sharing the same context." << std::endl; 
//    throw std::logic_error( msg.str().c_str() );
//  }

//  const Int M = A.Height();
//  const Int N = A.Width();
//  const Int K = B.Width();
//  const Int contxt = A.Context();

//  SCALAPACK(pdgemm)(&transA, &transB,
//      &M, &N, &K, 
//      &alpha,
//      A.Data(), &I_ONE, &I_ONE, A.Desc().Values(), 
//      B.Data(), &I_ONE, &I_ONE, B.Desc().Values(), 
//      &beta,
//      C.Data(), &I_ONE, &I_ONE, C.Desc().Values(), 
//      &contxt );	
//#ifndef _RELEASE_
//  PopCallStack();
//#endif
//  return;
//}   // -----  end of function Gemm  ----- 


// huwei
void
Lacpy( char uplo, 
       Int m, Int n, 
       double* A, Int ia, Int ja, Int* desca, 
       double* B, Int ib, Int jb, Int* descb){

#ifndef _RELEASE_
  PushCallStack("scalapack::Lacpy");
#endif

SCALAPACK(pdlacpy)( &uplo,
      &m, &n,
      A, &ia, &ja, desca, 
      B, &ib, &jb, descb );
  
#ifndef _RELEASE_
  PopCallStack();
#endif
  return;
}   // -----  end of function Lacpy  ----- 

// huwei
void
Gemm( char transA, char transB,
      Int m, Int n, Int k,
      double alpha,
      double* A, Int ia, Int ja, Int* desca, 
      double* B, Int ib, Int jb, Int* descb,
      double beta,
      double* C, Int ic, Int jc, Int* descc,
      Int contxt){
  
#ifndef _RELEASE_
  PushCallStack("scalapack::Gemm");
#endif

SCALAPACK(pdgemm)( &transA, &transB,
      &m, &n, &k,
      &alpha,
      A, &ia, &ja, desca, 
      B, &ib, &jb, descb,
      &beta,
      C, &ic, &jc, descc,
      &contxt );
  
#ifndef _RELEASE_
  PopCallStack();
#endif
  return;
}   // -----  end of function Gemm  ----- 



void
Gemr2d(const ScaLAPACKMatrix<double>& A, ScaLAPACKMatrix<double>& B){
#ifndef _RELEASE_
  PushCallStack("scalapack::Gemr2d");
#endif
  if( A.Height() != B.Height() || A.Width() != B.Width() ){
    std::ostringstream msg;
    msg 
      << "Gemr2d:: Global matrix dimension does not match\n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl
      << "The dimension of B is " << B.Height() << " x " << B.Width() << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }

  if( A.Context() != B.Context() ){
    std::ostringstream msg;
    msg << "Gemr2d:: A and B are not sharing the same context." << std::endl; 
    throw std::logic_error( msg.str().c_str() );
  }

  const Int M = A.Height();
  const Int N = A.Width();
  const Int contxt = A.Context();

  SCALAPACK(pdgemr2d)(&M, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), 
      B.Data(), &I_ONE, &I_ONE, 
      B.Desc().Values(), &contxt );	
#ifndef _RELEASE_
  PopCallStack();
#endif
  return;
}   // -----  end of function Gemr2d  ----- 



void
Trsm( char side, char uplo, char trans, char diag, double alpha,
    const ScaLAPACKMatrix<double>& A, ScaLAPACKMatrix<double>& B )
{
#ifndef _RELEASE_
	PushCallStack("scalapack::Trsm");
#endif
  const Int M = B.Height(); // const Int M = A.Height();
  const Int N = A.Width();

  SCALAPACK(pdtrsm)(&side, &uplo, &trans, &diag, &M, &N, &alpha,
      A.Data(), &I_ONE, &I_ONE, A.Desc().Values(), 
      B.Data(), &I_ONE, &I_ONE, B.Desc().Values());

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
}		// -----  end of function Trsm  ----- 


void
Syev(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs){
#ifndef _RELEASE_
  PushCallStack("scalapack::Syev");
#endif

  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syev: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }

  char jobz = 'N';
  Int lwork = -1, info;
  std::vector<double> work(1);
  Int N = A.Height();

  eigs.resize(N);
  ScaLAPACKMatrix<double> dummyZ;

  SCALAPACK(pdsyev)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], dummyZ.Data(),
      &I_ONE, &I_ONE, dummyZ.Desc().Values(), &work[0],
      &lwork, &info);
  lwork = (Int)work[0];
  work.resize(lwork);

  SCALAPACK(pdsyev)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], dummyZ.Data(),
      &I_ONE, &I_ONE, dummyZ.Desc().Values(), &work[0],
      &lwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyev: logic error. Info = " << info << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyev: runtime error. Info = " << info << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }
#ifndef _RELEASE_
  PopCallStack();
#endif
  return;
}   // -----  end of function Syev ----- 



void
Syev(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z){
#ifndef _RELEASE_
  PushCallStack("scalapack::Syev");
#endif
  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syev: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }

  char jobz = 'V';
  Int lwork = -1, info;
  std::vector<double> work(1);
  Int N = A.Height();

  eigs.resize(N);
  Z.SetDescriptor(A.Desc());

  SCALAPACK(pdsyev)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), &work[0],
      &lwork, &info);
  lwork = (Int)work[0];
  work.resize(lwork);

  SCALAPACK(pdsyev)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), &work[0],
      &lwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyev: logic error. Info = " << info << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyev: runtime error. Info = " << info << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }
#ifndef _RELEASE_
  PopCallStack();
#endif
  return;
}   // -----  end of function Syev ----- 

void
Syevd(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z){
#ifndef _RELEASE_
  PushCallStack("scalapack::Syevd");
#endif

  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syevd: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }

  char jobz = 'V';
  Int  liwork = -1, lwork = -1, info;
  std::vector<double> work(1);
  std::vector<Int>    iwork(1);
  Int N = A.Height();

  eigs.resize(N);
  Z.SetDescriptor(A.Desc());

  SCALAPACK(pdsyevd)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork,&iwork[0], &liwork, &info);
  lwork = (Int)work[0];
  // NOTE: Buggy memory allocation in pdsyevd?
  lwork = lwork+500;
  work.resize(lwork);
  liwork = iwork[0];
  // NOTE: Buggy memory allocation in pdsyevd?
  liwork = liwork+500;
  iwork.resize(liwork);

  SCALAPACK(pdsyevd)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork,&iwork[0], &liwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevd: logic error. Info = " << info << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevd: runtime error. Info = " << info << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }
#ifndef _RELEASE_
  PopCallStack();
#endif
  return;
}   // -----  end of function Syevd ----- 


void
Syevr(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z){
#ifndef _RELEASE_
  PushCallStack("scalapack::Syevr");
#endif

  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syevr: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }

  char jobz = 'V';
  char range = 'A'; // Compute all eigenvalues

  Int  liwork = -1, lwork = -1, info;
  std::vector<double> work(1);
  std::vector<Int>    iwork(1);
  Int N = A.Height();

  eigs.resize(N);
  Z.SetDescriptor(A.Desc());
  double dummyV = 0.0;
  Int dummyI = 0;
  Int numEigValueFound, numEigVectorFound;


  SCALAPACK(pdsyevr)(&jobz, &range, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &dummyV, &dummyV, 
      &dummyI, &dummyI, &numEigValueFound, &numEigVectorFound,
      &eigs[0], Z.Data(), &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork, &iwork[0], &liwork, &info);
  lwork = (Int)work[0];
  work.resize(lwork);
  liwork = iwork[0];
  iwork.resize(liwork);

  SCALAPACK(pdsyevr)(&jobz, &range, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &dummyV, &dummyV, 
      &dummyI, &dummyI, &numEigValueFound, &numEigVectorFound,
      &eigs[0], Z.Data(), &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork, &iwork[0], &liwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevr: logic error. Info = " << info << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevr: runtime error. Info = " << info << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }

  if( numEigValueFound != N ){
    std::ostringstream msg;
    msg 
      << "pdsyevr: Not all eigenvalues are found.\n " 
      << "Found " << numEigValueFound << " eigenvalues, " << 
      N << " eigenvalues in total." << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }
  if( numEigVectorFound != N ){
    std::ostringstream msg;
    msg 
      << "pdsyevr: Not all eigenvectors are found.\n " 
      << "Found " << numEigVectorFound << " eigenvectors, " << 
      N << " eigenvectors in total." << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }
#ifndef _RELEASE_
  PopCallStack();
#endif
  return;
}   // -----  end of function Syevr ----- 

void
Syevr(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z,
    Int il,
    Int iu){
#ifndef _RELEASE_
  PushCallStack("scalapack::Syevr");
#endif

  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syevr: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }

  char jobz = 'V';
  char range = 'I'; // Compute selected range of eigenvalues

  Int  liwork = -1, lwork = -1, info;
  std::vector<double> work(1);
  std::vector<Int>    iwork(1);
  Int N = A.Height();
  Int numEigValue = std::min( N, iu - il + 1 );

  eigs.resize( N );
  Z.SetDescriptor(A.Desc());
  double dummyV = 0.0;
  Int dummyI = 0;
  Int numEigValueFound, numEigVectorFound;


  SCALAPACK(pdsyevr)(&jobz, &range, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &dummyV, &dummyV, 
      &il, &iu, &numEigValueFound, &numEigVectorFound,
      &eigs[0], Z.Data(), &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork,&iwork[0], &liwork, &info);
  lwork = (Int)work[0];
  work.resize(lwork);
  liwork = iwork[0];
  iwork.resize(liwork);

  SCALAPACK(pdsyevr)(&jobz, &range, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &dummyV, &dummyV, 
      &il, &iu, &numEigValueFound, &numEigVectorFound,
      &eigs[0], Z.Data(), &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork,&iwork[0], &liwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevr: logic error. Info = " << info << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevr: runtime error. Info = " << info << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }

  if( numEigValueFound != numEigValue ){
    std::ostringstream msg;
    msg 
      << "pdsyevr: Not all eigenvalues are found.\n " 
      << "Found " << numEigValueFound << " eigenvalues, " << 
      N << " eigenvalues in total." << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }
  if( numEigVectorFound != numEigValue ){
    std::ostringstream msg;
    msg 
      << "pdsyevr: Not all eigenvectors are found.\n " 
      << "Found " << numEigVectorFound << " eigenvectors, " << 
      N << " eigenvectors in total." << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }

  // Post processing of eigs by resize (not destroying the computed
  // eigenvalues) 
  eigs.resize( numEigValue );


#ifndef _RELEASE_
  PopCallStack();
#endif
  return;
}   // -----  end of function Syevr ----- 


void
Potrf( char uplo, ScaLAPACKMatrix<double>& A )
{
#ifndef _RELEASE_
  PushCallStack("Potrf");
#endif
  Int info;

  Int N = A.Height();

  SCALAPACK(pdpotrf)(&uplo, &N, A.Data(), &I_ONE,
      &I_ONE, A.Desc().Values(), &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdpotrf: logic error. Info = " << info << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdpotrf: runtime error. Info = " << info << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }

#ifndef _RELEASE_
  PopCallStack();
#endif

  return ;
} 	// -----  end of function Potrf  ----- 


void 
Sygst( Int ibtype, char uplo, ScaLAPACKMatrix<double>& A,
   ScaLAPACKMatrix<double>& B )
{
#ifndef _RELEASE_
	PushCallStack("Sygst");
#endif

  Int info;
  double scale;
  Int N = A.Height();

  if( A.Height() != B.Height() ){
    throw std::logic_error("A and B are not of the same size.");
  }

  SCALAPACK(pdsygst)(&ibtype, &uplo, &N, A.Data(), 
      &I_ONE, &I_ONE, A.Desc().Values(),
      B.Data(), &I_ONE, &I_ONE, B.Desc().Values(), &scale, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsygst: logic error. Info = " << info << std::endl;
    throw std::logic_error( msg.str().c_str() );
  }

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
}		// -----  end of function Sygst  ----- 




} // namespace scalapack
} // namespace dgdft
