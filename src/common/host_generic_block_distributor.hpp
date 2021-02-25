//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: David Williams-Young

#pragma once
#include "block_distributor_impl.hpp"

#include "numvec_impl.hpp"
#include "nummat_impl.hpp"
#include "utility.hpp"

namespace scales {

template <typename T>
class HostGenericBlockDistributor : public detail::BlockDistributorImpl<T> {

protected:

  IntNumVec sendcounts_, recvcounts_, senddispls_, recvdispls_;
  IntNumMat sendk_, recvk_;

  NumVec<T> recvbuf_, sendbuf_;

public:

  HostGenericBlockDistributor( MPI_Comm comm, Int M, Int N ) :
    detail::BlockDistributorImpl<T>( comm, M, N ),
    sendcounts_( this->comm_size_ ),
    recvcounts_( this->comm_size_ ),
    senddispls_( this->comm_size_ ),
    recvdispls_( this->comm_size_ ),
    sendk_( this->M_,      this->NLocal_ ),
    recvk_( this->MLocal_, this->N_      ),
    sendbuf_( this->M_ * this->NLocal_ ),
    recvbuf_( this->MLocal_ * this->N_ ) 
  {


    std::fill_n( sendcounts_.Data(), this->comm_size_, 
                 this->MBlock_ * this->NLocal_ );
    std::fill_n( recvcounts_.Data(), this->comm_size_, 
                 this->MLocal_ * this->NBlock_ );

    for( Int k = 0; k < this->MRem_; k++ ){ 
      sendcounts_[k] += this->NLocal_;
    };
    for( Int k = 0; k < this->NRem_; k++ ){ 
      recvcounts_[k] += this->MLocal_;
    };


    senddispls_[0] = 0;
    recvdispls_[0] = 0;
    for( Int k = 1; k < this->comm_size_; k++ ){ 
      senddispls_[k] = senddispls_[k-1] + sendcounts_[k-1];
      recvdispls_[k] = recvdispls_[k-1] + recvcounts_[k-1];
    }

    if(!this->MRem_) {

      for( Int j = 0; j < this->NLocal_; j++ )
      for( Int i = 0; i < this->M_;      i++ ) {
        sendk_(i, j) = senddispls_[i / this->MBlock_] + 
                       j * this->MBlock_              + 
                       i % this->MBlock_;
      }

    } else {

      for( Int j = 0; j < this->NLocal_; j++ ){ 
      for( Int i = 0; i < this->M_;      i++ ){

        if( i < ((this->MRem_) * (this->MBlock_+1)) ) {

          sendk_(i, j) = senddispls_[i / (this->MBlock_+1)] + 
                         j * (this->MBlock_+1)              + 
                         i % (this->MBlock_+1);

        } else {

          sendk_(i, j) = senddispls_[(this->MRem_) + (i-(this->MRem_)*(this->MBlock_+1))/this->MBlock_]
            + j * this->MBlock_ + (i-(this->MRem_)*(this->MBlock_+1)) % this->MBlock_;

        }

      }
      }

    }

    for( Int j = 0; j < this->N_; j++ ) 
    for( Int i = 0; i < this->MLocal_; i++ ){
      recvk_(i, j) = recvdispls_[j % this->comm_size_] + (j / this->comm_size_) * this->MLocal_ + i;
    }

  }


  void redistribute_row_to_col( const NumMat<T>& row_data, 
                                      NumMat<T>& col_data ) override { 

    Real timeSta, timeEnd;

    GetTime( timeSta );
    for( Int j = 0; j < this->N_;      j++ )
    for( Int i = 0; i < this->MLocal_; i++ ){
      recvbuf_[recvk_(i, j)] = row_data(i, j);
    }
    GetTime( timeEnd );

    Real r2c_pack_dur = timeEnd - timeSta;

    MPI_Alltoallv( recvbuf_.Data(), recvcounts_.Data(), recvdispls_.Data(),
                   MPI_DOUBLE, // FIXME: base on template param
                   sendbuf_.Data(), sendcounts_.Data(), senddispls_.Data(),
                   MPI_DOUBLE, // FIXME: base on template param
                   this->comm_ );

    GetTime( timeSta );
    for( Int j = 0; j < this->NLocal_; j++ )
    for( Int i = 0; i < this->M_;      i++ ){
      col_data(i, j) = sendbuf_[sendk_(i, j)];
    }
    GetTime( timeEnd );

    Real r2c_unpack_dur = timeEnd - timeSta;

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "R2C PACK DUR   = " << r2c_pack_dur   << std::endl;
    statusOFS << "R2C UNPACK DUR = " << r2c_unpack_dur << std::endl;
#endif
  };



  void redistribute_col_to_row( const NumMat<T>& col_data, 
                                      NumMat<T>& row_data ) override {

    for( Int j = 0; j < this->NLocal_; j++ )
    for( Int i = 0; i < this->M_;      i++ ) {
      sendbuf_[sendk_(i, j)] = col_data(i, j);
    }

    MPI_Alltoallv( sendbuf_.Data(), sendcounts_.Data(), senddispls_.Data(),
                   MPI_DOUBLE, // FIXME: base on template param
                   recvbuf_.Data(), recvcounts_.Data(), recvdispls_.Data(),
                   MPI_DOUBLE, // FIXME: base on template param
                   this->comm_ );
                   
    for( Int j = 0; j < this->N_;      j++ )
    for( Int i = 0; i < this->MLocal_; i++ ) {
      row_data(i, j) = recvbuf_[recvk_(i, j)];
    }
    
  }; // redistribute_col_to_row

}; // HostGenericBlockDistributor

}

