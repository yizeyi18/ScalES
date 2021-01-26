#pragma once
#include "block_distributor_impl.hpp"

#include "numvec_impl.hpp"
#include "nummat_impl.hpp"
#include "utility.hpp"

#include <lapack.hh>

namespace scales {

template <typename T>
class HostOptPackBlockDistributor : public detail::BlockDistributorImpl<T> {

protected:

  IntNumVec sendcounts_, recvcounts_, senddispls_, recvdispls_;

  NumVec<T> recvbuf_, sendbuf_;

public:

  HostOptPackBlockDistributor( MPI_Comm comm, Int M, Int N ) :
    detail::BlockDistributorImpl<T>( comm, M, N ),
    sendcounts_( this->comm_size_ ),
    recvcounts_( this->comm_size_ ),
    senddispls_( this->comm_size_ ),
    recvdispls_( this->comm_size_ ),
    sendbuf_( this->M_ * this->NLocal_ ),
    recvbuf_( this->MLocal_ * this->N_ ) 
  {

    statusOFS << "Using OptPack Block Distributor" << std::endl;

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


  }


  void redistribute_row_to_col( const NumMat<T>& row_data, 
                                      NumMat<T>& col_data ) override { 

    Real timeSta, timeEnd;

    GetTime( timeSta );
    // Pack row_data into buffer
    Int recvbuf_off = 0;
    for( Int irank = 0; irank < this->comm_size_; ++irank ) {
      Int col_rank_local = this->NBlock_;
      if( irank < this->NRem_ ) col_rank_local++;
      lapack::lacpy( lapack::MatrixType::General, this->MLocal_, col_rank_local, 
                     &row_data(0,irank), this->comm_size_*this->MLocal_, 
                     &recvbuf_(recvbuf_off), this->MLocal_ ); 

      recvbuf_off += col_rank_local * this->MLocal_;
    }
    GetTime( timeEnd );

    Real r2c_pack_dur = timeEnd - timeSta;

    MPI_Alltoallv( recvbuf_.Data(), recvcounts_.Data(), recvdispls_.Data(),
                   MPI_DOUBLE, // FIXME: base on template param
                   sendbuf_.Data(), sendcounts_.Data(), senddispls_.Data(),
                   MPI_DOUBLE, // FIXME: base on template param
                   this->comm_ );

    GetTime( timeSta );
    Int row_offset  = 0;
    Int sendbuf_off = 0;
    for( Int irank = 0; irank < this->comm_size_; ++irank ) {
      Int row_rank_local = this->MBlock_;
      if( irank < this->MRem_ ) row_rank_local++;
      
      lapack::lacpy( lapack::MatrixType::General, row_rank_local, this->NLocal_, 
                     &sendbuf_(sendbuf_off), row_rank_local,
                     &col_data(row_offset,0), col_data.m() );

      sendbuf_off += this->NLocal_ * row_rank_local;
      row_offset  += row_rank_local;
    }
    GetTime( timeEnd );

    Real r2c_unpack_dur = timeEnd - timeSta;

    statusOFS << "R2C PACK DUR   = " << r2c_pack_dur   << std::endl;
    statusOFS << "R2C UNPACK DUR = " << r2c_unpack_dur << std::endl;

  };



  void redistribute_col_to_row( const NumMat<T>& col_data, 
                                      NumMat<T>& row_data ) override {

    // Pack matrices into contiguous buffers
    Int row_offset  = 0;
    Int sendbuf_off = 0;
    for( Int irank = 0; irank < this->comm_size_; ++irank ) {
      Int row_rank_local = this->MBlock_;
      if( irank < this->MRem_ ) row_rank_local++;
      
      lapack::lacpy( lapack::MatrixType::General, row_rank_local, 
                     this->NLocal_, &col_data(row_offset,0),
                     col_data.m(), &sendbuf_(sendbuf_off), row_rank_local );

      sendbuf_off += this->NLocal_ * row_rank_local;
      row_offset  += row_rank_local;
    }

    MPI_Alltoallv( sendbuf_.Data(), sendcounts_.Data(), senddispls_.Data(),
                   MPI_DOUBLE, // FIXME: base on template param
                   recvbuf_.Data(), recvcounts_.Data(), recvdispls_.Data(),
                   MPI_DOUBLE, // FIXME: base on template param
                   this->comm_ );
                   
    // Unpack buffers into row_data
    Int recvbuf_off = 0;
    for( Int irank = 0; irank < this->comm_size_; ++irank ) {
      Int col_rank_local = this->NBlock_;
      if( irank < this->NRem_ ) col_rank_local++;
      lapack::lacpy( lapack::MatrixType::General, this->MLocal_, col_rank_local, 
                     &recvbuf_(recvbuf_off), this->MLocal_, 
                     &row_data(0,irank), this->comm_size_*this->MLocal_ );

      recvbuf_off+= col_rank_local * this->MLocal_;
    }
    
  }; // redistribute_col_to_row

}; // HostOptPackBlockDistributor

}

