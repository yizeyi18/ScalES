#pragma once
#include "block_distributor_impl.hpp"

#include "numvec_impl.hpp"
#include "nummat_impl.hpp"


namespace dgdft {

template <typename T>
class RMAHostBlockDistributor : public detail::BlockDistributorImpl<T> {

protected:

  IntNumVec sendcounts_, recvcounts_, senddispls_, recvdispls_;
  IntNumMat sendk_, recvk_;

  NumVec<T> recvbuf_, sendbuf_;
public:

  RMAHostBlockDistributor( MPI_Comm comm, Int M, Int N ) :
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

    // Make row_data publically accessible
    // TODO: Cache row window internally to avoid recreating buffer
    MPI_Win window;
    MPI_Aint win_size = row_data.Size() * sizeof(T);
    MPI_Win_create( row_data.Data(), win_size, sizeof(T), 
                    MPI_INFO_NULL, this->comm_, &window );



    // The first col block is responsible for 
    // col_data(:, (rank*NLocal):((rank+1)*NLocal) )

/*
    NumMat<T> col_block( this->MBlock_, this->NBlock_ );
    for( int irank = 0; irank < this->comm_size_; ++irank )
    if( irank != this->comm_rank_ ) { // No remote get for local data

      MPI_Aint remote_disp =  this->MBlock_ * irank;
      MPI_Get( col_block.Data(), col_block.Size(), MPI_DOUBLE,
               irank, remote_disp, col_block.Size(), MPI_DOUBLE,
               window );

      for( int64_t j = 0; j < this->NBlock_; ++j )
      for( int64_t i = 0; j < this->MBlock_; ++i ) {
        col_data( this->
      }
    } else {

      // TODO: LACOPY

    }
*/

    // Loop over local column indices
    for( Int j = 0; j < this->NLocal_; ++j ) { 
      Int J = j*mpisize; // Compute global index
    }


    // Free Window
    MPI_Win_free( &window );



    for( Int j = 0; j < this->N_;      j++ )
    for( Int i = 0; i < this->MLocal_; i++ ){
      recvbuf_[recvk_(i, j)] = row_data(i, j);
    }

    MPI_Alltoallv( recvbuf_.Data(), recvcounts_.Data(), recvdispls_.Data(),
                   MPI_DOUBLE, // FIXME: base on template param
                   sendbuf_.Data(), sendcounts_.Data(), senddispls_.Data(),
                   MPI_DOUBLE, // FIXME: base on template param
                   this->comm_ );
    
    for( Int j = 0; j < this->NLocal_; j++ )
    for( Int i = 0; i < this->M_;      i++ ){
      col_data(i, j) = sendbuf_[sendk_(i, j)];
    }

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

  const IntNumVec* sendcounts() const override { return &sendcounts_; }
  const IntNumVec* recvcounts() const override { return &recvcounts_;}
  const IntNumVec* senddispls() const override { return &senddispls_;}
  const IntNumVec* recvdispls() const override { return &recvdispls_;}
  const IntNumMat* sendk()      const override { return &sendk_   ;}
  const IntNumMat* recvk()      const override { return &recvk_   ;}
}; // HostBlockDistributor


namespace detail {


template <typename T>
std::unique_ptr< BlockDistributorImpl<T> > 
  make_default_host_distributor( MPI_Comm comm, Int M, Int N ) {

  return std::unique_ptr< BlockDistributorImpl<T> >(
    new HostBlockDistributor<T>( comm, M, N )
  );

}

}

}
