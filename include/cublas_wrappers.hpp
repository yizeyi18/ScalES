#pragma once
#include "cuda_type_wrappers.hpp"

namespace cublas {

namespace detail {
  struct handle_pimpl;
  handle_pimpl* get_handle( handle& );
}



class handle {

  friend cuda::stream;
  friend detail::handle_pimpl* detail::get_handle( handle& );;

  std::shared_ptr< detail::handle_pimpl > pimpl_;

public:

  handle();
  ~handle() noexcept;

  handle( const handle& );
  handle( handle&&      ) noexcept;

  cuda::stream get_stream() const;
  void set_stream( const cuda::stream& );

};






namespace blas {

template <typename T>
void gemm( handle& h,
  char TRANSA, char TRANSB, int M, int N, int K, 
  T ALPHA, const T* A_device, int LDA, const T* B_device,
  int LDB, T BETA, T* C_device, int LDC );


template <typename T>
void gemm_batched( handle& h,
  char TRANSA, char TRANSB, int M, int N, int K, 
  T ALPHA, T** A_device, int LDA, T** B_device,
  int LDB, T BETA, T** C_device, int LDC, int batch_count );

template <typename T>
void axpy( handle& h,
  int N, T ALPHA, const T* X, int INCX, T* Y, int INCY 

);

}


}
