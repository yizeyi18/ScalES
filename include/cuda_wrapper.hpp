#pragma once
#include <cstddef>
#include <memory>
#if __cplusplus >= 201703L
#include "sfinae.hpp"
#endif

namespace cuda {

namespace wrappers {

void memcpy_h2d( void* dest, const void* src, size_t len );
void memcpy_d2h( void* dest, const void* src, size_t len );

void* malloc( size_t len );
void  free( void* ptr ); 

void device_sync();

void memset( void* data, int val, size_t len );

}

template <typename T>
inline void memcpy_h2d( T* dest, const T* src, size_t len ) {
  wrappers::memcpy_h2d( dest, src, len * sizeof(T) );
}

template <typename T>
inline void memcpy_d2h( T* dest, const T* src, size_t len ) {
  wrappers::memcpy_d2h( dest, src, len * sizeof(T) );
}

inline void* malloc( size_t len ) {
  return wrappers::malloc( len );
}

inline void  free( void* ptr ) { 
  wrappers::free( ptr );
}


inline void device_sync() {
  wrappers::device_sync();
}


#if __cplusplus >= 201703L


template <typename DestContainer, typename SrcContainer>
std::enable_if_t<
  (dgdft::has_data_member_v<DestContainer> or 
   dgdft::has_Data_member_v<DestContainer>) and
  (dgdft::has_data_member_v<SrcContainer> or 
   dgdft::has_Data_member_v<SrcContainer>) and
  (dgdft::has_size_member_v<DestContainer> or 
   dgdft::has_Size_member_v<DestContainer>) and
  (dgdft::has_size_member_v<SrcContainer> or 
   dgdft::has_Size_member_v<SrcContainer>)
> memcpy_h2d( DestContainer& dest, const SrcContainer& src ) {

  auto size_dest = dgdft::get_size_member( dest );
  auto size_src  = dgdft::get_size_member( src );

  auto* data_dest = dgdft::get_data_member( dest );
  auto* data_src  = dgdft::get_data_member( src );

  assert( size_dest == size_src );
  assert( data_dest != data_src );

  memcpy_h2d( data_dest, data_src, size_dest );

}

template <typename DestContainer, typename SrcContainer>
std::enable_if_t<
  (dgdft::has_data_member_v<DestContainer> or 
   dgdft::has_Data_member_v<DestContainer>) and
  (dgdft::has_data_member_v<SrcContainer> or 
   dgdft::has_Data_member_v<SrcContainer>) and
  (dgdft::has_size_member_v<DestContainer> or 
   dgdft::has_Size_member_v<DestContainer>) and
  (dgdft::has_size_member_v<SrcContainer> or 
   dgdft::has_Size_member_v<SrcContainer>)
> memcpy_d2h( DestContainer& dest, const SrcContainer& src ) {

  auto size_dest = dgdft::get_size_member( dest );
  auto size_src  = dgdft::get_size_member( src );

  auto* data_dest = dgdft::get_data_member( dest );
  auto* data_src  = dgdft::get_data_member( src );

  assert( size_dest == size_src );
  assert( data_dest != data_src );

  memcpy_d2h( data_dest, data_src, size_dest );

}



#endif





namespace detail {

  struct cuda_event_pimpl;
  struct cuda_stream_pimpl;
  struct cublas_handle_pimpl;

}

class cuda_stream;
class cuda_event;
class cublas_handle;

class cublas_handle {

  friend cuda_stream;

  std::unique_ptr< detail::cublas_handle_pimpl > pimpl_;

public:

  cublas_handle();
  ~cublas_handle() noexcept;

  cublas_handle( const cublas_handle& ) = delete;
  cublas_handle( cublas_handle&&      ) noexcept;

  decltype(pimpl_.get()) pimpl() const;

};


class cuda_stream {

  friend cuda_event;

  std::unique_ptr< detail::cuda_stream_pimpl > pimpl_;

public:

  cuda_stream();
  ~cuda_stream() noexcept;

  cuda_stream( const cuda_stream& ) = delete;
  cuda_stream( cuda_stream&&      ) noexcept;

  void synchronize() const ;

};

class cuda_event {

  std::unique_ptr< detail::cuda_event_pimpl > pimpl_;

public:

  cuda_event();
  ~cuda_event() noexcept;

  cuda_event( const cuda_event& ) = delete;
  cuda_event( cuda_event&&      ) noexcept;

  void record( const cuda_stream& );
  void record( );

  void synchronize() const;

  static float elapsed_time( const cuda_event&, const cuda_event& );

};






template <typename T>
void cublas_gemm( cublas_handle& handle,
  char TRANSA, char TRANSB, int M, int N, int K, 
  T ALPHA, const T* A_device, int LDA, const T* B_device,
  int LDB, T BETA, T* C_device, int LDC );


template <typename T>
void cublas_gemm_batched( cublas_handle& handle,
  char TRANSA, char TRANSB, int M, int N, int K, 
  T ALPHA, T** A_device, int LDA, T** B_device,
  int LDB, T BETA, T** C_device, int LDC, int batch_count );

template <typename T>
void cublas_axpy( cublas_handle& handle,
  int N, T ALPHA, const T* X, int INCX, T* Y, int INCY 
);

template <typename T>
void axpby_device( 
  int N, T ALPHA, const T* X, int INCX, T BETA, T* Y, int INCY,
  cuda_stream& stream 
);
template <typename T>
void axpby_device( 
  int N, T ALPHA, const T* X, int INCX, T BETA, T* Y, int INCY
);




template <typename T>
void chebyshev_filter_device( int N, T sigma, T sigma_new, T e, T c, T* HY, const T* Y, const T* X ); 



}

