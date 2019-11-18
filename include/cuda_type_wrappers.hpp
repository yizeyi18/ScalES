#pragma once
#include <memory>
#include "cuda_type_fwd.hpp"
#include "cublas_type_fwd.hpp"

namespace cuda {

namespace detail {

  struct event_pimpl;
  struct stream_pimpl;

}

class stream {

  friend event;
  friend cublas::handle;

  std::shared_ptr< detail::stream_pimpl > pimpl_;

  stream( decltype(pimpl_)&& );

public:

  stream();
  ~stream() noexcept;

  stream( const stream& );
  stream( stream&&      ) noexcept;

  void synchronize() const ;

};

class event {

  std::shared_ptr< detail::event_pimpl > pimpl_;

public:

  event();
  ~event() noexcept;

  event( const event& );
  event( event&&      ) noexcept;

  void record( const stream& );
  void record( );

  void synchronize() const;

  static float elapsed_time( const event&, const event& );

};

}
