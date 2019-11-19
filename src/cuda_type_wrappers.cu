#include <cuda_type_wrappers.hpp>
#include <exceptions.hpp>

#include "cuda_type_pimpl.hpp"

namespace cuda {


stream::stream() :
  pimpl_( std::make_shared<detail::managed_stream_pimpl>() ){ }
stream::stream( std::shared_ptr<detail::stream_pimpl>&& p ) :
  pimpl_(std::move(p)){ }

stream::~stream() noexcept = default;
stream::stream( const stream& ) = default;
stream::stream( stream&& ) noexcept = default;






void stream::synchronize() const {
  CUDA_THROW( cudaStreamSynchronize( pimpl_->stream ) );
}






event::event() :
  pimpl_( std::make_shared<detail::event_pimpl>() ){ }

event::~event() noexcept = default;
event::event( event&& ) noexcept = default;
event::event( const event& ) = default;

void event::record( const stream& stream ) {
  CUDA_THROW( cudaEventRecord( pimpl_->event, stream.pimpl_->stream ) );
}

void event::record() {
  CUDA_THROW( cudaEventRecord( pimpl_->event ) );
}

void event::synchronize() const {
  CUDA_THROW( cudaEventSynchronize( pimpl_->event ) );
}


float event::elapsed_time( const event& first, const event& second ) {
  float time;
  CUDA_THROW( cudaEventElapsedTime( &time, first.pimpl_->event, second.pimpl_->event ) );
  return time;
}

}
