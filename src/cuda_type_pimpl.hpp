#include <iostream>
#include <cuda_type_wrappers.hpp>
namespace cuda {
namespace detail {

  struct event_pimpl {

    cudaEvent_t event;

    event_pimpl() {
      CUDA_THROW( cudaEventCreate( &this->event ) );
    }

    ~event_pimpl() noexcept {
      CUDA_ASSERT( cudaEventDestroy( this->event ) );
    }

  };


  struct stream_pimpl {
    cudaStream_t stream;
  };


  struct managed_stream_pimpl : public stream_pimpl {

    managed_stream_pimpl() {
      std::cout << "CREATE" << std::endl;
      CUDA_THROW( cudaStreamCreate( &this->stream ) );
    }

    ~managed_stream_pimpl() noexcept {
      std::cout << "DESTROY" << std::endl;
      CUDA_ASSERT( cudaStreamDestroy( this->stream ) );
    }

  };

}
}
