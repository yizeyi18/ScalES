#include "cuda_api_wrappers.hpp"

namespace dgdft {
namespace device {

template <typename T>
void axpby_device( 
  int N, T ALPHA, const T* X, int INCX, T BETA, T* Y, int INCY
);

template <typename T>
void axpby_combined_device( 
  int N, T c, T e, T sigma, T sigma_new, 
	const T* X, int INCX, const T* M, int INCM, T* Y, int INCY
);
}
}
