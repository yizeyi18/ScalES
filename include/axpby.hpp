#include "cuda_api_wrappers.hpp"

namespace dgdft {
namespace device {

template <typename T>
void axpby_device( 
  int N, T ALPHA, const T* X, int INCX, T BETA, T* Y, int INCY
);

}
}
