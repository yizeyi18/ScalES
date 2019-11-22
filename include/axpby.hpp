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


// res = beta * res + \sum_k vec_k 
//HA: I have to make vecs not constant. 
//Because of this error: types "const T" and "double" have incompatible cv-qualifiers
template <typename T>
void add_vecs_device( int nVec, int N, T** vecs, T BETA, T* result ); 

}





}
