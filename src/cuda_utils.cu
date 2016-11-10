#ifndef  _CUDA_UTILS_CU_
#define  _CUDA_UTILS_CU_
#include "cuda_utils.h"

#define DIM 128

 __device__ inline cuDoubleComplex operator* (const cuDoubleComplex & x,const cuDoubleComplex & y) {
    return cuCmul(x,y);
 }

  __device__ inline cuDoubleComplex operator+ (const cuDoubleComplex & x,const cuDoubleComplex & y) {
    return cuCadd(x,y);
 }

   __device__ inline cuDoubleComplex operator- (const cuDoubleComplex & x,const cuDoubleComplex & y) {
    return cuCsub(x,y);
 }

 __device__ inline cuDoubleComplex operator* (const double & a,const cuDoubleComplex & x) {
    return make_cuDoubleComplex (a*cuCreal(x), a*cuCimag(x));
 }

 __device__ inline cuDoubleComplex operator* (const cuDoubleComplex & x,const double & a) {
    return make_cuDoubleComplex (a*cuCreal(x), a*cuCimag(x));
 }

 __device__ inline cuDoubleComplex operator+ (const double & a,const cuDoubleComplex & x) {
    return make_cuDoubleComplex (a+cuCreal(x), cuCimag(x));
 }

 __device__ inline cuDoubleComplex operator+ (const cuDoubleComplex & x,const double & a) {
    return make_cuDoubleComplex (a+cuCreal(x), cuCimag(x));
 }
  __device__ inline double Norm_2(const cuDoubleComplex & x) {
    return (cuCreal(x)*cuCreal(x)) + (cuCimag(x)*cuCimag(x));
 }


__global__ void gpu_setValue( float* dev, float val, int len)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < len)
		dev[tid] = val;
}

__global__ void gpu_setValue( double* dev, double val, int len)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < len)
		dev[tid] = val;
}
__global__ void gpu_setValue( cuDoubleComplex* dev, cuDoubleComplex val, int len)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < len){
		dev[tid].x = val.x;
		dev[tid].y = val.y;
	}
}
__global__ void gpu_setValue( cuComplex* dev, cuComplex val, int len)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < len){
		dev[tid].x = val.x;
		dev[tid].y = val.y;
	}
}
__global__ void  gpu_interpolate_wf_C2F( cuDoubleComplex * coarse_psi, cuDoubleComplex* fine_psi, int *index, int len, double factor)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len)
	{
		int idx = index[tid];
		fine_psi[idx] = coarse_psi[tid] * factor;
	}
}

__global__ void  gpu_interpolate_wf_F2C( cuDoubleComplex * fine_psi, cuDoubleComplex* coarse_psi, int *index, int len, double factor)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len)
	{
		int idx = index[tid];
		coarse_psi[tid] = coarse_psi[tid] + fine_psi[idx] * factor;
	}
}
__global__ void gpu_laplacian ( cuDoubleComplex * psi, double * gkk, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len)
	{
		psi[tid] = psi[tid] * gkk[tid];
	}
}

__global__ void gpu_vtot( double* psi, double * gkk, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len)
	{
		psi[tid] = psi[tid] * gkk[tid];
	}
}


#if 0
float* cuda_malloc( float* ptr, size_t size)
{
  	printf("cudaMalloc float the pointer %d \n", size);
	CUDA_CALL( cudaMalloc( &ptr, size ) );
	return ptr;
}
double* cuda_malloc( double* ptr, size_t size)
{
  	printf("cudaMalloc double the pointer %d \n", size);
	CUDA_CALL( cudaMalloc( &ptr, size ) );
	return ptr;
}
cuComplex* cuda_malloc( cuComplex* ptr, size_t size)
{
  	printf("cudaMalloc complex the pointer %d \n", size);
	CUDA_CALL( cudaMalloc( &ptr, size ) );
	return ptr;
}
cuDoubleComplex* cuda_malloc( cuDoubleComplex* ptr, size_t size)
{
  	printf("cudaMalloc double complex the pointer %d \n", size);
	CUDA_CALL( cudaMalloc( &ptr, size ) );
	return ptr;
}
#endif


void cuda_free( void *ptr)
{
	CUDA_CALL( cudaFree(ptr) );
}

void cuda_memcpy_CPU2GPU( void *gpu, void * cpu, size_t size )
{
	CUDA_CALL( cudaMemcpy(gpu, cpu, size, cudaMemcpyHostToDevice ); );
	//std::flush(std::cout);
}

void cuda_memcpy_GPU2CPU( void *cpu, void * gpu, size_t size )
{
	CUDA_CALL( cudaMemcpy(cpu, gpu, size, cudaMemcpyDeviceToHost); );
}

void cuda_memcpy_GPU2GPU( void * dest, void * src, size_t size)
{
	CUDA_CALL( cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice); );
}

void cuda_setValue( float* dev, float val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	gpu_setValue<<<ndim, DIM>>>(dev, val, len);
}

void cuda_setValue( double* dev, double val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	gpu_setValue<<<ndim, DIM>>>(dev, val, len);
}
void cuda_setValue( cuDoubleComplex* dev, cuDoubleComplex val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	gpu_setValue<<<ndim, DIM>>>(dev, val, len);
}

void cuda_setValue( cuComplex* dev, cuComplex val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	gpu_setValue<<<ndim, DIM>>>(dev, val, len);
}

void cuda_interpolate_wf_C2F( cuDoubleComplex * coarse_psi, cuDoubleComplex * fine_psi, int * index, int len, double factor)
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_interpolate_wf_C2F<<< ndim, DIM>>> ( coarse_psi, fine_psi, index, len, factor);
}
void cuda_interpolate_wf_F2C( cuDoubleComplex * fine_psi, cuDoubleComplex * coarse_psi, int * index, int len, double factor)
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_interpolate_wf_F2C<<< ndim, DIM>>> ( fine_psi, coarse_psi, index, len, factor);
}

void *cuda_malloc( size_t size)
{
	void *ptr;
	CUDA_CALL( cudaMalloc( &ptr, size ) );
	return ptr;
}
void cuda_laplacian( cuDoubleComplex* psi, double * gkk, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_laplacian<<< ndim, DIM>>> ( psi, gkk, len);
	
}
void cuda_vtot( double* psi, double * vtot, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_vtot<<< ndim, DIM>>> ( psi, vtot, len);
} 

#endif
