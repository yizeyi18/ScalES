#ifndef  _CUDA_UTILS_CU_
#define  _CUDA_UTILS_CU_
#include "cuda_utils.h"

#define DIM 128
#define LDIM  256
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

__global__ void gpu_mapping_to_buf( double *buf, double * psi, int *index, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int x;
	if(tid < len)
	{
		x = index[tid];
		buf[x] = psi[tid];
	}
}

__global__ void gpu_mapping_from_buf( double *psi, double * buf, int *index, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int x;
	if(tid < len)
	{
		x = index[tid];
		psi[tid] = buf[x];
	}
}
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
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
__global__ void gpu_teter( cuDoubleComplex * psi, double * teter, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len)
	{
		psi[tid] = psi[tid] * teter[tid];
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
__global__ void gpu_update_psiUpdate( double *psiUpdate, double* NL, int * parts, int *index, double *atom_weight, double* weight)
{
	int start = parts[blockIdx.x];
	int end   = parts[blockIdx.x+1];
	int len   = end - start;
	int tid = threadIdx.x;
	double w = weight[blockIdx.x] * atom_weight[blockIdx.x];
	double s;

	while(tid < len)
	{
		int j = start + tid;
		int i = index[j];
		s = NL[j] * w;
		atomicAdd(&psiUpdate[i], s);
		tid += blockDim.x;
	}
}

template<unsigned int blockSize>
__global__ void gpu_cal_weight( double * psi, double * NL, int * parts, int * index, double * weight)
{
	//first get the starting and ending point and length
        __shared__ double sdata[DIM];
	int start = parts[blockIdx.x];
	int end   = parts[blockIdx.x+1];
	int len   = end - start;
	
	int tid   = threadIdx.x;
	double s = 0.0;
	while(tid < len)
	{
		int j = start + tid;
		int i = index[j];
		s += psi[i] * NL[j];
 		tid += blockDim.x;
	}

	sdata[threadIdx.x] =  s;
 	double mySum = s;
	__syncthreads();

	tid = threadIdx.x;
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();
	
	if ((blockSize >= 256) &&(tid < 128))
	{
	        sdata[tid] = mySum = mySum + sdata[tid + 128];
	}
	
	 __syncthreads();
	
	if ((blockSize >= 128) && (tid <  64))
	{
	   sdata[tid] = mySum = mySum + sdata[tid +  64];
	}
	
	__syncthreads();

#if (__CUDA_ARCH__ >= 300 )
	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize/2; offset > 0; offset /= 2)
		{
		    mySum += __shfl_down(mySum, offset);
		}
	}
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

	if( tid == 0) weight[blockIdx.x] = mySum;
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

void cuda_memory(void)
{
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	cudaMemGetInfo(&free_mem, &total_mem);
	cudaThreadSynchronize();
	printf("free  memory is: %zu MB\n", free_mem/1000000);
	printf("total memory is: %zu MB\n", total_mem/1000000);
	fflush(stdout);
} 

void cuda_calculate_nonlocal( double * psiUpdate, double * psi, double * NL, int * index, int * parts,  double * atom_weight, double * weight, int blocks)
{
	// two steps. 
        // 1. calculate the weight.
        gpu_cal_weight<DIM><<<blocks, DIM, DIM * sizeof(double) >>>( psi, NL, parts, index, weight);

        // 2. update the psiUpdate.
	gpu_update_psiUpdate<<<blocks, LDIM>>>( psiUpdate, NL, parts, index, atom_weight, weight);
}
void cuda_teter( cuDoubleComplex* psi, double * vtot, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_teter<<< ndim, DIM>>> ( psi, vtot, len);
}

void cuda_mapping_from_buf( double * psi, double * buf, int * index, int len )
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_mapping_from_buf<<< ndim, DIM>>>( psi, buf, index, len);	
}

void cuda_mapping_to_buf( double * buf, double * psi, int * index, int len )
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_mapping_to_buf<<< ndim, DIM>>>( buf, psi, index, len);	
}
#endif
