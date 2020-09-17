#ifndef  _CUDA_UTILS_CU_
#define  _CUDA_UTILS_CU_
#include "device_utils.h"
#include <sys/time.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;
#define DIM   128
#define LDIM  256
#define LEN   512

double * dev_vtot;
double * dev_gkkR2C;
int    * dev_idxFineGridR2C;
int    * dev_NLindex;
int    * dev_NLpart;
double * dev_NLvecFine;
double * dev_atom_weight;
double * dev_temp_weight;
cuDoubleComplex* dev_temp_weight_complex;
double * dev_TeterPrecond;

bool vtot_gpu_flag;
bool NL_gpu_flag;
bool teter_gpu_flag;
int totPart_gpu;


#ifdef _PROFILING_
double CPU2GPUTime = 0.0;
double GPU2CPUTime = 0.0;
double GPU2GPUTime = 0.0;
#endif

#define gpuErrchk(ans) {gpuAssert((ans),__FILE__,__LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if(code != cudaSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if(abort) exit(code);
	}
}
/*
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
*/
__global__ void gpu_X_Equal_AX_minus_X_eigVal(cuDoubleComplex* Xtemp, cuDoubleComplex* AX, cuDoubleComplex *X, double *eigen, int len ,int bandLen)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int bid = tid / bandLen;
	if(tid < len)
	{
		Xtemp[tid] = AX[tid] - X[tid] * eigen[bid];
	}
}

__global__ void gpu_X_Equal_AX_minus_X_eigVal(double* Xtemp, double *AX, double *X, double *eigen, int len ,int bandLen)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int bid = tid / bandLen;
	if(tid < len)
	{
		Xtemp[tid] = AX[tid] - X[tid] * eigen[bid];
	}
}
__global__ void gpu_batch_Scal( cuDoubleComplex *psi, double *vec, int bandLen, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int iband = tid / bandLen;
	if(tid < len)
	{
		double alpha = 1.0 / sqrt( vec[iband] );
		psi[tid] = psi[tid] * alpha;
	}
}

__global__ void gpu_batch_Scal( double *psi, double *vec, int bandLen, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int iband = tid / bandLen;
	if(tid < len)
	{
		double alpha = 1.0 / sqrt( vec[iband] );
		psi[tid] = psi[tid] * alpha;
	}
}
	template<unsigned int blockSize>
__global__ void gpu_energy( cuDoubleComplex * psi, double * energy, int len)
{
	__shared__ double sdata[DIM];
	int offset = blockIdx.x * len;
	int tid = threadIdx.x;
	double s = 0.0;
	cufftDoubleComplex px, py;

	while ( tid < len)
	{
		int index = tid + offset;
		px = psi[index];
		py.x = px.x;
		py.y = -px.y;
		s += (px*py).x;
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
        thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block()); 
	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize/2; offset > 0; offset /= 2)
		{
			//mySum += __shfl_down(mySum, offset);
                        mySum += tile32.shfl_down(mySum, offset);
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

	if( tid == 0) energy[blockIdx.x] = mySum;

}
	template<unsigned int blockSize>
__global__ void gpu_reduce( double * density, double * sum_den, int len)
{
	__shared__ double sdata[DIM];
	int offset = blockIdx.x * len;
	int tid = threadIdx.x;
	double s = 0.0;

	while ( tid < len)
	{
		int index = tid + offset;
		s += density[index];
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
        thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block()); 
	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize/2; offset > 0; offset /= 2)
		{
                        mySum += tile32.shfl_down(mySum, offset);
			//mySum += __shfl_down(mySum, offset);
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

	if( tid == 0) sum_den[blockIdx.x] = mySum;

}

	template<unsigned int blockSize>
__global__ void gpu_energy( double * psi, double * energy, int len)
{
	__shared__ double sdata[DIM];
	int offset = blockIdx.x * len;
	int tid = threadIdx.x;
	double s = 0.0;

	while ( tid < len)
	{
		int index = tid + offset;
		s += psi[index] * psi[index];
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
        thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block()); 
	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize/2; offset > 0; offset /= 2)
		{
			//mySum += __shfl_down(mySum, offset);
                        mySum += tile32.shfl_down(mySum, offset);
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

	if( tid == 0) energy[blockIdx.x] = mySum;

}
	template<class T> 
__global__ void gpu_mapping_to_buf( T* buf, T * psi, int *index, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int x;
	if(tid < len)
	{
		x = index[tid];
		buf[x] = psi[tid];
	}
}

	template<class T> 
__global__ void gpu_mapping_from_buf( T *psi, T * buf, int *index, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int x;
	if(tid < len)
	{
		x = index[tid];
		psi[tid] = buf[x];
	}
}
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
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
__global__ void gpu_teter( cuDoubleComplex * psi, cuDoubleComplex * teter, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len)
	{
		psi[tid] = psi[tid] * teter[tid];
	}
}
__global__ void gpu_teter( cuDoubleComplex * psi, double * teter, int len, int nbands)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int bid = blockIdx.y;
	long offset = bid * len + tid;
	if(tid < len)
	{
		psi[offset] = psi[offset] * teter[tid];
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
	template<class T>
__global__ void gpu_vtot( T* psi, cuDoubleComplex * gkk, int len, int nbands)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int bid = blockIdx.y;
	cuDoubleComplex vtot;
	long offset = bid * len + tid;
	if(tid < len)
	{
		vtot = gkk[tid];
		vtot.y = - vtot.y;
		psi[offset] = psi[offset] * vtot;
	}
}

	template<class T>
__global__ void gpu_vtot( T* psi, cuDoubleComplex * gkk, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	cuDoubleComplex vtot;
	if(tid < len)
	{
		vtot = gkk[tid];
		vtot.y = - vtot.y;
		psi[tid] = psi[tid] * vtot;
	}
}

	template<class T>
__global__ void gpu_vtot( T* psi, double * gkk, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len)
	{
		psi[tid] = psi[tid] * gkk[tid];
	}
}
__global__ void gpu_update_psiUpdate( cuDoubleComplex *psiUpdate, double* NL, int * parts, int *index, double *atom_weight, cuDoubleComplex* weight)
{
	int start = parts[blockIdx.x];
	int end   = parts[blockIdx.x+1];
	int len   = end - start;
	int tid = threadIdx.x;
	cuDoubleComplex w = weight[blockIdx.x] * atom_weight[blockIdx.x];
	cuDoubleComplex s;

	while(tid < len)
	{
		int j = start + tid;
		int i = index[j];
		s = NL[j] * w;
		atomicAdd(&psiUpdate[i].x, s.x);
		atomicAdd(&psiUpdate[i].y, s.y);
		tid += blockDim.x;
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
__global__ void gpu_cal_weight( cuDoubleComplex* psi, double * NL, int * parts, int * index, cuDoubleComplex * weight)
{
	//first get the starting and ending point and length
	__shared__ cuDoubleComplex sdata[DIM];
	int start = parts[blockIdx.x];
	int end   = parts[blockIdx.x+1];
	int len   = end - start;

	int tid   = threadIdx.x;
	cuDoubleComplex s = make_cuDoubleComplex (0.0, 0.0);
	while(tid < len)
	{
		int j = start + tid;
		int i = index[j];
		s = s + psi[i] * NL[j];
		tid += blockDim.x;
	}

	sdata[threadIdx.x] =  s;
	cuDoubleComplex mySum = s;
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

	if( tid == 0) weight[blockIdx.x] = mySum;
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
        thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block()); 
	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize/2; offset > 0; offset /= 2)
		{
			//mySum += __shfl_down(mySum, offset);
                        mySum += tile32.shfl_down(mySum, offset);
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
float* device_malloc( float* ptr, size_t size)
{
	printf("cudaMalloc float the pointer %d \n", size);
	device_CALL( cudaMalloc( &ptr, size ) );
	return ptr;
}
double* device_malloc( double* ptr, size_t size)
{
	printf("cudaMalloc double the pointer %d \n", size);
	CUDA_CALL( cudaMalloc( &ptr, size ) );
	return ptr;
}
cuComplex* device_malloc( cuComplex* ptr, size_t size)
{
	printf("cudaMalloc complex the pointer %d \n", size);
	CUDA_CALL( cudaMalloc( &ptr, size ) );
	return ptr;
}
cuDoubleComplex* device_malloc( cuDoubleComplex* ptr, size_t size)
{
	printf("cudaMalloc double complex the pointer %d \n", size);
	CUDA_CALL( cudaMalloc( &ptr, size ) );
	return ptr;
}
#endif

void reset_time()
{
#ifdef _PROFILING_
	CPU2GPUTime = 0.0;
	GPU2CPUTime = 0.0;
	GPU2GPUTime = 0.0;
#endif
}

void print_timing()
{
#ifdef _PROFILING_
	printf( " cudaMemcpy CPU2GPU %lf \n", CPU2GPUTime );
	printf( " cudaMemcpy GPU2CPU %lf \n", GPU2CPUTime );
	printf( " cudaMemcpy GPU2GPU %lf \n", GPU2GPUTime );
	printf( " cudaMemcpy Total time %lf \n", GPU2GPUTime + CPU2GPUTime + GPU2CPUTime);
	fflush(stdout);
#endif
}

void device_free( void *ptr)
{
	CUDA_CALL( cudaFree(ptr) );
}

void device_memcpy_Async_HOST2DEVICE( void *gpu, void * cpu, size_t size )
{
#ifdef _PROFILING_
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
#endif
	CUDA_CALL( cudaMemcpyAsync(gpu, cpu, size, cudaMemcpyHostToDevice, 0); );

#ifdef _PROFILING_
	gettimeofday(&end, NULL);
	double time = 1000000* (end.tv_sec - begin.tv_sec) + end.tv_usec - begin.tv_usec;
	time /= 1000000;
	CPU2GPUTime += time;
#endif
}

void device_memcpy_HOST2DEVICE( void *gpu, void * cpu, size_t size )
{
#ifdef _PROFILING_
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
#endif
	CUDA_CALL( cudaMemcpy(gpu, cpu, size, cudaMemcpyHostToDevice ); );
#ifdef _PROFILING_
	gettimeofday(&end, NULL);
	double time = 1000000* (end.tv_sec - begin.tv_sec) + end.tv_usec - begin.tv_usec;
	time /= 1000000;
	CPU2GPUTime += time;
#endif
}

void device_memcpy_DEVICE2HOST( void *cpu, void * gpu, size_t size )
{
#ifdef _PROFILING_
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
#endif
	CUDA_CALL( cudaMemcpy(cpu, gpu, size, cudaMemcpyDeviceToHost); );
#ifdef _PROFILING_
	gettimeofday(&end, NULL);
	double time = 1000000* (end.tv_sec - begin.tv_sec) + end.tv_usec - begin.tv_usec;
	time /= 1000000;
	GPU2CPUTime += time;
#endif
}

void device_memcpy_DEVICE2DEVICE( void * dest, void * src, size_t size)
{
#ifdef _PROFILING_
	assert(cudaThreadSynchronize() == cudaSuccess );
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
#endif
	CUDA_CALL( cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice); );
#ifdef _PROFILING_
	assert(cudaThreadSynchronize() == cudaSuccess );
	gettimeofday(&end, NULL);
	double time = 1000000* (end.tv_sec - begin.tv_sec) + end.tv_usec - begin.tv_usec;
	time /= 1000000;
	GPU2GPUTime += time;
#endif
}

void device_setValue( float* dev, float val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	gpu_setValue<<<ndim, DIM>>>(dev, val, len);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_setValue( double* dev, double val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	gpu_setValue<<<ndim, DIM>>>(dev, val, len);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}
void device_setValue( cuDoubleComplex* dev, cuDoubleComplex val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	gpu_setValue<<<ndim, DIM>>>(dev, val, len);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_setValue( cuComplex* dev, cuComplex val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	gpu_setValue<<<ndim, DIM>>>(dev, val, len);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_interpolate_wf_C2F( cuDoubleComplex * coarse_psi, cuDoubleComplex * fine_psi, int * index, int len, double factor)
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_interpolate_wf_C2F<<< ndim, DIM>>> ( coarse_psi, fine_psi, index, len, factor);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}
void device_interpolate_wf_F2C( cuDoubleComplex * fine_psi, cuDoubleComplex * coarse_psi, int * index, int len, double factor)
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_interpolate_wf_F2C<<< ndim, DIM>>> ( fine_psi, coarse_psi, index, len, factor);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void *device_malloc( size_t size)
{
	void *ptr;
	CUDA_CALL( cudaMalloc( &ptr, size ) );
	return ptr;
}
void device_laplacian( cuDoubleComplex* psi, double * gkk, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_laplacian<<< ndim, DIM>>> ( psi, gkk, len);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_vtot( cuDoubleComplex * psi, cuDoubleComplex* vtot, int len, int nbands)
{
	int ndim = (len + DIM - 1) / DIM;
	dim3 dim(ndim, nbands);
	gpu_vtot<<< dim, DIM>>> ( psi, vtot, len, nbands);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_vtot( cuDoubleComplex * psi, cuDoubleComplex* vtot, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_vtot<cuDoubleComplex><<< ndim, DIM>>> ( psi, vtot, len);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_vtot( cuDoubleComplex * psi, double * vtot, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_vtot<cuDoubleComplex><<< ndim, DIM>>> ( psi, vtot, len);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_vtot( double* psi, double * vtot, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_vtot<double><<< ndim, DIM>>> ( psi, vtot, len);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_memory(void)
{
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	cudaMemGetInfo(&free_mem, &total_mem);
	assert(cudaThreadSynchronize() == cudaSuccess );
	printf("free  memory is: %zu MB\n", free_mem/1000000);
	printf("total memory is: %zu MB\n", total_mem/1000000);
	fflush(stdout);
} 
void device_calculate_nonlocal( cuDoubleComplex* psiUpdate, cuDoubleComplex* psi, double * NL, int * index, int * parts,  double * atom_weight, cuDoubleComplex* weight, int blocks)
{
	// two steps. 
	// 1. calculate the weight.
	gpu_cal_weight<DIM><<<blocks, DIM, DIM * sizeof(double) >>>( psi, NL, parts, index, weight);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif

	// 2. update the psiUpdate.
	gpu_update_psiUpdate<<<blocks, LDIM>>>( psiUpdate, NL, parts, index, atom_weight, weight);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_calculate_nonlocal( double * psiUpdate, double * psi, double * NL, int * index, int * parts,  double * atom_weight, double * weight, int blocks)
{
	// two steps. 
	// 1. calculate the weight.
	gpu_cal_weight<DIM><<<blocks, DIM, DIM * sizeof(double) >>>( psi, NL, parts, index, weight);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif

	// 2. update the psiUpdate.
	gpu_update_psiUpdate<<<blocks, LDIM>>>( psiUpdate, NL, parts, index, atom_weight, weight);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}
void device_teter( cuDoubleComplex* psi, cuDoubleComplex* vtot, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_teter<<< ndim, DIM>>> ( psi, vtot, len);
#ifdef SYNC
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}
void device_teter( cuDoubleComplex* psi, double * vtot, int len, int nbands)
{
	int ndim = (len + DIM - 1) / DIM;
	dim3 dim(ndim, nbands);
	gpu_teter<<< dim, DIM>>> ( psi, vtot, len, nbands);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_teter( cuDoubleComplex* psi, double * vtot, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_teter<<< ndim, DIM>>> ( psi, vtot, len);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_mapping_from_buf( cuDoubleComplex * psi, cuDoubleComplex * buf, int * index, int len )
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_mapping_from_buf<cuDoubleComplex><<< ndim, DIM>>>( psi, buf, index, len);	
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_mapping_from_buf( double * psi, double * buf, int * index, int len )
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_mapping_from_buf<double><<< ndim, DIM>>>( psi, buf, index, len);	
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}
void device_mapping_to_buf( cuDoubleComplex * buf, cuDoubleComplex * psi, int * index, int len )
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_mapping_to_buf<cuDoubleComplex><<< ndim, DIM>>>( buf, psi, index, len);	
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_mapping_to_buf( double * buf, double * psi, int * index, int len )
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_mapping_to_buf<double><<< ndim, DIM>>>( buf, psi, index, len);	
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}
void device_calculate_Energy( cuDoubleComplex * psi, double * energy, int nbands, int bandLen)
{
	// calculate  nbands psi Energy. 
	gpu_energy<DIM><<<nbands, DIM, DIM*sizeof(double)>>>( psi, energy, bandLen);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}
void device_reduce( double * density, double * sum, int nbands, int bandLen)
{
	gpu_reduce<DIM><<<nbands, DIM, DIM*sizeof(double)>>>( density, sum, bandLen);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_calculate_Energy( double * psi, double * energy, int nbands, int bandLen)
{
	// calculate  nbands psi Energy. 
	gpu_energy<DIM><<<nbands, DIM, DIM*sizeof(double)>>>( psi, energy, bandLen);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}
void device_batch_Scal( cuDoubleComplex * psi, double* vec, int nband, int bandLen)
{
	int ndim = ( nband * bandLen + DIM - 1) / DIM;
	int len = nband * bandLen;
	gpu_batch_Scal<<< ndim, DIM >>> ( psi, vec, bandLen, len);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_batch_Scal( double * psi, double * vec, int nband, int bandLen)
{
	int ndim = ( nband * bandLen + DIM - 1) / DIM;
	int len = nband * bandLen;
	gpu_batch_Scal<<< ndim, DIM >>> ( psi, vec, bandLen, len);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}
void device_X_Equal_AX_minus_X_eigVal( cuDoubleComplex* Xtemp, cuDoubleComplex * AX, cuDoubleComplex * X, double * eigen, int nbands, int bandLen)
{
	int ndim = ( nbands * bandLen + DIM - 1 ) / DIM;
	int len = nbands * bandLen;
	gpu_X_Equal_AX_minus_X_eigVal<<< ndim, DIM >>>( Xtemp, AX, X, eigen, len, bandLen );
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_X_Equal_AX_minus_X_eigVal( double * Xtemp, double * AX, double * X, double * eigen, int nbands, int bandLen)
{
	int ndim = ( nbands * bandLen + DIM - 1 ) / DIM;
	int len = nbands * bandLen;
	gpu_X_Equal_AX_minus_X_eigVal<<< ndim, DIM >>>( Xtemp, AX, X, eigen, len, bandLen );
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_init_vtot()
{
	dev_vtot           = NULL;
	dev_gkkR2C         = NULL;
	dev_idxFineGridR2C = NULL;
	dev_NLindex        = NULL;
	dev_NLpart         = NULL;
	dev_NLvecFine      = NULL;
	dev_atom_weight    = NULL;
	dev_temp_weight    = NULL;
	dev_TeterPrecond   = NULL;
	vtot_gpu_flag      = false;
	NL_gpu_flag        = false;
	teter_gpu_flag     = false;
}
void device_clean_vtot()
{
	device_free(dev_vtot);
	device_free(dev_gkkR2C);
	device_free(dev_idxFineGridR2C);
	device_free(dev_NLindex);
	device_free(dev_NLpart);
	device_free(dev_NLvecFine);
	device_free(dev_atom_weight);
	device_free(dev_temp_weight);
	device_free(dev_temp_weight_complex);
	device_free(dev_TeterPrecond);
}
void device_reset_vtot_flag()
{
	device_free(dev_vtot);
	dev_vtot       = NULL;
	vtot_gpu_flag  = false;
}
void device_reset_nonlocal_flag()
{
	// Note, the gkk is malloc and freed with the nonlocal part.
	// since GKK will not change until atom moves. 
	device_free(dev_NLvecFine);
	device_free(dev_NLpart);
	device_free(dev_NLindex);
	device_free(dev_atom_weight);
	device_free(dev_temp_weight_complex);
	device_free(dev_gkkR2C);
	device_free(dev_idxFineGridR2C);

	dev_gkkR2C         = NULL;
	dev_idxFineGridR2C = NULL;
	dev_NLindex        = NULL;
	dev_NLpart         = NULL;
	dev_NLvecFine      = NULL;
	dev_atom_weight    = NULL;
	dev_temp_weight    = NULL;

	NL_gpu_flag = false;
}

__global__ void gpu_matrix_add( double * A, double * B, int length )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < length )
	{
		A[tid] = A[tid] + B[tid];
	}
}

void device_DMatrix_Add( double * A , double * B, int m, int n)
{
	// Matrix A and B are double dimension m,n; A = A + B
	int ndim = (m * n + DIM - 1) / DIM;
	int length = m*n;
	gpu_matrix_add<<< ndim, DIM >>>( A, B, length);

#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}
__global__ void gpu_alpha_X_plus_beta_Y_multiply_Z( double * X, double alpha, double * Y, double beta, double * Z, int length)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < length)
	{
		X[tid] = alpha * X[tid] + beta * Y[tid] * Z[tid];
	}
}
__global__ void gpu_alpha_X_plus_beta_Y_multiply_Z( cuDoubleComplex * X, double alpha, cuDoubleComplex * Y, double beta, cuDoubleComplex* Z, int length, int nbands)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int bid = blockIdx.y;
	long offset = bid * length + tid;
	if(tid < length)
	{
		X[offset] = alpha * X[offset] + beta * Y[offset] * Z[tid];
	}
}

__global__ void gpu_alpha_X_plus_beta_Y_multiply_Z( cuDoubleComplex * X, double alpha, cuDoubleComplex * Y, double beta, cuDoubleComplex* Z, int length)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < length)
	{
		X[tid] = alpha * X[tid] + beta * Y[tid] * Z[tid];
	}
}
void device_Axpyz( cuDoubleComplex * X, double alpha, cuDoubleComplex * Y, double beta, cuDoubleComplex * Z, int length, int nbands)
{
	int ndim = ( length + DIM - 1) / DIM ;
	dim3 dim( ndim, nbands);
	gpu_alpha_X_plus_beta_Y_multiply_Z <<< dim, DIM >>> (X, alpha, Y, beta, Z, length, nbands);

#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_Axpyz( cuDoubleComplex * X, double alpha, cuDoubleComplex * Y, double beta, cuDoubleComplex * Z, int length)
{
	int ndim = ( length + DIM - 1) / DIM ;
	gpu_alpha_X_plus_beta_Y_multiply_Z <<< ndim, DIM >>> (X, alpha, Y, beta, Z, length);

#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_Axpyz( double * X, double alpha, double * Y, double beta, double * Z, int length)
{
	int ndim = ( length + DIM - 1) / DIM ;
	gpu_alpha_X_plus_beta_Y_multiply_Z <<< ndim, DIM >>> (X, alpha, Y, beta, Z, length);

#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

__global__ void gpu_cal_sendk( int *sendk, int * senddisps, int widthLocal, int height, int heightBlockSize, int mpisize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < height* widthLocal)
	{
		int i = tid % height;
		int j = tid / height;

		if(height % mpisize == 0){
			sendk[tid] = senddisps[i/heightBlockSize] + j * heightBlockSize + i % heightBlockSize;
		}
		else{
			if( i < ((height%mpisize) * (heightBlockSize + 1)) ) {
				sendk[tid] = senddisps[i/(heightBlockSize + 1)] + j * ( heightBlockSize + 1) + i % ( heightBlockSize + 1);
			}
			else{
				sendk[tid] = senddisps[(height % mpisize) + (i-(height % mpisize)*(heightBlockSize+1))/heightBlockSize] 
					+ j * heightBlockSize + (i-(height % mpisize)*(heightBlockSize+1)) % heightBlockSize;
			}
		}
	}
}

__global__ void gpu_cal_recvk( int *recvk, int * recvdisps, int width, int heightLocal, int mpisize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < heightLocal* width)
	{
		int i = tid % heightLocal;
		int j = tid / heightLocal;

		recvk[tid] = recvdisps[j%mpisize] + ( j/mpisize) * heightLocal + i;
	}
}

void device_cal_sendk( int * sendk, int * senddispl, int widthLocal, int height, int heightBlockSize, int mpisize)
{
	int total = widthLocal * height;
	int dim = (total + LEN - 1) / LEN;

	gpu_cal_sendk<<< dim, LEN>>> ( sendk, senddispl, widthLocal, height, heightBlockSize, mpisize );
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_cal_recvk( int * recvk, int * recvdisp, int width, int heightLocal, int mpisize)
{
	int total = width * heightLocal;
	int dim = ( total + LEN - 1 ) / LEN;

	gpu_cal_recvk<<< dim, LEN>>> ( recvk, recvdisp, width, heightLocal, mpisize );
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}
	template < class T >
__global__ void gpu_hadamard_product ( T* dev_A, T* dev_B, T * dev_result, int length)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < length)
	{
		dev_result[tid] = dev_A[tid] * dev_B[tid];
	}
}

void device_hadamard_product( double * in1, double * in2, double * out, int length)
{
	int dim = ( length + LEN - 1 ) / LEN;
	gpu_hadamard_product<double> <<< dim, LEN >>> ( in1, in2, out, length );

#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}
	template <class T>
__global__ void gpu_set_vector( T* out, T* in , int length)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < length)
	{
		out[tid] = in[tid];
	}
}
void device_set_vector( cuComplex* out, cuComplex *in, int length)
{
	int dim = (length + LEN - 1) / LEN;
	gpu_set_vector< cuComplex> <<< dim, LEN >>>( out, in, length);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}


void device_set_vector( cuDoubleComplex* out, cuDoubleComplex *in, int length)
{
	int dim = (length + LEN - 1) / LEN;
	gpu_set_vector< cuDoubleComplex> <<< dim, LEN >>>( out, in, length);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_set_vector( double * out, double *in, int length)
{
	int dim = (length + LEN - 1) / LEN;
	gpu_set_vector< double> <<< dim, LEN >>>( out, in, length);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

__global__ void gpu_tddft_prec( cuDoubleComplex * prec, cuDoubleComplex * gkk, cuDoubleComplex trace, cuDoubleComplex factor, double width, int ntot)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < ntot)
	{
		//prec[tid] = 1.0 / (1.0 + factor*( gkk[tid] - trace/width));
	}
}

void device_tddft_prec( cuDoubleComplex * prec, cuDoubleComplex * gkk, cuDoubleComplex trace, cuDoubleComplex factor, int width, int ntot)
{
	int dim = (ntot + LEN - 1) /LEN;
	gpu_tddft_prec<<< dim, LEN>>>( prec, gkk, trace, factor, (double) width, ntot);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

	template <class T>
__global__ void gpu_compress( double *source, T* dest, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len)
	{
		dest[tid] = source[tid];
	}
}

void device_compress_d2f( float *out, double *in, int length)
{
	int dim = ( length + LEN - 1) / LEN;
	gpu_compress<float> <<< dim, LEN>>> ( in, out, length);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

	template <class T>
__global__ void gpu_decompress( T *source, double* dest, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < len)
	{
		dest[tid] = source[tid];
	}
}

void device_decompress_f2d( double *out, float *in, int length)
{
	int dim = ( length + LEN - 1) / LEN;
	gpu_decompress<float> <<< dim, LEN>>> ( in, out, length);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}
__global__ void gpu_XTX ( cuDoubleComplex *X, double * Y, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	cuDoubleComplex x, y;
	if(tid < len)
	{
		x = X[tid];
		y = x;
		y.y = - x.y;
		Y[tid] = (x * y).x;
	}
}
void device_XTX( cuDoubleComplex * X, double * Y, int length)
{
	int dim = ( length + LEN - 1) / LEN;
	gpu_XTX<<< dim, LEN>>>( X, Y, length);
#ifdef SYNC 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
#endif
}

void device_sync()
{
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaThreadSynchronize() == cudaSuccess );
}
#endif
