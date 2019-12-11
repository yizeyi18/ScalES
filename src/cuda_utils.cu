#include <hip/hip_runtime.h>
#ifndef  _CUDA_UTILS_CU_
#define  _CUDA_UTILS_CU_
#include "cuda_utils.h"
//#include "hip_complex.h"
#define DIM   128
#define LDIM  256
#define LEN   512
typedef  hipDoubleComplex cuDoubleComplex ;
typedef  hipFloatComplex cuComplex;
double * dev_vtot;
double * dev_gkkR2C;
int    * dev_idxFineGridR2C;
int    * dev_NLindex;
int    * dev_NLpart;
double * dev_NLvecFine;
double * dev_atom_weight;
double * dev_temp_weight;
double * dev_TeterPrecond;

bool vtot_gpu_flag;
bool NL_gpu_flag;
bool teter_gpu_flag;
int totPart_gpu;

#define gpuErrchk(ans) {gpuAssert((ans),__FILE__,__LINE__);}
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
	if(code != hipSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
		if(abort) exit(code);
	}
}
/*
__device__ inline cuDoubleComplex operator* (const cuDoubleComplex & x,const cuDoubleComplex & y) {
	return hipCmul(x,y);
}

__device__ inline cuDoubleComplex operator+ (const cuDoubleComplex & x,const cuDoubleComplex & y) {
	return hipCadd(x,y);
}

__device__ inline cuDoubleComplex operator- (const cuDoubleComplex & x,const cuDoubleComplex & y) {
	return hipCsub(x,y);
}

__device__ inline cuDoubleComplex operator* (const double & a,const cuDoubleComplex & x) {
	return make_hipDoubleComplex (a*hipCreal(x), a*hipCimag(x));
}

__device__ inline cuDoubleComplex operator* (const cuDoubleComplex & x,const double & a) {
	return make_hipDoubleComplex (a*hipCreal(x), a*hipCimag(x));
}

__device__ inline cuDoubleComplex operator+ (const double & a,const cuDoubleComplex & x) {
	return make_hipDoubleComplex (a+hipCreal(x), hipCimag(x));
}

__device__ inline cuDoubleComplex operator+ (const cuDoubleComplex & x,const double & a) {
	return make_hipDoubleComplex (a+hipCreal(x), hipCimag(x));
}
*/
__device__ inline double Norm_2(const cuDoubleComplex & x) {
	return (hipCreal(x)*hipCreal(x)) + (hipCimag(x)*hipCimag(x));
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
	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize/2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down_sync(0xFFFFFFFF, mySum, offset);
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
//#if __CUDA_ARCH__ < 600
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
			mySum += __shfl_down_sync(0xFFFFFFFF, mySum, offset);
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
	printf("hipMalloc float the pointer %d \n", size);
	CUDA_CALL( hipMalloc( &ptr, size ) );
	return ptr;
}
double* cuda_malloc( double* ptr, size_t size)
{
	printf("hipMalloc double the pointer %d \n", size);
	CUDA_CALL( hipMalloc( &ptr, size ) );
	return ptr;
}
cuComplex* cuda_malloc( cuComplex* ptr, size_t size)
{
	printf("hipMalloc complex the pointer %d \n", size);
	CUDA_CALL( hipMalloc( &ptr, size ) );
	return ptr;
}
cuDoubleComplex* cuda_malloc( cuDoubleComplex* ptr, size_t size)
{
	printf("hipMalloc double complex the pointer %d \n", size);
	CUDA_CALL( hipMalloc( &ptr, size ) );
	return ptr;
}
#endif


void cuda_free( void *ptr)
{
	//CUDA_CALL( hipFree(ptr) );
        hipFree(ptr);
}

void cuda_memcpy_CPU2GPU( void *gpu, void * cpu, size_t size )
{
	//CUDA_CALL( hipMemcpy(gpu, cpu, size, hipMemcpyHostToDevice ); );
        hipMemcpy(gpu, cpu, size, hipMemcpyHostToDevice );
	//std::flush(std::cout);
}

void cuda_memcpy_GPU2CPU( void *cpu, void * gpu, size_t size )
{
	//CUDA_CALL( hipMemcpy(cpu, gpu, size, hipMemcpyDeviceToHost); );
        hipMemcpy(cpu, gpu, size, hipMemcpyDeviceToHost); 
}

void cuda_memcpy_GPU2GPU( void * dest, void * src, size_t size)
{
	//CUDA_CALL( hipMemcpy(dest, src, size, hipMemcpyDeviceToDevice); );
        hipMemcpy(dest, src, size, hipMemcpyDeviceToDevice);
}

void cuda_setValue( float* dev, float val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	hipLaunchKernelGGL((gpu_setValue), dim3(ndim), dim3(DIM), 0, 0, dev, val, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_setValue( double* dev, double val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	hipLaunchKernelGGL((gpu_setValue), dim3(ndim), dim3(DIM), 0, 0, dev, val, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
void cuda_setValue( cuDoubleComplex* dev, cuDoubleComplex val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	hipLaunchKernelGGL((gpu_setValue), dim3(ndim), dim3(DIM), 0, 0, dev, val, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_setValue( cuComplex* dev, cuComplex val, int len )
{
	int ndim = len / DIM;
	if(len % DIM) ndim++;
	hipLaunchKernelGGL((gpu_setValue), dim3(ndim), dim3(DIM), 0, 0, dev, val, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_interpolate_wf_C2F( cuDoubleComplex * coarse_psi, cuDoubleComplex * fine_psi, int * index, int len, double factor)
{
	int ndim = (len + DIM - 1) / DIM;
	hipLaunchKernelGGL((gpu_interpolate_wf_C2F), dim3(ndim), dim3(DIM), 0, 0,  coarse_psi, fine_psi, index, len, factor);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
void cuda_interpolate_wf_F2C( cuDoubleComplex * fine_psi, cuDoubleComplex * coarse_psi, int * index, int len, double factor)
{
	int ndim = (len + DIM - 1) / DIM;
	hipLaunchKernelGGL((gpu_interpolate_wf_F2C), dim3(ndim), dim3(DIM), 0, 0,  fine_psi, coarse_psi, index, len, factor);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void *cuda_malloc( size_t size)
{
	void *ptr;
	//CUDA_CALL( hipMalloc( &ptr, size ) );
        hipMalloc( &ptr, size );
	return ptr;
}
void cuda_laplacian( cuDoubleComplex* psi, double * gkk, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	hipLaunchKernelGGL((gpu_laplacian), dim3(ndim), dim3(DIM), 0, 0,  psi, gkk, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
void cuda_vtot( double* psi, double * vtot, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	hipLaunchKernelGGL((gpu_vtot), dim3(ndim), dim3(DIM), 0, 0,  psi, vtot, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_memory(void)
{
	size_t free_mem, total_mem;
	hipMemGetInfo(&free_mem, &total_mem);
	hipMemGetInfo(&free_mem, &total_mem);
	assert(hipDeviceSynchronize() == hipSuccess );
	printf("free  memory is: %zu MB\n", free_mem/1000000);
	printf("total memory is: %zu MB\n", total_mem/1000000);
	fflush(stdout);
} 

void cuda_calculate_nonlocal( double * psiUpdate, double * psi, double * NL, int * index, int * parts,  double * atom_weight, double * weight, int blocks)
{
	// two steps. 
	// 1. calculate the weight.
	hipLaunchKernelGGL((gpu_cal_weight<DIM>), dim3(blocks), dim3(DIM), DIM * sizeof(double) , 0,  psi, NL, parts, index, weight);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif

	// 2. update the psiUpdate.
	hipLaunchKernelGGL((gpu_update_psiUpdate), dim3(blocks), dim3(LDIM), 0, 0,  psiUpdate, NL, parts, index, atom_weight, weight);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
void cuda_teter( cuDoubleComplex* psi, double * vtot, int len)
{
	int ndim = (len + DIM - 1) / DIM;
	hipLaunchKernelGGL((gpu_teter), dim3(ndim), dim3(DIM), 0, 0,  psi, vtot, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_mapping_from_buf( double * psi, double * buf, int * index, int len )
{
	int ndim = (len + DIM - 1) / DIM;
	hipLaunchKernelGGL((gpu_mapping_from_buf), dim3(ndim), dim3(DIM), 0, 0,  psi, buf, index, len);	
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_mapping_to_buf( double * buf, double * psi, int * index, int len )
{
	int ndim = (len + DIM - 1) / DIM;
	hipLaunchKernelGGL((gpu_mapping_to_buf), dim3(ndim), dim3(DIM), 0, 0,  buf, psi, index, len);	
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
void cuda_calculate_Energy( double * psi, double * energy, int nbands, int bandLen)
{
	// calculate  nbands psi Energy. 
	hipLaunchKernelGGL((gpu_energy<DIM>), dim3(nbands), dim3(DIM), DIM*sizeof(double), 0,  psi, energy, bandLen);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_batch_Scal( double * psi, double * vec, int nband, int bandLen)
{
	int ndim = ( nband * bandLen + DIM - 1) / DIM;
	int len = nband * bandLen;
	hipLaunchKernelGGL((gpu_batch_Scal), dim3(ndim), dim3(DIM ), 0, 0,  psi, vec, bandLen, len);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}
void cu_X_Equal_AX_minus_X_eigVal( double * Xtemp, double * AX, double * X, double * eigen, int nbands, int bandLen)
{
	int ndim = ( nbands * bandLen + DIM - 1 ) / DIM;
	int len = nbands * bandLen;
	hipLaunchKernelGGL((gpu_X_Equal_AX_minus_X_eigVal), dim3(ndim), dim3(DIM ), 0, 0,  Xtemp, AX, X, eigen, len, bandLen );
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_init_vtot()
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
void cuda_clean_vtot()
{
	cuda_free(dev_vtot);
	cuda_free(dev_gkkR2C);
	cuda_free(dev_idxFineGridR2C);
	cuda_free(dev_NLindex);
	cuda_free(dev_NLpart);
	cuda_free(dev_NLvecFine);
	cuda_free(dev_atom_weight);
	cuda_free(dev_temp_weight);
	cuda_free(dev_TeterPrecond);
}
void cuda_set_vtot_flag()
{
	vtot_gpu_flag  = false;
}

__global__ void gpu_matrix_add( double * A, double * B, int length )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < length )
	{
		A[tid] = A[tid] + B[tid];
	}
}

void cuda_DMatrix_Add( double * A , double * B, int m, int n)
{
	// Matrix A and B are double dimension m,n; A = A + B
	int ndim = (m * n + DIM - 1) / DIM;
	int length = m*n;
	hipLaunchKernelGGL((gpu_matrix_add), dim3(ndim), dim3(DIM ), 0, 0,  A, B, length);

#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
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
void cuda_Axpyz( double * X, double alpha, double * Y, double beta, double * Z, int length)
{
	int ndim = ( length + DIM - 1) / DIM ;
	hipLaunchKernelGGL((gpu_alpha_X_plus_beta_Y_multiply_Z), dim3(ndim), dim3(DIM ), 0, 0, X, alpha, Y, beta, Z, length);

#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
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

void cuda_cal_sendk( int * sendk, int * senddispl, int widthLocal, int height, int heightBlockSize, int mpisize)
{
	int total = widthLocal * height;
	int dim = (total + LEN - 1) / LEN;
	
       	hipLaunchKernelGGL((gpu_cal_sendk), dim3(dim), dim3(LEN), 0, 0,  sendk, senddispl, widthLocal, height, heightBlockSize, mpisize );
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_cal_recvk( int * recvk, int * recvdisp, int width, int heightLocal, int mpisize)
{
	int total = width * heightLocal;
	int dim = ( total + LEN - 1 ) / LEN;
	
	hipLaunchKernelGGL((gpu_cal_recvk), dim3(dim), dim3(LEN), 0, 0,  recvk, recvdisp, width, heightLocal, mpisize );
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
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

void cuda_hadamard_product( double * in1, double * in2, double * out, int length)
{
	int dim = ( length + LEN - 1 ) / LEN;
	hipLaunchKernelGGL((gpu_hadamard_product<double>), dim3(dim), dim3(LEN ), 0, 0,  in1, in2, out, length );

#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
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

void cuda_set_vector( double * out, double *in, int length)
{
	int dim = (length + LEN - 1) / LEN;
	hipLaunchKernelGGL((gpu_set_vector< double>), dim3(dim), dim3(LEN ), 0, 0,  out, in, length);
#ifdef SYNC 
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
#endif
}

void cuda_sync()
{
	gpuErrchk(hipPeekAtLastError());
	gpuErrchk(hipDeviceSynchronize());
	assert(hipDeviceSynchronize() == hipSuccess );
}
#endif
