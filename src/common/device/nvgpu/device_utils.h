//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

/** \file cuda_utils.h
 * \brief Definitions of persistent cuda pointers. 
 *
 * Each type internally the same, just typed differently.
 * Remember to also include cuda_utils.cu, which 
 * contains manipulation functions.
 */
#ifndef _CUDA_UTILS_
#define _CUDA_UTILS_
#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "cuComplex.h"
#define NSTREAM 1
#if 0
#define HOST2DEVICE cudaMemcpyHostToDevice 
#define DEVICE2HOST cudaMemcpyDeviceToHost 
#define DEVICE2DEVICE cudaMemcpyDeviceToDevice 
#endif 

#ifdef _PROFILING_
extern double CPU2GPUTime ;
extern double GPU2CPUTime ;
extern double GPU2GPUTime ;
#endif
//Macros for n-dimensional array access (column major format)
#define DIM2(x, y, xdim)                                       ((y * xdim) + x)
#define DIM3(x, y, z, xdim, ydim)                              ((((z * ydim) + y) * xdim) + x)
#define DIM4(x1, x2, x3, x4, x1dim, x2dim, x3dim)              ((((((x4 * x3dim) + x3) * x2dim) + x2) * x1dim) + x2)
#define DIM5(x1, x2, x3, x4, x5, x1dim, x2dim, x3dim, x4dim)   ((((((((x5 * x4dim) + x4) * x3dim) + x3) * x2dim) + x2) * x1dim) + x1)

#define CUDA_CALL(function) {\
cudaError_t err = function; \
if (err != cudaSuccess) \
  fprintf(stderr, "CURROR [%s,%d]: %s \n", \
  __FILE__,  __LINE__, cudaGetErrorString(err)); \
}

#define CUDA_FFT_CALL(function) {\
cufftResult err = function; \
if (err != CUFFT_SUCCESS) \
  fprintf(stderr, "CURROR [%s,%d]: %d \n", \
  __FILE__,  __LINE__, err); \
}

#if 0
/** N-dimensional double pointer */
typedef struct ndim_double {
  unsigned size;
  unsigned ndims;
  unsigned *dim;     //GPU
  unsigned lead_dim_size;
  double *ptr;      //current position, GPU
  double *start_pt;  //Starting point for access, GPU
} *ndim_double_ptr;

/** N dimensional cuDoubleComplex ptr */
typedef struct ndim_complex {
  unsigned size;
  unsigned ndims;
  unsigned *dim;
  unsigned lead_dim_size;
  cuDoubleComplex *ptr;        
  cuDoubleComplex *start_pt;
} *ndim_complex_ptr;

/**  int ptr */
typedef struct ndim_int {
  unsigned size;
  unsigned ndims;
  unsigned *dim;
  unsigned lead_dim_size;
  int *ptr;
  int *start_pt;
} *ndim_int_ptr;
#endif
/** void pointer, used mostly for casting */
//typedef struct void_p{
//  unsigned int size;
//  void* ptr;
//} void_p;
//
///** double pointer */
//typedef struct double_p{
//  unsigned int size;
//  double* ptr;
//} double_p;
//
///** float pointer */
//typedef struct float_p{
//  unsigned int size;
//  float* ptr;
//} float_p;
//
///** int pointer */
//typedef struct int_p{
//  unsigned int size;
//  int* ptr;
//} int_p;
//
///** Double Complex pointer (foo.x is real, foo.y is imag) */
//typedef struct cuDoubleComplex_p{
//  unsigned int size;
//  cuDoubleComplex* ptr;
//} cuDoubleComplex_p;
//
///** Complex pointer (foo.x is real, foo.y is imag) */
//typedef struct cuComplex_p{
//  unsigned int size;
//  cuComplex* ptr;
//} cuComplex_p;
//
//extern "C"{
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

extern double * dev_vtot;
extern double * dev_gkkR2C;
extern int    * dev_idxFineGridR2C;

extern int    * dev_NLindex;
extern int    * dev_NLpart;
extern double * dev_NLvecFine;
extern double * dev_atom_weight;
extern double * dev_temp_weight;
extern cuDoubleComplex * dev_temp_weight_complex;
extern double * dev_TeterPrecond;
extern int totPart_gpu;

extern bool vtot_gpu_flag;
extern bool NL_gpu_flag;
extern bool teter_gpu_flag;

void device_setValue( cuComplex* dev, cuComplex val, int len );
void device_setValue( cuDoubleComplex* dev, cuDoubleComplex val, int len );
void device_setValue( double* dev, double val, int len );
void device_setValue( float* dev, float val, int len );
void device_memcpy_DEVICE2DEVICE( void * dest, void * src, size_t size);
void device_memcpy_DEVICE2HOST( void *cpu, void * gpu, size_t size );
void device_memcpy_HOST2DEVICE( void *gpu, void * cpu, size_t size );
void device_free( void *ptr);
void *device_malloc( size_t size);
void device_interpolate_wf_C2F( cuDoubleComplex * coarse_psi, cuDoubleComplex * fine_psi, int * index, int len, double factor);
void device_interpolate_wf_F2C( cuDoubleComplex * fine_psi, cuDoubleComplex * coarse_psi, int * index, int len, double factor);
void device_laplacian( cuDoubleComplex* psi, double * gkk, int len);
void device_vtot( double* psi, double * vtot, int len);
void device_vtot( cuDoubleComplex* psi, double * vtot, int len);
void device_vtot( cuDoubleComplex* psi, cuDoubleComplex* vtot, int len);
void device_memory(void);
void device_calculate_nonlocal( double * psiUpdate, double * psi, double * NL, int * index, int * parts,  double * atom_weight, double * weight, int blocks);
void device_calculate_nonlocal( cuDoubleComplex* psiUpdate, cuDoubleComplex* psi, double * NL, int * index, int * parts,  double * atom_weight, cuDoubleComplex* weight, int blocks);
void device_teter( cuDoubleComplex* psi, double * vtot, int len);
void device_teter( cuDoubleComplex* psi, cuDoubleComplex * vtot, int len);
void device_mapping_to_buf( double * buf, double * psi, int * index, int len );
void device_mapping_to_buf( cuDoubleComplex * buf, cuDoubleComplex * psi, int * index, int len );
void device_mapping_from_buf( double * psi, double * buf, int * index, int len );
void device_mapping_from_buf( cuDoubleComplex * psi, cuDoubleComplex* buf, int * index, int len );
void device_calculate_Energy( double * psi, double * energy, int nbands, int bandLen);
void device_calculate_Energy( cuDoubleComplex* psi, double * energy, int nbands, int bandLen);
void device_batch_Scal( double * psi, double * vec, int nband, int bandLen);
void device_batch_Scal( cuDoubleComplex* psi, double * vec, int nband, int bandLen);
void device_X_Equal_AX_minus_X_eigVal( double * Xtemp, double * AX, double * X, double * eigen, int nbands, int bandLen);
void device_X_Equal_AX_minus_X_eigVal( cuDoubleComplex* Xtemp, cuDoubleComplex* AX, cuDoubleComplex* X, double * eigen, int nbands, int bandLen);
void device_init_vtot();
void device_clean_vtot();
void device_reset_vtot_flag();
void device_reset_nonlocal_flag();
void device_DMatrix_Add( double * A , double * B, int m, int n);
void device_Axpyz( double * X, double alpha, double * Y, double beta, double * Z, int length);
void device_Axpyz( cuDoubleComplex* X, double alpha, cuDoubleComplex* Y, double beta, cuDoubleComplex* Z, int length);
void device_cal_recvk( int * recvk, int * recvdisp, int width, int heightLocal, int mpisize);
void device_cal_sendk( int * sendk, int * senddispl, int widthLocal, int height, int heightBlockSize, int mpisize);
void device_hadamard_product( double * in1, double * in2, double * out, int length);
void device_set_vector( double * out, double *in, int length);
void device_set_vector( cuDoubleComplex* out, cuDoubleComplex *in, int length);
void device_decompress_f2d( double *out, float *in, int length);
void device_compress_d2f( float *out, double *in, int length);
void device_XTX( cuDoubleComplex * X, double * Y, int length);
void device_reduce( double * density, double * sum, int nbands, int bandLen);
void device_Axpyz( cuDoubleComplex * X, double alpha, cuDoubleComplex * Y, double beta, cuDoubleComplex * Z, int length, int nbands);
void device_teter( cuDoubleComplex* psi, double * vtot, int len, int nbands);
void device_vtot( cuDoubleComplex * psi, cuDoubleComplex* vtot, int len, int nbands);
void device_sync();
void device_set_vector( cuComplex* out, cuComplex *in, int length);
void device_memcpy_Async_HOST2DEVICE( void *gpu, void * cpu, size_t size );
void print_timing();
void reset_time();
//}
#endif
