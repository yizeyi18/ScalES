// File: device_errors.c
// CUBLAS and CUFFT error checking.

// includes standard headers

#ifdef DEVICE 

#ifndef _DEVICE_ERRORS_HPP__
#define _DEVICE_ERRORS_HPP__
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cufft.h>

/******************************************************/
// CUBLAS and CUFFT error checking, in library

// returns string for CUBLAS API error
char *deviceBLASGetErrorString(cublasStatus_t error);
char *deviceFFTGetErrorString(cufftResult error);
#endif
#endif
/******************************************************/
