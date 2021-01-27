//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

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
const char *deviceBLASGetErrorString(cublasStatus_t error);
const char *deviceFFTGetErrorString(cufftResult error);
#endif
#endif
/******************************************************/
