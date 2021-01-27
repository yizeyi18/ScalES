//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

/// @file device_fft.cpp
/// @brief Sequential and Distributed device_fft wrapper.
/// @date 2020-08-12
#ifndef _DEVICE_FFT_HPP_
#define _DEVICE_FFT_HPP_

#include  "device_utils.h"
#include  "environment.hpp"
#include  "domain.hpp"
#include  "numvec_decl.hpp"
#include  "device_numvec_impl.hpp"
#include  "fourier.hpp"
#include  "assert.h"
#include  <cuda_runtime.h>

extern  cufftHandle cuPlanR2C[NSTREAM];
extern  cufftHandle cuPlanR2CFine[NSTREAM];
extern  cufftHandle cuPlanC2R[NSTREAM];
extern  cufftHandle cuPlanC2RFine[NSTREAM];
extern  cufftHandle cuPlanC2CFine[NSTREAM];
extern  cufftHandle cuPlanC2C[NSTREAM];

namespace scales{

// *********************************************************************
// NVIDIA GPU FFT Thin interface
// *********************************************************************
  namespace device_fft{

  void Init(const Domain & dm);
  void Destroy(); 
}

  void deviceFFTExecuteInverse( Fourier& fft, cufftHandle &plan, int fft_type, deviceCpxNumVec &cu_psi_in, deviceCpxNumVec &cu_psi_out );
  void deviceFFTExecuteInverse( Fourier& fft, cufftHandle &plan, int fft_type, deviceCpxNumVec &cu_psi_in, deviceCpxNumVec &cu_psi_out , int nbands);
  void deviceFFTExecuteInverse2( Fourier& fft, cufftHandle &plan, int fft_type, deviceCpxNumVec &cu_psi_in, deviceCpxNumVec &cu_psi_out );
  void deviceFFTExecuteForward( Fourier& fft, cufftHandle &plan, int fft_type, deviceCpxNumVec &cu_psi_in, deviceCpxNumVec &cu_psi_out );
  void deviceFFTExecuteForward2( Fourier& fft, cufftHandle &plan, int fft_type, deviceCpxNumVec &cu_psi_in, deviceCpxNumVec &cu_psi_out );
  void deviceFFTExecuteInverse( Fourier& fft, cufftHandle &plan, int fft_type, deviceDblNumVec &cu_psi_in, deviceDblNumVec &cu_psi_out );
  void deviceFFTExecuteForward( Fourier& fft, cufftHandle &plan, int fft_type, deviceDblNumVec &cu_psi_in, deviceDblNumVec &cu_psi_out );

    
} // namespace scales


#endif // _DEIVCE_FFT_HPP_
