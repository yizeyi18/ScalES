/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Weile Jia

This file is part of ScalES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
(2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to Lawrence Berkeley National
Laboratory, without imposing a separate written license agreement for such
Enhancements, then you hereby grant the following license: a non-exclusive,
royalty-free perpetual license to install, use, modify, prepare derivative
works, incorporate into other computer software, distribute, and sublicense
such enhancements or derivative works thereof, in binary and source code form.
 */
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
