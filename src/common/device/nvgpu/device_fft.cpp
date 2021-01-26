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

#include  "domain.hpp"
#include  "device_fft.hpp"
#include  "fourier.hpp"
#include  "blas.hpp"
#include  "esdf.hpp"
#include "device_blas.hpp"
#include "mpi_interf.hpp"

cufftHandle cuPlanR2C[NSTREAM];
cufftHandle cuPlanR2CFine[NSTREAM];
cufftHandle cuPlanC2R[NSTREAM];
cufftHandle cuPlanC2RFine[NSTREAM];
cufftHandle cuPlanC2CFine[NSTREAM];
cufftHandle cuPlanC2C[NSTREAM];

namespace scales{

using namespace scales::esdf;


namespace device_fft{

  void Init(const Domain & dm){

    Int i;
    Domain  domain;
    domain = dm;
    Index3& numGrid = domain.numGrid;
    Index3& numGridFine = domain.numGridFine;

#if ( _DEBUGlevel_ >= 1 )
    std::cout << "init the R2C cufftPlan: "<< numGrid[2] << " " << numGrid[1] <<" " << numGrid[0]<<std::endl;
#endif
    for(i = 0; i < NSTREAM; i++)
    {
      cufftPlan3d(&cuPlanR2C[i], numGrid[2], numGrid[1], numGrid[0], CUFFT_D2Z);
      cufftPlan3d(&cuPlanC2R[i], numGrid[2], numGrid[1], numGrid[0], CUFFT_Z2D);
      cufftPlan3d(&cuPlanC2C[i], numGrid[2], numGrid[1], numGrid[0], CUFFT_Z2Z);

      cufftPlan3d(&cuPlanR2CFine[i], numGridFine[2], numGridFine[1], numGridFine[0], CUFFT_D2Z);
      cufftPlan3d(&cuPlanC2RFine[i], numGridFine[2], numGridFine[1], numGridFine[0], CUFFT_Z2D);
      cufftPlan3d(&cuPlanC2CFine[i], numGridFine[2], numGridFine[1], numGridFine[0], CUFFT_Z2Z);
    }
  }
  
  void Destroy() {

#if ( _DEBUGlevel_ >= 1 )
    std::cout << "Destroy cufftPlan...... "<< std::endl;
#endif

    Int i;
    for(i =0; i < NSTREAM; i++)
    {
       cufftDestroy(cuPlanR2C[i]);
       cufftDestroy(cuPlanR2CFine[i]);
       cufftDestroy(cuPlanC2R[i]);
       cufftDestroy(cuPlanC2RFine[i]);
       cufftDestroy(cuPlanC2C[i]);
       cufftDestroy(cuPlanC2CFine[i]);
    }
  }

}

void deviceFFTExecuteForward2( Fourier& fft, cufftHandle &plan, int fft_type, deviceCpxNumVec &cu_psi_in, deviceCpxNumVec &cu_psi_out )
{
   Index3& numGrid = fft.domain.numGrid;
   Index3& numGridFine = fft.domain.numGridFine;
   Real vol      = fft.domain.Volume();
   Int ntot      = fft.domain.NumGridTotal();
   Int ntotFine  = fft.domain.NumGridTotalFine();
   Real factor;
   if(fft_type > 0) // fine grid FFT.
   {
      factor = vol/ntotFine;
      assert( cufftExecZ2Z(plan, cu_psi_in.Data(), cu_psi_out.Data(), CUFFT_FORWARD)  == CUFFT_SUCCESS );
      DEVICE_BLAS::Scal( ntotFine, &factor, cu_psi_out.Data(),1); 
   }
   else // coarse grid FFT.
   {
      factor = vol/ntot;
      assert( cufftExecZ2Z(plan, cu_psi_in.Data(), cu_psi_out.Data(), CUFFT_FORWARD)  == CUFFT_SUCCESS );
      DEVICE_BLAS::Scal(ntot, &factor, cu_psi_out.Data(), 1); 
   }
}

void deviceFFTExecuteForward( Fourier& fft, cufftHandle &plan, int fft_type, deviceCpxNumVec &cu_psi_in, deviceCpxNumVec &cu_psi_out )
{
   Index3& numGrid = fft.domain.numGrid;
   Index3& numGridFine = fft.domain.numGridFine;
   Real vol      = fft.domain.Volume();
   Int ntot      = fft.domain.NumGridTotal();
   Int ntotFine  = fft.domain.NumGridTotalFine();
   Real factor;
   if(fft_type > 0) // fine grid FFT.
   {
      factor = vol/ntotFine;
      assert( cufftExecZ2Z(plan, cu_psi_in.Data(), cu_psi_out.Data(), CUFFT_FORWARD)  == CUFFT_SUCCESS );
      DEVICE_BLAS::Scal( ntotFine, &factor, cu_psi_out.Data(),1); 
   }
   else // coarse grid FFT.
   {
      //factor = vol/ntot;
      assert( cufftExecZ2Z(plan, cu_psi_in.Data(), cu_psi_out.Data(), CUFFT_FORWARD)  == CUFFT_SUCCESS );
      //DEVICE_BLAS::Scal(ntot, &factor, cu_psi_out.Data(), 1); 
   }
}
void deviceFFTExecuteInverse( Fourier& fft, cufftHandle &plan, int fft_type, deviceCpxNumVec &cu_psi_in, deviceCpxNumVec &cu_psi_out , int nbands)
{
   Index3& numGrid = fft.domain.numGrid;
   Index3& numGridFine = fft.domain.numGridFine;
   Real vol      = fft.domain.Volume();
   Int ntot      = fft.domain.NumGridTotal();
   Int ntotFine  = fft.domain.NumGridTotalFine();
   Real factor;
   if(fft_type > 0) // fine grid FFT.
   {
      factor = 1.0 / vol;
      assert( cufftExecZ2Z(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data(), CUFFT_INVERSE) == CUFFT_SUCCESS );
      DEVICE_BLAS::Scal(ntotFine, &factor, cu_psi_out.Data(),1); 
   }
   else // coarse grid FFT.
   {
      //factor = 1.0 / vol;
      factor = 1.0 / Real(ntot*nbands);
      assert( cufftExecZ2Z(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data(), CUFFT_INVERSE) == CUFFT_SUCCESS );
      DEVICE_BLAS::Scal(nbands*ntot, &factor, cu_psi_out.Data(), 1); 
   }
}

void deviceFFTExecuteInverse( Fourier& fft, cufftHandle &plan, int fft_type, deviceCpxNumVec &cu_psi_in, deviceCpxNumVec &cu_psi_out )
{
   Index3& numGrid = fft.domain.numGrid;
   Index3& numGridFine = fft.domain.numGridFine;
   Real vol      = fft.domain.Volume();
   Int ntot      = fft.domain.NumGridTotal();
   Int ntotFine  = fft.domain.NumGridTotalFine();
   Real factor;
   if(fft_type > 0) // fine grid FFT.
   {
      factor = 1.0 / vol;
      assert( cufftExecZ2Z(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data(), CUFFT_INVERSE) == CUFFT_SUCCESS );
      DEVICE_BLAS::Scal(ntotFine, &factor, cu_psi_out.Data(),1); 
   }
   else // coarse grid FFT.
   {
      //factor = 1.0 / vol;
      factor = 1.0 / Real(ntot);
      assert( cufftExecZ2Z(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data(), CUFFT_INVERSE) == CUFFT_SUCCESS );
      DEVICE_BLAS::Scal(ntot, &factor, cu_psi_out.Data(), 1); 
   }
}
void deviceFFTExecuteInverse2( Fourier& fft, cufftHandle &plan, int fft_type, deviceCpxNumVec &cu_psi_in, deviceCpxNumVec &cu_psi_out )
{
   Index3& numGrid = fft.domain.numGrid;
   Index3& numGridFine = fft.domain.numGridFine;
   Real vol      = fft.domain.Volume();
   Int ntot      = fft.domain.NumGridTotal();
   Int ntotFine  = fft.domain.NumGridTotalFine();
   Real factor;
   if(fft_type > 0) // fine grid FFT.
   {
      factor = 1.0 / vol;
      assert( cufftExecZ2Z(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data(), CUFFT_INVERSE) == CUFFT_SUCCESS );
      DEVICE_BLAS::Scal(ntotFine, &factor, cu_psi_out.Data(),1); 
   }
   else // coarse grid FFT.
   {
      //factor = 1.0 / vol;
      //factor = 1.0 / Real(ntot);
      assert( cufftExecZ2Z(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data(), CUFFT_INVERSE) == CUFFT_SUCCESS );
      //DEVICE_BLAS::Scal(ntot, &factor, cu_psi_out.Data(), 1); 
   }
}



void deviceFFTExecuteForward( Fourier& fft, cufftHandle &plan, int fft_type, deviceDblNumVec &cu_psi_in, deviceDblNumVec &cu_psi_out )
{
   Index3& numGrid = fft.domain.numGrid;
   Index3& numGridFine = fft.domain.numGridFine;
   Real vol      = fft.domain.Volume();
   Int ntot      = fft.domain.NumGridTotal();
   Int ntotFine  = fft.domain.NumGridTotalFine();
   Int ntotR2C = (numGrid[0]/2+1) * numGrid[1] * numGrid[2];
   Int ntotR2CFine = (numGridFine[0]/2+1) * numGridFine[1] * numGridFine[2];
   Real factor;
   if(fft_type > 0) // fine grid FFT.
   {
      factor = vol/ntotFine;
      assert( cufftExecD2Z(plan, cu_psi_in.Data(), reinterpret_cast<cuDoubleComplex*> (cu_psi_out.Data())) == CUFFT_SUCCESS );
      DEVICE_BLAS::Scal(2*ntotR2CFine, &factor, cu_psi_out.Data(),1); 
   }
   else // coarse grid FFT.
   {
      factor = vol/ntot;
      assert( cufftExecD2Z(plan, cu_psi_in.Data(), reinterpret_cast<cuDoubleComplex*> (cu_psi_out.Data())) == CUFFT_SUCCESS );
      DEVICE_BLAS::Scal(2*ntotR2C, &factor, cu_psi_out.Data(), 1); 
   }
}
void deviceFFTExecuteInverse( Fourier& fft, cufftHandle &plan, int fft_type, deviceDblNumVec &cu_psi_in, deviceDblNumVec &cu_psi_out )
{
   Index3& numGrid = fft.domain.numGrid;
   Index3& numGridFine = fft.domain.numGridFine;
   Real vol      = fft.domain.Volume();
   Int ntot      = fft.domain.NumGridTotal();
   Int ntotFine  = fft.domain.NumGridTotalFine();
   Int ntotR2C = (numGrid[0]/2+1) * numGrid[1] * numGrid[2];
   Int ntotR2CFine = (numGridFine[0]/2+1) * numGridFine[1] * numGridFine[2];
   Real factor;
   if(fft_type > 0) // fine grid FFT.
   {
      factor = 1.0 / vol;
      assert( cufftExecZ2D(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data()) == CUFFT_SUCCESS );
      DEVICE_BLAS::Scal(ntotFine, &factor, cu_psi_out.Data(),1); 
   }
   else // coarse grid FFT.
   {
      factor = 1.0 / vol;
      assert( cufftExecZ2D(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data()) == CUFFT_SUCCESS );
      DEVICE_BLAS::Scal(ntot, &factor, cu_psi_out.Data(), 1); 
   }
}




} // namespace scales
