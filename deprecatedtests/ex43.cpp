/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin

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
/// @file ex43.cpp
/// @brief Testing the multi-threaded version of many fftws using
/// OpenMP. Also test the alignment
/// @date 2016-04-24
#include<iostream>
#include<complex>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
// FFTW libraries
#include <fftw3.h>
#include <fftw3-mpi.h>
#include "mpi.h"
#include <omp.h>

int FFTWInit(){

  std::cout << "FFTW uses 1 thread only." << std::endl;

  return 0;
}

int main(int argc, char **argv) 
{
  MPI_Init(&argc, &argv);
  // FIXME
  //    int provided, threads_ok;
  //    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  //    threads_ok = (provided >= MPI_THREAD_FUNNELED);
  //    std::cout << "threads_ok = " << threads_ok << std::endl;

  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
  double timeSta, timeEnd;


  int N = 96;
  int NtotR = N*N*N;
  int NtotC = (N/2+1)*N*N;
  int howmany = 100;
  std::vector<double> a1(NtotR*howmany);
  std::vector<std::complex<double> > a2(NtotC*howmany);
  for( int i = 0; i < NtotR*howmany; i++ ){
    a1[i] = drand48();
  }

  FFTWInit();
  // unsigned plannerFlag = FFTW_MEASURE;
  unsigned plannerFlag = FFTW_ESTIMATE;
  std::vector<int> nR2C(3);
  nR2C[0] = N;
  nR2C[1] = N;
  nR2C[2] = N;


  fftw_plan forwardPlanR2C = fftw_plan_dft_r2c_3d( 
      N, N, N, ( &a1[0] ), 
      reinterpret_cast<fftw_complex*>( &a2[0] ),
      plannerFlag );
  fftw_plan backwardPlanR2C = fftw_plan_dft_c2r_3d(
      N, N, N,
      reinterpret_cast<fftw_complex*>( &a2[0] ),
      &a1[0],
      plannerFlag);

  timeSta = MPI_Wtime();
#pragma omp parallel
  {
    // Private arrays
    double* b1;
    fftw_complex* b2;
    b1 = fftw_alloc_real(NtotR);
    b2 = fftw_alloc_complex(NtotC);
#pragma omp for schedule (dynamic,1) nowait
    for( int i = 0; i < howmany; i++ ){
      for( int j = 0; j < NtotR; j++ ){
        b1[j] = a1[j+i*NtotR];
      }
      fftw_execute_dft_r2c( forwardPlanR2C,
          &b1[0],
          &b2[0] );
      fftw_execute_dft_c2r( backwardPlanR2C,
          &b2[0],
          &b1[0] );
      for( int j = 0; j < NtotR; j++ ){
        a1[j+i*NtotR] = b1[j];
      }
    }
    fftw_free(b1);
    fftw_free(b2);
  } // end omp
  timeEnd = MPI_Wtime();
  std::cout << "Time for FFT is " << timeEnd - timeSta << std::endl;


  fftw_destroy_plan( backwardPlanR2C );
  fftw_destroy_plan( forwardPlanR2C );

  // Finalize 
  MPI_Finalize();

  return 0;
}
