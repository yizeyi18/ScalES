/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Weile Jia

This file is part of DGDFT. All rights reserved.

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
/// @file device_utility.cpp
/// @brief DEVICE_UTILITY subroutines.
/// @date 2020-08-12
#include "device_utility.hpp"

namespace dgdft{


void device_AlltoallForward( deviceDblNumMat& cu_A, deviceDblNumMat& cu_B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = cu_A.m();
  Int widthTemp = cu_A.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }
  
  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }
  
  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  deviceIntNumMat  cu_sendk( height, widthLocal );
  deviceIntNumMat  cu_recvk( heightLocal, width );
  deviceIntNumVec  cu_senddispls(mpisize);
  deviceIntNumVec  cu_recvdispls(mpisize);
  deviceDblNumVec  cu_recvbuf(heightLocal*width);
  deviceDblNumVec  cu_sendbuf(height*widthLocal); 

  cu_senddispls.CopyFrom( senddispls );
  cu_recvdispls.CopyFrom( recvdispls );
 
  device_cal_sendk( cu_sendk.Data(), cu_senddispls.Data(), widthLocal, height, heightBlocksize, mpisize );
  device_cal_recvk( cu_recvk.Data(), cu_recvdispls.Data(), width, heightLocal, mpisize ); 

  device_mapping_to_buf( cu_sendbuf.Data(), cu_A.Data(), cu_sendk.Data(), height*widthLocal);
  cu_sendbuf.CopyTo( sendbuf );
  
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, comm );

  cu_recvbuf.CopyFrom( recvbuf );
  device_mapping_from_buf(cu_B.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);
 

  return ;
}        // -----  end of function device_AlltoallForward ----- 


void device_AlltoallBackward( deviceDblNumMat& cu_A, deviceDblNumMat& cu_B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = cu_B.m();
  Int widthTemp = cu_B.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  deviceIntNumMat  cu_sendk( height, widthLocal );
  deviceIntNumMat  cu_recvk( heightLocal, width );
  deviceIntNumVec  cu_senddispls(mpisize);
  deviceIntNumVec  cu_recvdispls(mpisize);
  deviceDblNumVec  cu_recvbuf(heightLocal*width);
  deviceDblNumVec  cu_sendbuf(height*widthLocal); 

  cu_senddispls.CopyFrom( senddispls );
  cu_recvdispls.CopyFrom( recvdispls );
 
  device_cal_sendk( cu_sendk.Data(), cu_senddispls.Data(), widthLocal, height, heightBlocksize, mpisize );
  device_cal_recvk( cu_recvk.Data(), cu_recvdispls.Data(), width, heightLocal, mpisize ); 

  device_mapping_to_buf( cu_recvbuf.Data(), cu_A.Data(), cu_recvk.Data(), heightLocal*width);
  cu_recvbuf.CopyTo( recvbuf );
  
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, comm );

  cu_sendbuf.CopyFrom( sendbuf );
  device_mapping_from_buf(cu_B.Data(), cu_sendbuf.Data(), cu_sendk.Data(), height*widthLocal);
  
  return ;
}        // -----  end of function device_AlltoallBackward ----- 


}  // namespace dgdft
