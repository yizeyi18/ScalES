//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

/// @file device_utility.cpp
/// @brief DEVICE_UTILITY subroutines.
/// @date 2020-08-12
#include "device_utility.h"

namespace scales{


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


}  // namespace scales
