//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin 

/// @file mpi_interf.cpp
/// @brief Inerface with MPI to facilitate communication.
/// @date 2012-11-03
#include  "mpi_interf.hpp"
#include  "utility.hpp"
namespace scales{

// *********************************************************************
// Constants
// *********************************************************************

namespace mpi{

// *********************************************************************
// Gather
// *********************************************************************
double allreduceTime = 0.0;
double bcastTime = 0.0 ;
double allgatherTime = 0.0;

void reset_mpi_time()
{
	allreduceTime = 0.0;
	bcastTime = 0.0 ;
	allgatherTime = 0.0;
}

void
  Allgatherv ( 
      std::vector<Int>& localVec, 
      std::vector<Int>& allVec,
      MPI_Comm          comm )
  {
    Int mpirank, mpisize;
    MPI_Comm_rank( comm, &mpirank );
    MPI_Comm_size( comm, &mpisize );

    Int localSize = localVec.size();
    std::vector<Int>  localSizeVec( mpisize );
    std::vector<Int>  localSizeDispls( mpisize );
    MPI_Allgather( &localSize, 1, MPI_INT, &localSizeVec[0], 1, MPI_INT, comm );
    localSizeDispls[0] = 0;
    for( Int ip = 1; ip < mpisize; ip++ ){
      localSizeDispls[ip] = localSizeDispls[ip-1] + localSizeVec[ip-1];
    }
    Int totalSize = localSizeDispls[mpisize-1] + localSizeVec[mpisize-1];

    allVec.clear();
    allVec.resize( totalSize );

    MPI_Allgatherv( &localVec[0], localSize, MPI_INT, &allVec[0], 
        &localSizeVec[0], &localSizeDispls[0], MPI_INT, comm    );


    return ;
  }        // -----  end of function Allgatherv  ----- 


// *********************************************************************
// Send / Recv
// *********************************************************************
void 
  Send( std::stringstream& sstm, Int dest, Int tagSize, Int tagContent, 
      MPI_Comm comm ){
    std::vector<char> sstr;
    sstr.resize( Size( sstm ) );
    Int sizeStm = sstr.size();
    sstm.read( &sstr[0], sizeStm );
    MPI_Send( &sizeStm, 1, MPI_INT,  dest, tagSize, comm );
    MPI_Send( (void*)&sstr[0], sizeStm, MPI_BYTE, dest, tagContent, comm );
    return; 
  } // -----  end of function Send ----- 


void
  Recv ( std::stringstream& sstm, Int src, Int tagSize, Int tagContent, 
      MPI_Comm comm, MPI_Status& statSize, MPI_Status& statContent )
  {
    std::vector<char> sstr;
    Int sizeStm;
    MPI_Recv( &sizeStm, 1, MPI_INT, src, tagSize, comm, &statSize );
    sstr.resize( sizeStm );
    MPI_Recv( (void*) &sstr[0], sizeStm, MPI_BYTE, src, tagContent, comm, &statContent );
    sstm.write( &sstr[0], sizeStm );

    return ;
  }        // -----  end of function Recv  ----- 

void
  Recv ( std::stringstream& sstm, Int src, Int tagSize, Int tagContent, 
      MPI_Comm comm )
  {
    std::vector<char> str;
    Int sizeStm;
    MPI_Recv( &sizeStm, 1, MPI_INT, src, tagSize, comm, MPI_STATUS_IGNORE );
    str.resize( sizeStm );
    MPI_Recv( (void*) &str[0], sizeStm, MPI_BYTE, src, tagContent, comm, MPI_STATUS_IGNORE );
    sstm.write( &str[0], sizeStm );

    return ;
  }        // -----  end of function Recv  ----- 



// *********************************************************************
// Wait
// *********************************************************************


void
  Wait    ( MPI_Request& req  )
  {
    MPI_Wait( &req, MPI_STATUS_IGNORE );

    return ;
  }         // -----  end of method Wait  ----- 

void
  Waitall ( std::vector<MPI_Request>& reqs, std::vector<MPI_Status>& stats )
  {
    if( reqs.size() != stats.size() ){
      ErrorHandling( "MPI_Request does not have the same as as MPI_Status." );
    }
    for( Int i = 0; i < reqs.size(); i++ ){
      MPI_Wait( &reqs[i], &stats[i] );
    }

    return ;
  }        // -----  end of function Waitall  ----- 

void
  Waitall ( std::vector<MPI_Request>& reqs )
  {
    for( Int i = 0; i < reqs.size(); i++ ){
      MPI_Wait( &reqs[i], MPI_STATUS_IGNORE );
    }

    return ;
  }        // -----  end of function Waitall  ----- 


// *********************************************************************
// Reduce
// *********************************************************************


void
  Reduce ( Real* sendbuf, Real* recvbuf, Int count, MPI_Op op, Int root, MPI_Comm comm )
  {
    MPI_Reduce( sendbuf,  recvbuf, count, MPI_DOUBLE, op, root, comm );

    return ;
  }        // -----  end of function Reduce  ----- 

void
  Reduce ( Complex* sendbuf, Complex* recvbuf, Int count, MPI_Op op, Int root, MPI_Comm comm )
  {
    MPI_Reduce( (Real*)sendbuf,  (Real*)recvbuf, 2 * count, MPI_DOUBLE, op, root, comm );

    return ;
  }        // -----  end of function Reduce  ----- 

void
  Allreduce ( Int* sendbuf, Int* recvbuf, Int count, MPI_Op op, MPI_Comm comm )
  {
#ifdef _PROFILING_
  Real timeSta, timeEnd;
  MPI_Barrier( comm );
  cuda_sync();
  GetTime( timeSta );
#endif
    MPI_Allreduce( sendbuf,  recvbuf, count, MPI_INT, 
        op, comm );

#ifdef _PROFILING_
  MPI_Barrier( comm );
  cuda_sync();
  GetTime( timeEnd );
  allreduceTime += timeEnd - timeSta;
#endif
    return ;
  }        // -----  end of function Allreduce  ----- 


void
  Allreduce ( Real* sendbuf, Real* recvbuf, Int count, MPI_Op op, MPI_Comm comm )
  {

#ifdef _PROFILING_
  Real timeSta, timeEnd;
  MPI_Barrier( comm );
  cuda_sync();
  GetTime( timeSta );
#endif

    MPI_Allreduce( sendbuf,  recvbuf, count, MPI_DOUBLE, 
        op, comm );

#ifdef _PROFILING_
  MPI_Barrier( comm );
  cuda_sync();
  GetTime( timeEnd );
  allreduceTime += timeEnd - timeSta;
#endif

    return ;
  }        // -----  end of function Allreduce  ----- 


void
  Allreduce ( Complex* sendbuf, Complex* recvbuf, Int count, MPI_Op op, MPI_Comm comm )
  {
#ifdef _PROFILING_
  Real timeSta, timeEnd;
  MPI_Barrier( comm );
  cuda_sync();
  GetTime( timeSta );
#endif

    MPI_Allreduce( (Real*)sendbuf, (Real*) recvbuf, 2*count, MPI_DOUBLE, 
        op, comm );

#ifdef _PROFILING_
  MPI_Barrier( comm );
  cuda_sync();
  GetTime( timeEnd );
  allreduceTime += timeEnd - timeSta;
#endif

    return ;
  }        // -----  end of function Allreduce  ----- 


// *********************************************************************
// Alltoall
// *********************************************************************

void
  Alltoallv ( Int *bufSend, Int *sizeSend, Int *displsSend, 
      Int *bufRecv, Int *sizeRecv, 
      Int *displsRecv, MPI_Comm comm )
  {
    MPI_Alltoallv( bufSend, sizeSend, displsSend, MPI_INT,
        bufRecv, sizeRecv, displsRecv, MPI_INT, comm );    
    return ;
  }        // -----  end of function Alltoallv  ----- 


void
  Alltoallv ( Real *bufSend, Int *sizeSend, Int *displsSend, 
      Real *bufRecv, Int *sizeRecv, 
      Int *displsRecv, MPI_Comm comm )
  {
    MPI_Alltoallv( bufSend, sizeSend, displsSend, MPI_DOUBLE,
        bufRecv, sizeRecv, displsRecv, MPI_DOUBLE, comm );    
    return ;
  }        // -----  end of function Alltoallv  ----- 

void
  Alltoallv ( Complex *bufSend, Int *sizeSend, Int *displsSend, 
      Complex *bufRecv, Int *sizeRecv, 
      Int *displsRecv, MPI_Comm comm )
  {
    Int mpisize; 
    MPI_Comm_size( comm, &mpisize );
    std::vector<Int> dblSizeSend( mpisize );
    std::vector<Int> dblDisplsSend( mpisize ); 
    std::vector<Int> dblSizeRecv( mpisize );
    std::vector<Int> dblDisplsRecv( mpisize );

    for( Int ip = 0; ip < mpisize; ip++ ){
      dblSizeSend[ip] = 2 * sizeSend[ip];
      dblSizeRecv[ip] = 2 * sizeRecv[ip];
      dblDisplsSend[ip] = 2 * displsSend[ip];
      dblDisplsRecv[ip] = 2 * displsRecv[ip];
    }

    MPI_Alltoallv( 
        (Real*)bufSend, &dblSizeSend[0], &dblDisplsSend[0], MPI_DOUBLE, 
        (Real*)bufRecv, &dblSizeRecv[0], &dblDisplsRecv[0], MPI_DOUBLE, comm );    

    return ;
  }        // -----  end of function Alltoallv  ----- 

} // namespace mpi

} // namespace scales
