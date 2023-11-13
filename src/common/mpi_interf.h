//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin 

/// @file mpi_interf.hpp
/// @brief Interface with MPI to facilitate communication.
/// @date 2012-11-03
#ifndef _MPI_INTERF_HPP_
#define _MPI_INTERF_HPP_

#include  "environment.h"

namespace scales{

/// @namespace mpi
///
/// @brief Interface with MPI to facilitate communication.
namespace mpi{

#ifdef _PROFILING_
extern double allreduceTime;
extern double bcastTime;
extern double allgatherTime;
void reset_mpi_time();
#endif
// *********************************************************************
// Allgatherv
//
// NOTE: The interface is quite preliminary.
// *********************************************************************
void Allgatherv( 
    std::vector<Int>& localVec, 
    std::vector<Int>& allVec,
    MPI_Comm          comm );


// *********************************************************************
// Send / Recv for stringstream 
//
// Isend / Irecv is not here because the size and content has to be 
// communicated separately for non-blocking communication.
// *********************************************************************

void Send( std::stringstream& sstm, Int dest, Int tagSize, Int tagContent, 
    MPI_Comm comm );

void Recv ( std::stringstream& sstm, Int src, Int tagSize, Int tagContent, 
    MPI_Comm comm, MPI_Status& statSize, MPI_Status& statContent );

void Recv ( std::stringstream& sstm, Int src, Int tagSize, Int tagContent, 
    MPI_Comm comm );


// *********************************************************************
// Waitall
// *********************************************************************

void
  Wait    ( MPI_Request& req  );

void
  Waitall ( std::vector<MPI_Request>& reqs, std::vector<MPI_Status>& stats );

void
  Waitall ( std::vector<MPI_Request>& reqs );

// *********************************************************************
// Reduce
// *********************************************************************

void
  Reduce ( Real* sendbuf, Real* recvbuf, Int count, MPI_Op op, Int root, MPI_Comm comm );

void
  Reduce ( Complex* sendbuf, Complex* recvbuf, Int count, MPI_Op op, Int root, MPI_Comm comm );

void
  Allreduce ( Int* sendbuf, Int* recvbuf, Int count, MPI_Op op, MPI_Comm comm );

void
  Allreduce ( Real* sendbuf, Real* recvbuf, Int count, MPI_Op op, MPI_Comm comm );

void
  Allreduce ( Complex* sendbuf, Complex* recvbuf, Int count, MPI_Op op, MPI_Comm comm );

// *********************************************************************
// Alltoall
// *********************************************************************

void
  Alltoallv ( Int *bufSend, Int *sizeSend, Int *displsSend, 
      Int *bufRecv, Int *sizeRecv, 
      Int *displsRecv, MPI_Comm comm );

void
  Alltoallv ( Real *bufSend, Int *sizeSend, Int *displsSend, 
      Real *bufRecv, Int *sizeRecv, 
      Int *displsRecv, MPI_Comm comm );

void
  Alltoallv ( Complex *bufSend, Int *sizeSend, Int *displsSend, 
      Complex *bufRecv, Int *sizeRecv, 
      Int *displsRecv, MPI_Comm comm );

} // namespace mpi

} // namespace scales



#endif // _MPI_INTERF_HPP_

