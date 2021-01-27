//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin 

/// @file global.cpp
/// @brief Global structure.
/// @date 2012-08-01
/// @date 2016-07-12 Update to the coredumper model for debugging
#include    "environment.hpp"
#include    "esdf.hpp"
#include    "periodtable.hpp"
#ifdef _COREDUMPER_
#include <google/coredumper.h>
#endif

namespace scales{

// *********************************************************************
// Input parameters
// *********************************************************************
namespace esdf{
ESDFInputParam  esdfParam;
}

// *********************************************************************
// IO
// *********************************************************************
std::ofstream  statusOFS;

// *********************************************************************
// Error handling
// *********************************************************************
void ErrorHandling( const char * msg ){
  statusOFS << std::endl << "ERROR!" << std::endl 
    << msg << std::endl << std::endl;
#ifdef _COREDUMPER_
  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
  char filename[100];
  sprintf(filename, "core_%d_%d", mpirank, mpisize);

  if( WriteCoreDump(filename) ==0 ) {   
    statusOFS << "success: WriteCoreDump to " << filename << std::endl;
  } else {  
    statusOFS << "failed:  WriteCoreDump to " << filename << std::endl;
  }     
#endif // #ifdef _COREDUMPER_
  throw std::runtime_error( msg );
}


} // namespace scales
